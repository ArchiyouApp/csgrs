//! Edge projection with BVH-accelerated hidden-line removal (HLR).
//!
//! Inspired by [`three-edge-projection`](https://github.com/gkjohnson/three-edge-projection).
//!
//! # Pipeline
//! 1. Triangulate the mesh and record each edge's adjacent face normals.
//! 2. Classify edges as **boundary**, **silhouette**, or **feature/crease**.
//!    Back-facing smooth interior edges are discarded.
//! 3. For each surviving edge, sample `n_samples` positions along it, cast a
//!    ray toward the viewer against all occluder TriMeshes, and tag each sample
//!    visible/hidden.
//! 4. Consecutive same-visibility samples are merged into projected polylines.

use std::collections::HashMap;
use std::fmt::Debug;

use nalgebra::{Point3, Vector3};

use crate::float_types::{
    parry3d::query::RayCast,
    rapier3d::prelude::{Ray, TriMesh},
    {Real, tolerance},
};
use crate::mesh::Mesh;

// ─── public result types ─────────────────────────────────────────────────────

/// A polyline of 3-D points that lie on the projection plane.
pub type Polyline3D = Vec<Point3<Real>>;

/// Output of [`Mesh::project_edges`].
#[derive(Debug, Clone, Default)]
pub struct EdgeProjectionResult {
    /// Polylines whose samples are unoccluded (visible to the viewer).
    pub visible_polylines: Vec<Polyline3D>,
    /// Polylines whose samples are occluded (hidden behind other geometry).
    pub hidden_polylines: Vec<Polyline3D>,
}

/// Output of [`Mesh::project_edges_section`].
#[cfg(feature = "sketch")]
#[derive(Debug, Clone)]
pub struct SectionElevationResult<S: Clone + Debug + Send + Sync> {
    /// 2-D sketch of the cut outline produced by slicing the mesh.
    pub cut: crate::sketch::Sketch<S>,
    /// Visible projected edge polylines.
    pub visible_polylines: Vec<Polyline3D>,
    /// Hidden projected edge polylines.
    pub hidden_polylines: Vec<Polyline3D>,
}

// ─── Mesh impl ───────────────────────────────────────────────────────────────

impl<S: Clone + Send + Sync + Debug> Mesh<S> {
    /// Project silhouette, boundary, and feature edges onto a plane with
    /// full BVH-accelerated hidden-line removal.
    ///
    /// # Parameters
    /// - `view_normal` — direction toward the viewer (used for silhouette
    ///   classification and as the ray direction for HLR).
    /// - `plane_origin` / `plane_normal` — defines the projection plane.
    ///   Both endpoints of each edge are projected onto this plane before
    ///   chaining into polylines.
    /// - `feature_angle_deg` — minimum dihedral angle (degrees) between
    ///   adjacent face normals for an edge to be considered a feature crease.
    ///   Typical value: `15.0` (matches three-edge-projection default).
    /// - `n_samples` — number of visibility samples per edge.
    ///   More samples give finer HLR at the cost of more ray casts.
    /// - `occluders` — additional meshes that can occlude edges of `self`.
    ///   The mesh itself is always included as an occluder.
    pub fn project_edges(
        &self,
        view_normal: &Vector3<Real>,
        plane_origin: &Point3<Real>,
        plane_normal: &Vector3<Real>,
        feature_angle_deg: Real,
        n_samples: usize,
        occluders: &[&Mesh<S>],
    ) -> EdgeProjectionResult {
        // Build TriMesh for self + all additional occluders.
        let mut trimeshes: Vec<TriMesh> = Vec::new();
        if let Some(t) = self.to_trimesh() {
            trimeshes.push(t);
        }
        for m in occluders {
            if let Some(t) = m.to_trimesh() {
                trimeshes.push(t);
            }
        }

        let view_dir = view_normal.normalize();
        let plane_n = plane_normal.normalize();
        let feature_thresh =
            (feature_angle_deg * std::f64::consts::PI as Real / 180.0).max(0.0);
        let n = n_samples.max(2);

        // Pre-compute a `far_dist` large enough to start rays beyond the entire
        // scene along the view direction.  Used by the back-to-front HLR casts.
        let far_dist: Real = {
            let mut max_d: Real = 1.0;
            for tm in &trimeshes {
                for v in tm.vertices() {
                    let d = v.coords.dot(&view_dir).abs();
                    if d > max_d { max_d = d; }
                }
            }
            max_d * 4.0 + 1.0
        };

        let raw_edges = extract_edges(self);
        // Merge collinear edge segments that have the same face-normal signature.
        // BSP splitting creates extra vertices on existing edges (e.g. a cube edge
        // split at an intersecting BSP plane), producing collinear sub-edges that
        // would otherwise appear as multiple separate polylines.
        let edges = merge_collinear_edges(raw_edges);
        let mut result = EdgeProjectionResult::default();

        for (_key, edge) in &edges {
            // Skip degenerate edges (zero-length or near-zero after merging)
            if (edge.v1 - edge.v0).norm() < 1e-9 {
                continue;
            }
            if !should_keep_edge(&edge.face_normals, &view_dir, feature_thresh) {
                continue;
            }

            let vis = hlr_sample_edge(&edge.v0, &edge.v1, &trimeshes, &view_dir, n, far_dist);

            let proj_v0 = project_point(&edge.v0, plane_origin, &plane_n);
            let proj_v1 = project_point(&edge.v1, plane_origin, &plane_n);

            chain_segments(&vis, &proj_v0, &proj_v1, &mut result);
        }

        result
    }

    /// Slice the mesh at `section_plane` and project its visible edges.
    ///
    /// Returns the cut `Sketch`, visible edge polylines, and hidden edge
    /// polylines, suitable for architectural section-elevation drawings.
    #[cfg(feature = "sketch")]
    pub fn project_edges_section(
        &self,
        section_normal: &Vector3<Real>,
        section_offset: Real,
        view_normal: &Vector3<Real>,
        plane_origin: &Point3<Real>,
        plane_normal: &Vector3<Real>,
        feature_angle_deg: Real,
        n_samples: usize,
        occluders: &[&Mesh<S>],
    ) -> SectionElevationResult<S> {
        use crate::mesh::plane::Plane;
        let cut_plane = Plane::from_normal(*section_normal, section_offset);
        let cut = self.slice(cut_plane);
        let edge_result = self.project_edges(
            view_normal,
            plane_origin,
            plane_normal,
            feature_angle_deg,
            n_samples,
            occluders,
        );
        SectionElevationResult {
            cut,
            visible_polylines: edge_result.visible_polylines,
            hidden_polylines: edge_result.hidden_polylines,
        }
    }
}

// ─── internal record ─────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct EdgeRecord {
    v0: Point3<Real>,
    v1: Point3<Real>,
    /// Normals of the faces adjacent to this edge.
    face_normals: Vec<Vector3<Real>>,
}

// Canonical edge key: two quantised vertex triples in lexicographic order.
type EdgeKey = (i64, i64, i64, i64, i64, i64);

/// Quantise a coordinate to a stable integer (1 µm grid).
#[inline]
fn q(x: Real) -> i64 {
    (x * 1_000_000.0).round() as i64
}

#[inline]
fn vkey(p: &Point3<Real>) -> (i64, i64, i64) {
    (q(p.x), q(p.y), q(p.z))
}

// ─── helpers: edge extraction ─────────────────────────────────────────────────

/// Extract edges from polygon **boundaries** (not triangulation edges).
///
/// Using polygon boundary edges rather than triangle edges prevents collinear
/// interior triangulation edges — commonly introduced by CSG operations — from
/// appearing as separate feature edges and inflating the edge count.
///
/// Each polygon contributes:
/// - Its outer boundary edges (consecutive vertex pairs, wrapping around).
/// - Each hole's boundary edges.
///
/// The polygon's face normal is recorded against every one of its edges so that
/// `should_keep_edge` can classify silhouette / feature / back-facing edges.
fn extract_edges<S: Clone + Debug + Send + Sync>(
    mesh: &Mesh<S>,
) -> HashMap<EdgeKey, EdgeRecord> {
    let mut edges: HashMap<EdgeKey, EdgeRecord> = HashMap::new();

    for poly in &mesh.polygons {
        // Use the polygon's own plane normal (already computed and normalised).
        let face_normal: Vector3<Real> = poly.plane.normal().normalize();

        // Collect all boundary rings: outer vertices + each hole.
        let rings: Vec<&Vec<crate::vertex::Vertex>> =
            std::iter::once(&poly.vertices)
                .chain(poly.holes.iter())
                .collect();

        for ring in rings {
            let n = ring.len();
            if n < 2 { continue; }
            for i in 0..n {
                let a = ring[i].position;
                let b = ring[(i + 1) % n].position;
                let (ka, kb) = (vkey(&a), vkey(&b));
                if ka == kb { continue; } // zero-length edge
                let key: EdgeKey = if ka <= kb {
                    (ka.0, ka.1, ka.2, kb.0, kb.1, kb.2)
                } else {
                    (kb.0, kb.1, kb.2, ka.0, ka.1, ka.2)
                };
                let rec = edges.entry(key).or_insert_with(|| EdgeRecord {
                    v0: a,
                    v1: b,
                    face_normals: Vec::new(),
                });
                rec.face_normals.push(face_normal);
            }
        }
    }

    edges
}

// ─── helpers: collinear edge merging ─────────────────────────────────────────

/// Quantise a direction vector to a canonical signed integer triple.
/// Two edges are collinear iff their direction vectors (or negations) have the
/// same canonical key.  The sign convention: first non-zero component is positive.
#[inline]
fn dir_key(d: &Vector3<Real>) -> (i64, i64, i64) {
    // Normalise then quantise to 1e-4 grid.
    let len = d.norm();
    if len < 1e-12 {
        return (0, 0, 0);
    }
    let n = d / len;
    let ix = (n.x * 10_000.0).round() as i64;
    let iy = (n.y * 10_000.0).round() as i64;
    let iz = (n.z * 10_000.0).round() as i64;
    // Make canonical: flip so first non-zero component is positive.
    if ix < 0 || (ix == 0 && iy < 0) || (ix == 0 && iy == 0 && iz < 0) {
        (-ix, -iy, -iz)
    } else {
        (ix, iy, iz)
    }
}

/// Normal signature for an edge: sorted pair of quantised normal keys.
/// Used to group edges that lie between the same pair of geometric faces.
type NormalSig = ((i64, i64, i64), (i64, i64, i64));

fn normal_sig(face_normals: &[Vector3<Real>]) -> NormalSig {
    if face_normals.is_empty() {
        return ((0, 0, 0), (0, 0, 0));
    }
    let k0 = {
        let n = face_normals[0];
        let s = if n.x < 0.0 || (n.x == 0.0 && n.y < 0.0) || (n.x == 0.0 && n.y == 0.0 && n.z < 0.0) { -1.0 } else { 1.0 };
        ((n.x * s * 1000.0).round() as i64, (n.y * s * 1000.0).round() as i64, (n.z * s * 1000.0).round() as i64)
    };
    if face_normals.len() < 2 {
        let mut pair = (k0, k0);
        if pair.0 > pair.1 { std::mem::swap(&mut pair.0, &mut pair.1); }
        return pair;
    }
    let k1 = {
        let n = face_normals[1];
        let s = if n.x < 0.0 || (n.x == 0.0 && n.y < 0.0) || (n.x == 0.0 && n.y == 0.0 && n.z < 0.0) { -1.0 } else { 1.0 };
        ((n.x * s * 1000.0).round() as i64, (n.y * s * 1000.0).round() as i64, (n.z * s * 1000.0).round() as i64)
    };
    let mut pair = (k0, k1);
    if pair.0 > pair.1 { std::mem::swap(&mut pair.0, &mut pair.1); }
    pair
}

/// Merge collinear consecutive edge segments that share the same face-normal signature.
///
/// After CSG/BSP operations, a single logical edge (e.g. a cube edge) may be split into
/// several collinear segments by the BSP planes of the operand mesh.  This function
/// stitches them back together so each logical edge appears as one `EdgeRecord`.
fn merge_collinear_edges(
    edges: HashMap<EdgeKey, EdgeRecord>,
) -> HashMap<EdgeKey, EdgeRecord> {
    // ── 1. Group edges by (direction, normal_signature) ───────────────────
    // key: (dir_key, normal_sig)
    type GroupKey = ((i64, i64, i64), NormalSig);
    let mut groups: HashMap<GroupKey, Vec<EdgeRecord>> = HashMap::new();

    for (_, rec) in edges {
        let dir = rec.v1 - rec.v0;
        let dk = dir_key(&dir);
        let ns = normal_sig(&rec.face_normals);
        groups.entry((dk, ns)).or_default().push(rec);
    }

    let mut result: HashMap<EdgeKey, EdgeRecord> = HashMap::new();

    for ((dk, _ns), group) in groups {
        if dk == (0, 0, 0) {
            // degenerate
            for rec in group {
                let (ka, kb) = (vkey(&rec.v0), vkey(&rec.v1));
                let key = if ka <= kb { (ka.0,ka.1,ka.2,kb.0,kb.1,kb.2) } else { (kb.0,kb.1,kb.2,ka.0,ka.1,ka.2) };
                result.insert(key, rec);
            }
            continue;
        }

        // ── 2. Build adjacency: endpoint → list of edge indices ───────────
        // Only edges that are actually collinear (same direction key) are in this group.
        // We want to chain them: find sequences where v1 of one = v0 of next.

        // Build a map from quantised vertex key to edge index (both endpoints)
        let mut endpoint_map: HashMap<(i64,i64,i64), Vec<usize>> = HashMap::new();
        for (i, rec) in group.iter().enumerate() {
            endpoint_map.entry(vkey(&rec.v0)).or_default().push(i);
            endpoint_map.entry(vkey(&rec.v1)).or_default().push(i);
        }

        let mut used = vec![false; group.len()];

        // ── 3. Walk chains from endpoints (vertices with degree == 1) ─────
        // A chain start is a vertex that appears in only one edge of this group.
        let chain_starts: Vec<usize> = (0..group.len())
            .filter(|&i| {
                let k0 = vkey(&group[i].v0);
                let k1 = vkey(&group[i].v1);
                endpoint_map[&k0].len() == 1 || endpoint_map[&k1].len() == 1
            })
            .collect();

        // Helper: walk from an edge in a direction to build the merged span.
        let walk_chain = |start_idx: usize, used: &mut Vec<bool>, group: &[EdgeRecord], ep_map: &HashMap<(i64,i64,i64), Vec<usize>>| -> EdgeRecord {
            used[start_idx] = true;
            let mut chain_start = group[start_idx].v0;
            let mut chain_end = group[start_idx].v1;
            let face_normals = group[start_idx].face_normals.clone();

            // Extend forward (from chain_end)
            loop {
                let ek = vkey(&chain_end);
                let next = ep_map.get(&ek).and_then(|idxs| {
                    idxs.iter().find(|&&j| !used[j]).copied()
                });
                match next {
                    Some(j) => {
                        used[j] = true;
                        // chain_end is either group[j].v0 or group[j].v1
                        if vkey(&group[j].v0) == ek {
                            chain_end = group[j].v1;
                        } else {
                            chain_end = group[j].v0;
                        }
                    }
                    None => break,
                }
            }

            // Extend backward (from chain_start)
            loop {
                let sk = vkey(&chain_start);
                let prev = ep_map.get(&sk).and_then(|idxs| {
                    idxs.iter().find(|&&j| !used[j]).copied()
                });
                match prev {
                    Some(j) => {
                        used[j] = true;
                        if vkey(&group[j].v1) == sk {
                            chain_start = group[j].v0;
                        } else {
                            chain_start = group[j].v1;
                        }
                    }
                    None => break,
                }
            }

            EdgeRecord { v0: chain_start, v1: chain_end, face_normals }
        };

        // Start walks from chain start edges first
        let mut processed_starts: std::collections::HashSet<usize> = std::collections::HashSet::new();
        for &si in &chain_starts {
            if used[si] { continue; }
            processed_starts.insert(si);
            let merged = walk_chain(si, &mut used, &group, &endpoint_map);
            let (ka, kb) = (vkey(&merged.v0), vkey(&merged.v1));
            let key = if ka <= kb { (ka.0,ka.1,ka.2,kb.0,kb.1,kb.2) } else { (kb.0,kb.1,kb.2,ka.0,ka.1,ka.2) };
            result.insert(key, merged);
        }

        // Handle any remaining (e.g. isolated single edges or loops)
        for i in 0..group.len() {
            if used[i] { continue; }
            let merged = walk_chain(i, &mut used, &group, &endpoint_map);
            let (ka, kb) = (vkey(&merged.v0), vkey(&merged.v1));
            let key = if ka <= kb { (ka.0,ka.1,ka.2,kb.0,kb.1,kb.2) } else { (kb.0,kb.1,kb.2,ka.0,ka.1,ka.2) };
            result.insert(key, merged);
        }
    }

    result
}

// ─── helpers: edge classification ─────────────────────────────────────────────

/// Returns `true` if this edge should be drawn from `view_dir`.
///
/// Rules (in priority order):
/// - **Boundary** (one adjacent face): always keep.
/// - **Coplanar check first**: if adjacent faces lie on the same geometric plane
///   (cross-product norm ≈ 0) the edge is a BSP-split artifact or interior
///   tessellation edge — always skip regardless of normal orientation.
/// - **Silhouette**: adjacent faces straddle the silhouette plane
///   (`dot(n0, view) × dot(n1, view) ≤ 0`).
/// - **Feature/crease**: angle between adjacent face normals ≥ `feature_thresh`.
///   Kept for all orientations; HLR will classify as visible or hidden.
/// - Otherwise: skip.
fn should_keep_edge(
    face_normals: &[Vector3<Real>],
    view_dir: &Vector3<Real>,
    feature_thresh: Real,
) -> bool {
    match face_normals.len() {
        0 => false,
        1 => true, // boundary edge — naked edge of an open mesh
        _ => {
            let n0 = face_normals[0];
            let n1 = face_normals[1];

            // Coplanar guard: |n0 × n1| ≈ sin(angle_between_normals).
            // Both parallel (same-direction coplanar) and anti-parallel (opposite
            // winding — a common CSG/BSP artifact) give |cross| ≈ 0.
            // These are always interior / split edges and must be skipped before
            // the silhouette test (anti-parallel normals would pass d0*d1 ≤ 0
            // as a false silhouette).
            let sin_angle_sq = n0.cross(&n1).norm_squared();
            let sin_thresh = feature_thresh.sin();
            if sin_angle_sq < sin_thresh * sin_thresh {
                return false;
            }

            let d0 = n0.dot(view_dir);
            let d1 = n1.dot(view_dir);

            // Silhouette: sign change across view direction
            if d0 * d1 <= 0.0 {
                return true;
            }

            // Feature crease: dihedral angle ≥ threshold (already ensured by the
            // coplanar guard above).
            // Keep back-facing feature edges too — HLR will classify as hidden.
            true
        }
    }
}

// ─── helpers: orthographic projection ─────────────────────────────────────────

/// Project `p` orthographically onto the plane `(origin, normal)`.
#[inline]
fn project_point(
    p: &Point3<Real>,
    origin: &Point3<Real>,
    normal: &Vector3<Real>,
) -> Point3<Real> {
    let d = (*p - *origin).dot(normal);
    Point3::from(p.coords - normal * d)
}

// ─── helpers: HLR ray sampling ───────────────────────────────────────────────

/// Sample `n` positions along edge `(v0, v1)` and return a visibility flag
/// for each (`true` = visible, `false` = hidden).
///
/// Rays are cast **back-to-front**: from `far_dist` units ahead of each sample
/// in the view direction, back toward the sample, stopping `min_gap` before the
/// sample to avoid self-intersection.  Because the ray always originates outside
/// the mesh, only front-face intersections need to be checked (`solid = false`).
///
/// This correctly classifies both:
/// - Front-facing visible edges (no occluder between sample and viewer).
/// - Back-facing hidden edges (the mesh itself occludes the sample — the reverse
///   ray hits a front face of the closed shell before reaching the back edge).
fn hlr_sample_edge(
    v0: &Point3<Real>,
    v1: &Point3<Real>,
    trimeshes: &[TriMesh],
    view_dir: &Vector3<Real>,
    n: usize,
    far_dist: Real,
) -> Vec<bool> {
    // Stop just before reaching the sample to skip the surface self-intersection.
    let min_gap = tolerance() * 100.0;
    let toi_limit = far_dist - min_gap;

    // Inset endpoints by a small fraction to avoid placing samples exactly at
    // mesh vertices.  Vertex-coincident samples are unreliable: the back-to-front
    // ray reaches the face plane at toi == far_dist, which is outside toi_limit,
    // so the face is missed and the vertex looks visible even when it should be
    // hidden.  A 2% inset keeps samples on the interior of each edge.
    let t_min = 0.02;
    let t_max = 0.98;

    (0..n)
        .map(|i| {
            let t_raw = i as Real / (n - 1) as Real;
            let t = t_min + t_raw * (t_max - t_min);
            let p = Point3::from(v0.coords + (v1.coords - v0.coords) * t);
            // Origin is far in front of the viewer; ray travels back toward sample.
            let ray_origin = Point3::from(p.coords + view_dir * far_dist);
            let ray = Ray::new(ray_origin, -(*view_dir));
            // visible = no occluder intersected before the ray reaches the sample
            !trimeshes
                .iter()
                .any(|tm| tm.cast_local_ray(&ray, toi_limit, false).is_some())
        })
        .collect()
}

// ─── helpers: segment chaining ───────────────────────────────────────────────

/// Convert the per-sample visibility flags into polylines and append them to
/// `result`.
///
/// Consecutive samples with the same visibility are merged into one polyline.
/// Sample positions are linearly interpolated `proj_v0 → proj_v1`.
fn chain_segments(
    vis: &[bool],
    proj_v0: &Point3<Real>,
    proj_v1: &Point3<Real>,
    result: &mut EdgeProjectionResult,
) {
    if vis.is_empty() {
        return;
    }
    let n = vis.len();
    let sample_pt = |i: usize| -> Point3<Real> {
        let t = i as Real / (n - 1) as Real;
        Point3::from(proj_v0.coords + (proj_v1.coords - proj_v0.coords) * t)
    };

    let mut run_start = 0usize;
    for i in 1..=n {
        let end_of_run = i == n || vis[i] != vis[i - 1];
        if end_of_run {
            let end = i.min(n - 1);
            if run_start < end {
                let pts: Vec<Point3<Real>> =
                    (run_start..=end).map(sample_pt).collect();
                if pts.len() >= 2 {
                    if vis[run_start] {
                        result.visible_polylines.push(pts);
                    } else {
                        result.hidden_polylines.push(pts);
                    }
                }
            }
            run_start = i;
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::csg::CSG;
    use nalgebra::{Point3, Vector3};

    fn translate(
        m: crate::mesh::Mesh<()>,
        dx: Real, dy: Real, dz: Real,
    ) -> crate::mesh::Mesh<()> {
        crate::mesh::Mesh {
            polygons: m.polygons.into_iter().map(|p| {
                let verts: Vec<_> = p.vertices.iter().map(|v| {
                    let mut vv = *v;
                    vv.position = Point3::new(v.position.x + dx, v.position.y + dy, v.position.z + dz);
                    vv
                }).collect();
                crate::polygon::Polygon::new(verts, p.metadata.clone())
            }).collect(),
            bounding_box: std::sync::OnceLock::new(),
            metadata: m.metadata,
        }
    }

    #[test]
    fn edge_count_plain_cube() {
        let c = translate(crate::mesh::Mesh::<()>::cube(10.0, None), -5.0, -5.0, -5.0);
        let view = Vector3::new(1.0_f64, 1.0, 1.0).normalize();
        let origin = Point3::new(0.0, 0.0, 0.0);
        let r = c.project_edges(&view, &origin, &view, 15.0, 8, &[]);
        eprintln!("cube: vis={} hid={}", r.visible_polylines.len(), r.hidden_polylines.len());
        assert_eq!(r.visible_polylines.len(), 9, "cube should have 9 visible edges");
        assert_eq!(r.hidden_polylines.len(), 3, "cube should have 3 hidden edges");
    }

    #[test]
    fn edge_count_subtracted_box() {
        let c1 = translate(crate::mesh::Mesh::<()>::cube(10.0, None), -5.0, -5.0, -5.0);
        let c2 = translate(crate::mesh::Mesh::<()>::cube(2.0, None), 4.0, 4.0, 4.0);
        let sub = c1.difference(&c2);

        eprintln!("Polygon count: {}", sub.polygons.len());
        for (i, poly) in sub.polygons.iter().enumerate() {
            let n = poly.plane.normal();
            eprintln!("  poly[{}] verts={} n=({:.3},{:.3},{:.3})", i, poly.vertices.len(), n.x, n.y, n.z);
        }

        let view = Vector3::new(1.0_f64, 1.0, 1.0).normalize();
        let origin = Point3::new(0.0, 0.0, 0.0);
        let r = sub.project_edges(&view, &origin, &view, 15.0, 8, &[]);

        eprintln!("sub: vis={} hid={}", r.visible_polylines.len(), r.hidden_polylines.len());
        for (i, pl) in r.visible_polylines.iter().enumerate() {
            if let (Some(p0), Some(p1)) = (pl.first(), pl.last()) {
                eprintln!("  vis[{}]: ({:.2},{:.2},{:.2})->({:.2},{:.2},{:.2})", i,
                    p0.x,p0.y,p0.z, p1.x,p1.y,p1.z);
            }
        }

        assert_eq!(r.visible_polylines.len(), 18, "sub should have 18 visible edges");
        assert_eq!(r.hidden_polylines.len(), 3, "sub should have 3 hidden edges");
    }
}
