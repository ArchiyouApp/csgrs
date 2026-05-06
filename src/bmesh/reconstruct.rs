//! Post-boolean polygon reconstruction for BMesh.
//!
//! Uses `Manifold::coplanar` (pre-computed by boolmesh) to group triangles into
//! coplanar connected sets, then reconstructs each group's boundary into an N-gon.

use std::collections::HashMap;
use std::fmt::Debug;

use boolmesh::prelude::Manifold;
use boolmesh::Vec3 as BVec3;
use nalgebra::{Point3, Vector3};

use crate::float_types::Real;
use crate::polygon::Polygon;
use crate::vertex::Vertex;

/// Reconstruct N-gon polygons from a boolmesh `Manifold` using its precomputed
/// `coplanar` face grouping.  Triangles in the same coplanar group are merged into
/// a single boundary polygon (with holes when needed).  Degenerate/isolated faces
/// (`coplanar[f] == -1`) are emitted as individual triangles.
pub fn reconstruct_polygons<S>(m: &Manifold, metadata: &Option<S>) -> Vec<Polygon<S>>
where
    S: Clone + Send + Sync + Debug,
{
    // --- Phase 1: group faces by coplanar root ---
    let mut groups: HashMap<i32, Vec<usize>> = HashMap::new();
    let mut degenerate: Vec<usize> = Vec::new();

    for f in 0..m.nf {
        let root = m.coplanar[f];
        if root < 0 {
            degenerate.push(f);
        } else {
            groups.entry(root).or_default().push(f);
        }
    }

    let mut polygons: Vec<Polygon<S>> = Vec::with_capacity(groups.len() + degenerate.len());

    // --- Degenerate triangles: emit as-is ---
    for f in degenerate {
        if let Some(poly) = triangle_polygon(m, f, metadata) {
            polygons.push(poly);
        }
    }

    // --- Phase 2-4: reconstruct each coplanar group ---
    for (root, faces) in &groups {
        let root = *root as usize;

        // Use the face normal from boolmesh for the group.
        // All faces in the group share the same plane by construction.
        let fn_ = m.face_normals[root];
        let normal = Vector3::new(fn_.x as Real, fn_.y as Real, fn_.z as Real);
        let normal = if let Some(n) = normal.try_normalize(1e-12) { n } else { continue; };

        // Phase 2: collect boundary half-edges (those crossing into a different group)
        let mut boundary_hes: Vec<usize> = Vec::new();
        for &f in faces {
            for k in 0..3usize {
                let h = f * 3 + k;
                let pair = m.hs[h].pair;
                let is_boundary = pair >= m.hs.len()
                    || m.coplanar[pair / 3] != m.coplanar[f];
                if is_boundary {
                    boundary_hes.push(h);
                }
            }
        }

        if boundary_hes.is_empty() {
            continue;
        }

        // Phase 3: build directed edge map tail → head, then walk loops
        let mut edge_map: HashMap<usize, usize> = HashMap::with_capacity(boundary_hes.len());
        for &h in &boundary_hes {
            let tail = m.hs[h].tail;
            let head = m.hs[h].head;
            edge_map.insert(tail, head);
        }

        let loops = walk_loops(&edge_map);
        if loops.is_empty() {
            continue;
        }

        // Phase 4: classify loops → outer boundary + holes
        let (outer_idx, hole_indices) = classify_loops(m, &loops, &normal);
        let outer_loop = &loops[outer_idx];

        let outer_verts = loop_to_verts(m, outer_loop, &normal);
        if outer_verts.len() < 3 {
            continue;
        }

        let hole_verts: Vec<Vec<Vertex>> = hole_indices
            .iter()
            .filter_map(|&hi| {
                let v = loop_to_verts(m, &loops[hi], &normal);
                if v.len() >= 3 { Some(v) } else { None }
            })
            .collect();

        let poly = if hole_verts.is_empty() {
            Polygon::new(outer_verts, metadata.clone())
        } else {
            Polygon::new_with_holes(outer_verts, hole_verts, metadata.clone())
        };
        polygons.push(poly);
    }

    polygons
}

// --- Helpers ---

fn triangle_polygon<S>(m: &Manifold, f: usize, metadata: &Option<S>) -> Option<Polygon<S>>
where
    S: Clone + Send + Sync + Debug,
{
    let base = f * 3;
    if base + 2 >= m.hs.len() {
        return None;
    }
    let i0 = m.hs[base].tail;
    let i1 = m.hs[base + 1].tail;
    let i2 = m.hs[base + 2].tail;
    if i0 >= m.ps.len() || i1 >= m.ps.len() || i2 >= m.ps.len() {
        return None;
    }
    let p0 = to_point(&m.ps[i0]);
    let p1 = to_point(&m.ps[i1]);
    let p2 = to_point(&m.ps[i2]);
    let e1 = p1 - p0;
    let e2 = p2 - p0;
    let raw = e1.cross(&e2);
    let n = raw.try_normalize(1e-12)?;
    Some(Polygon::new(
        vec![Vertex::new(p0, n), Vertex::new(p1, n), Vertex::new(p2, n)],
        metadata.clone(),
    ))
}

fn walk_loops(edge_map: &HashMap<usize, usize>) -> Vec<Vec<usize>> {
    let mut visited: HashMap<usize, bool> = HashMap::with_capacity(edge_map.len());
    let mut loops: Vec<Vec<usize>> = Vec::new();

    for &start in edge_map.keys() {
        if visited.contains_key(&start) {
            continue;
        }

        let mut ring: Vec<usize> = Vec::new();
        let mut current = start;

        loop {
            if visited.contains_key(&current) {
                break;
            }
            visited.insert(current, true);
            ring.push(current);
            match edge_map.get(&current) {
                Some(&next) => current = next,
                None => break,
            }
            if ring.len() > edge_map.len() {
                break; // cycle guard
            }
        }

        if ring.len() >= 3 {
            loops.push(ring);
        }
    }

    loops
}

/// Returns (outer_loop_index, vec of hole loop indices).
/// The loop with the largest absolute 2D signed area is the outer boundary.
fn classify_loops(
    m: &Manifold,
    loops: &[Vec<usize>],
    normal: &Vector3<Real>,
) -> (usize, Vec<usize>) {
    if loops.len() == 1 {
        return (0, vec![]);
    }

    // Build an orthonormal 2D basis for the face plane
    let (u, v) = build_basis(normal);

    let areas: Vec<f64> = loops
        .iter()
        .map(|lp| signed_area_2d(m, lp, &u, &v))
        .collect();

    // Largest absolute area is the outer boundary
    let outer_idx = areas
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.abs().partial_cmp(&b.abs()).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0);

    let holes: Vec<usize> = (0..loops.len()).filter(|&i| i != outer_idx).collect();
    (outer_idx, holes)
}

fn signed_area_2d(
    m: &Manifold,
    loop_verts: &[usize],
    u: &Vector3<Real>,
    v: &Vector3<Real>,
) -> f64 {
    let pts: Vec<(Real, Real)> = loop_verts
        .iter()
        .map(|&vi| {
            let p = to_point(&m.ps[vi]);
            let pv = Vector3::new(p.x, p.y, p.z);
            (pv.dot(u), pv.dot(v))
        })
        .collect();

    let n = pts.len();
    let mut area = 0.0;
    for i in 0..n {
        let (x0, y0) = pts[i];
        let (x1, y1) = pts[(i + 1) % n];
        area += (x0 * y1 - x1 * y0) as f64;
    }
    area * 0.5
}

fn build_basis(normal: &Vector3<Real>) -> (Vector3<Real>, Vector3<Real>) {
    let u = if normal.x.abs() < 0.9 {
        normal.cross(&Vector3::x()).normalize()
    } else {
        normal.cross(&Vector3::y()).normalize()
    };
    let v = normal.cross(&u);
    (u, v)
}

fn loop_to_verts(m: &Manifold, loop_verts: &[usize], normal: &Vector3<Real>) -> Vec<Vertex> {
    loop_verts
        .iter()
        .filter_map(|&vi| {
            if vi < m.ps.len() {
                Some(Vertex::new(to_point(&m.ps[vi]), *normal))
            } else {
                None
            }
        })
        .collect()
}

#[inline]
fn to_point(v: &BVec3) -> Point3<Real> {
    Point3::new(v.x as Real, v.y as Real, v.z as Real)
}

#[cfg(test)]
mod tests {
    use crate::bmesh::BMesh;
    use crate::csg::CSG;
    use crate::mesh::Mesh;

    #[test]
    fn cube_difference_reconstructs_polygons() {
        // Big 10×10×10 cube minus a small 2×2×2 cube at one corner.
        // The 5 untouched faces should reconstruct to quad (4-vertex) polygons.
        let big = Mesh::<()>::cube(10.0, None);
        let small = Mesh::<()>::cube(2.0, None).translate(4.0, 4.0, 4.0);

        let big_bm = BMesh::from(big);
        let small_bm = BMesh::from(small);
        let result_bm = big_bm.difference(&small_bm);

        let soup = Mesh::from(result_bm.clone());
        let tri_count = soup.polygons.len();

        let reconstructed = result_bm.to_mesh_reconstructed();
        let poly_count = reconstructed.polygons.len();

        assert!(tri_count > 6, "boolean should produce more than 6 triangles, got {tri_count}");
        assert!(
            poly_count < tri_count,
            "reconstructed ({poly_count}) should be fewer than triangles ({tri_count})"
        );

        for (i, poly) in reconstructed.polygons.iter().enumerate() {
            assert!(poly.vertices.len() >= 3, "polygon {i} has < 3 vertices");

            if poly.vertices.len() >= 4 {
                let n = poly.plane.normal();
                let p0 = poly.vertices[0].position;
                for (j, v) in poly.vertices.iter().skip(1).enumerate() {
                    let dist = (v.position - p0).dot(&n).abs();
                    assert!(dist < 1e-6, "polygon {i} vertex {j} not coplanar: dist={dist}");
                }
            }
        }
    }

    #[test]
    fn two_cubes_union_reconstruction() {
        // Union of two overlapping cubes: resulting flat faces should reconstruct to N-gons,
        // and reconstruction should produce fewer polygons than the triangle soup.
        let a = Mesh::<()>::cube(10.0, None);
        let b = Mesh::<()>::cube(10.0, None).translate(5.0, 0.0, 0.0);

        let a_bm = BMesh::from(a);
        let b_bm = BMesh::from(b);
        let result_bm = a_bm.union(&b_bm);

        let soup = Mesh::from(result_bm.clone());
        let reconstructed = result_bm.to_mesh_reconstructed();

        for (i, poly) in reconstructed.polygons.iter().enumerate() {
            assert!(poly.vertices.len() >= 3, "polygon {i} has only {} vertices", poly.vertices.len());
        }
        assert!(
            reconstructed.polygons.len() < soup.polygons.len(),
            "reconstructed ({}) should be fewer than triangle soup ({})",
            reconstructed.polygons.len(), soup.polygons.len()
        );
    }
}
