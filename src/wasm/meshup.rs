// Some special WASM bindings for Meshup TS library
// - expose from curvo: NurbsCurve and CompoundCurve
// - TODO: BHV

use crate::float_types::Real;
use nalgebra::{ Point3, Point4, Rotation3, Translation3, Matrix4, Vector3 };
use curvo::prelude::{ NurbsCurve2D, NurbsCurve3D, CompoundCurve2D, Tessellation, BoundingBox, Transformable,
        CompoundCurve3D, Fillet, FilletRadiusOption, FilletRadiusParameterOption,
        CurveOffsetOption, CurveOffsetCornerType, Offset };

use super::point_js::{ Point3Js };
use super::vector_js::{ Vector3Js };

use wasm_bindgen::prelude::*;


//// Point4Js for weighted control points ////

#[wasm_bindgen]
pub struct Point4Js {
    pub(crate) inner: Point4<Real>,
}

#[wasm_bindgen]
impl Point4Js {
    #[wasm_bindgen(constructor)]
    pub fn new(x: f64, y: f64, z: f64, w: f64) -> Point4Js {
        Point4Js {
            inner: Point4::new(x as Real, y as Real, z as Real, w as Real),
        }
    }

    #[wasm_bindgen(getter)]
    pub fn x(&self) -> f64 {
        self.inner.x as f64
    }

    #[wasm_bindgen(getter)]
    pub fn y(&self) -> f64 {
        self.inner.y as f64
    }

    #[wasm_bindgen(getter)]
    pub fn z(&self) -> f64 {
        self.inner.z as f64
    }

    #[wasm_bindgen(getter)]
    pub fn w(&self) -> f64 {
        self.inner.w as f64
    }

    #[wasm_bindgen(js_name = toString)]
    pub fn to_string_js(&self) -> String {
        format!("<Point4({}, {}, {}, {})>", self.inner.x, self.inner.y, self.inner.z, self.inner.w)
    }
}

// Rust-only conversions (not visible to JS)
impl From<Point4<Real>> for Point4Js {
    fn from(p: Point4<Real>) -> Self {
        Point4Js { inner: p }
    }
}

impl From<&Point4Js> for Point4<Real> {
    fn from(p: &Point4Js) -> Self {
        p.inner
    }
}


//// NURBSCURVE3DJS ////
// 
// NURBS Curve 3D JavaScript bindings
//
// NOTES:
//  - NurbsCurves control points have an extra weight coordinate,
//     so 3D curves use Point4Js for weighted control points

#[wasm_bindgen]
pub struct NurbsCurve3DJs 
{
    pub(crate) inner: NurbsCurve3D<Real>,
}

#[wasm_bindgen]
impl NurbsCurve3DJs 
{
    #[wasm_bindgen(constructor)]
    pub fn new(degree: usize, control_points: Vec<Point3Js>, weights: Option<Vec<Real>>, knots: Vec<Real>) -> Result<NurbsCurve3DJs, JsValue>
    {   
        let num_points = control_points.len();
        
        // Use provided weights or default to all 1.0
        let weights_vec = weights.unwrap_or_else(|| vec![1.0; num_points]);
        
        // Check that control points and weights have the same length
        if weights_vec.len() != num_points {
            return Err(JsValue::from_str(&format!(
                "Control points count ({}) must match weights count ({})", 
                num_points, 
                weights_vec.len()
            )));
        }
        
        // Combine points and weights into Point4 (homogeneous coordinates)
        let points4d: Vec<Point4<Real>> = control_points.into_iter()
            .zip(weights_vec.into_iter())
            .map(|(p, w)| Point4::new(p.inner.x, p.inner.y, p.inner.z, w))
            .collect();

        match NurbsCurve3D::try_new(degree, points4d, knots) 
        {
            Ok(curve) => Ok(Self { inner: curve }),
            Err(e) => Err(JsValue::from_str(&format!("Failed to create NURBS curve: {:?}", e)))
        }
    }

    pub fn clone(&self) -> Self {
        Self { inner: self.inner.clone() }
    }

    // Create a polyline NurbsCurve3D from a list of points
    #[wasm_bindgen(js_name = makePolyline)]
    pub fn make_polyline(points: Vec<Point3Js>, normalize: bool) -> Self
    {
        let control_points: Vec<Point3<Real>> = points.into_iter()
            .map(|p| p.inner)
            .collect();

        // NOTE: Future versions have try_polyline that can fail
        Self { inner: NurbsCurve3D::polyline(&control_points, normalize) }
    }

    /// Create NURBS Curve (degree 3) passing through given control points
    ///
    ///  # Arguments
    /// 
    /// * `points` - Control points to interpolate through (x,y,z)
    /// 
    #[wasm_bindgen(js_name = makeInterpolated)]
    pub fn make_interpolated(points: Vec<Point3Js>, degree: usize) -> Result<NurbsCurve3DJs, JsValue>
    {
        let control_points: Vec<Point3<Real>> = points.into_iter()
            .map(|p| p.inner)
            .collect();

        match NurbsCurve3D::try_interpolate(&control_points, degree) 
        {
            Ok(curve) => Ok(Self { inner: curve }),
            Err(e) => Err(JsValue::from_str(&format!("Interpolation failed: {:?}", e)))
        }
    }

    /// PROPERTIES ///

    #[wasm_bindgen(js_name = controlPoints)]
    pub fn control_points(&self) -> Vec<Point3Js> 
    {
        // IMPORTANT: Curvo returns control points including weight component
        self.inner.control_points().iter()
            .map(|p| Point3Js::new(p.x, p.y, p.z)) // Remove the weight factor
            .collect()
    }

    pub fn weights(&self) -> Vec<Real> {
        self.inner.weights().to_vec()
    }

    pub fn knots(&self) -> Vec<Real> {
        self.inner.knots().to_vec()
    }

    #[wasm_bindgen(js_name = knotsDomain)]
    pub fn knots_domain(&self) -> Vec<Real> {
        let domain = self.inner.knots_domain();
        vec![domain.0, domain.1]
    }

    /// Get the degree of the curve
    pub fn degree(&self) -> usize {
        self.inner.degree()
    }

    //// CALCULATED PROPERTIES ////

    pub fn length(&self) -> Real {
        self.inner.try_length()
            .expect("Failed to compute curve length")
    }

    #[wasm_bindgen(js_name = paramAtLength)]
    pub fn param_at_length(&self, length: Real) -> Result<Real, JsValue> 
    {
        match self.inner.try_parameter_at_length(length, Some(1e-4)) 
        {
            Ok(param) => Ok(param),
            Err(e) => Err(JsValue::from_str(&format!("Failed to compute parameter at given length: {:?}", e)))
        }
    }

    #[wasm_bindgen(js_name = paramClosestToPoint)]
    pub fn param_closest_to_point(&self, point: &Point3Js) -> Result<Real, JsValue> 
    {
        match self.inner.find_closest_parameter(&point.inner) 
        {
            Ok(param) => Ok(param),
            Err(e) => Err(JsValue::from_str(&format!("Failed to compute closest parameter to point: {:?}", e)))
        }
    }

    /// Get the point at given parameter
    #[wasm_bindgen(js_name = pointAtParam)]
    pub fn point_at_param(&self, param: Real) -> Point3Js {
        Point3Js::from(self.inner.point_at(param))
    }

    // Get tangent at given parameter
    #[wasm_bindgen(js_name = tangentAt)]
    pub fn tangent_at(&self, param: Real) -> Vector3Js {
        Vector3Js::from(self.inner.tangent_at(param))
    }

    pub fn bbox(&self) -> Vec<Point3Js> {
        let bbox = BoundingBox::from(&self.inner);
        let min = bbox.min();
        let max = bbox.max();
        vec![
            Point3Js::new(min.x as f64, min.y as f64, min.z as f64),
            Point3Js::new(max.x as f64, max.y as f64, max.z as f64),
        ]
    }

    #[wasm_bindgen(js_name = filletAtParams)]
    pub fn fillet_at_params(&self, radius: Real, at: Vec<Real>) -> Result<CompoundCurve3DJs, JsValue> {
        let knots = self.inner.knots();
        
        // Find closest knot value for each parameter and return slightly less for robustness
        let adjusted_params: Vec<Real> = at.into_iter()
            .map(|p| {
                // Find the closest knot value
                let closest_knot = knots.iter()
                    .min_by(|a, b| {
                        let diff_a = (*a - p).abs();
                        let diff_b = (*b - p).abs();
                        diff_a.partial_cmp(&diff_b).unwrap()
                    })
                    .copied()
                    .unwrap_or(p);
                
                // Return slightly less than the knot value for robustness
                closest_knot - 1e-6
            })
            .collect();
        
        let fillet_options = FilletRadiusParameterOption::new(radius, adjusted_params);
        match self.inner.fillet(fillet_options) {
            Ok(compound_curve) => Ok(CompoundCurve3DJs::from(compound_curve)),
            Err(e) => Err(JsValue::from_str(&format!("Failed to fillet curve: {:?}", e)))
        }
    }

    // Fillet sharp corner(s) of Curve. Optionally only at given point(s)
    pub fn fillet(&self, radius: Real, at: Option<Vec<Point3Js>>) -> Result<CompoundCurve3DJs, JsValue> {
        
        match at {
            Some(points) => {
  
                let params: Result<Vec<Real>, JsValue> = points.into_iter()
                    .map(|p| self.param_closest_to_point(&p))
                    .collect();
                let params = params?;
                self.fillet_at_params(radius, params)
            },
            None => {
                // Fillet all sharp corners
                let fillet_options = FilletRadiusOption::new(radius);
                match self.inner.fillet(fillet_options) {
                    Ok(compound_curve) => Ok(CompoundCurve3DJs::from(compound_curve)),
                    Err(e) => Err(JsValue::from_str(&format!("NurbsCurve3DJs::fillet(): Failed to fillet curve: {:?}", e)))
                }
            }
        }
    }

    /// Offset the curve by a distance in the specified corner type ('sharp','round', 'smooth')
    pub fn offset(&self, distance: Real, corner_type: &str) -> Result<CompoundCurve3DJs, JsValue> {

        // map strings: 'sharp','round','smooth' to CurveOffsetCornerType
        let corner_type = match corner_type {
            "sharp" => CurveOffsetCornerType::Sharp,
            "round" => CurveOffsetCornerType::Round,
            "smooth" => CurveOffsetCornerType::Smooth,
            _ => CurveOffsetCornerType::Sharp,
        };

        // Check if curve is planar, get plane axes
        let plane = self.get_on_plane(None);
        if plane.len() != 3 {
            return Err(JsValue::from_str("Cannot offset a non-planar 3D curve"));
        }
        let local_x: Vector3<Real> = (&plane[1]).into();
        let local_y: Vector3<Real> = (&plane[2]).into();

        // Use first control point as projection origin
        let first_cp = self.inner.control_points()[0];
        let w = first_cp.w;
        let origin = Point3::new(first_cp.x / w, first_cp.y / w, first_cp.z / w);

        // Project 3D curve to 2D
        let curve_2d = self.project_to_2d(&origin, &local_x, &local_y)
            .map_err(|e| JsValue::from_str(&e))?;

        let option = CurveOffsetOption::default()
            .with_corner_type(corner_type)
            .with_distance(distance)
            .with_normal_tolerance(1e-4);

        // Offset in 2D
        let offset_results = curve_2d.offset(option)
            .map_err(|e| JsValue::from_str(&format!("Failed to offset curve: {:?}", e)))?;

        // Convert each 2D offset span back to 3D and collect into a single CompoundCurve3D
        let mut all_3d_spans: Vec<NurbsCurve3D<Real>> = Vec::new();
        for compound_2d in &offset_results {
            for span_2d in compound_2d.spans() {
                // Build a temporary wrapper to use from_2d
                let curve_3d = NurbsCurve3DJs::from_2d(&span_2d, &origin, &local_x, &local_y)
                    .map_err(|e| JsValue::from_str(&e))?;
                all_3d_spans.push(curve_3d.inner);
            }
        }

        if all_3d_spans.is_empty() {
            return Err(JsValue::from_str("Offset produced no curves"));
        }

        // If single span, wrap it into a CompoundCurve3D; otherwise join them
        match CompoundCurve3D::try_new(all_3d_spans) {
            Ok(compound) => Ok(CompoundCurve3DJs::from(compound)),
            Err(e) => Err(JsValue::from_str(&format!("Failed to create compound curve from offset result: {:?}", e)))
        }
    }
    

    /// Rotate the curve by Euler angles (in radians) around the X, Y, and Z axes
    pub fn rotate(&self, ax: Real, ay: Real, az: Real) -> NurbsCurve3DJs {
        let rotation = Rotation3::from_euler_angles(ax, ay, az);
        // Convert 3x3 rotation to 4x4 homogeneous matrix for Curvo's Transformable
        let mat4 = rotation.to_homogeneous();
        let mut curve = self.inner.clone();
        curve.transform(&mat4);
        NurbsCurve3DJs { inner: curve }
    }

    /// Scale the curve by factors along the X, Y, and Z axes
    pub fn scale(&self, sx: Real, sy: Real, sz: Real) -> NurbsCurve3DJs {
        #[rustfmt::skip]
        let mat4 = Matrix4::new(
            sx,   0.0,  0.0,  0.0,
            0.0,  sy,   0.0,  0.0,
            0.0,  0.0,  sz,   0.0,
            0.0,  0.0,  0.0,  1.0,
        );
        let mut curve = self.inner.clone();
        curve.transform(&mat4);
        NurbsCurve3DJs { inner: curve }
    }

    /// Translate the curve by a Vector3Js offset
    #[wasm_bindgen(js_name = translate)]
    pub fn translate(&self, offset: &Vector3Js) -> NurbsCurve3DJs {
        let v: Vector3<Real> = offset.into();
        let translation = Translation3::new(v.x, v.y, v.z);
        let mat4 = translation.to_homogeneous();
        let mut curve = self.inner.clone();
        curve.transform(&mat4);
        NurbsCurve3DJs { inner: curve }
    }

    // Tessellate curve into evenly spaced points by count
    #[wasm_bindgen(js_name = tessellate)]
    pub fn tessellate(&self, tol: Option<f64>) -> Vec<Point3Js> {
        let t = tol.unwrap_or(1e-4);
        return
            self.inner.tessellate(Some(t))
            .iter()
            .map(|p| Point3Js::new(p.x as f64, p.y as f64, p.z as f64))
            .collect();
    }

    /// Check if all control points lie on a single plane
    #[wasm_bindgen(js_name = isPlanar)]
    pub fn is_planar(&self, tolerance: Option<f64>) -> bool {
        self.get_on_plane(tolerance).len() == 3
    }

    /// Get the plane the curve lies on, returned as [normal, localX, localY].
    /// Returns an empty array if the curve is not planar.
    /// Local axes are aligned to the closest global axes for consistency.
    #[wasm_bindgen(js_name = getOnPlane)]
    pub fn get_on_plane(&self, tolerance: Option<f64>) -> Vec<Vector3Js> {
        let tol = tolerance.unwrap_or(1e-6);

        // Dehomogenize control points to get 3D positions
        let pts: Vec<Point3<Real>> = self.inner.control_points().iter()
            .map(|p4| {
                let w = p4.w;
                Point3::new(p4.x / w, p4.y / w, p4.z / w)
            })
            .collect();

        if pts.len() < 3 {
            // Degenerate: default to XY plane
            return vec![
                Vector3Js::new(0.0, 0.0, 1.0),
                Vector3Js::new(1.0, 0.0, 0.0),
                Vector3Js::new(0.0, 1.0, 0.0),
            ];
        }

        let p0 = &pts[0];

        // Find a plane normal from two non-degenerate edge vectors
        let mut normal: Option<Vector3<Real>> = None;
        'outer: for i in 1..pts.len() {
            let v1 = pts[i] - p0;
            if v1.norm() < tol { continue; }
            for j in (i + 1)..pts.len() {
                let v2 = pts[j] - p0;
                let n = v1.cross(&v2);
                if n.norm() > tol {
                    normal = Some(n.normalize());
                    break 'outer;
                }
            }
        }

        // All points collinear: default normal to Z-up
        let normal = normal.unwrap_or(Vector3::new(0.0, 0.0, 1.0));

        // Verify all points lie on the plane
        for p in &pts {
            let v = p - p0;
            if v.dot(&normal).abs() >= tol {
                return vec![]; // Not planar
            }
        }

        // Choose local axes aligned to closest global axes
        let global_axes = [
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
            Vector3::new(0.0, 0.0, 1.0),
        ];

        // Pick the global axis most perpendicular to the normal (largest cross product)
        let mut best_axis = global_axes[0];
        let mut best_cross_len: Real = 0.0;
        for axis in &global_axes {
            let c = axis.cross(&normal);
            let len = c.norm();
            if len > best_cross_len {
                best_cross_len = len;
                best_axis = *axis;
            }
        }

        let mut local_x = best_axis.cross(&normal).normalize();
        let mut local_y = normal.cross(&local_x).normalize();

        // Swap so that local_x is closest to global X axis
        let angle_x_to_gx = local_x.angle(&global_axes[0]);
        let angle_y_to_gx = local_y.angle(&global_axes[0]);
        if angle_x_to_gx > angle_y_to_gx {
            std::mem::swap(&mut local_x, &mut local_y);
        }

        // Make absolute for consistency (curve coords get mapped via dot products anyway)
        local_x = Vector3::new(local_x.x.abs(), local_x.y.abs(), local_x.z.abs());
        local_y = Vector3::new(local_y.x.abs(), local_y.y.abs(), local_y.z.abs());

        vec![
            Vector3Js::from(normal),
            Vector3Js::from(local_x),
            Vector3Js::from(local_y),
        ]
    }

}

///// NURBSCURVE3DJS NON-WASM  ////


impl NurbsCurve3DJs {
    /// Project this 3D NURBS curve onto a 2D plane, producing a NurbsCurve2D.
    /// The plane is defined by an origin point and two orthonormal axis vectors (localX, localY).
    /// Each 3D control point is projected to 2D via dot products with the local axes.
    /// Degree, knots, and weights are preserved.
    pub(crate) fn project_to_2d(&self, origin: &Point3<Real>, local_x: &Vector3<Real>, local_y: &Vector3<Real>) -> Result<NurbsCurve2D<Real>, String> {
        let o = origin;
        let lx = local_x;
        let ly = local_y;

        // Project each homogeneous 3D control point (Point4: x*w, y*w, z*w, w) to homogeneous 2D (Point3: x*w, y*w, w)
        let control_points_2d: Vec<Point3<Real>> = self.inner.control_points().iter()
            .map(|p4| {
                let w = p4.w;
                // Dehomogenize to get the actual 3D coordinates
                let p3 = Point3::new(p4.x / w, p4.y / w, p4.z / w);
                let diff = p3 - o;
                let x_2d = diff.dot(lx);
                let y_2d = diff.dot(ly);
                // Re-homogenize for 2D: (x*w, y*w, w)
                Point3::new(x_2d * w, y_2d * w, w)
            })
            .collect();

        let degree = self.inner.degree();
        let knots = self.inner.knots().to_vec();

        NurbsCurve2D::try_new(degree, control_points_2d, knots)
            .map_err(|e| format!("Failed to project curve to 2D: {:?}", e))
    }

    /// Reconstruct a 3D NURBS curve from a 2D curve and its projection plane.
    /// This is the inverse of `project_to_2d`: given a NurbsCurve2D and the plane
    /// (origin + localX + localY) it was projected onto, map each 2D control point
    /// back to 3D space.
    pub(crate) fn from_2d(curve_2d: &NurbsCurve2D<Real>, origin: &Point3<Real>, local_x: &Vector3<Real>, local_y: &Vector3<Real>) -> Result<NurbsCurve3DJs, String> {
        let o = origin;
        let lx = local_x;
        let ly = local_y;

        // Map each 2D homogeneous control point (Point3: x*w, y*w, w) back to 3D homogeneous (Point4: x*w, y*w, z*w, w)
        let control_points_3d: Vec<Point4<Real>> = curve_2d.control_points().iter()
            .map(|p3| {
                let w = p3.z;
                let x_2d = p3.x / w;
                let y_2d = p3.y / w;
                // Reconstruct the 3D point on the plane
                let p = o + lx * x_2d + ly * y_2d;
                Point4::new(p.x * w, p.y * w, p.z * w, w)
            })
            .collect();

        let degree = curve_2d.degree();
        let knots = curve_2d.knots().to_vec();

        NurbsCurve3D::try_new(degree, control_points_3d, knots)
            .map(|c| NurbsCurve3DJs { inner: c })
            .map_err(|e| format!("Failed to project curve to 3D: {:?}", e))
    }
}


impl From<NurbsCurve3D<Real>> for NurbsCurve3DJs {
    fn from(curve: NurbsCurve3D<Real>) -> Self {
        NurbsCurve3DJs { inner: curve }
    }
}


//// COMPOUNDCURVE3DJS ////

// A composite curve made of separate NurbsCurve3DJs (called spans)

#[wasm_bindgen]
pub struct CompoundCurve3DJs 
{
    pub(crate) inner: CompoundCurve3D<Real>,
}

#[wasm_bindgen]
impl CompoundCurve3DJs 
{
    #[wasm_bindgen(constructor)]
    pub fn new(spans: Vec<NurbsCurve3DJs>) -> Result<CompoundCurve3DJs, JsValue>
    {
        match CompoundCurve3D::try_new(spans.into_iter().map(|s| s.inner).collect()) 
        {
            Ok(curve) => Ok(Self { inner: curve }),
            Err(e) => Err(JsValue::from_str(&format!("Failed to create Compound NURBS curve: {:?}", e)))
        }
    }

    pub fn clone(&self) -> Self {
        Self { inner: self.inner.clone() }
    }

    /// PROPERTIES ///

    pub fn spans(&self) -> Vec<NurbsCurve3DJs> 
    {
        self.inner.clone().into_spans().iter()
            .map(|s| NurbsCurve3DJs::from(s.clone()))
            .collect()
    }

    /// Get all unique control points across all spans
    #[wasm_bindgen(js_name = controlPoints)]
    pub fn control_points(&self) -> Vec<Point3Js> 
    {
        let spans = self.inner.clone().into_spans();
        let mut points: Vec<Point3Js> = Vec::new();

        for (i, span) in spans.iter().enumerate() {
            let pts = span.control_points();
            // Skip the first point of subsequent spans to avoid duplicates at joins
            let start = if i == 0 { 0 } else { 1 };
            for p in &pts[start..] {
                points.push(Point3Js::new(p.x as f64, p.y as f64, p.z as f64));
            }
        }

        points
    }

    

    #[wasm_bindgen(js_name = knotsDomain)]
    pub fn knots_domain(&self) -> Vec<Real> {
        let domain = self.inner.knots_domain();
        vec![domain.0, domain.1]
    }

    pub fn find_span(&self, param: Real) -> NurbsCurve3DJs {
        NurbsCurve3DJs { inner: self.inner.find_span(param).clone() }
    }

    //// CALCULATED PROPERTIES ////

    pub fn length(&self) -> Real {
        self.inner.try_length()
            .expect("Failed to compute curve length")
    }

    pub fn closed(&self, tol: Option<Real>) -> bool {
        let t = tol.unwrap_or(1e-4);
        self.inner.is_closed(Some(t))
    }

    #[wasm_bindgen(js_name = closestPoint)]
    pub fn closest_point(&self, point: &Point3Js) -> Result<Point3Js, JsValue> {
        match self.inner.find_closest_point(&point.inner) {
            Ok(p) => Ok(Point3Js::from(p)),
            Err(e) => Err(JsValue::from_str(&format!("Can't get closest point: {:?}", e)))
        }
    }

    /// Get the point at given parameter
    #[wasm_bindgen(js_name = pointAtParam)]
    pub fn point_at_param(&self, param: Real) -> Point3Js {
        Point3Js::from(self.inner.point_at(param))
    }

    // Get tangent at given parameter
    #[wasm_bindgen(js_name = tangentAt)]
    pub fn tangent_at(&self, param: Real) -> Vector3Js {
        Vector3Js::from(self.inner.tangent_at(param))
    }

    pub fn bbox(&self) -> Vec<Point3Js> {
        let bbox = BoundingBox::from(&self.inner);
        let min = bbox.min();
        let max = bbox.max();
        vec![
            Point3Js::new(min.x as f64, min.y as f64, min.z as f64),
            Point3Js::new(max.x as f64, max.y as f64, max.z as f64),
        ]
    }
    

    //// OPERATIONS ////
    
    /// Translate the compound curve by a Vector3Js offset
    #[wasm_bindgen(js_name = translate)]
    pub fn translate(&self, offset: &Vector3Js) -> CompoundCurve3DJs {
        let v: Vector3<Real> = offset.into();
        let translation = Translation3::new(v.x, v.y, v.z);
        let mat4 = translation.to_homogeneous();
        let mut curve = self.inner.clone();
        curve.transform(&mat4);
        CompoundCurve3DJs { inner: curve }
    }

    /// Rotate the compound curve by Euler angles (in radians) around the X, Y, and Z axes
    pub fn rotate(&self, ax: Real, ay: Real, az: Real) -> CompoundCurve3DJs {
        let rotation = Rotation3::from_euler_angles(ax, ay, az);
        let mat4 = rotation.to_homogeneous();
        let mut curve = self.inner.clone();
        curve.transform(&mat4);
        CompoundCurve3DJs { inner: curve }
    }

    /// Scale the compound curve by factors along the X, Y, and Z axes
    pub fn scale(&self, sx: Real, sy: Real, sz: Real) -> CompoundCurve3DJs {
        #[rustfmt::skip]
        let mat4 = Matrix4::new(
            sx,   0.0,  0.0,  0.0,
            0.0,  sy,   0.0,  0.0,
            0.0,  0.0,  sz,   0.0,
            0.0,  0.0,  0.0,  1.0,
        );
        let mut curve = self.inner.clone();
        curve.transform(&mat4);
        CompoundCurve3DJs { inner: curve }
    }

    /// Tessellate curve into evenly spaced points by count
    #[wasm_bindgen(js_name = tessellate)]
    pub fn tessellate(&self, tol: Option<f64>) -> Vec<Point3Js> {
        let t = tol.unwrap_or(1e-4);
        return
            self.inner.tessellate(Some(t))
            .iter()
            .map(|p| Point3Js::new(p.x as f64, p.y as f64, p.z as f64))
            .collect();
    }

    /// Offset the compound curve by a distance with the specified corner type ('sharp','round','smooth')
    pub fn offset(&self, distance: Real, corner_type: &str) -> Result<CompoundCurve3DJs, JsValue> {
        let corner_type = match corner_type {
            "sharp" => CurveOffsetCornerType::Sharp,
            "round" => CurveOffsetCornerType::Round,
            "smooth" => CurveOffsetCornerType::Smooth,
            _ => CurveOffsetCornerType::Sharp,
        };

        // Check planarity using control points from all spans
        let spans_js: Vec<NurbsCurve3DJs> = self.inner.clone().into_spans().iter()
            .map(|s| NurbsCurve3DJs::from(s.clone()))
            .collect();

        if spans_js.is_empty() {
            return Err(JsValue::from_str("Cannot offset an empty compound curve"));
        }

        // Use the first span's plane
        let plane = spans_js[0].get_on_plane(None);
        if plane.len() != 3 {
            return Err(JsValue::from_str("Cannot offset a non-planar compound curve"));
        }
        let local_x: Vector3<Real> = (&plane[1]).into();
        let local_y: Vector3<Real> = (&plane[2]).into();

        // Use first control point as projection origin
        let first_cp = spans_js[0].inner.control_points()[0];
        let w = first_cp.w;
        let origin = Point3::new(first_cp.x / w, first_cp.y / w, first_cp.z / w);

        // Project compound curve to 2D
        let compound_2d = self.project_to_2d(&origin, &local_x, &local_y)
            .map_err(|e| JsValue::from_str(&e))?;

        let option = CurveOffsetOption::default()
            .with_corner_type(corner_type)
            .with_distance(distance)
            .with_normal_tolerance(1e-4);

        // Offset in 2D
        let offset_results = compound_2d.offset(option)
            .map_err(|e| JsValue::from_str(&format!("Failed to offset compound curve: {:?}", e)))?;

        // Convert 2D offset results back to 3D
        let mut all_3d_spans: Vec<NurbsCurve3D<Real>> = Vec::new();
        for compound_2d_result in &offset_results {
            for span_2d in compound_2d_result.spans() {
                let curve_3d = NurbsCurve3DJs::from_2d(&span_2d, &origin, &local_x, &local_y)
                    .map_err(|e| JsValue::from_str(&e))?;
                all_3d_spans.push(curve_3d.inner);
            }
        }

        if all_3d_spans.is_empty() {
            return Err(JsValue::from_str("Offset produced no curves"));
        }

        match CompoundCurve3D::try_new(all_3d_spans) {
            Ok(compound) => Ok(CompoundCurve3DJs::from(compound)),
            Err(e) => Err(JsValue::from_str(&format!("Failed to create compound curve from offset result: {:?}", e)))
        }
    }
}

// COMPOUND CURVE3D JS - NON-WASM
impl CompoundCurve3DJs {
    /// Project this 3D compound curve onto a 2D plane, producing a CompoundCurve2D.
    /// Each span is projected individually and then combined.
    pub(crate) fn project_to_2d(&self, origin: &Point3<Real>, local_x: &Vector3<Real>, local_y: &Vector3<Real>) -> Result<CompoundCurve2D<Real>, String> {
        let spans_3d = self.inner.clone().into_spans();
        let mut spans_2d: Vec<NurbsCurve2D<Real>> = Vec::new();

        for span in &spans_3d {
            let wrapper = NurbsCurve3DJs { inner: span.clone() };
            let span_2d = wrapper.project_to_2d(origin, local_x, local_y)?;
            spans_2d.push(span_2d);
        }

        CompoundCurve2D::try_new(spans_2d)
            .map_err(|e| format!("Failed to create 2D compound curve: {:?}", e))
    }
}

impl From<CompoundCurve3D<Real>> for CompoundCurve3DJs {
    fn from(curve: CompoundCurve3D<Real>) -> Self {
        CompoundCurve3DJs { inner: curve }
    }
}


