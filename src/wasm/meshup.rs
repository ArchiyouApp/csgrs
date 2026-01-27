// Some special WASM bindings for Meshup TS library
// - expose from curvo: NurbsCurve and CompoundCurve
// - TODO: BHV

use crate::float_types::Real;
use nalgebra::{ Point3, Point4 };
use curvo::prelude::{ NurbsCurve3D, Tessellation, BoundingBox, 
        CompoundCurve3D, Fillet, FilletRadiusOption, FilletRadiusParameterOption };

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

//// curvo: NurbsCurveJs and CompoundCurveJs and related functions ////
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
}

impl From<NurbsCurve3D<Real>> for NurbsCurve3DJs {
    fn from(curve: NurbsCurve3D<Real>) -> Self {
        NurbsCurve3DJs { inner: curve }
    }
}


//// COMPOUND CURVE ////

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

    /// PROPERTIES ///

    pub fn spans(&self) -> Vec<NurbsCurve3DJs> 
    {
        self.inner.clone().into_spans().iter()
            .map(|s| NurbsCurve3DJs::from(s.clone()))
            .collect()
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
}

impl From<CompoundCurve3D<Real>> for CompoundCurve3DJs {
    fn from(curve: CompoundCurve3D<Real>) -> Self {
        CompoundCurve3DJs { inner: curve }
    }
}


