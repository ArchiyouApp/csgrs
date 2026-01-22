// Some special WASM bindings for Meshup TS library
// - expose from curvo: NurbsCurve and CompoundCurve
// - TODO: BHV

use crate::float_types::Real;
use nalgebra::{ Point2, Point3, Vector2 };
use curvo::prelude::{ NurbsCurve2D, Tessellation, BoundingBox };
use super::point_js::{ Point3Js };

use wasm_bindgen::prelude::*;

//// curvo: NurbsCurveJs and CompoundCurveJs and related functions ////
// NOTES:
//  - NurbsCurves control points have an extra weight coordinate,
//     so 2D curves use Point3Js and 2D Point4Js

#[wasm_bindgen]
pub struct NurbsCurve2DJs 
{
    pub(crate) inner: NurbsCurve2D<Real>,
}

#[wasm_bindgen]
impl NurbsCurve2DJs 
{
    #[wasm_bindgen(constructor)]
    pub fn new(degree: usize, control_points: Vec<Point3Js>, knots: Vec<Real>) -> Result<NurbsCurve2DJs, JsValue>
    {   
        let points2d: Vec<Point3<Real>> = control_points.into_iter()
            .map(|p| p.inner) // Point3Js wraps Point3<Real>
            .collect();

        match NurbsCurve2D::try_new(degree, points2d, knots) 
        {
            Ok(curve) => Ok(Self { inner: curve }),
            Err(e) => Err(JsValue::from_str(&format!("Failed to create NURBS curve: {:?}", e)))
        }
    }

    // Create a polyline NurbsCurve2D from a list of points
    #[wasm_bindgen(js_name = makePolyline)]
    pub fn make_polyline(points: Vec<Point2Js>, normalize: bool) -> Self
    {
        let control_points: Vec<Point2<Real>> = points.into_iter()
            .map(|p| p.inner)
            .collect();

        // NOTE: Future versions have try_polyline that can fail
        Self { inner: NurbsCurve2D::polyline(&control_points, normalize) }
    }

    /// Create NURBS Curve (degree 3) passing through given control points
    ///
    ///  # Arguments
    /// 
    /// * `points` - Control points to interpolate through (x,y,w=0)
    /// 
    #[wasm_bindgen(js_name = makeInterpolated)]
    pub fn make_interpolated(points: Vec<Point2Js>, degree: usize) -> Result<NurbsCurve2DJs, JsValue>
    {
        let control_points: Vec<Point2<Real>> = points.into_iter()
            .map(|p| p.inner)
            .collect();

        match NurbsCurve2D::try_interpolate(&control_points, degree) 
        {
            Ok(curve) => Ok(Self { inner: curve }),
            Err(e) => Err(JsValue::from_str(&format!("Interpolation failed: {:?}", e)))
        }
    }

    // NOTE: bring back when updating curvo version (now 0.1.52)
    // Create BezierCurve with control points
    // #[wasm_bindgen(js_name = makeBezier)]
    // pub fn make_bezier(points: Vec<Point2Js>) -> Result<NurbsCurve2DJs, JsValue>
    // {
    //    let control_points: Vec<Point2<Real>> = points.into_iter()
    //        .map(|p| p.inner)
    //        .collect();

    //    match NurbsCurve2D::try_bezier(&control_points) 
    //    {
    //        Ok(curve) => Ok(Self { inner: curve }),
    //        Err(e) => Err(JsValue::from_str(&format!("Failed to create Bezier curve: {:?}", e)))
    //    }
    //}

    /// PROPERTIES ///

    #[wasm_bindgen(js_name = controlPoints)]
    pub fn control_points(&self) -> Vec<Point2Js> 
    {
        // IMPORTANT: Curvo returns control points including weight component
        self.inner.control_points().iter()
            .map(|p| Point2Js::new(p.x, p.y)) // Remove the weight factor
            .collect()
    }

    pub fn weights(&self) -> Vec<Real> {
        self.inner.weights().to_vec()
    }

    pub fn knots(&self) -> Vec<Real> {
        self.inner.knots().to_vec()
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
    pub fn param_closest_to_point(&self, point: &Point2Js) -> Result<Real, JsValue> 
    {
        match self.inner.find_closest_parameter(&point.inner) 
        {
            Ok(param) => Ok(param),
            Err(e) => Err(JsValue::from_str(&format!("Failed to compute closest parameter to point: {:?}", e)))
        }
    }

    /// Get the point at given parameter
    #[wasm_bindgen(js_name = pointAtParam)]
    pub fn point_at_param(&self, param: Real) -> Point2Js {
        Point2Js::from(self.inner.point_at(param))
    }

    // Get tangent at given parameter
    #[wasm_bindgen(js_name = tangentAt)]
    pub fn tangent_at(&self, param: Real) -> Vector2Js {
        Vector2Js::from(self.inner.tangent_at(param))
    }

    pub fn bbox(&self) -> Vec<Point2Js> {
        let bbox = BoundingBox::from(&self.inner);
        let min = bbox.min();
        let max = bbox.max();
        vec![
            Point2Js::new(min.x as f64, min.y as f64),
            Point2Js::new(max.x as f64, max.y as f64),
        ]
    }

    /// Tessellate curve into evenly spaced points by count
    #[wasm_bindgen(js_name = tessellate)]
    pub fn tessellate(&self, tol: Option<f64>) -> Vec<Point2Js> {
        let t = tol.unwrap_or(1e-4);
        return
            self.inner.tessellate(Some(t))
            .iter()
            .map(|p| Point2Js::new(p.x as f64, p.y as f64))
            .collect();
    }


}

//// ADDED Point2Js ////

#[wasm_bindgen]
pub struct Point2Js {
    pub(crate) inner: Point2<Real>,
}

#[wasm_bindgen]
impl Point2Js {
    #[wasm_bindgen(constructor)]
    pub fn new(x: f64, y: f64) -> Point2Js {
        Point2Js {
            inner: Point2::new(x as Real, y as Real),
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

    #[wasm_bindgen(js_name = toString)]
    pub fn to_string_js(&self) -> String {
        format!("<Point2({}, {})>", self.inner.x, self.inner.y)
    }
}

// Rust-only conversions (not visible to JS)
impl From<Point2<Real>> for Point2Js {
    fn from(p: Point2<Real>) -> Self {
        Point2Js { inner: p }
    }
}

impl From<&Point2Js> for Point2<Real> {
    fn from(p: &Point2Js) -> Self {
        p.inner
    }
}

/// Vector2Js for 2D vectors

#[wasm_bindgen]
pub struct Vector2Js {
    pub(crate) inner: Vector2<Real>,
}

#[wasm_bindgen]
impl Vector2Js {
    #[wasm_bindgen(constructor)]
    pub fn new(x: f64, y: f64) -> Vector2Js {
        Vector2Js {
            inner: Vector2::new(x as Real, y as Real),
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

    #[wasm_bindgen(js_name = toString)]
    pub fn to_string_js(&self) -> String {
        format!("<Vector2({}, {})>", self.inner.x, self.inner.y)
    }
}

// Rust-only conversions
impl From<Vector2<Real>> for Vector2Js {
    fn from(v: Vector2<Real>) -> Self {
        Vector2Js { inner: v }
    }
}

impl From<&Vector2Js> for Vector2<Real> {
    fn from(v: &Vector2Js) -> Self {
        v.inner
    }
}
