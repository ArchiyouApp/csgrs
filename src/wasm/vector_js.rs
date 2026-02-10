use crate::float_types::Real;
use nalgebra::{Vector3, Rotation3, Unit};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct Vector3Js {
    pub(crate) inner: Vector3<Real>,
}

#[wasm_bindgen]
impl Vector3Js {
    #[wasm_bindgen(constructor)]
    pub fn new(x: f64, y: f64, z: f64) -> Vector3Js {
        Vector3Js {
            inner: Vector3::new(x as Real, y as Real, z as Real),
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

    /* Meshup: basic Vector calculation methods */
    pub fn length(&self) -> f64
    {
        self.inner.norm() as f64
    }

    pub fn normalize(&self) -> Vector3Js
    {
        Vector3Js {
            inner: self.inner.normalize()
        }
    }

    #[wasm_bindgen(js_name = isOrthogonal)]
    pub fn is_orthogonal(&self, tolerance: f64) -> bool
    {
        self.inner.is_orthogonal(tolerance)
    }

    pub fn abs(&self) -> Vector3Js
    {
        Vector3Js {
            inner: self.inner.abs()
        }
    }

    pub fn reverse(&self) -> Vector3Js
    {
        Vector3Js {
            inner: self.inner.scale(-1.0) // TODO: fix neg ()
        }
    }

    pub fn add(&self, other: &Vector3Js) -> Vector3Js
    {
        Vector3Js {
            inner: self.inner + other.inner
        }
    }

    pub fn subtract(&self, other: &Vector3Js) -> Vector3Js
    {
        Vector3Js {
            inner: self.inner - other.inner
        }
    }

    pub fn dot(&self, other: &Vector3Js) -> f64 
    {
        self.inner.dot(&other.inner) as f64
    }

    pub fn equals(&self, other: &Vector3Js) -> bool
    {
        self.inner.eq(&other.inner)
    }

    pub fn angle(&self, other: &Vector3Js) -> f64 
    {
        self.inner.angle(&other.inner) as f64
    }

    pub fn scale(&self, factor: f64) -> Vector3Js
    {
        Vector3Js {
            inner: self.inner.scale(factor as Real)
        }
    }

    // Rotate the vector around a given axis by a certain angle in radians
    pub fn rotate(&self, axis: &Vector3Js, angle: f64) -> Vector3Js
    {
        let axis_norm = Unit::new_normalize(axis.inner);
        let rotation = Rotation3::from_axis_angle(&axis_norm, angle as Real);
        Vector3Js {
            inner: rotation * self.inner
        }
    }

    #[wasm_bindgen(js_name = rotateEuler)]
    pub fn rotate_euler(&self, roll: f64, pitch: f64, yaw: f64) -> Vector3Js
    {
        let rotation = Rotation3::from_euler_angles(roll as Real, pitch as Real, yaw as Real);
        Vector3Js {
            inner: rotation * self.inner
        }
    }

    pub fn cross(&self, other: &Vector3Js) -> Vector3Js
    {
        Vector3Js {
            inner: self.inner.cross(&other.inner)
        }
    }

}

// Rust-only conversions
impl From<Vector3<Real>> for Vector3Js {
    fn from(v: Vector3<Real>) -> Self {
        Vector3Js { inner: v }
    }
}

impl From<&Vector3Js> for Vector3<Real> {
    fn from(v: &Vector3Js) -> Self {
        v.inner
    }
}
