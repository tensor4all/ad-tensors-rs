use num_complex::{Complex32, Complex64};

use crate::{AdMode, AdValue};

use super::{DynScalar, ScalarType};

/// Runtime AD scalar value wrapper.
///
/// # Examples
///
/// ```rust
/// use ad_tensors_rs::{AdMode, AdValue, DynAdValue};
///
/// let x: DynAdValue = AdValue::forward(2.0_f64, 1.0_f64).into();
/// assert_eq!(x.mode(), AdMode::Forward);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub enum DynAdValue {
    F32(AdValue<f32>),
    F64(AdValue<f64>),
    C32(AdValue<Complex32>),
    C64(AdValue<Complex64>),
}

impl DynAdValue {
    /// Returns runtime scalar type.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ad_tensors_rs::{AdValue, DynAdValue, ScalarType};
    ///
    /// let x: DynAdValue = AdValue::primal(1.0_f32).into();
    /// assert_eq!(x.scalar_type(), ScalarType::F32);
    /// ```
    pub fn scalar_type(&self) -> ScalarType {
        match self {
            Self::F32(_) => ScalarType::F32,
            Self::F64(_) => ScalarType::F64,
            Self::C32(_) => ScalarType::C32,
            Self::C64(_) => ScalarType::C64,
        }
    }

    /// Returns AD mode.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ad_tensors_rs::{AdMode, AdValue, DynAdValue};
    ///
    /// let x: DynAdValue = AdValue::forward(2.0_f64, 1.0_f64).into();
    /// assert_eq!(x.mode(), AdMode::Forward);
    /// ```
    pub fn mode(&self) -> AdMode {
        match self {
            Self::F32(v) => v.mode(),
            Self::F64(v) => v.mode(),
            Self::C32(v) => v.mode(),
            Self::C64(v) => v.mode(),
        }
    }

    /// Returns primal part as dynamic scalar.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ad_tensors_rs::{AdValue, DynAdValue, DynScalar};
    ///
    /// let x: DynAdValue = AdValue::primal(3.0_f64).into();
    /// assert_eq!(x.primal(), DynScalar::F64(3.0));
    /// ```
    pub fn primal(&self) -> DynScalar {
        match self {
            Self::F32(v) => DynScalar::F32(*v.primal_ref()),
            Self::F64(v) => DynScalar::F64(*v.primal_ref()),
            Self::C32(v) => DynScalar::C32(*v.primal_ref()),
            Self::C64(v) => DynScalar::C64(*v.primal_ref()),
        }
    }

    /// Returns tangent part as dynamic scalar when available.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ad_tensors_rs::{AdValue, DynAdValue, DynScalar};
    ///
    /// let x: DynAdValue = AdValue::forward(3.0_f64, 0.5_f64).into();
    /// assert_eq!(x.tangent(), Some(DynScalar::F64(0.5)));
    /// ```
    pub fn tangent(&self) -> Option<DynScalar> {
        match self {
            Self::F32(v) => v.tangent_ref().copied().map(DynScalar::F32),
            Self::F64(v) => v.tangent_ref().copied().map(DynScalar::F64),
            Self::C32(v) => v.tangent_ref().copied().map(DynScalar::C32),
            Self::C64(v) => v.tangent_ref().copied().map(DynScalar::C64),
        }
    }

    /// Returns typed AD value ref when dtype is `f32`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ad_tensors_rs::{AdValue, DynAdValue};
    ///
    /// let x: DynAdValue = AdValue::primal(1.0_f32).into();
    /// assert!(x.as_f32().is_some());
    /// ```
    pub fn as_f32(&self) -> Option<&AdValue<f32>> {
        if let Self::F32(v) = self {
            Some(v)
        } else {
            None
        }
    }

    /// Returns typed AD value ref when dtype is `f64`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ad_tensors_rs::{AdValue, DynAdValue};
    ///
    /// let x: DynAdValue = AdValue::primal(1.0_f64).into();
    /// assert!(x.as_f64().is_some());
    /// ```
    pub fn as_f64(&self) -> Option<&AdValue<f64>> {
        if let Self::F64(v) = self {
            Some(v)
        } else {
            None
        }
    }

    /// Returns typed AD value ref when dtype is `Complex32`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ad_tensors_rs::{AdValue, DynAdValue};
    /// use num_complex::Complex32;
    ///
    /// let x: DynAdValue = AdValue::primal(Complex32::new(1.0, 0.0)).into();
    /// assert!(x.as_c32().is_some());
    /// ```
    pub fn as_c32(&self) -> Option<&AdValue<Complex32>> {
        if let Self::C32(v) = self {
            Some(v)
        } else {
            None
        }
    }

    /// Returns typed AD value ref when dtype is `Complex64`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ad_tensors_rs::{AdValue, DynAdValue};
    /// use num_complex::Complex64;
    ///
    /// let x: DynAdValue = AdValue::primal(Complex64::new(1.0, 0.0)).into();
    /// assert!(x.as_c64().is_some());
    /// ```
    pub fn as_c64(&self) -> Option<&AdValue<Complex64>> {
        if let Self::C64(v) = self {
            Some(v)
        } else {
            None
        }
    }
}

impl From<AdValue<f32>> for DynAdValue {
    fn from(value: AdValue<f32>) -> Self {
        Self::F32(value)
    }
}

impl From<AdValue<f64>> for DynAdValue {
    fn from(value: AdValue<f64>) -> Self {
        Self::F64(value)
    }
}

impl From<AdValue<Complex32>> for DynAdValue {
    fn from(value: AdValue<Complex32>) -> Self {
        Self::C32(value)
    }
}

impl From<AdValue<Complex64>> for DynAdValue {
    fn from(value: AdValue<Complex64>) -> Self {
        Self::C64(value)
    }
}
