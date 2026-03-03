use num_complex::{Complex32, Complex64};

/// Runtime scalar type tag used by all `Dyn*` wrappers.
///
/// # Examples
///
/// ```rust
/// use ad_tensors_rs::ScalarType;
///
/// assert_eq!(ScalarType::F64, ScalarType::F64);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalarType {
    /// `f32`
    F32,
    /// `f64`
    F64,
    /// `Complex32`
    C32,
    /// `Complex64`
    C64,
}

/// Runtime scalar wrapper for a fixed supported dtype set.
///
/// # Examples
///
/// ```rust
/// use ad_tensors_rs::{DynScalar, ScalarType};
///
/// let x: DynScalar = 2.0_f32.into();
/// assert_eq!(x.scalar_type(), ScalarType::F32);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DynScalar {
    F32(f32),
    F64(f64),
    C32(Complex32),
    C64(Complex64),
}

impl DynScalar {
    /// Returns runtime scalar type.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ad_tensors_rs::{DynScalar, ScalarType};
    ///
    /// let x = DynScalar::F64(1.0);
    /// assert_eq!(x.scalar_type(), ScalarType::F64);
    /// ```
    pub fn scalar_type(&self) -> ScalarType {
        match self {
            Self::F32(_) => ScalarType::F32,
            Self::F64(_) => ScalarType::F64,
            Self::C32(_) => ScalarType::C32,
            Self::C64(_) => ScalarType::C64,
        }
    }

    /// Returns the `f32` value when this scalar is `F32`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ad_tensors_rs::DynScalar;
    ///
    /// let x = DynScalar::F32(3.0);
    /// assert_eq!(x.as_f32(), Some(3.0));
    /// ```
    pub fn as_f32(&self) -> Option<f32> {
        if let Self::F32(v) = self {
            Some(*v)
        } else {
            None
        }
    }

    /// Returns the `f64` value when this scalar is `F64`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ad_tensors_rs::DynScalar;
    ///
    /// let x = DynScalar::F64(3.0);
    /// assert_eq!(x.as_f64(), Some(3.0));
    /// ```
    pub fn as_f64(&self) -> Option<f64> {
        if let Self::F64(v) = self {
            Some(*v)
        } else {
            None
        }
    }

    /// Returns the `Complex32` value when this scalar is `C32`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ad_tensors_rs::DynScalar;
    /// use num_complex::Complex32;
    ///
    /// let x = DynScalar::C32(Complex32::new(1.0, 2.0));
    /// assert_eq!(x.as_c32(), Some(Complex32::new(1.0, 2.0)));
    /// ```
    pub fn as_c32(&self) -> Option<Complex32> {
        if let Self::C32(v) = self {
            Some(*v)
        } else {
            None
        }
    }

    /// Returns the `Complex64` value when this scalar is `C64`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ad_tensors_rs::DynScalar;
    /// use num_complex::Complex64;
    ///
    /// let x = DynScalar::C64(Complex64::new(1.0, 2.0));
    /// assert_eq!(x.as_c64(), Some(Complex64::new(1.0, 2.0)));
    /// ```
    pub fn as_c64(&self) -> Option<Complex64> {
        if let Self::C64(v) = self {
            Some(*v)
        } else {
            None
        }
    }
}

impl From<f32> for DynScalar {
    fn from(value: f32) -> Self {
        Self::F32(value)
    }
}

impl From<f64> for DynScalar {
    fn from(value: f64) -> Self {
        Self::F64(value)
    }
}

impl From<Complex32> for DynScalar {
    fn from(value: Complex32) -> Self {
        Self::C32(value)
    }
}

impl From<Complex64> for DynScalar {
    fn from(value: Complex64) -> Self {
        Self::C64(value)
    }
}
