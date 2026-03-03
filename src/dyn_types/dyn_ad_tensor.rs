use num_complex::{Complex32, Complex64};

use crate::{AdMode, AdTensor};

use super::ScalarType;

/// Runtime AD tensor wrapper.
///
/// # Examples
///
/// ```rust
/// use ad_tensors_rs::{AdTensor, DynAdTensor, ScalarType};
/// use tenferro_tensor::{MemoryOrder, Tensor};
///
/// let t = Tensor::<f64>::from_slice(&[1.0], &[1], MemoryOrder::ColumnMajor).unwrap();
/// let x: DynAdTensor = AdTensor::new_primal(t).into();
/// assert_eq!(x.scalar_type(), ScalarType::F64);
/// ```
#[derive(Clone)]
pub enum DynAdTensor {
    F32(AdTensor<f32>),
    F64(AdTensor<f64>),
    C32(AdTensor<Complex32>),
    C64(AdTensor<Complex64>),
}

impl DynAdTensor {
    /// Returns runtime scalar type.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ad_tensors_rs::{AdTensor, DynAdTensor, ScalarType};
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let t = Tensor::<f32>::from_slice(&[1.0], &[1], MemoryOrder::ColumnMajor).unwrap();
    /// let x: DynAdTensor = AdTensor::new_primal(t).into();
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
    /// use ad_tensors_rs::{AdMode, AdTensor, DynAdTensor};
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let t = Tensor::<f64>::from_slice(&[1.0], &[1], MemoryOrder::ColumnMajor).unwrap();
    /// let x: DynAdTensor = AdTensor::new_primal(t).into();
    /// assert_eq!(x.mode(), AdMode::Primal);
    /// ```
    pub fn mode(&self) -> AdMode {
        match self {
            Self::F32(v) => v.mode(),
            Self::F64(v) => v.mode(),
            Self::C32(v) => v.mode(),
            Self::C64(v) => v.mode(),
        }
    }

    /// Returns primal tensor dimensions.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ad_tensors_rs::{AdTensor, DynAdTensor};
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let t = Tensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();
    /// let x: DynAdTensor = AdTensor::new_primal(t).into();
    /// assert_eq!(x.dims(), &[2]);
    /// ```
    pub fn dims(&self) -> &[usize] {
        match self {
            Self::F32(v) => v.dims(),
            Self::F64(v) => v.dims(),
            Self::C32(v) => v.dims(),
            Self::C64(v) => v.dims(),
        }
    }

    /// Returns primal tensor rank.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ad_tensors_rs::{AdTensor, DynAdTensor};
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let t = Tensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();
    /// let x: DynAdTensor = AdTensor::new_primal(t).into();
    /// assert_eq!(x.ndim(), 1);
    /// ```
    pub fn ndim(&self) -> usize {
        self.dims().len()
    }

    /// Returns primal tensor element count.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ad_tensors_rs::{AdTensor, DynAdTensor};
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let t = Tensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();
    /// let x: DynAdTensor = AdTensor::new_primal(t).into();
    /// assert_eq!(x.len(), 2);
    /// ```
    pub fn len(&self) -> usize {
        self.dims().iter().product()
    }

    /// Returns true when primal tensor has zero elements.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ad_tensors_rs::{AdTensor, DynAdTensor};
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let t = Tensor::<f64>::from_slice(&[], &[0], MemoryOrder::ColumnMajor).unwrap();
    /// let x: DynAdTensor = AdTensor::new_primal(t).into();
    /// assert!(x.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns typed AD tensor ref when dtype is `f32`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ad_tensors_rs::{AdTensor, DynAdTensor};
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let t = Tensor::<f32>::from_slice(&[1.0], &[1], MemoryOrder::ColumnMajor).unwrap();
    /// let x: DynAdTensor = AdTensor::new_primal(t).into();
    /// assert!(x.as_f32().is_some());
    /// ```
    pub fn as_f32(&self) -> Option<&AdTensor<f32>> {
        if let Self::F32(v) = self {
            Some(v)
        } else {
            None
        }
    }

    /// Returns typed AD tensor ref when dtype is `f64`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ad_tensors_rs::{AdTensor, DynAdTensor};
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let t = Tensor::<f64>::from_slice(&[1.0], &[1], MemoryOrder::ColumnMajor).unwrap();
    /// let x: DynAdTensor = AdTensor::new_primal(t).into();
    /// assert!(x.as_f64().is_some());
    /// ```
    pub fn as_f64(&self) -> Option<&AdTensor<f64>> {
        if let Self::F64(v) = self {
            Some(v)
        } else {
            None
        }
    }

    /// Returns typed AD tensor ref when dtype is `Complex32`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ad_tensors_rs::{AdTensor, DynAdTensor};
    /// use num_complex::Complex32;
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let t = Tensor::<Complex32>::from_slice(
    ///     &[Complex32::new(1.0, 0.0)],
    ///     &[1],
    ///     MemoryOrder::ColumnMajor,
    /// )
    /// .unwrap();
    /// let x: DynAdTensor = AdTensor::new_primal(t).into();
    /// assert!(x.as_c32().is_some());
    /// ```
    pub fn as_c32(&self) -> Option<&AdTensor<Complex32>> {
        if let Self::C32(v) = self {
            Some(v)
        } else {
            None
        }
    }

    /// Returns typed AD tensor ref when dtype is `Complex64`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ad_tensors_rs::{AdTensor, DynAdTensor};
    /// use num_complex::Complex64;
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let t = Tensor::<Complex64>::from_slice(
    ///     &[Complex64::new(1.0, 0.0)],
    ///     &[1],
    ///     MemoryOrder::ColumnMajor,
    /// )
    /// .unwrap();
    /// let x: DynAdTensor = AdTensor::new_primal(t).into();
    /// assert!(x.as_c64().is_some());
    /// ```
    pub fn as_c64(&self) -> Option<&AdTensor<Complex64>> {
        if let Self::C64(v) = self {
            Some(v)
        } else {
            None
        }
    }
}

impl From<AdTensor<f32>> for DynAdTensor {
    fn from(value: AdTensor<f32>) -> Self {
        Self::F32(value)
    }
}

impl From<AdTensor<f64>> for DynAdTensor {
    fn from(value: AdTensor<f64>) -> Self {
        Self::F64(value)
    }
}

impl From<AdTensor<Complex32>> for DynAdTensor {
    fn from(value: AdTensor<Complex32>) -> Self {
        Self::C32(value)
    }
}

impl From<AdTensor<Complex64>> for DynAdTensor {
    fn from(value: AdTensor<Complex64>) -> Self {
        Self::C64(value)
    }
}
