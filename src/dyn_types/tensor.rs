use num_complex::{Complex32, Complex64};
use tenferro_tensor::Tensor;

use super::ScalarType;

/// Runtime tensor wrapper for a fixed supported dtype set.
///
/// # Examples
///
/// ```rust
/// use ad_tensors_rs::{DynTensor, ScalarType};
/// use tenferro_tensor::{MemoryOrder, Tensor};
///
/// let t = Tensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();
/// let x: DynTensor = t.into();
/// assert_eq!(x.scalar_type(), ScalarType::F64);
/// ```
#[derive(Clone)]
pub enum DynTensor {
    F32(Tensor<f32>),
    F64(Tensor<f64>),
    C32(Tensor<Complex32>),
    C64(Tensor<Complex64>),
}

impl DynTensor {
    /// Returns runtime scalar type.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ad_tensors_rs::{DynTensor, ScalarType};
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let t = Tensor::<f32>::from_slice(&[1.0], &[1], MemoryOrder::ColumnMajor).unwrap();
    /// let x: DynTensor = t.into();
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

    /// Returns dimensions of the underlying tensor.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ad_tensors_rs::DynTensor;
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let t = Tensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();
    /// let x: DynTensor = t.into();
    /// assert_eq!(x.dims(), &[2]);
    /// ```
    pub fn dims(&self) -> &[usize] {
        match self {
            Self::F32(t) => t.dims(),
            Self::F64(t) => t.dims(),
            Self::C32(t) => t.dims(),
            Self::C64(t) => t.dims(),
        }
    }

    /// Returns rank.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ad_tensors_rs::DynTensor;
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let t = Tensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();
    /// let x: DynTensor = t.into();
    /// assert_eq!(x.ndim(), 1);
    /// ```
    pub fn ndim(&self) -> usize {
        self.dims().len()
    }

    /// Returns number of elements.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ad_tensors_rs::DynTensor;
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let t = Tensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();
    /// let x: DynTensor = t.into();
    /// assert_eq!(x.len(), 2);
    /// ```
    pub fn len(&self) -> usize {
        self.dims().iter().product()
    }

    /// Returns true when tensor has zero elements.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ad_tensors_rs::DynTensor;
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let t = Tensor::<f64>::from_slice(&[], &[0], MemoryOrder::ColumnMajor).unwrap();
    /// let x: DynTensor = t.into();
    /// assert!(x.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns typed tensor ref when dtype is `f32`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ad_tensors_rs::DynTensor;
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let t = Tensor::<f32>::from_slice(&[1.0], &[1], MemoryOrder::ColumnMajor).unwrap();
    /// let x: DynTensor = t.into();
    /// assert!(x.as_f32().is_some());
    /// ```
    pub fn as_f32(&self) -> Option<&Tensor<f32>> {
        if let Self::F32(t) = self {
            Some(t)
        } else {
            None
        }
    }

    /// Returns typed tensor ref when dtype is `f64`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ad_tensors_rs::DynTensor;
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let t = Tensor::<f64>::from_slice(&[1.0], &[1], MemoryOrder::ColumnMajor).unwrap();
    /// let x: DynTensor = t.into();
    /// assert!(x.as_f64().is_some());
    /// ```
    pub fn as_f64(&self) -> Option<&Tensor<f64>> {
        if let Self::F64(t) = self {
            Some(t)
        } else {
            None
        }
    }

    /// Returns typed tensor ref when dtype is `Complex32`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ad_tensors_rs::DynTensor;
    /// use num_complex::Complex32;
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let t = Tensor::<Complex32>::from_slice(
    ///     &[Complex32::new(1.0, 0.0)],
    ///     &[1],
    ///     MemoryOrder::ColumnMajor,
    /// )
    /// .unwrap();
    /// let x: DynTensor = t.into();
    /// assert!(x.as_c32().is_some());
    /// ```
    pub fn as_c32(&self) -> Option<&Tensor<Complex32>> {
        if let Self::C32(t) = self {
            Some(t)
        } else {
            None
        }
    }

    /// Returns typed tensor ref when dtype is `Complex64`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ad_tensors_rs::DynTensor;
    /// use num_complex::Complex64;
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let t = Tensor::<Complex64>::from_slice(
    ///     &[Complex64::new(1.0, 0.0)],
    ///     &[1],
    ///     MemoryOrder::ColumnMajor,
    /// )
    /// .unwrap();
    /// let x: DynTensor = t.into();
    /// assert!(x.as_c64().is_some());
    /// ```
    pub fn as_c64(&self) -> Option<&Tensor<Complex64>> {
        if let Self::C64(t) = self {
            Some(t)
        } else {
            None
        }
    }
}

impl From<Tensor<f32>> for DynTensor {
    fn from(value: Tensor<f32>) -> Self {
        Self::F32(value)
    }
}

impl From<Tensor<f64>> for DynTensor {
    fn from(value: Tensor<f64>) -> Self {
        Self::F64(value)
    }
}

impl From<Tensor<Complex32>> for DynTensor {
    fn from(value: Tensor<Complex32>) -> Self {
        Self::C32(value)
    }
}

impl From<Tensor<Complex64>> for DynTensor {
    fn from(value: Tensor<Complex64>) -> Self {
        Self::C64(value)
    }
}
