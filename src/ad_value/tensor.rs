use tenferro_algebra::Scalar;
use tenferro_tensor::Tensor;

use super::{AdMode, AdValue, NodeId, TapeId};

/// Tensor newtype carrying AD mode information.
///
/// # Examples
///
/// ```rust
/// use ad_tensors_rs::{AdMode, AdTensor};
/// use tenferro_tensor::{MemoryOrder, Tensor};
///
/// let t = Tensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();
/// let x: AdTensor<f64> = t.into();
/// assert_eq!(x.mode(), AdMode::Primal);
/// ```
#[derive(Clone)]
pub struct AdTensor<T: Scalar>(pub AdValue<Tensor<T>>);

impl<T: Scalar> AdTensor<T> {
    /// Creates a primal tensor.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ad_tensors_rs::AdTensor;
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let t = Tensor::<f64>::from_slice(&[1.0], &[1], MemoryOrder::ColumnMajor).unwrap();
    /// let x = AdTensor::new_primal(t);
    /// assert_eq!(x.dims(), &[1]);
    /// ```
    pub fn new_primal(tensor: Tensor<T>) -> Self {
        Self(AdValue::primal(tensor))
    }

    /// Creates a forward-mode tensor.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ad_tensors_rs::{AdMode, AdTensor};
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let primal = Tensor::<f64>::from_slice(&[1.0], &[1], MemoryOrder::ColumnMajor).unwrap();
    /// let tangent = Tensor::<f64>::from_slice(&[0.1], &[1], MemoryOrder::ColumnMajor).unwrap();
    /// let x = AdTensor::new_forward(primal, tangent);
    /// assert_eq!(x.mode(), AdMode::Forward);
    /// ```
    pub fn new_forward(primal: Tensor<T>, tangent: Tensor<T>) -> Self {
        Self(AdValue::forward(primal, tangent))
    }

    /// Creates a reverse-mode tensor.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ad_tensors_rs::{AdMode, AdTensor, NodeId, TapeId};
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let primal = Tensor::<f64>::from_slice(&[1.0], &[1], MemoryOrder::ColumnMajor).unwrap();
    /// let x = AdTensor::new_reverse(primal, NodeId(8), TapeId(3), None);
    /// assert_eq!(x.mode(), AdMode::Reverse);
    /// ```
    pub fn new_reverse(
        primal: Tensor<T>,
        node: NodeId,
        tape: TapeId,
        tangent: Option<Tensor<T>>,
    ) -> Self {
        Self(AdValue::reverse(primal, node, tape, tangent))
    }

    /// Returns AD mode.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ad_tensors_rs::{AdMode, AdTensor};
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let t = Tensor::<f64>::from_slice(&[1.0], &[1], MemoryOrder::ColumnMajor).unwrap();
    /// let x = AdTensor::new_primal(t);
    /// assert_eq!(x.mode(), AdMode::Primal);
    /// ```
    pub fn mode(&self) -> AdMode {
        self.0.mode()
    }

    /// Returns reference to underlying [`AdValue`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ad_tensors_rs::{AdTensor, AdValue};
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let t = Tensor::<f64>::from_slice(&[1.0], &[1], MemoryOrder::ColumnMajor).unwrap();
    /// let x = AdTensor::new_primal(t);
    /// assert!(matches!(x.as_value(), AdValue::Primal(_)));
    /// ```
    pub fn as_value(&self) -> &AdValue<Tensor<T>> {
        &self.0
    }

    /// Consumes wrapper and returns the underlying [`AdValue`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ad_tensors_rs::{AdTensor, AdValue};
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let t = Tensor::<f64>::from_slice(&[1.0], &[1], MemoryOrder::ColumnMajor).unwrap();
    /// let x = AdTensor::new_primal(t).into_value();
    /// assert!(matches!(x, AdValue::Primal(_)));
    /// ```
    pub fn into_value(self) -> AdValue<Tensor<T>> {
        self.0
    }

    /// Returns primal tensor reference.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ad_tensors_rs::AdTensor;
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let t = Tensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();
    /// let x = AdTensor::new_primal(t);
    /// assert_eq!(x.primal().dims(), &[2]);
    /// ```
    pub fn primal(&self) -> &Tensor<T> {
        self.0.primal_ref()
    }

    /// Returns tangent tensor reference when available.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ad_tensors_rs::AdTensor;
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let primal = Tensor::<f64>::from_slice(&[1.0], &[1], MemoryOrder::ColumnMajor).unwrap();
    /// let tangent = Tensor::<f64>::from_slice(&[0.5], &[1], MemoryOrder::ColumnMajor).unwrap();
    /// let x = AdTensor::new_forward(primal, tangent);
    /// assert_eq!(x.tangent().unwrap().dims(), &[1]);
    /// ```
    pub fn tangent(&self) -> Option<&Tensor<T>> {
        self.0.tangent_ref()
    }

    /// Returns dimensions of the primal tensor.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ad_tensors_rs::AdTensor;
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let t = Tensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();
    /// let x = AdTensor::new_primal(t);
    /// assert_eq!(x.dims(), &[2]);
    /// ```
    pub fn dims(&self) -> &[usize] {
        self.primal().dims()
    }

    /// Returns number of dimensions of the primal tensor.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ad_tensors_rs::AdTensor;
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let t = Tensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();
    /// let x = AdTensor::new_primal(t);
    /// assert_eq!(x.ndim(), 1);
    /// ```
    pub fn ndim(&self) -> usize {
        self.dims().len()
    }

    /// Returns total number of elements in the primal tensor.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ad_tensors_rs::AdTensor;
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let t = Tensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();
    /// let x = AdTensor::new_primal(t);
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
    /// use ad_tensors_rs::AdTensor;
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let t = Tensor::<f64>::from_slice(&[], &[0], MemoryOrder::ColumnMajor).unwrap();
    /// let x = AdTensor::new_primal(t);
    /// assert!(x.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<T: Scalar> From<Tensor<T>> for AdTensor<T> {
    fn from(value: Tensor<T>) -> Self {
        Self(AdValue::Primal(value))
    }
}

impl<T: Scalar> From<AdValue<Tensor<T>>> for AdTensor<T> {
    fn from(value: AdValue<Tensor<T>>) -> Self {
        Self(value)
    }
}

impl<T: Scalar> From<AdTensor<T>> for AdValue<Tensor<T>> {
    fn from(value: AdTensor<T>) -> Self {
        value.0
    }
}
