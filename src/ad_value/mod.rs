mod mode;
mod scalar;
mod tensor;
mod value;

pub use mode::{AdMode, NodeId, TapeId};
pub use scalar::AdScalar;
pub use tensor::AdTensor;
pub use value::AdValue;

#[cfg(test)]
mod tests {
    use super::*;
    use tenferro_tensor::{MemoryOrder, Tensor};

    #[test]
    fn ad_value_map_preserves_mode() {
        let x = AdValue::forward(2_i32, 3_i32);
        let y = x.map(|v| v as f64);
        assert_eq!(y.mode(), AdMode::Forward);
        assert_eq!(y.primal_ref(), &2.0_f64);
        assert_eq!(y.tangent_ref(), Some(&3.0_f64));
    }

    #[test]
    fn ad_tensor_metadata() {
        let tensor =
            Tensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();
        let ad = AdTensor::new_primal(tensor);
        assert_eq!(ad.mode(), AdMode::Primal);
        assert_eq!(ad.dims(), &[2]);
        assert_eq!(ad.ndim(), 1);
        assert_eq!(ad.len(), 2);
    }
}
