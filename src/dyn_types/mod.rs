mod dyn_ad_tensor;
mod dyn_ad_value;
mod scalar;
mod tensor;

pub use dyn_ad_tensor::DynAdTensor;
pub use dyn_ad_value::DynAdValue;
pub use scalar::{DynScalar, ScalarType};
pub use tensor::DynTensor;

#[cfg(test)]
mod tests {
    use tenferro_tensor::{MemoryOrder, Tensor};

    use super::*;
    use crate::{AdMode, AdTensor, AdValue};

    #[test]
    fn dyn_scalar_metadata() {
        let x: DynScalar = 1.0_f64.into();
        assert_eq!(x.scalar_type(), ScalarType::F64);
        assert_eq!(x.as_f64(), Some(1.0));
    }

    #[test]
    fn dyn_ad_value_mode_and_tangent() {
        let x: DynAdValue = AdValue::forward(2.0_f32, 0.5_f32).into();
        assert_eq!(x.scalar_type(), ScalarType::F32);
        assert_eq!(x.mode(), AdMode::Forward);
        assert_eq!(x.tangent(), Some(DynScalar::F32(0.5)));
    }

    #[test]
    fn dyn_tensor_and_dyn_ad_tensor_dims() {
        let t = Tensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();
        let d: DynTensor = t.clone().into();
        assert_eq!(d.dims(), &[2]);

        let ad = AdTensor::new_primal(t);
        let dad: DynAdTensor = ad.into();
        assert_eq!(dad.dims(), &[2]);
        assert_eq!(dad.mode(), AdMode::Primal);
    }
}
