use std::collections::HashMap;

use tenferro_algebra::{Algebra, HasAlgebra, Scalar};
use tenferro_linalg::backend::TensorLinalgContextFor;
use tenferro_linalg::{LinalgScalar, SvdOptions, SvdResult};
use tenferro_prims::TensorPrims;
use tenferro_tensor::Tensor;

use crate::context::with_global_context;
use crate::{Error, Result};

/// Explicit-context einsum API (shape-compatible with tenferro-einsum).
///
/// This POC only defines the interface and currently returns
/// [`Error::NotImplemented`].
pub fn einsum<Alg, Backend>(
    ctx: &mut Backend::Context,
    subscripts: &str,
    operands: &[&Tensor<Alg::Scalar>],
    size_dict: Option<&HashMap<u32, usize>>,
) -> Result<Tensor<Alg::Scalar>>
where
    Alg: Algebra,
    Alg::Scalar: Scalar + HasAlgebra<Algebra = Alg>,
    Backend: TensorPrims<Alg>,
{
    let _ = (ctx, subscripts, operands, size_dict);
    Err(Error::NotImplemented { op: "einsum" })
}

/// Global-context convenience wrapper for [`einsum`].
pub fn einsum_auto<Alg, Backend>(
    subscripts: &str,
    operands: &[&Tensor<Alg::Scalar>],
    size_dict: Option<&HashMap<u32, usize>>,
) -> Result<Tensor<Alg::Scalar>>
where
    Alg: Algebra,
    Alg::Scalar: Scalar + HasAlgebra<Algebra = Alg>,
    Backend: TensorPrims<Alg>,
    Backend::Context: 'static,
{
    with_global_context::<Backend::Context, _>(|ctx| {
        einsum::<Alg, Backend>(ctx, subscripts, operands, size_dict)
    })
}

/// Explicit-context SVD API (shape-compatible with tenferro-linalg).
///
/// This POC only defines the interface and currently returns
/// [`Error::NotImplemented`].
pub fn svd<T: LinalgScalar, C>(
    ctx: &mut C,
    tensor: &Tensor<T>,
    options: Option<&SvdOptions>,
) -> Result<SvdResult<T, T::Real>>
where
    C: TensorLinalgContextFor<T>,
{
    let _ = (ctx, tensor, options);
    Err(Error::NotImplemented { op: "svd" })
}

/// Global-context convenience wrapper for [`svd`].
pub fn svd_auto<T: LinalgScalar, C>(
    tensor: &Tensor<T>,
    options: Option<&SvdOptions>,
) -> Result<SvdResult<T, T::Real>>
where
    C: TensorLinalgContextFor<T> + 'static,
{
    with_global_context::<C, _>(|ctx| svd::<T, C>(ctx, tensor, options))
}
