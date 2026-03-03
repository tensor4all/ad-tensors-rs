use std::collections::HashMap;

use tenferro_algebra::{Algebra, HasAlgebra, Scalar};
use tenferro_linalg::backend::TensorLinalgContextFor;
use tenferro_linalg::{LinalgScalar, SvdOptions, SvdResult};
use tenferro_prims::TensorPrims;
use tenferro_tensor::Tensor;

use crate::context::with_global_context;
use crate::{AdTensor, Error, Result};

/// Explicit-context einsum API for primal tensors.
///
/// This POC defines signatures only and currently returns
/// [`Error::NotImplemented`].
///
/// # Examples
///
/// ```rust
/// # use ad_tensors_rs::{einsum, Result};
/// # use tenferro_algebra::{Algebra, HasAlgebra, Scalar};
/// # use tenferro_prims::TensorPrims;
/// # use tenferro_tensor::Tensor;
/// # fn demo<Alg, Backend>(
/// #     ctx: &mut Backend::Context,
/// #     a: &Tensor<Alg::Scalar>,
/// #     b: &Tensor<Alg::Scalar>,
/// # ) -> Result<()>
/// # where
/// #     Alg: Algebra,
/// #     Alg::Scalar: Scalar + HasAlgebra<Algebra = Alg>,
/// #     Backend: TensorPrims<Alg>,
/// # {
/// let _out = einsum::<Alg, Backend>(ctx, "ij,jk->ik", &[a, b], None)?;
/// # Ok(())
/// # }
/// ```
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
///
/// # Examples
///
/// ```rust
/// # use ad_tensors_rs::{einsum_auto, Result};
/// # use tenferro_algebra::{Algebra, HasAlgebra, Scalar};
/// # use tenferro_prims::TensorPrims;
/// # use tenferro_tensor::Tensor;
/// # fn demo<Alg, Backend>(a: &Tensor<Alg::Scalar>, b: &Tensor<Alg::Scalar>) -> Result<()>
/// # where
/// #     Alg: Algebra,
/// #     Alg::Scalar: Scalar + HasAlgebra<Algebra = Alg>,
/// #     Backend: TensorPrims<Alg>,
/// #     Backend::Context: 'static,
/// # {
/// let _out = einsum_auto::<Alg, Backend>("ij,jk->ik", &[a, b], None)?;
/// # Ok(())
/// # }
/// ```
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

/// Explicit-context einsum API for AD tensors.
///
/// This POC defines signatures only and currently returns
/// [`Error::NotImplemented`].
///
/// # Examples
///
/// ```rust
/// # use ad_tensors_rs::{einsum_ad, AdTensor, Result};
/// # use tenferro_algebra::{Algebra, HasAlgebra, Scalar};
/// # use tenferro_prims::TensorPrims;
/// # fn demo<Alg, Backend>(
/// #     ctx: &mut Backend::Context,
/// #     a: &AdTensor<Alg::Scalar>,
/// #     b: &AdTensor<Alg::Scalar>,
/// # ) -> Result<()>
/// # where
/// #     Alg: Algebra,
/// #     Alg::Scalar: Scalar + HasAlgebra<Algebra = Alg>,
/// #     Backend: TensorPrims<Alg>,
/// # {
/// let _out = einsum_ad::<Alg, Backend>(ctx, "ij,jk->ik", &[a, b], None)?;
/// # Ok(())
/// # }
/// ```
pub fn einsum_ad<Alg, Backend>(
    ctx: &mut Backend::Context,
    subscripts: &str,
    operands: &[&AdTensor<Alg::Scalar>],
    size_dict: Option<&HashMap<u32, usize>>,
) -> Result<AdTensor<Alg::Scalar>>
where
    Alg: Algebra,
    Alg::Scalar: Scalar + HasAlgebra<Algebra = Alg>,
    Backend: TensorPrims<Alg>,
{
    let _ = (ctx, subscripts, operands, size_dict);
    Err(Error::NotImplemented { op: "einsum_ad" })
}

/// Global-context convenience wrapper for [`einsum_ad`].
///
/// # Examples
///
/// ```rust
/// # use ad_tensors_rs::{einsum_ad_auto, AdTensor, Result};
/// # use tenferro_algebra::{Algebra, HasAlgebra, Scalar};
/// # use tenferro_prims::TensorPrims;
/// # fn demo<Alg, Backend>(a: &AdTensor<Alg::Scalar>, b: &AdTensor<Alg::Scalar>) -> Result<()>
/// # where
/// #     Alg: Algebra,
/// #     Alg::Scalar: Scalar + HasAlgebra<Algebra = Alg>,
/// #     Backend: TensorPrims<Alg>,
/// #     Backend::Context: 'static,
/// # {
/// let _out = einsum_ad_auto::<Alg, Backend>("ij,jk->ik", &[a, b], None)?;
/// # Ok(())
/// # }
/// ```
pub fn einsum_ad_auto<Alg, Backend>(
    subscripts: &str,
    operands: &[&AdTensor<Alg::Scalar>],
    size_dict: Option<&HashMap<u32, usize>>,
) -> Result<AdTensor<Alg::Scalar>>
where
    Alg: Algebra,
    Alg::Scalar: Scalar + HasAlgebra<Algebra = Alg>,
    Backend: TensorPrims<Alg>,
    Backend::Context: 'static,
{
    with_global_context::<Backend::Context, _>(|ctx| {
        einsum_ad::<Alg, Backend>(ctx, subscripts, operands, size_dict)
    })
}

/// Explicit-context SVD API (shape-compatible with `tenferro-linalg`).
///
/// This POC defines signatures only and currently returns
/// [`Error::NotImplemented`].
///
/// # Examples
///
/// ```rust
/// # use ad_tensors_rs::{svd, Result};
/// # use tenferro_linalg::{LinalgScalar, SvdOptions};
/// # use tenferro_linalg::backend::TensorLinalgContextFor;
/// # use tenferro_tensor::Tensor;
/// # fn demo<T, C>(ctx: &mut C, t: &Tensor<T>, opts: Option<&SvdOptions>) -> Result<()>
/// # where
/// #     T: LinalgScalar,
/// #     C: TensorLinalgContextFor<T>,
/// # {
/// let _ = svd::<T, C>(ctx, t, opts)?;
/// # Ok(())
/// # }
/// ```
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
///
/// # Examples
///
/// ```rust
/// # use ad_tensors_rs::{svd_auto, Result};
/// # use tenferro_linalg::{LinalgScalar, SvdOptions};
/// # use tenferro_linalg::backend::TensorLinalgContextFor;
/// # use tenferro_tensor::Tensor;
/// # fn demo<T, C>(t: &Tensor<T>, opts: Option<&SvdOptions>) -> Result<()>
/// # where
/// #     T: LinalgScalar,
/// #     C: TensorLinalgContextFor<T> + 'static,
/// # {
/// let _ = svd_auto::<T, C>(t, opts)?;
/// # Ok(())
/// # }
/// ```
pub fn svd_auto<T: LinalgScalar, C>(
    tensor: &Tensor<T>,
    options: Option<&SvdOptions>,
) -> Result<SvdResult<T, T::Real>>
where
    C: TensorLinalgContextFor<T> + 'static,
{
    with_global_context::<C, _>(|ctx| svd::<T, C>(ctx, tensor, options))
}
