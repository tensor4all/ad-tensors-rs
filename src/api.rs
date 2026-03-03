use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

use chainrules_core::Differentiable;
use tenferro_algebra::{Algebra, HasAlgebra, Scalar};
use tenferro_einsum as tf_einsum;
use tenferro_linalg::backend::TensorLinalgContextFor;
use tenferro_linalg::{LinalgScalar, SvdOptions, SvdResult};
use tenferro_prims::TensorPrims;
use tenferro_tensor::Tensor;

use crate::ad_value::{AdValue, NodeId};
use crate::context::with_global_context;
use crate::{AdTensor, Error, Result, TapeId};

/// Explicit-context einsum API for primal tensors.
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
    tf_einsum::einsum::<Alg, Backend>(ctx, subscripts, operands, size_dict).map_err(Error::from)
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

fn collect_ad_tangents<'a, S: Scalar>(operands: &[&'a AdTensor<S>]) -> Vec<Option<&'a Tensor<S>>> {
    operands
        .iter()
        .map(|op| match op.as_value() {
            AdValue::Primal(_) => None,
            AdValue::Forward { tangent, .. } => Some(tangent),
            AdValue::Reverse { tangent, .. } => tangent.as_ref(),
        })
        .collect()
}

fn sum_einsum_tangent_terms<Alg, Backend>(
    ctx: &mut Backend::Context,
    subscripts: &str,
    primals: &[&Tensor<Alg::Scalar>],
    tangents: &[Option<&Tensor<Alg::Scalar>>],
    size_dict: Option<&HashMap<u32, usize>>,
) -> Result<Option<Tensor<Alg::Scalar>>>
where
    Alg: Algebra,
    Alg::Scalar: Scalar + HasAlgebra<Algebra = Alg>,
    Backend: TensorPrims<Alg>,
{
    let mut out_tangent: Option<Tensor<Alg::Scalar>> = None;

    for (k, tangent_opt) in tangents.iter().enumerate() {
        let Some(tangent_k) = tangent_opt else {
            continue;
        };

        let mut term_operands: Vec<&Tensor<Alg::Scalar>> = primals.to_vec();
        term_operands[k] = tangent_k;
        let term = tf_einsum::einsum::<Alg, Backend>(ctx, subscripts, &term_operands, size_dict)
            .map_err(Error::from)?;

        out_tangent = Some(match out_tangent {
            None => term,
            Some(existing) => Tensor::<Alg::Scalar>::accumulate_tangent(existing, &term),
        });
    }

    Ok(out_tangent)
}

fn derive_reverse_tape<S: Scalar>(operands: &[&AdTensor<S>]) -> Result<Option<TapeId>> {
    let mut tape: Option<TapeId> = None;

    for op in operands {
        if let AdValue::Reverse { tape: current, .. } = op.as_value() {
            if let Some(expected) = tape {
                if expected != *current {
                    return Err(Error::MixedReverseTape {
                        expected: expected.0,
                        found: current.0,
                    });
                }
            } else {
                tape = Some(*current);
            }
        }
    }

    Ok(tape)
}

fn derive_reverse_node<S: Scalar>(
    subscripts: &str,
    operands: &[&AdTensor<S>],
    output_dims: &[usize],
    tape: TapeId,
) -> NodeId {
    let mut hasher = DefaultHasher::new();
    subscripts.hash(&mut hasher);
    output_dims.hash(&mut hasher);
    tape.0.hash(&mut hasher);

    for (idx, op) in operands.iter().enumerate() {
        idx.hash(&mut hasher);
        match op.as_value() {
            AdValue::Primal(_) => {
                0_u8.hash(&mut hasher);
            }
            AdValue::Forward { .. } => {
                1_u8.hash(&mut hasher);
            }
            AdValue::Reverse { node, .. } => {
                2_u8.hash(&mut hasher);
                node.0.hash(&mut hasher);
            }
        }
        op.dims().hash(&mut hasher);
    }

    NodeId(hasher.finish())
}

/// Explicit-context einsum API for AD tensors.
///
/// AD mode propagation rules:
/// - If any input is `Reverse`, output is `Reverse`.
/// - Else if any input is `Forward`, output is `Forward`.
/// - Else output is `Primal`.
///
/// Tangents are propagated by summing per-input directional terms.
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
    let primals: Vec<&Tensor<Alg::Scalar>> = operands.iter().map(|op| op.primal()).collect();
    let primal_out = tf_einsum::einsum::<Alg, Backend>(ctx, subscripts, &primals, size_dict)
        .map_err(Error::from)?;

    let has_reverse = operands
        .iter()
        .any(|op| matches!(op.as_value(), AdValue::Reverse { .. }));
    let has_forward = operands
        .iter()
        .any(|op| matches!(op.as_value(), AdValue::Forward { .. }));

    if !has_reverse && !has_forward {
        return Ok(AdTensor::new_primal(primal_out));
    }

    let tangents = collect_ad_tangents(operands);
    let tangent_out =
        sum_einsum_tangent_terms::<Alg, Backend>(ctx, subscripts, &primals, &tangents, size_dict)?;

    if has_reverse {
        let tape = derive_reverse_tape(operands)?.ok_or_else(|| Error::InvalidAdTensor {
            message: "reverse-mode output requested but no reverse tape found".to_string(),
        })?;
        let node = derive_reverse_node(subscripts, operands, primal_out.dims(), tape);
        return Ok(AdTensor::from(AdValue::Reverse {
            primal: primal_out,
            node,
            tape,
            tangent: tangent_out,
        }));
    }

    let tangent = tangent_out.ok_or_else(|| Error::InvalidAdTensor {
        message: "forward-mode inputs must provide at least one tangent".to_string(),
    })?;

    Ok(AdTensor::new_forward(primal_out, tangent))
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
    tenferro_linalg::svd(ctx, tensor, options).map_err(Error::from)
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

#[cfg(test)]
mod tests {
    use super::*;
    use tenferro_algebra::Standard;
    use tenferro_prims::{CpuBackend, CpuContext};
    use tenferro_tensor::MemoryOrder;

    fn as_slice<'a, T: Scalar>(t: &'a Tensor<T>) -> &'a [T] {
        t.buffer()
            .as_slice()
            .unwrap_or_else(|| panic!("expected CPU-backed contiguous tensor"))
    }

    #[test]
    fn einsum_primal_matches_tenferro_einsum() {
        let mut ctx = CpuContext::new(1);
        let a = Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2], MemoryOrder::ColumnMajor)
            .unwrap();
        let b = Tensor::<f64>::from_slice(&[5.0, 6.0, 7.0, 8.0], &[2, 2], MemoryOrder::ColumnMajor)
            .unwrap();

        let got =
            einsum::<Standard<f64>, CpuBackend>(&mut ctx, "ij,jk->ik", &[&a, &b], None).unwrap();
        let expected =
            tf_einsum::einsum::<Standard<f64>, CpuBackend>(&mut ctx, "ij,jk->ik", &[&a, &b], None)
                .unwrap();

        assert_eq!(got.dims(), expected.dims());
        assert_eq!(as_slice(&got), as_slice(&expected));
    }

    #[test]
    fn einsum_ad_forward_propagates_tangent() {
        let mut ctx = CpuContext::new(1);
        let a = Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2], MemoryOrder::ColumnMajor)
            .unwrap();
        let da =
            Tensor::<f64>::from_slice(&[0.5, 0.0, 0.0, 0.5], &[2, 2], MemoryOrder::ColumnMajor)
                .unwrap();
        let b = Tensor::<f64>::from_slice(&[2.0, 0.0, 0.0, 2.0], &[2, 2], MemoryOrder::ColumnMajor)
            .unwrap();

        let ad_a = AdTensor::new_forward(a.clone(), da.clone());
        let ad_b = AdTensor::new_primal(b.clone());
        let out =
            einsum_ad::<Standard<f64>, CpuBackend>(&mut ctx, "ij,jk->ik", &[&ad_a, &ad_b], None)
                .unwrap();

        match out.as_value() {
            AdValue::Forward { primal, tangent } => {
                let expected_primal = tf_einsum::einsum::<Standard<f64>, CpuBackend>(
                    &mut ctx,
                    "ij,jk->ik",
                    &[&a, &b],
                    None,
                )
                .unwrap();
                let expected_tangent = tf_einsum::einsum::<Standard<f64>, CpuBackend>(
                    &mut ctx,
                    "ij,jk->ik",
                    &[&da, &b],
                    None,
                )
                .unwrap();
                assert_eq!(primal.dims(), expected_primal.dims());
                assert_eq!(tangent.dims(), expected_tangent.dims());
                assert_eq!(as_slice(primal), as_slice(&expected_primal));
                assert_eq!(as_slice(tangent), as_slice(&expected_tangent));
            }
            _ => panic!("expected forward output"),
        }
    }

    #[test]
    fn einsum_ad_reverse_checks_tape_and_keeps_reverse_mode() {
        let mut ctx = CpuContext::new(1);
        let a = Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2], MemoryOrder::ColumnMajor)
            .unwrap();
        let b = Tensor::<f64>::from_slice(&[5.0, 6.0, 7.0, 8.0], &[2, 2], MemoryOrder::ColumnMajor)
            .unwrap();

        let r0 = AdTensor::new_reverse(a.clone(), NodeId(10), TapeId(2), None);
        let r1 = AdTensor::new_reverse(b.clone(), NodeId(11), TapeId(2), None);
        let out = einsum_ad::<Standard<f64>, CpuBackend>(&mut ctx, "ij,jk->ik", &[&r0, &r1], None)
            .unwrap();
        match out.as_value() {
            AdValue::Reverse { tape, .. } => assert_eq!(*tape, TapeId(2)),
            _ => panic!("expected reverse output"),
        }

        let r_bad = AdTensor::new_reverse(b, NodeId(12), TapeId(9), None);
        let err =
            einsum_ad::<Standard<f64>, CpuBackend>(&mut ctx, "ij,jk->ik", &[&r0, &r_bad], None)
                .err();
        assert!(matches!(err, Some(Error::MixedReverseTape { .. })));
    }

    #[test]
    fn auto_context_wrappers_work() {
        let a = Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2], MemoryOrder::ColumnMajor)
            .unwrap();
        let b = Tensor::<f64>::from_slice(&[5.0, 6.0, 7.0, 8.0], &[2, 2], MemoryOrder::ColumnMajor)
            .unwrap();

        let guard = crate::set_global_context::<CpuContext>(CpuContext::new(1));
        let out = einsum_auto::<Standard<f64>, CpuBackend>("ij,jk->ik", &[&a, &b], None).unwrap();
        assert_eq!(out.dims(), &[2, 2]);

        let ad_a = AdTensor::new_primal(a);
        let ad_b = AdTensor::new_primal(b);
        let ad_out =
            einsum_ad_auto::<Standard<f64>, CpuBackend>("ij,jk->ik", &[&ad_a, &ad_b], None)
                .unwrap();
        assert_eq!(ad_out.dims(), &[2, 2]);

        drop(guard);
    }

    #[test]
    fn svd_wrapper_runs() {
        let mut ctx = CpuContext::new(1);
        let t = Tensor::<f64>::from_slice(&[1.0, 0.0, 0.0, 1.0], &[2, 2], MemoryOrder::ColumnMajor)
            .unwrap();
        let result = svd::<f64, CpuContext>(&mut ctx, &t, None).unwrap();
        assert_eq!(result.s.dims(), &[2]);
    }
}
