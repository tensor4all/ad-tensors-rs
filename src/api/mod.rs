use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

use chainrules_core::Differentiable as _;
use num_complex::Complex;
use num_traits::Float;
use tenferro_algebra::{HasAlgebra, Scalar, Standard};
use tenferro_einsum as tf_einsum;
use tenferro_linalg::backend::CpuLinalgScalar;
use tenferro_linalg::{
    EigResult, EigenResult, LinalgScalar, LstsqResult, LuPivot, LuResult, NormKind, QrResult,
    SlogdetResult, SvdOptions, SvdResult,
};
use tenferro_prims::{CpuBackend, CpuContext, TensorPrims};
use tenferro_tensor::{MemoryOrder, Tensor};

use crate::ad_value::{AdValue, NodeId};
use crate::runtime::{with_default_runtime, RuntimeContext};
use crate::{AdTensor, Error, Result, TapeId};

mod ad_results;

pub use ad_results::{
    AdEigResult, AdEigenResult, AdLstsqResult, AdLuResult, AdQrResult, AdSlogdetResult, AdSvdResult,
};

fn with_cpu_runtime<R>(
    op: &'static str,
    f: impl FnOnce(&mut CpuContext) -> Result<R>,
) -> Result<R> {
    with_default_runtime(|runtime| match runtime {
        RuntimeContext::Cpu(ctx) => f(ctx),
        RuntimeContext::Cuda(_) => Err(Error::UnsupportedRuntimeOp {
            op,
            runtime: "cuda",
        }),
        RuntimeContext::Rocm(_) => Err(Error::UnsupportedRuntimeOp {
            op,
            runtime: "rocm",
        }),
    })
}

fn has_forward<S: Scalar>(operands: &[&AdTensor<S>]) -> bool {
    operands
        .iter()
        .any(|op| matches!(op.as_value(), AdValue::Forward { .. }))
}

fn has_reverse<S: Scalar>(operands: &[&AdTensor<S>]) -> bool {
    operands
        .iter()
        .any(|op| matches!(op.as_value(), AdValue::Reverse { .. }))
}

fn has_any_tangent<S: Scalar>(operands: &[&AdTensor<S>]) -> bool {
    operands.iter().any(|op| op.tangent().is_some())
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
    op_name: &str,
    operands: &[&AdTensor<S>],
    output_dims: &[usize],
    output_tag: u64,
    tape: TapeId,
) -> NodeId {
    let mut hasher = DefaultHasher::new();
    op_name.hash(&mut hasher);
    output_dims.hash(&mut hasher);
    output_tag.hash(&mut hasher);
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

fn zero_like<T: Scalar>(tensor: &Tensor<T>) -> Tensor<T> {
    Tensor::zeros(
        tensor.dims(),
        tensor.logical_memory_space(),
        MemoryOrder::ColumnMajor,
    )
}

fn wrap_ad_output<TIn: Scalar, TOut: Scalar>(
    op_name: &'static str,
    inputs: &[&AdTensor<TIn>],
    primal: Tensor<TOut>,
    tangent: Option<Tensor<TOut>>,
    output_tag: u64,
) -> Result<AdTensor<TOut>> {
    if has_reverse(inputs) {
        let tape = derive_reverse_tape(inputs)?.ok_or_else(|| Error::InvalidAdTensor {
            message: "reverse-mode output requested but no reverse tape found".to_string(),
        })?;
        let node = derive_reverse_node(op_name, inputs, primal.dims(), output_tag, tape);
        return Ok(AdTensor::new_reverse(primal, node, tape, tangent));
    }

    if has_forward(inputs) {
        let tangent = tangent.ok_or_else(|| Error::InvalidAdTensor {
            message: "forward-mode inputs must provide tangent output".to_string(),
        })?;
        return Ok(AdTensor::new_forward(primal, tangent));
    }

    Ok(AdTensor::new_primal(primal))
}

mod einsum;
mod linalg_ad_decomp;
mod linalg_ad_matrix;
mod linalg_primal_decomp;
mod linalg_primal_matrix;

pub use einsum::*;
pub use linalg_ad_decomp::*;
pub use linalg_ad_matrix::*;
pub use linalg_primal_decomp::*;
pub use linalg_primal_matrix::*;

fn run_unary_tensor_ad<T, FPrimal, FFrule>(
    op_name: &'static str,
    input: &AdTensor<T>,
    primal_fn: FPrimal,
    frule_fn: FFrule,
) -> Result<AdTensor<T>>
where
    T: LinalgScalar<Real = T> + Float + CpuLinalgScalar,
    FPrimal: FnOnce(&mut CpuContext, &Tensor<T>) -> Result<Tensor<T>>,
    FFrule: FnOnce(
        &mut CpuContext,
        &Tensor<T>,
        &Tensor<T>,
    )
        -> std::result::Result<(Tensor<T>, Tensor<T>), chainrules_core::AutodiffError>,
{
    let operands = [input];
    let needs_tangent = has_forward(&operands) || has_any_tangent(&operands);

    let (primal, tangent) = if needs_tangent {
        let in_tangent = input
            .tangent()
            .cloned()
            .unwrap_or_else(|| zero_like(input.primal()));
        let (p, d) = with_cpu_runtime(op_name, |ctx| {
            frule_fn(ctx, input.primal(), &in_tangent).map_err(Error::from)
        })?;
        (p, Some(d))
    } else {
        (
            with_cpu_runtime(op_name, |ctx| primal_fn(ctx, input.primal()))?,
            None,
        )
    };

    wrap_ad_output(op_name, &operands, primal, tangent, 0)
}

fn run_binary_tensor_ad<T, FPrimal, FFrule>(
    op_name: &'static str,
    a: &AdTensor<T>,
    b: &AdTensor<T>,
    primal_fn: FPrimal,
    frule_fn: FFrule,
) -> Result<AdTensor<T>>
where
    T: LinalgScalar<Real = T> + Float + CpuLinalgScalar,
    FPrimal: FnOnce(&mut CpuContext, &Tensor<T>, &Tensor<T>) -> Result<Tensor<T>>,
    FFrule: FnOnce(
        &mut CpuContext,
        &Tensor<T>,
        &Tensor<T>,
        &Tensor<T>,
        &Tensor<T>,
    )
        -> std::result::Result<(Tensor<T>, Tensor<T>), chainrules_core::AutodiffError>,
{
    let operands = [a, b];
    let needs_tangent = has_forward(&operands) || has_any_tangent(&operands);

    let (primal, tangent) = if needs_tangent {
        let ta = a
            .tangent()
            .cloned()
            .unwrap_or_else(|| zero_like(a.primal()));
        let tb = b
            .tangent()
            .cloned()
            .unwrap_or_else(|| zero_like(b.primal()));
        let (p, d) = with_cpu_runtime(op_name, |ctx| {
            frule_fn(ctx, a.primal(), b.primal(), &ta, &tb).map_err(Error::from)
        })?;
        (p, Some(d))
    } else {
        (
            with_cpu_runtime(op_name, |ctx| primal_fn(ctx, a.primal(), b.primal()))?,
            None,
        )
    };

    wrap_ad_output(op_name, &operands, primal, tangent, 0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tenferro_tensor::MemoryOrder;

    fn as_slice<T: Scalar>(t: &Tensor<T>) -> &[T] {
        t.buffer()
            .as_slice()
            .unwrap_or_else(|| panic!("expected CPU-backed contiguous tensor"))
    }

    #[test]
    fn run_requires_runtime() {
        let t = Tensor::<f64>::from_slice(&[1.0, 0.0, 0.0, 1.0], &[2, 2], MemoryOrder::ColumnMajor)
            .unwrap();
        let err = qr(&t).run().err();
        assert!(matches!(err, Some(Error::RuntimeNotConfigured)));
    }

    #[test]
    fn primal_einsum_builder_runs() {
        let _guard = crate::set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));
        let a = Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2], MemoryOrder::ColumnMajor)
            .unwrap();
        let b = Tensor::<f64>::from_slice(&[5.0, 6.0, 7.0, 8.0], &[2, 2], MemoryOrder::ColumnMajor)
            .unwrap();
        let out = einsum("ij,jk->ik", &[&a, &b]).run().unwrap();
        assert_eq!(out.dims(), &[2, 2]);
        assert_eq!(as_slice(&out).len(), 4);
    }

    #[test]
    fn primal_qr_builder_runs() {
        let _guard = crate::set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));
        let t = Tensor::<f64>::from_slice(&[1.0, 0.0, 0.0, 1.0], &[2, 2], MemoryOrder::ColumnMajor)
            .unwrap();
        let out = qr(&t).run().unwrap();
        assert_eq!(out.q.dims(), &[2, 2]);
        assert_eq!(out.r.dims(), &[2, 2]);
    }

    #[test]
    fn solve_triangular_ad_rejects_non_primal() {
        let _guard = crate::set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));
        let a = Tensor::<f64>::from_slice(&[1.0, 0.0, 0.0, 1.0], &[2, 2], MemoryOrder::ColumnMajor)
            .unwrap();
        let b = Tensor::<f64>::from_slice(&[1.0, 1.0], &[2], MemoryOrder::ColumnMajor).unwrap();
        let da =
            Tensor::<f64>::from_slice(&[0.1, 0.0, 0.0, 0.1], &[2, 2], MemoryOrder::ColumnMajor)
                .unwrap();

        let ad_a = AdTensor::new_forward(a, da);
        let ad_b = AdTensor::new_primal(b);
        let err = solve_triangular_ad(&ad_a, &ad_b).run().err();
        assert!(matches!(err, Some(Error::UnsupportedAdOp { .. })));
    }

    fn assert_primal_mode(t: &AdTensor<f64>) {
        assert!(matches!(t.as_value(), AdValue::Primal(_)));
    }

    #[test]
    fn primal_linalg_builders_cover_all_ops() {
        let _guard = crate::set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));

        let a = Tensor::<f64>::from_slice(&[4.0, 1.0, 1.0, 3.0], &[2, 2], MemoryOrder::ColumnMajor)
            .unwrap();
        let b = Tensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();
        let tri =
            Tensor::<f64>::from_slice(&[2.0, 0.0, 1.0, 3.0], &[2, 2], MemoryOrder::ColumnMajor)
                .unwrap();
        let a_general =
            Tensor::<f64>::from_slice(&[0.0, 1.0, -1.0, 0.0], &[2, 2], MemoryOrder::ColumnMajor)
                .unwrap();
        let a_rect = Tensor::<f64>::from_slice(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[2, 3],
            MemoryOrder::ColumnMajor,
        )
        .unwrap();
        let a_ls = Tensor::<f64>::from_slice(
            &[1.0, 0.0, 1.0, 0.0, 1.0, 1.0],
            &[3, 2],
            MemoryOrder::ColumnMajor,
        )
        .unwrap();
        let b_ls =
            Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0], &[3], MemoryOrder::ColumnMajor).unwrap();

        let out_svd = svd(&a).run().unwrap();
        assert_eq!(out_svd.s.dims(), &[2]);
        let out_qr = qr(&a).run().unwrap();
        assert_eq!(out_qr.q.dims(), &[2, 2]);
        let out_lu = lu(&a).pivot(LuPivot::Partial).run().unwrap();
        assert_eq!(out_lu.l.dims(), &[2, 2]);
        let out_eigen = eigen(&a).run().unwrap();
        assert_eq!(out_eigen.values.dims(), &[2]);
        let out_lstsq = lstsq(&a_ls, &b_ls).run().unwrap();
        assert_eq!(out_lstsq.x.dims(), &[2]);
        let out_cholesky = cholesky(&a).run().unwrap();
        assert_eq!(out_cholesky.dims(), &[2, 2]);
        let out_solve = solve(&a, &b).run().unwrap();
        assert_eq!(out_solve.dims(), &[2]);
        let out_inv = inv(&a).run().unwrap();
        assert_eq!(out_inv.dims(), &[2, 2]);
        let out_det = det(&a).run().unwrap();
        assert_eq!(out_det.dims(), &[]);
        let out_slogdet = slogdet(&a).run().unwrap();
        assert_eq!(out_slogdet.sign.dims(), &[]);
        let out_eig = eig(&a_general).run().unwrap();
        assert_eq!(out_eig.values.dims(), &[2]);
        let out_pinv = pinv(&a_rect).rcond(1e-12).run().unwrap();
        assert_eq!(out_pinv.dims(), &[3, 2]);
        let out_exp = matrix_exp(&a).run().unwrap();
        assert_eq!(out_exp.dims(), &[2, 2]);
        let out_tri = solve_triangular(&tri, &b).upper(true).run().unwrap();
        assert_eq!(out_tri.dims(), &[2]);
        let out_norm = norm(&a).kind(NormKind::Fro).run().unwrap();
        assert_eq!(out_norm.dims(), &[]);
    }

    #[test]
    fn ad_linalg_builders_cover_all_ops_in_primal_mode() {
        let _guard = crate::set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));

        let a = Tensor::<f64>::from_slice(&[4.0, 1.0, 1.0, 3.0], &[2, 2], MemoryOrder::ColumnMajor)
            .unwrap();
        let b = Tensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();
        let tri =
            Tensor::<f64>::from_slice(&[2.0, 0.0, 1.0, 3.0], &[2, 2], MemoryOrder::ColumnMajor)
                .unwrap();
        let a_general =
            Tensor::<f64>::from_slice(&[0.0, 1.0, -1.0, 0.0], &[2, 2], MemoryOrder::ColumnMajor)
                .unwrap();
        let a_rect = Tensor::<f64>::from_slice(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[2, 3],
            MemoryOrder::ColumnMajor,
        )
        .unwrap();
        let a_ls = Tensor::<f64>::from_slice(
            &[1.0, 0.0, 1.0, 0.0, 1.0, 1.0],
            &[3, 2],
            MemoryOrder::ColumnMajor,
        )
        .unwrap();
        let b_ls =
            Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0], &[3], MemoryOrder::ColumnMajor).unwrap();

        let ad_a = AdTensor::new_primal(a);
        let ad_b = AdTensor::new_primal(b);
        let ad_tri = AdTensor::new_primal(tri);
        let ad_general = AdTensor::new_primal(a_general);
        let ad_rect = AdTensor::new_primal(a_rect);
        let ad_ls_a = AdTensor::new_primal(a_ls);
        let ad_ls_b = AdTensor::new_primal(b_ls);

        let out_svd = svd_ad(&ad_a).run().unwrap();
        assert_primal_mode(&out_svd.u);
        assert_primal_mode(&out_svd.s);
        assert_primal_mode(&out_svd.vt);

        let out_qr = qr_ad(&ad_a).run().unwrap();
        assert_primal_mode(&out_qr.q);
        assert_primal_mode(&out_qr.r);

        let out_lu = lu_ad(&ad_a).run().unwrap();
        assert_primal_mode(&out_lu.l);
        assert_primal_mode(&out_lu.u);

        let out_eigen = eigen_ad(&ad_a).run().unwrap();
        assert_primal_mode(&out_eigen.values);
        assert_primal_mode(&out_eigen.vectors);

        let out_lstsq = lstsq_ad(&ad_ls_a, &ad_ls_b).run().unwrap();
        assert_primal_mode(&out_lstsq.x);
        assert_primal_mode(&out_lstsq.residual);

        assert_primal_mode(&cholesky_ad(&ad_a).run().unwrap());
        assert_primal_mode(&solve_ad(&ad_a, &ad_b).run().unwrap());
        assert_primal_mode(&inv_ad(&ad_a).run().unwrap());
        assert_primal_mode(&det_ad(&ad_a).run().unwrap());

        let out_slogdet = slogdet_ad(&ad_a).run().unwrap();
        assert_primal_mode(&out_slogdet.sign);
        assert_primal_mode(&out_slogdet.logabsdet);

        let out_eig = eig_ad(&ad_general).run().unwrap();
        assert!(matches!(out_eig.values.as_value(), AdValue::Primal(_)));
        assert!(matches!(out_eig.vectors.as_value(), AdValue::Primal(_)));

        assert_primal_mode(&pinv_ad(&ad_rect).run().unwrap());
        assert_primal_mode(&matrix_exp_ad(&ad_a).run().unwrap());
        assert_primal_mode(&solve_triangular_ad(&ad_tri, &ad_b).run().unwrap());
        assert_primal_mode(&norm_ad(&ad_a).kind(NormKind::Fro).run().unwrap());
    }

    #[test]
    fn ad_mode_propagation_forward_and_reverse() {
        let _guard = crate::set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));

        let a = Tensor::<f64>::from_slice(&[4.0, 1.0, 1.0, 3.0], &[2, 2], MemoryOrder::ColumnMajor)
            .unwrap();
        let da =
            Tensor::<f64>::from_slice(&[0.1, 0.0, 0.0, 0.1], &[2, 2], MemoryOrder::ColumnMajor)
                .unwrap();
        let b = Tensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();

        let ad_a_fwd = AdTensor::new_forward(a.clone(), da);
        let ad_b = AdTensor::new_primal(b);
        let out_fwd = solve_ad(&ad_a_fwd, &ad_b).run().unwrap();
        assert!(matches!(out_fwd.as_value(), AdValue::Forward { .. }));

        let ad_a_rev = AdTensor::new_reverse(a.clone(), NodeId(1), TapeId(11), None);
        let ad_b_rev = AdTensor::new_reverse(a, NodeId(2), TapeId(11), None);
        let out_rev = einsum_ad("ij,jk->ik", &[&ad_a_rev, &ad_b_rev])
            .run()
            .unwrap();
        assert!(matches!(out_rev.as_value(), AdValue::Reverse { tape, .. } if *tape == TapeId(11)));
    }
}
