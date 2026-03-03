//! ad-tensors-rs: three-mode AD tensor-layer API skeleton.
//!
//! This crate defines interface-level types and function signatures for:
//! - `Primal` evaluation
//! - `Dual` forward-mode differentiation
//! - `Tracked` reverse-mode differentiation
//!
//! Numeric kernels are intentionally not implemented yet in this POC.

pub mod api;
pub mod context;
pub mod error;
pub mod mode;
pub mod policy;
pub mod scalar;
pub mod traits;

pub use api::{einsum, einsum_auto, svd, svd_auto};
pub use context::{
    set_global_context, try_with_global_context, with_global_context, GlobalContextGuard,
};
pub use error::{Error, Result};
pub use mode::{Dual, NodeId, Primal, TapeId, Tracked};
pub use policy::DiffPolicy;
pub use scalar::{AnyScalar, BaseScalar, BaseScalarLike};
pub use traits::{
    AdResult, AllowedPairs, Differentiable, FactorizeOptions, FactorizeResult, IndexLike, OpRule,
    TensorKernel,
};
