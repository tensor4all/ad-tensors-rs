//! `ad-tensors-rs`: AD-aware tensor interface skeleton on top of `tenferro-rs`.
//!
//! This crate exposes two layers:
//! - Generic, user-extensible AD values via [`AdValue`]
//! - Runtime dtype wrappers via `Dyn*` enums for FFI and dynamic dispatch
//!
//! Numeric kernels are intentionally not implemented yet in this POC.

pub mod ad_value;
pub mod api;
pub mod context;
pub mod dyn_types;
pub mod error;
pub mod policy;
pub mod traits;

pub use ad_value::{AdMode, AdScalar, AdTensor, AdValue, NodeId, TapeId};
pub use api::{einsum, einsum_ad, einsum_ad_auto, einsum_auto, svd, svd_auto};
pub use context::{
    set_global_context, try_with_global_context, with_global_context, GlobalContextGuard,
};
pub use dyn_types::{DynAdTensor, DynAdValue, DynScalar, DynTensor, ScalarType};
pub use error::{Error, Result};
pub use policy::DiffPolicy;
pub use traits::{
    AdResult, AllowedPairs, Differentiable, FactorizeOptions, FactorizeResult, IndexLike, OpRule,
    TensorKernel,
};
