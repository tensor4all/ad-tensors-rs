use std::hash::Hash;

use crate::{BaseScalar, DiffPolicy, Dual, Primal, Result, Tracked};

/// AD rule method result type alias.
pub type AdResult<T> = Result<T>;

/// Trait bound for index-like labels used in contraction/factorization APIs.
pub trait IndexLike: Clone + Eq + Hash {}

impl<T> IndexLike for T where T: Clone + Eq + Hash {}

/// A differentiable value with an associated tangent type.
pub trait Differentiable: Clone {
    type Tangent: Clone;
}

impl<T: Clone> Differentiable for Primal<T> {
    type Tangent = T;
}

impl<T: Clone> Differentiable for Dual<T> {
    type Tangent = T;
}

impl<T: Clone> Differentiable for Tracked<T> {
    type Tangent = T;
}

/// Allowed index-pair restrictions for contraction planning.
#[derive(Debug, Clone, Copy)]
pub struct AllowedPairs<'a> {
    pub pairs: &'a [(usize, usize)],
}

/// Factorization options for generic tensor-kernel APIs.
#[derive(Debug, Clone)]
pub struct FactorizeOptions {
    pub max_rank: Option<usize>,
    pub diff_policy: DiffPolicy,
}

impl Default for FactorizeOptions {
    fn default() -> Self {
        Self {
            max_rank: None,
            diff_policy: DiffPolicy::StopGradient,
        }
    }
}

/// Generic factorization output container.
#[derive(Debug, Clone)]
pub struct FactorizeResult<T> {
    pub left: T,
    pub right: T,
}

/// Numeric tensor-kernel boundary (primal kernels only).
pub trait TensorKernel: Clone {
    type Index: IndexLike;

    fn contract(tensors: &[&Self], allowed: AllowedPairs<'_>) -> Result<Self>;
    fn factorize(
        &self,
        left_inds: &[Self::Index],
        options: &FactorizeOptions,
    ) -> Result<FactorizeResult<Self>>;
    fn axpby(&self, a: BaseScalar, other: &Self, b: BaseScalar) -> Result<Self>;
    fn scale(&self, a: BaseScalar) -> Result<Self>;
    fn inner_product(&self, other: &Self) -> Result<BaseScalar>;
}

/// Operation-level AD rules (rrule/frule/hvp).
pub trait OpRule<V: Differentiable> {
    fn eval(&self, inputs: &[&V]) -> Result<V>;
    fn rrule(&self, inputs: &[&V], out: &V, cotangent: &V::Tangent) -> AdResult<Vec<V::Tangent>>;
    fn frule(&self, inputs: &[&V], tangents: &[Option<&V::Tangent>]) -> AdResult<V::Tangent>;
    fn hvp(
        &self,
        inputs: &[&V],
        cotangent: &V::Tangent,
        cotangent_tangent: Option<&V::Tangent>,
        input_tangents: &[Option<&V::Tangent>],
    ) -> AdResult<Vec<(V::Tangent, V::Tangent)>>;
}
