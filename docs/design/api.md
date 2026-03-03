# Public API

## Operation entry points

Current POC signatures include:

- `einsum` / `einsum_auto` for primal tensors
- `einsum_ad` / `einsum_ad_auto` for AD tensors
- `svd` / `svd_auto` for linalg factorization surface

All currently return `Error::NotImplemented`.

## Context model

`*_auto` functions use thread-local context utilities:

- `set_global_context`
- `with_global_context`
- `try_with_global_context`

The context key is the concrete Rust type.

## Extension interfaces

- `TensorKernel`: numeric kernel boundary
- `Differentiable`: abstraction over values that can expose `AdValue`
- `OpRule`: operation-level AD rules (`rrule`, `frule`, `hvp`)

## Documentation rule

Every public signature is documented with a minimal usage example in rustdoc.
