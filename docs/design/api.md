# Public API

## Operation entry points

Current operation entry points:

- `einsum` / `einsum_auto` for primal tensors
- `einsum_ad` / `einsum_ad_auto` for AD tensors
- `svd` / `svd_auto` for linalg factorization surface

Implementation status:

- `einsum` / `einsum_auto`: implemented with `tenferro-einsum`
- `einsum_ad` / `einsum_ad_auto`: implemented with AD mode propagation
- `svd` / `svd_auto`: implemented with `tenferro-linalg`

`einsum_ad` uses the following mode precedence:

- if any operand is `Reverse`, output is `Reverse`
- else if any operand is `Forward`, output is `Forward`
- else output is `Primal`

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
