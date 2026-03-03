# Public API

## Operation entry points

Current operation entry points:

- `einsum` (builder) for primal tensors
- `einsum_ad` (builder) for AD tensors
- Full linalg builder surface for primal tensors:
  - `svd`, `qr`, `lu`, `eigen`, `lstsq`, `cholesky`, `solve`
  - `inv`, `det`, `slogdet`, `eig`, `pinv`, `matrix_exp`
  - `solve_triangular`, `norm`
- Full linalg builder surface for AD tensors:
  - `svd_ad`, `qr_ad`, `lu_ad`, `eigen_ad`, `lstsq_ad`, `cholesky_ad`, `solve_ad`
  - `inv_ad`, `det_ad`, `slogdet_ad`, `eig_ad`, `pinv_ad`, `matrix_exp_ad`
  - `solve_triangular_ad`, `norm_ad`

Implementation status:

- `einsum`: implemented with `tenferro-einsum`
- `einsum_ad`: implemented with AD mode propagation
- linalg builders: implemented with `tenferro-linalg` primary APIs

`einsum_ad` uses the following mode precedence:

- if any operand is `Reverse`, output is `Reverse`
- else if any operand is `Forward`, output is `Forward`
- else output is `Primal`

## Context model

Builder `.run()` resolves a thread-local default runtime:

- `set_default_runtime`
- `with_default_runtime`
- `RuntimeContext` (`Cpu`, `Cuda`, `Rocm`)

Current execution is CPU-backed; unsupported runtime/op combinations return
explicit runtime errors.

## Extension interfaces

- `TensorKernel`: numeric kernel boundary
- `Differentiable`: abstraction over values that can expose `AdValue`
- `OpRule`: operation-level AD rules (`rrule`, `frule`, `hvp`)

## Documentation rule

Every public signature is documented with a minimal usage example in rustdoc.
