# ad-tensors-rs API

`ad-tensors-rs` is an interface-level crate for AD-aware tensor modeling.

## Main exports

- `AdValue<T>`: generic AD payload model
- `AdScalar<T>`, `AdTensor<T>`: semantic wrappers
- `DynScalar`, `DynTensor`, `DynAdValue`, `DynAdTensor`: runtime dtype wrappers
- `einsum`, `einsum_ad`, `svd` and auto-context variants
- `TensorKernel`, `OpRule`, `Differentiable`

## Status

Operation wrappers are implemented and delegate to `tenferro-einsum` and
`tenferro-linalg`. AD-mode-aware einsum (`einsum_ad`) is implemented with
mode/tangent propagation and reverse-tape consistency checks.
