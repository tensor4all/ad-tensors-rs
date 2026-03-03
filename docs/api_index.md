# ad-tensors-rs API

`ad-tensors-rs` is an interface-level crate for AD-aware tensor modeling.

## Main exports

- `AdValue<T>`: generic AD payload model
- `AdScalar<T>`, `AdTensor<T>`: semantic wrappers
- `DynScalar`, `DynTensor`, `DynAdValue`, `DynAdTensor`: runtime dtype wrappers
- `einsum`, `einsum_ad`, `svd` and auto-context variants
- `TensorKernel`, `OpRule`, `Differentiable`

## Status

Numeric kernels are not implemented in this POC. Operation functions return
`Error::NotImplemented`.
