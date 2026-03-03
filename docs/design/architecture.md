# Architecture

`ad-tensors-rs` is an interface crate that sits on top of `tenferro-rs`.

## Layers

1. **Core AD values (`ad_value`)**
   - `AdValue<T>`: mode + payload (`Primal` / `Forward` / `Reverse`)
   - `AdScalar<T>`, `AdTensor<T>`: semantic wrappers over `AdValue<T>` and `AdValue<Tensor<T>>`

2. **Runtime dtype wrappers (`dyn_types`)**
   - `DynScalar`, `DynTensor`, `DynAdValue`, `DynAdTensor`
   - fixed runtime dtype set (`f32`, `f64`, `Complex32`, `Complex64`)

3. **Operational boundaries (`api`, `traits`)**
   - POC operation signatures (`einsum`, `einsum_ad`, `svd`, and auto-context wrappers)
   - abstraction traits (`TensorKernel`, `OpRule`, `Differentiable`)

4. **Execution context (`context`)**
   - thread-local context registration and lookup for `*_auto` APIs

## Non-goals in this phase

- No numerical kernels.
- No backend-specific scheduling.
- No mixed-dtype arithmetic rules.

All operation entry points currently return `Error::NotImplemented`.
