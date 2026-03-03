# ad-tensors-rs Documentation

This site provides design documents for `ad-tensors-rs`.

## Entry points

- [Design documents](./design/index.html)
- [Rust API docs](../api/index.html)

## Scope

`ad-tensors-rs` defines an interface-level AD tensor layer with two public planes:

1. `AdValue<T>`-based generic abstractions for user-extensible scalar/tensor payloads.
2. `Dyn*` wrappers (`DynScalar`, `DynTensor`, `DynAdValue`, `DynAdTensor`) for runtime dtype dispatch.

Numeric kernels are intentionally left unimplemented in this POC.
