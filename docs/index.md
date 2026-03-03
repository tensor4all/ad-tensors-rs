# ad-tensors-rs Documentation

This site provides design documents for `ad-tensors-rs`.

## Entry points

- [Design documents](./design/index.html)
- [Rust API docs](../api/index.html)

## Scope

`ad-tensors-rs` provides:

1. `AdValue<T>`-based generic AD abstractions and tensor wrappers.
2. `Dyn*` wrappers (`DynScalar`, `DynTensor`, `DynAdValue`, `DynAdTensor`) for runtime dtype dispatch.
3. Builder-based operation API (`einsum`, `*_ad`, and full linalg surface) executed via `run()`.
