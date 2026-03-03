# Type System

## Design goals

- Keep the generic extension point open for downstream users.
- Keep runtime dispatch explicit and bounded.
- Make AD mode explicit in type names.

## Generic plane

The primary public abstraction is:

```rust
pub enum AdValue<T> {
    Primal(T),
    Forward { primal: T, tangent: T },
    Reverse { primal: T, node: NodeId, tape: TapeId, tangent: Option<T> },
}
```

This allows users to use custom payload types `T` directly.

`AdScalar<T>` and `AdTensor<T>` are semantic wrappers around this model.

## Runtime plane

Runtime wrappers are intentionally finite:

- `DynScalar`
- `DynTensor`
- `DynAdValue`
- `DynAdTensor`

Their dtype tag is `ScalarType` with variants:

- `F32`
- `F64`
- `C32`
- `C64`

This plane is intended for dynamic boundaries (FFI, plugin-style dispatch, dynamic pipelines).

## Why both planes exist

- Generic plane: open-world extension and compile-time type safety.
- Runtime plane: closed-world dynamic dispatch for integration boundaries.
