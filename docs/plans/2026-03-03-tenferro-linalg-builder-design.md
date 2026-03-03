# Tenferro Linalg Builder API Design

## Goal
Implement the full `tenferro-linalg` primary operation surface in `ad-tensors-rs` with a unified builder-style API for both `Tensor<T>` and `AdTensor<T>`.

The API removes `*_auto` variants and uses `run()` as the only execution terminal.

## Decisions (Approved)
- Public API is limited to operation functions over `Tensor<T>` / `AdTensor<T>`.
- `rrule` / `frule` functions are not re-exported as public API.
- Builder pattern is used for all operations, including `einsum`.
- Context specification is omitted at call site and unified under runtime resolution.
- `run()` is submit-only (no forced synchronization).
- If runtime is GPU and the operation cannot run on that runtime, return runtime error.

## Scope
Primary linalg operations to expose in builder style:
- `svd`, `qr`, `lu`, `eigen`, `lstsq`, `cholesky`, `solve`, `inv`, `det`, `slogdet`, `eig`, `pinv`, `matrix_exp`, `solve_triangular`, `norm`

Einsum operations to migrate to builder style:
- `einsum`, `einsum_ad`

## Public API Shape

### Primal
```rust
let qr_result = qr(&a).run()?;
let inv_a = inv(&a).run()?;
let out = einsum("ij,jk->ik", &[&a, &b])
    .size_dict(&size_dict)
    .run()?;
```

### AD
```rust
let qr_result = qr_ad(&ad_a).run()?;
let out = einsum_ad("ij,jk->ik", &[&ad_a, &ad_b])
    .size_dict(&size_dict)
    .run()?;
```

### Setter Examples (order-independent)
```rust
let result = svd(&a)
    .options(&svd_options)
    .run()?;

let result = lu(&a)
    .pivot(LuPivot::Partial)
    .run()?;
```

## Builder Semantics
- Operation constructor captures required operands.
- Optional configuration is applied through setter methods.
- `run()` executes the operation and returns operation-specific result type.
- Setter order is arbitrary.

## Runtime Model
- `run()` resolves execution runtime from global runtime state.
- No per-call context argument (`&mut C`) is exposed in public API.
- Missing runtime configuration returns `Error::RuntimeNotConfigured`.
- Runtime-op mismatch returns `Error::UnsupportedRuntimeOp`.

### GPU Behavior
- If GPU runtime is selected and operation path is unavailable, `run()` must fail fast with explicit runtime error.
- No implicit CPU fallback for omitted context.

## AD Model
- For each primal op, provide corresponding `*_ad` operation constructor.
- AD mode precedence:
  - if any operand is reverse -> output reverse
  - else if any operand is forward -> output forward
  - else output primal
- Reverse tape consistency is enforced across operands.
- Tangent propagation is implemented where operation AD support exists.

### `solve_triangular_ad`
- Current `tenferro-linalg` has no AD rule support for triangular solve.
- `solve_triangular_ad(...).run()` accepts primal input.
- For forward/reverse inputs, return `Error::UnsupportedAdOp` and mention upstream tracking issue.

Upstream tracking:
- `tenferro-rs#257`: add `solve_triangular` AD rules
- `tenferro-rs#258`: remove `CpuContext`-locked signatures and unify generic context

## Result Types
- Primal operations return `tenferro-linalg` result types directly.
- AD operations return structured `Ad*Result` types for multi-output ops:
  - `AdSvdResult`, `AdQrResult`, `AdLuResult`, `AdEigenResult`, `AdEigResult`, `AdSlogdetResult`, `AdLstsqResult`
- Single-output AD ops return `AdTensor<T>`.

## Internal Structure
Planned module split:
- `src/api/mod.rs`
- `src/api/builders/macros.rs`
- `src/api/builders/common.rs`
- `src/api/builders/primal.rs`
- `src/api/builders/ad.rs`
- `src/api/ad_results.rs`

## Error Surface
Add or refine errors for:
- runtime not configured
- runtime/op incompatibility
- unsupported AD operation
- mixed reverse tapes
- invalid builder configuration

All errors should remain explicit and non-panicking.

## Testing Strategy
- Keep dimensions small for speed and reproducibility.
- Add tests for each public operation constructor + `run()` path.
- Compare primal builder results against direct `tenferro-linalg` calls.
- Validate AD mode propagation and tape checks.
- Ensure every public signature has minimal rustdoc usage example.

## Migration Strategy
1. Introduce builder API in parallel.
2. Migrate existing `einsum*` / `svd*` callers in crate tests/docs.
3. Remove deprecated direct `*_auto` exports in one cleanup commit.
4. Keep docs and API index aligned.

## Non-Goals
- Re-exporting `rrule` / `frule` APIs.
- Implicit CPU fallback when GPU runtime is active.
- Full runtime backend abstraction redesign inside `tenferro-rs`.
