# Tenferro Linalg Builder API Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace function-style `*_auto` APIs with builder-based `run()` APIs and implement the full `tenferro-linalg` primary operation surface for both `Tensor<T>` and `AdTensor<T>`.

**Architecture:** Introduce operation-specific builders that collect operands/options and execute through a global runtime context in `run()`. Keep primal and AD builders parallel, share mode/tape/runtime logic in common helpers/macros, and expose structured AD result types for multi-output ops.

**Tech Stack:** Rust 2021, `tenferro-*` crates, `thiserror`, rustdoc examples, cargo test/clippy/fmt.

---

### Task 1: Runtime Context Abstraction For Context-Omitted Execution

**Files:**
- Create: `src/runtime.rs`
- Modify: `src/context.rs`
- Modify: `src/error.rs`
- Modify: `src/lib.rs`
- Test: `tests/runtime_context_tests.rs`

**Step 1: Write failing runtime tests**

```rust
#[test]
fn run_fails_when_runtime_not_configured() {
    // expect Error::RuntimeNotConfigured
}

#[test]
fn set_default_runtime_cpu_allows_run_resolution() {
    // set runtime and resolve mutable CPU context
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test --release --test runtime_context_tests -v`
Expected: FAIL with missing `RuntimeContext` APIs and/or error variants.

**Step 3: Implement runtime context API**

```rust
pub enum RuntimeContext {
    Cpu(tenferro_prims::CpuContext),
    #[allow(dead_code)]
    Cuda(tenferro_prims::CudaContext),
    #[allow(dead_code)]
    Rocm(tenferro_prims::RocmContext),
}

pub fn set_default_runtime(ctx: RuntimeContext) -> GlobalContextGuard<RuntimeContext> { ... }
pub fn with_default_runtime<R>(f: impl FnOnce(&mut RuntimeContext) -> Result<R>) -> Result<R> { ... }
```

Add error variants in `Error`:
- `RuntimeNotConfigured`
- `UnsupportedRuntimeOp { op: &'static str, runtime: &'static str }`
- `UnsupportedAdOp { op: &'static str }`

**Step 4: Run test to verify it passes**

Run: `cargo test --release --test runtime_context_tests -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/runtime.rs src/context.rs src/error.rs src/lib.rs tests/runtime_context_tests.rs
git commit -m "feat: add runtime context abstraction for builder run"
```

### Task 2: Introduce Builder Scaffolding And Migrate Einsum APIs

**Files:**
- Create: `src/api/mod.rs`
- Create: `src/api/builders/mod.rs`
- Create: `src/api/builders/common.rs`
- Create: `src/api/builders/macros.rs`
- Create: `src/api/builders/einsum.rs`
- Modify: `src/lib.rs`
- Remove/replace: `src/api.rs`
- Test: `tests/einsum_builder_tests.rs`

**Step 1: Write failing einsum builder tests**

```rust
#[test]
fn einsum_builder_runs_primal() {
    // einsum("ij,jk->ik", ...).run()
}

#[test]
fn einsum_builder_applies_size_dict_setter() {
    // .size_dict(&map)
}

#[test]
fn einsum_ad_builder_propagates_forward_mode() {
    // einsum_ad(...).run()
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test --release --test einsum_builder_tests -v`
Expected: FAIL because builder constructors are missing.

**Step 3: Implement einsum builders**

```rust
pub fn einsum<'a, T>(subscripts: &'a str, operands: &'a [&'a Tensor<T>]) -> EinsumBuilder<'a, T>;
pub fn einsum_ad<'a, T>(subscripts: &'a str, operands: &'a [&'a AdTensor<T>]) -> EinsumAdBuilder<'a, T>;

impl EinsumBuilder<'_, T> {
    pub fn size_dict(mut self, size_dict: &'a HashMap<u32, usize>) -> Self { ... }
    pub fn run(self) -> Result<Tensor<T>> { ... }
}
```

Use global runtime dispatch in `run()`. Return explicit runtime error for unsupported runtime path.

**Step 4: Run test to verify it passes**

Run: `cargo test --release --test einsum_builder_tests -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/api src/lib.rs tests/einsum_builder_tests.rs
git commit -m "refactor: migrate einsum APIs to builder run model"
```

### Task 3: Add Full Primal Linalg Builder Surface

**Files:**
- Create: `src/api/builders/linalg_primal.rs`
- Modify: `src/api/builders/mod.rs`
- Modify: `src/lib.rs`
- Test: `tests/linalg_primal_builder_tests.rs`

**Step 1: Write failing primal linalg builder tests**

```rust
#[test]
fn qr_builder_matches_direct_call() { ... }
#[test]
fn lu_builder_accepts_pivot_setter() { ... }
#[test]
fn svd_builder_accepts_options_setter() { ... }
#[test]
fn solve_triangular_builder_accepts_upper_setter() { ... }
#[test]
fn norm_builder_accepts_kind_setter() { ... }
```

Include coverage for all primary operations listed in scope.

**Step 2: Run test to verify it fails**

Run: `cargo test --release --test linalg_primal_builder_tests -v`
Expected: FAIL because constructors/builders are incomplete.

**Step 3: Implement primal linalg builders**

Provide constructor + `run()` for:
- `svd`, `qr`, `lu`, `eigen`, `lstsq`, `cholesky`, `solve`, `inv`, `det`, `slogdet`, `eig`, `pinv`, `matrix_exp`, `solve_triangular`, `norm`

Use setters for optional parameters:
- `.options(&SvdOptions)`
- `.pivot(LuPivot)`
- `.rcond(f64)`
- `.upper(bool)`
- `.kind(NormKind)`

**Step 4: Run test to verify it passes**

Run: `cargo test --release --test linalg_primal_builder_tests -v`
Expected: PASS on CPU runtime.

**Step 5: Commit**

```bash
git add src/api/builders/linalg_primal.rs src/api/builders/mod.rs src/lib.rs tests/linalg_primal_builder_tests.rs
git commit -m "feat: implement full primal linalg builder API"
```

### Task 4: Add AD Linalg Builders And Structured AD Result Types

**Files:**
- Create: `src/api/ad_results.rs`
- Create: `src/api/builders/linalg_ad.rs`
- Modify: `src/api/builders/common.rs`
- Modify: `src/api/builders/mod.rs`
- Modify: `src/lib.rs`
- Test: `tests/linalg_ad_builder_tests.rs`

**Step 1: Write failing AD tests**

```rust
#[test]
fn qr_ad_builder_preserves_primal_mode() { ... }
#[test]
fn svd_ad_builder_returns_structured_ad_result() { ... }
#[test]
fn reverse_inputs_require_single_tape() { ... }
#[test]
fn solve_triangular_ad_forward_returns_unsupported_error() { ... }
```

**Step 2: Run test to verify it fails**

Run: `cargo test --release --test linalg_ad_builder_tests -v`
Expected: FAIL because AD builders/result types are missing.

**Step 3: Implement AD result types and AD builders**

Add public AD result structs:
- `AdSvdResult`, `AdQrResult`, `AdLuResult`, `AdEigenResult`, `AdEigResult`, `AdSlogdetResult`, `AdLstsqResult`

Add `*_ad(...).run()` for all primal ops.

Rules:
- mode precedence: reverse > forward > primal
- enforce reverse tape consistency
- for `solve_triangular_ad`, return `Error::UnsupportedAdOp` for forward/reverse inputs and mention upstream issue `tensor4all/tenferro-rs#257` in docs

**Step 4: Run test to verify it passes**

Run: `cargo test --release --test linalg_ad_builder_tests -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/api/ad_results.rs src/api/builders/linalg_ad.rs src/api/builders/common.rs src/api/builders/mod.rs src/lib.rs tests/linalg_ad_builder_tests.rs
git commit -m "feat: add AD linalg builders and structured AD results"
```

### Task 5: Remove Legacy `*_auto` API And Update Re-exports

**Files:**
- Modify: `src/lib.rs`
- Modify: `src/api/mod.rs`
- Test: `tests/public_api_compile_tests.rs`

**Step 1: Write failing compile/public API tests**

```rust
#[test]
fn legacy_auto_symbols_are_absent() {
    // ensure old exports removed from public surface
}

#[test]
fn builder_symbols_are_public() {
    // ensure new constructors and result types are reachable
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test --release --test public_api_compile_tests -v`
Expected: FAIL until exports are updated.

**Step 3: Remove old exports and wire new public API**

- Remove `einsum_auto`, `einsum_ad_auto`, `svd_auto` exports.
- Export all new operation constructors and AD result types.

**Step 4: Run test to verify it passes**

Run: `cargo test --release --test public_api_compile_tests -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/lib.rs src/api/mod.rs tests/public_api_compile_tests.rs
git commit -m "refactor: remove auto APIs and finalize builder public surface"
```

### Task 6: Add Rustdoc Examples For All Public Signatures

**Files:**
- Modify: `src/api/builders/einsum.rs`
- Modify: `src/api/builders/linalg_primal.rs`
- Modify: `src/api/builders/linalg_ad.rs`
- Modify: `src/api/ad_results.rs`

**Step 1: Write missing-doc check list**

Create a checklist in the task branch notes for every `pub fn`, `pub struct`, `pub enum` in new API files and mark rustdoc examples required.

**Step 2: Add minimal runnable examples**

Each public signature gets a minimal but sufficient usage snippet showing constructor, setter (if any), and `run()`.

**Step 3: Run docs and doctests**

Run: `cargo test --release --doc -v`
Expected: PASS.

**Step 4: Commit**

```bash
git add src/api/builders/einsum.rs src/api/builders/linalg_primal.rs src/api/builders/linalg_ad.rs src/api/ad_results.rs
git commit -m "docs: add minimal rustdoc examples for all new public signatures"
```

### Task 7: End-to-End Validation And Final Cleanup

**Files:**
- Modify if needed: `README.md`
- Modify if needed: `docs/design/api.md`
- Modify if needed: `docs/index.md`

**Step 1: Run formatting**

Run: `cargo fmt --all`
Expected: no diff after formatting.

**Step 2: Run clippy**

Run: `cargo clippy --workspace`
Expected: no warnings/errors.

**Step 3: Run full tests**

Run: `cargo test --release --workspace`
Expected: PASS.

**Step 4: Commit final docs/cleanup**

```bash
git add README.md docs/design/api.md docs/index.md
git commit -m "docs: update API docs for builder-based linalg surface"
```

**Step 5: Prepare PR**

```bash
git push -u origin <branch>
gh pr create --base main --title "Builder-based full linalg API for Tensor/AdTensor" --body "Implements builder run API for all linalg ops and AD wrappers"
gh pr merge --auto --squash --delete-branch
```
