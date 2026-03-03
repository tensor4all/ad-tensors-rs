# ad-tensors-rs

Three-mode AD tensor-layer API skeleton on top of `tenferro-rs`.

## Status

This repository currently provides:

- Core scalar/mode model (`BaseScalar`, `AnyScalar`, `Primal`, `Dual`, `Tracked`)
- AD boundary traits (`TensorKernel`, `OpRule`, `Differentiable`)
- Explicit + global-context API signatures (`einsum`, `einsum_auto`, `svd`, `svd_auto`)
- Thread-local global context utilities

Numeric kernels and AD rules are not implemented yet. API functions return
`Error::NotImplemented` in this POC stage.

## Development

```bash
cargo fmt --all
cargo clippy --all-targets
cargo test --release
```

## Documentation

Build local docs site:

```bash
./scripts/build_docs_site.sh
```

Output:

- `target/docs-site/index.html` (top page)
- `target/docs-site/api/` (`cargo doc --workspace --no-deps` output)
