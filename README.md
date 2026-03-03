# ad-tensors-rs

AD-aware tensor interface layer on top of `tenferro-rs`.

## Status

This repository currently provides:

- Generic AD value model:
  - `AdValue<T>`
  - `AdScalar<T>`
  - `AdTensor<T>`
- Runtime dtype wrappers:
  - `DynScalar`, `DynTensor`
  - `DynAdValue`, `DynAdTensor`
  - `ScalarType` (`F32`, `F64`, `C32`, `C64`)
- AD boundary traits:
  - `Differentiable`, `TensorKernel`, `OpRule`
- API signatures:
  - `einsum`, `einsum_auto`
  - `einsum_ad`, `einsum_ad_auto`
  - `svd`, `svd_auto`
- Thread-local global context utilities

Implemented operation entry points:

- `einsum` / `einsum_auto`: primal einsum via `tenferro-einsum`
- `einsum_ad` / `einsum_ad_auto`: AD-mode-aware einsum (Primal/Forward/Reverse)
- `svd` / `svd_auto`: SVD via `tenferro-linalg`

`einsum_ad` propagates mode and tangent information. For reverse-mode inputs,
it enforces single-tape consistency and constructs output reverse metadata.

## Development

```bash
cargo fmt --all
cargo clippy --workspace
cargo test --release --workspace
```

## Documentation

Build local docs site:

```bash
./scripts/build_docs_site.sh
```

Output:

- `target/docs-site/index.html` (top page)
- `target/docs-site/api/` (`cargo doc --workspace --no-deps` output)
- `target/docs-site/design/` (rendered design docs)

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](./LICENSE-APACHE))
- MIT license ([LICENSE-MIT](./LICENSE-MIT))

at your option.
