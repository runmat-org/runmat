# RunMat AOT orchestration

This host-only crate links verified native user objects with a matching RunMat
runtime archive. Its bounded modules validate archive identity, target, ABI,
schema and builtin catalog; decode the payload; discover the linker; construct
response files; and publish outputs atomically with structured diagnostics.
Browser/WASM consumers use portable execution artifacts and report native
capability absence. This crate stays outside the portable dependency graph.

## Build and acquire the compile-time runtime

`scripts/build-runmat-with-aot-runtime.sh` and its PowerShell equivalent build a
light CLI plus a separate runtime pack and descriptor. The pack contains the
compressed native archive, its manifest, first-party license files and canonical
third-party notices. The CLI embeds the descriptor's exact pack digest and
native compatibility identity.

`runmat aot install` downloads and verifies the matching pack into the user cache.
`runmat aot install --offline --from PATH` provisions an already downloaded pack
through the same verifier. First compilation can request consent to install;
unattended callers explicitly install first or pass `--install-aot-runtime`.
Ordinary script execution does not acquire a compile-time pack. See the
[native compilation guide](../../docs/compiler/native-compilation.md) for cache,
offline and linking requirements.

The crate's build script supports three configurations:

- No runtime inputs: a developer build can execute ordinary code; stock native
  compilation reports that the build lacks its distribution descriptor.
- `RUNMAT_AOT_RUNTIME_MANIFEST` alone: embed the trusted distribution descriptor
  for a light CLI. The CLI owns acquisition and supplies a verified archive to
  the linker.
- Both `RUNMAT_AOT_RUNTIME_ARCHIVE` and `RUNMAT_AOT_RUNTIME_MANIFEST`: embed the
  legacy compressed payload and native archive manifest in the consumer.

An archive input without its manifest is rejected. Runtime packs provide
compile-time runtime inputs; external linker and system-library requirements
still apply.

## Selected paired products

`compile_operator_selected_pair` uses the bounded `selected_admission` loader
with a caller-issued host admission, selected-pack manifest and component root.
The CLI selected compiler uses the same loader. Selected packs require external
tool qualification, source-loading capability and storage allowance. Both
linked roles receive completed-image digests. Publishing the pair does not form
an atomic two-file transaction.

The packer's `--selected-tree` and `--selected-pack` modes produce exact
tool/sysroot metadata and runtime components from explicit build declarations.
Those modes do not qualify the products or issue host admission. Building a
stock compile-time runtime pack, including with `--profile dev`, does not supply
a selected pack or independent toolchain qualification.
