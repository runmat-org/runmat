# runmat-types

`runmat-types` owns RunMat's dependency-neutral semantic vocabulary: stable and product-local identities, immutable declarations, recursive static value facts, schema/version contracts, and their algebra.

It never owns or imports live runtime values, executable builtin bindings, compiler control-flow state, provider implementations, session registries, filesystem/network/process facilities, or host services. HIR, MIR, runtime adapters, LSP, native products, and WASM consume the same serializable contracts from this crate.

Modules are organized by domain rather than collected into a single authority file. Public facades compose focused identity, declaration, fact, contract, rule, symbolic, source, version, and codec modules. Contracts and rules are added only in their owning RM-1064 slice rather than as speculative empty scaffolding.
