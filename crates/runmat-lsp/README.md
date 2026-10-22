# Builtin documentation

The build script calls the shared `runmat-builtins::documentation_export`
presentation API and writes one aggregate into `OUT_DIR`. LSP consumers do not
enumerate, include, or merge individual JSON sidecars. Canonical catalog prose,
sections, examples and FAQs are preserved in hovers.

The presentation envelope is explicitly labeled `editor_prose` with
`executable_qualification: false`. It retains complete exported rows, including
fixture and verification metadata, but does not qualify executable examples.
The strict documentation CLI still uses its original validation and schema;
presentation envelopes cannot serve as documentation-cutover evidence.

During migration, remaining authoritative legacy prose is included through the
same export owner. Duplicate canonical/legacy owners, unresolved links and
duplicate identities or slugs remain errors. No canonical-only fallback is used
when the legacy assets are absent.

Standalone package-build limitation: remaining legacy assets live outside this
crate in `../../docs/builtins/reference`, and neither this package nor
`runmat-builtins` currently bundles those assets. This was already required by
the previous build script. A package without those assets fails with an explicit
error; this packet does not claim package closure or introduce copied shadow
documentation. Final removal of legacy documentation authority must also close
this package obligation.
