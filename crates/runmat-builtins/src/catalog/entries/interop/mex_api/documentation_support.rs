use crate::*;

pub(super) const NATIVE_COMPILER: BuiltinExampleRequirements = BuiltinExampleRequirements {
    host: BuiltinExampleHostRequirement::NativeOnly,
    engine: BuiltinExampleEngine::Default,
    compiler: &[BuiltinCompilerCapability::C],
    runtime: &[BuiltinRuntimeCapability::Mex],
    toolchain: &[BuiltinToolchainCapability::CCompiler],
};
pub(super) const NATIVE_TARGET: BuiltinExampleRequirements = BuiltinExampleRequirements {
    host: BuiltinExampleHostRequirement::NativeOnly,
    ..BuiltinExampleRequirements::NONE
};
pub(super) const CUDA_COMPILER: BuiltinExampleRequirements = BuiltinExampleRequirements {
    host: BuiltinExampleHostRequirement::NativeOnly,
    engine: BuiltinExampleEngine::Default,
    compiler: &[BuiltinCompilerCapability::Cuda],
    runtime: &[BuiltinRuntimeCapability::Mex],
    toolchain: &[BuiltinToolchainCapability::CudaToolkit],
};
pub(super) const MEX_LINK: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "MEX execution and build compatibility",
    target: BuiltinDocumentationLinkTarget::Documentation("/docs/execution/mex"),
};
pub(super) const EXTENSIONS_LINK: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "Native extension development",
    target: BuiltinDocumentationLinkTarget::Documentation("/docs/development/extensions"),
};
pub(super) const IMPLEMENTATION: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink {
        label: "MEX public call and error boundary",
        target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/interop/mex.rs"),
    },
    BuiltinDocumentationLink {
        label: "Build argument and API-pin parser",
        target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-mex/src/build/arguments.rs"),
    },
    BuiltinDocumentationLink {
        label: "Native target admission",
        target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-mex/src/build/target.rs"),
    },
    BuiltinDocumentationLink {
        label: "MEX artifact identity and validation",
        target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-mex/src/build/artifact.rs"),
    },
    BuiltinDocumentationLink {
        label: "C Matrix API array ownership and copy-on-write",
        target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-mex/src/libmx/api.rs"),
    },
    BuiltinDocumentationLink {
        label: "MEX callback host services",
        target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-mex/src/host/services.rs"),
    },
];
