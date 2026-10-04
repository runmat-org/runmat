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
pub(super) const IMPLEMENTATION: &[BuiltinDocumentationLink] = &[BuiltinDocumentationLink {
    label: "MEX builtin and build adapter",
    target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/interop/mex.rs"),
}];
