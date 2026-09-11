use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct BuiltinExampleRequirements {
    pub host: BuiltinExampleHostRequirement,
    pub engine: BuiltinExampleEngine,
    pub compiler: &'static [BuiltinCompilerCapability],
    pub runtime: &'static [BuiltinRuntimeCapability],
    pub toolchain: &'static [BuiltinToolchainCapability],
}

impl BuiltinExampleRequirements {
    pub const NONE: Self = Self {
        host: BuiltinExampleHostRequirement::Any,
        engine: BuiltinExampleEngine::Default,
        compiler: &[],
        runtime: &[],
        toolchain: &[],
    };
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub enum BuiltinExampleHostRequirement {
    Any,
    NativeOnly,
    DesktopHostOnly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub enum BuiltinExampleEngine {
    Default,
    Interpreter,
    Jit,
    Aot,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub enum BuiltinCompilerCapability {
    C,
    Cxx,
    Fortran,
    Cuda,
    JavaBytecode,
    PythonExtension,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub enum BuiltinRuntimeCapability {
    NativeDynamicLoader,
    Mex,
    NativeFfi,
    JavaVirtualMachine,
    Python,
    NumPy,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub enum BuiltinToolchainCapability {
    CCompiler,
    CxxCompiler,
    FortranCompiler,
    CudaToolkit,
    JavaDevelopmentKit,
    PythonInterpreter,
    PythonDevelopmentHeaders,
}
