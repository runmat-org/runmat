use runmat_types::ExecutionStackRequirement;
use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum BuiltinReachability {
    Always,
    Dynamic,
    Feature(&'static str),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub enum ForeignAdapterFamily {
    NativeFfi,
    Mex,
    Java,
    Python,
}

impl ForeignAdapterFamily {
    pub const fn adapter_id(self) -> &'static str {
        match self {
            Self::NativeFfi => "native-ffi",
            Self::Mex => "mex-c",
            Self::Java => "java",
            Self::Python => "python",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum BuiltinLinkPolicy {
    PortableRuntime,
    HostRuntime,
    NativeSymbol,
    ForeignRuntime(ForeignAdapterFamily),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct BuiltinLinkContract {
    pub reachability: BuiltinReachability,
    pub policy: BuiltinLinkPolicy,
    pub execution_stack: ExecutionStackRequirement,
    pub artifact_dependencies: &'static [&'static str],
}
