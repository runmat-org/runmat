use crate::*;
use runmat_types::ExecutionStackRequirement;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NativeBuildOption {
    Libraries,
    InterfaceName,
    IncludePath,
}

impl NativeBuildOption {
    pub fn parse(name: &str) -> Option<Self> {
        if name.eq_ignore_ascii_case("Libraries") {
            Some(Self::Libraries)
        } else if name.eq_ignore_ascii_case("InterfaceName") {
            Some(Self::InterfaceName)
        } else if name.eq_ignore_ascii_case("IncludePath")
            || name.eq_ignore_ascii_case("IncludeDirectories")
        {
            Some(Self::IncludePath)
        } else {
            None
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NativeLoadOption {
    Alias,
    IncludePath,
    AddHeader,
}

impl NativeLoadOption {
    pub fn parse(name: &str) -> Option<Self> {
        if name.eq_ignore_ascii_case("Alias") {
            Some(Self::Alias)
        } else if name.eq_ignore_ascii_case("IncludePath") {
            Some(Self::IncludePath)
        } else if name.eq_ignore_ascii_case("AddHeader") {
            Some(Self::AddHeader)
        } else {
            None
        }
    }

    pub const fn canonical_name(self) -> &'static str {
        match self {
            Self::Alias => "Alias",
            Self::IncludePath => "IncludePath",
            Self::AddHeader => "AddHeader",
        }
    }
}

pub(super) const HOST_PLACEMENT: BuiltinPlacementContract = BuiltinPlacementContract {
    preparation: BuiltinOperandPreparation::RuntimeOwned,
    portability: BuiltinPortability::NativeOnly,
    accelerator: BuiltinAcceleratorPolicy::Forbidden,
    residency: BuiltinResidencyPolicy::Host,
    fusion: BuiltinFusionPolicy::Boundary,
    distributed: BuiltinDistributedPolicy::Unsupported,
};

pub(super) const FOREIGN_LINK: BuiltinLinkContract = BuiltinLinkContract {
    reachability: BuiltinReachability::Always,
    policy: BuiltinLinkPolicy::ForeignRuntime(ForeignAdapterFamily::NativeFfi),
    execution_stack: ExecutionStackRequirement::Process,
    artifact_dependencies: &[],
};
