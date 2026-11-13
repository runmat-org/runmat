use crate::*;
use runmat_types::ExecutionStackRequirement;

pub const MEX_ERROR_UNAVAILABLE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MEX.UNAVAILABLE_TARGET",
    identifier: Some("RunMat:MEX:UnsupportedTarget"),
    when: "The current target cannot load native MEX modules.",
    message: "MEX modules are unavailable on this target.",
};
pub const MEX_ERROR_BUILD_ARGUMENTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MEX.INVALID_BUILD_ARGUMENTS",
    identifier: Some("RunMat:MEX:InvalidBuildArguments"),
    when: "The MEX command receives no source file or an invalid build option.",
    message: "mex: invalid build arguments.",
};
pub const MEX_ERROR_BUILD_FAILED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MEX.BUILD_FAILED",
    identifier: Some("RunMat:MEX:BuildFailed"),
    when: "The selected compiler cannot produce the requested MEX module.",
    message: "mex: C MEX compilation failed.",
};
pub const MEX_ERROR_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MEX.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:MEX:TooManyOutputs"),
    when: "The caller requests more results than the MEX build or target query returns.",
    message: "Too many MEX output arguments.",
};
pub(super) const BUILD_ERRORS: &[BuiltinErrorDescriptor] = &[
    MEX_ERROR_UNAVAILABLE,
    MEX_ERROR_BUILD_ARGUMENTS,
    MEX_ERROR_BUILD_FAILED,
    MEX_ERROR_OUTPUTS,
];
pub(super) const EXTENSION_ERRORS: &[BuiltinErrorDescriptor] =
    &[MEX_ERROR_UNAVAILABLE, MEX_ERROR_OUTPUTS];

pub(super) const BUILD_PLACEMENT: BuiltinPlacementContract = BuiltinPlacementContract {
    preparation: BuiltinOperandPreparation::RuntimeOwned,
    portability: BuiltinPortability::NativeOnly,
    accelerator: BuiltinAcceleratorPolicy::Forbidden,
    residency: BuiltinResidencyPolicy::Host,
    fusion: BuiltinFusionPolicy::Boundary,
    distributed: BuiltinDistributedPolicy::Unsupported,
};
pub(super) const BUILD_LINK: BuiltinLinkContract = BuiltinLinkContract {
    reachability: BuiltinReachability::Always,
    policy: BuiltinLinkPolicy::HostRuntime,
    execution_stack: ExecutionStackRequirement::Process,
    artifact_dependencies: &[],
};
pub(super) const EXTENSION_LINK: BuiltinLinkContract = BuiltinLinkContract {
    reachability: BuiltinReachability::Always,
    policy: BuiltinLinkPolicy::PortableRuntime,
    execution_stack: ExecutionStackRequirement::Any,
    artifact_dependencies: &[],
};

pub(super) const BUILD_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor =
    BuiltinIntegerAuditDescriptor {
        kind: BuiltinIntegerAuditKind::NotApplicable,
        canonical_builtin: None,
        notes: "MEX compilation consumes source paths and build options. Existing scalar numeric and logical arguments are converted to option text at the RunMat call boundary, not processed numerically.",
    };
pub(super) const EXTENSION_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor =
    BuiltinIntegerAuditDescriptor {
        kind: BuiltinIntegerAuditKind::NotApplicable,
        canonical_builtin: None,
        notes: "mexext accepts no inputs and returns a platform extension string.",
    };
