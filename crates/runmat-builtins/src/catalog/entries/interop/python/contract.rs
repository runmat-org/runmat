use crate::*;
use runmat_types::ExecutionStackRequirement;

pub const PYTHON_ENVIRONMENT_CLASS: runmat_types::StaticClassIdentity =
    runmat_types::StaticClassIdentity::new("py.PythonEnvironment");

pub(super) const INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "Python boundary conversion retains exact fixed-width integer signedness; pyargs stores values without conversion.",
};

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
    policy: BuiltinLinkPolicy::ForeignRuntime(ForeignAdapterFamily::Python),
    execution_stack: ExecutionStackRequirement::Process,
    artifact_dependencies: &[],
};
