use crate::{
    BuiltinAcceleratorPolicy, BuiltinErrorDescriptor, BuiltinFusionPolicy, BuiltinLinkContract,
    BuiltinLinkPolicy, BuiltinPlacementContract, BuiltinPortability, BuiltinReachability,
    BuiltinResidencyPolicy,
};
use runmat_types::{CapabilityRequirement, EffectKind, ExecutionStackRequirement};

pub(in crate::catalog::entries::parallel) const CURRENT_EFFECTS: [EffectKind; 1] =
    [EffectKind::MayThrow];
pub(in crate::catalog::entries::parallel) const POOL_READ_EFFECTS: [EffectKind; 3] = [
    EffectKind::EnvironmentRead,
    EffectKind::MaySuspend,
    EffectKind::MayThrow,
];
pub(in crate::catalog::entries::parallel) const POOL_WRITE_EFFECTS: [EffectKind; 4] = [
    EffectKind::EnvironmentRead,
    EffectKind::EnvironmentWrite,
    EffectKind::MaySuspend,
    EffectKind::MayThrow,
];
pub(in crate::catalog::entries::parallel) const SCHEDULE_EFFECTS: [EffectKind; 3] = [
    EffectKind::EnvironmentWrite,
    EffectKind::MaySuspend,
    EffectKind::MayThrow,
];
pub(in crate::catalog::entries::parallel) const FETCH_EFFECTS: [EffectKind; 2] =
    [EffectKind::MaySuspend, EffectKind::MayThrow];
pub(in crate::catalog::entries::parallel) const PARALLEL_EFFECTS: [EffectKind; 2] =
    [EffectKind::MaySuspend, EffectKind::MayThrow];
pub(in crate::catalog::entries::parallel) const PARALLEL_RUNTIME: [CapabilityRequirement; 1] =
    [CapabilityRequirement::ParallelRuntime];

pub(in crate::catalog::entries::parallel) const CURRENT_PLACEMENT: BuiltinPlacementContract =
    BuiltinPlacementContract {
        portability: BuiltinPortability::NativeAndWasm,
        accelerator: BuiltinAcceleratorPolicy::Forbidden,
        residency: BuiltinResidencyPolicy::Host,
        fusion: BuiltinFusionPolicy::Boundary,
        distributed: crate::BuiltinDistributedPolicy::Unsupported,
    };
pub(in crate::catalog::entries::parallel) const PARALLEL_PLACEMENT: BuiltinPlacementContract =
    BuiltinPlacementContract {
        portability: BuiltinPortability::NativeAndWasm,
        accelerator: BuiltinAcceleratorPolicy::Forbidden,
        residency: BuiltinResidencyPolicy::Dynamic,
        fusion: BuiltinFusionPolicy::Boundary,
        distributed: crate::BuiltinDistributedPolicy::Unsupported,
    };
pub(in crate::catalog::entries::parallel) const DISTRIBUTED_INSPECTION_PLACEMENT:
    BuiltinPlacementContract = BuiltinPlacementContract {
    distributed: crate::BuiltinDistributedPolicy::InspectHandles,
    ..PARALLEL_PLACEMENT
};

pub(in crate::catalog::entries::parallel) const CURRENT_LINK: BuiltinLinkContract =
    BuiltinLinkContract {
        reachability: BuiltinReachability::Always,
        policy: BuiltinLinkPolicy::PortableRuntime,
        execution_stack: ExecutionStackRequirement::Any,
        artifact_dependencies: &[],
    };
pub(in crate::catalog::entries::parallel) const PARALLEL_LINK: BuiltinLinkContract =
    BuiltinLinkContract {
        reachability: BuiltinReachability::Always,
        policy: BuiltinLinkPolicy::PortableRuntime,
        execution_stack: ExecutionStackRequirement::Any,
        artifact_dependencies: &[],
    };

pub(in crate::catalog::entries::parallel) const LOWERING_ERRORS: [BuiltinErrorDescriptor; 1] =
    [BuiltinErrorDescriptor {
    code: "RM.PARALLEL.LOWERING_REQUIRED",
    identifier: Some("RunMat:parallel:LoweringRequired"),
    when: "The operation is invoked without an active compiler-owned SPMD or distributed execution context.",
    message: "parallel operation requires executor-aware lowering",
    }];
