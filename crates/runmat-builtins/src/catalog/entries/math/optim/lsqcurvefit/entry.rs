use super::*;
use crate::*;
use runmat_types::{EffectKind, ExecutionStackRequirement};

pub const LSQCURVEFIT_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    audience: crate::BuiltinCatalogAudience::Public,
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity {
        name: "lsqcurvefit",
    },
    category: "math/optim",
    documentation: documentation::DOCUMENTATION,
    descriptor: &LSQCURVEFIT_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Incomplete,
        inference_rule: BuiltinInferenceRule::Identity(IdentityInferenceRule::new(
            inference::infer,
        )),
        compatibility: BuiltinCompatibility::Matlab,
        async_behavior: BuiltinAsyncBehavior::MaySuspend,
        purity: BuiltinPurity::Impure,
        semantic_kind: BuiltinSemanticKind::General,
        workspace_effect: None,
        environment_effect: None,
        effects: &[
            EffectKind::HostCallback,
            EffectKind::MaySuspend,
            EffectKind::MayThrow,
            EffectKind::Unknown,
        ],
        capabilities: &[],
    },
    placement: BuiltinPlacementContract {
        preparation: BuiltinOperandPreparation::RuntimeOwned,
        portability: BuiltinPortability::NativeAndWasm,
        accelerator: BuiltinAcceleratorPolicy::Forbidden,
        residency: BuiltinResidencyPolicy::Dynamic,
        fusion: BuiltinFusionPolicy::Boundary,
        distributed: BuiltinDistributedPolicy::Unsupported,
    },
    link: BuiltinLinkContract {
        reachability: BuiltinReachability::Always,
        policy: BuiltinLinkPolicy::PortableRuntime,
        execution_stack: ExecutionStackRequirement::Process,
        artifact_dependencies: &[],
    },
    bindings: &REQUIRED_DEFAULT_BINDING,
    extensions: &LSQCURVEFIT_EXTENSIONS,
    integer_capabilities: &LSQCURVEFIT_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};
