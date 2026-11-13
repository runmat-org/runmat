use crate::*;
use runmat_types::{CallInference, CallRequest, EffectKind, ExecutionStackRequirement};

pub(super) struct Definition {
    pub provenance: BuiltinCatalogProvenance,
    pub name: &'static str,
    pub descriptor: &'static BuiltinDescriptor,
    pub documentation: BuiltinDocumentation,
    pub inference: fn(&CallRequest, &BuiltinCatalogEntry) -> CallInference,
    pub compatibility: BuiltinCompatibility,
    pub residency: BuiltinResidencyPolicy,
    pub extensions: &'static [BuiltinExtensionDescriptor],
    pub integer_capabilities: &'static [BuiltinIntegerCapabilityDescriptor],
    pub integer_audit: Option<&'static BuiltinIntegerAuditDescriptor>,
}

pub(super) const fn entry(definition: Definition) -> BuiltinCatalogEntry {
    BuiltinCatalogEntry {
        audience: BuiltinCatalogAudience::Public,
        provenance: definition.provenance,
        identity: BuiltinCatalogIdentity {
            name: definition.name,
        },
        category: "array/sorting_sets",
        descriptor: definition.descriptor,
        documentation: definition.documentation,
        contract: BuiltinContractDeclaration {
            // The typed contract is explicit, but the public family still has
            // unsupported container/control/placement forms documented locally.
            maturity: BuiltinContractMaturity::Incomplete,
            inference_rule: BuiltinInferenceRule::Identity(IdentityInferenceRule::new(
                definition.inference,
            )),
            compatibility: definition.compatibility,
            async_behavior: BuiltinAsyncBehavior::MaySuspend,
            purity: BuiltinPurity::Pure,
            semantic_kind: BuiltinSemanticKind::General,
            workspace_effect: None,
            environment_effect: None,
            effects: &[EffectKind::MaySuspend, EffectKind::MayThrow],
            capabilities: &[],
        },
        placement: BuiltinPlacementContract {
            preparation: BuiltinOperandPreparation::RuntimeOwned,
            portability: BuiltinPortability::NativeAndWasm,
            accelerator: BuiltinAcceleratorPolicy::Optional,
            residency: definition.residency,
            fusion: BuiltinFusionPolicy::Boundary,
            distributed: BuiltinDistributedPolicy::Unsupported,
        },
        link: BuiltinLinkContract {
            reachability: BuiltinReachability::Always,
            policy: BuiltinLinkPolicy::PortableRuntime,
            execution_stack: ExecutionStackRequirement::Any,
            artifact_dependencies: &[],
        },
        bindings: &REQUIRED_DEFAULT_BINDING,
        extensions: definition.extensions,
        integer_capabilities: definition.integer_capabilities,
        integer_audit: definition.integer_audit,
        suppress_auto_output: false,
    }
}

pub(super) const BINARY_SET_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "A",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "A accepts every real integer class; unlike nondouble numeric inputs must use the same class, while double is the documented cross-class exception.",
    },
    BuiltinIntegerInputCapability {
        name: "B",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "B accepts every real integer class; unlike nondouble numeric inputs must use the same class, while double is the documented cross-class exception.",
    },
];
