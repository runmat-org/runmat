use crate::{
    BuiltinAcceleratorPolicy, BuiltinAsyncBehavior, BuiltinBindingDeclaration, BuiltinCatalogEntry,
    BuiltinCatalogIdentity, BuiltinCompatibility, BuiltinContractDeclaration,
    BuiltinContractMaturity, BuiltinDescriptor, BuiltinDistributedPolicy, BuiltinDocumentation,
    BuiltinExtensionDescriptor, BuiltinFusionPolicy, BuiltinInferenceRule,
    BuiltinIntegerAuditDescriptor, BuiltinIntegerCapabilityDescriptor, BuiltinLinkContract,
    BuiltinLinkPolicy, BuiltinPlacementContract, BuiltinPortability, BuiltinPurity,
    BuiltinReachability, BuiltinResidencyPolicy, BuiltinSemanticKind,
};
use runmat_types::{EffectKind, ExecutionStackRequirement};

const MAY_THROW: [EffectKind; 1] = [EffectKind::MayThrow];
const MAY_SUSPEND_AND_THROW: [EffectKind; 2] = [EffectKind::MaySuspend, EffectKind::MayThrow];

/// Identity-specific data for a pure, unary, elementwise numeric builtin.
///
/// This keeps the family invariants in one typed constructor while leaving
/// signatures, errors, extensions, and inference behavior explicit at each
/// builtin declaration. It is intentionally narrower than a general catalog
/// builder: APIs with output-count, mutation, workspace, or capability
/// semantics should use a family that represents those rules directly.
pub(super) struct UnaryNumericCatalogSpec {
    pub identity: BuiltinCatalogIdentity,
    pub documentation: BuiltinDocumentation,
    pub descriptor: &'static BuiltinDescriptor,
    pub inference_rule: BuiltinInferenceRule,
    pub bindings: &'static [BuiltinBindingDeclaration],
    pub extensions: &'static [BuiltinExtensionDescriptor],
    pub integer_capabilities: &'static [BuiltinIntegerCapabilityDescriptor],
    pub integer_audit: Option<&'static BuiltinIntegerAuditDescriptor>,
    pub fusion: BuiltinFusionPolicy,
}

pub(super) const fn unary_numeric_catalog_entry(
    spec: UnaryNumericCatalogSpec,
) -> BuiltinCatalogEntry {
    BuiltinCatalogEntry {
        identity: spec.identity,
        category: "math/elementwise",
        documentation: spec.documentation,
        descriptor: spec.descriptor,
        contract: BuiltinContractDeclaration {
            maturity: BuiltinContractMaturity::Complete,
            inference_rule: spec.inference_rule,
            compatibility: BuiltinCompatibility::Matlab,
            async_behavior: BuiltinAsyncBehavior::NeverSuspends,
            purity: BuiltinPurity::Pure,
            semantic_kind: BuiltinSemanticKind::General,
            workspace_effect: None,
            environment_effect: None,
            effects: &MAY_THROW,
            capabilities: &[],
        },
        placement: BuiltinPlacementContract {
            portability: BuiltinPortability::NativeAndWasm,
            accelerator: BuiltinAcceleratorPolicy::Optional,
            residency: BuiltinResidencyPolicy::Dynamic,
            fusion: spec.fusion,
            distributed: BuiltinDistributedPolicy::MapUnary,
        },
        link: BuiltinLinkContract {
            reachability: BuiltinReachability::Always,
            policy: BuiltinLinkPolicy::PortableRuntime,
            execution_stack: ExecutionStackRequirement::Any,
            artifact_dependencies: &[],
        },
        bindings: spec.bindings,
        extensions: spec.extensions,
        integer_capabilities: spec.integer_capabilities,
        integer_audit: spec.integer_audit,
        suppress_auto_output: false,
    }
}

/// Catalog entry for a unary numeric operation that can execute through an
/// asynchronous provider and preserves the input's physical residency.
pub(super) const fn provider_unary_numeric_catalog_entry(
    spec: UnaryNumericCatalogSpec,
) -> BuiltinCatalogEntry {
    let mut entry = unary_numeric_catalog_entry(spec);
    entry.contract.async_behavior = BuiltinAsyncBehavior::MaySuspend;
    entry.contract.effects = &MAY_SUSPEND_AND_THROW;
    entry.placement.residency = BuiltinResidencyPolicy::PreserveInputs;
    entry
}
