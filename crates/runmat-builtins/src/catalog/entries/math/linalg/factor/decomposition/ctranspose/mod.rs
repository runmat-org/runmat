mod documentation;
mod inference;
use super::super::decomposition::{integer_capabilities::*, integer_inputs::*, signatures::*};
use crate::*;
pub const DECOMPOSITION_CTRANSPOSE_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    audience: crate::BuiltinCatalogAudience::Public,
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity {
        name: "decomposition.ctranspose",
    },
    category: "math/linalg/factor",
    documentation: documentation::DOCUMENTATION,
    descriptor: &DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::DynamicByDesign,
        inference_rule: BuiltinInferenceRule::Identity(IdentityInferenceRule::new(
            inference::infer,
        )),
        compatibility: BuiltinCompatibility::Matlab,
        async_behavior: BuiltinAsyncBehavior::MaySuspend,
        purity: BuiltinPurity::Pure,
        semantic_kind: BuiltinSemanticKind::General,
        workspace_effect: None,
        environment_effect: None,
        effects: &[
            runmat_types::EffectKind::MaySuspend,
            runmat_types::EffectKind::MayThrow,
        ],
        capabilities: &[],
    },
    placement: BuiltinPlacementContract {
        preparation: crate::BuiltinOperandPreparation::RuntimeOwned,
        portability: BuiltinPortability::NativeAndWasm,
        accelerator: BuiltinAcceleratorPolicy::Optional,
        residency: BuiltinResidencyPolicy::Host,
        fusion: BuiltinFusionPolicy::Boundary,
        distributed: BuiltinDistributedPolicy::Unsupported,
    },
    link: BuiltinLinkContract {
        reachability: BuiltinReachability::Always,
        policy: BuiltinLinkPolicy::PortableRuntime,
        execution_stack: runmat_types::ExecutionStackRequirement::Any,
        artifact_dependencies: &[],
    },
    bindings: &REQUIRED_DEFAULT_BINDING,
    extensions: &DECOMPOSITION_METHOD_EXTENSIONS,
    integer_capabilities: &DECOMPOSITION_CTRANSPOSE_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};

const SIGNATURES: &[BuiltinSignatureDescriptor] = &[BuiltinSignatureDescriptor {
    label: "out = decomposition.ctranspose(dA)",
    inputs: &UNARY_INPUT,
    outputs: &OUTPUT,
}];
const DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: SIGNATURES,
    ..DECOMPOSITION_UNARY_DESCRIPTOR
};
