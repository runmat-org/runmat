//! Validated dependent access to the network-owned recurrent State table.
use crate::*;
use runmat_types::{
    standard, CallInference, CallRequest, DynamicReason, EffectKind, ExecutionStackRequirement,
    ValueFact, ValueKindFact,
};

const GET_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "network",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "A dlnetwork, SeriesNetwork, or DAGNetwork value object.",
}];
const SET_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "network",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "A dlnetwork, SeriesNetwork, or DAGNetwork value object.",
    },
    BuiltinParamDescriptor {
        name: "state",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "A Layer/Parameter/Value table matching the network state topology.",
    },
];
const GET_OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "state",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "The network's validated recurrent State table.",
}];
const SET_OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "network",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "The value network with replaced State.",
}];
const GET_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "state = get.State(network)",
    inputs: &GET_INPUTS,
    outputs: &GET_OUTPUTS,
}];
const SET_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "network = set.State(network, state)",
    inputs: &SET_INPUTS,
    outputs: &SET_OUTPUTS,
}];
const ERRORS: [BuiltinErrorDescriptor; 1] = [BuiltinErrorDescriptor {
    code: "RM.DEEP_LEARNING.INVALID_INPUT",
    identifier: Some("RunMat:deepLearning:InvalidInput"),
    when: "The State table does not match the network's recurrent state rows, class, or shape.",
    message: "network State access received invalid input",
}];
const GET_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &GET_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
const SET_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SET_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

const GET_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("get.State"), slug: Some("get.State"),
    summary: "Read a network's recurrent state rows.",
    description: "State is a Layer/Parameter/Value table. Recurrent HiddenState and CellState rows come from the network's canonical state store; stateless networks have an empty table.",
    keywords: &["dlnetwork", "State", "recurrent"], related: &["dlnetwork", "set.State"],
    sections: &[BuiltinDocumentationSection { heading: "Value ownership", paragraphs: &[
        "Reading State leaves the value network unchanged. State values retain their numeric class and resident allocation; reading does not download GPU data.",
        "RunMat exposes this table on SeriesNetwork and DAGNetwork values as an extension. State evolution by recurrent forward execution is documented with the recurrent execution support, not implied by this accessor alone.",
    ] }],
    examples: &[], example_exemption: Some("This is a dependent property accessor exercised by network state tests."),
    faqs: &[], links: &[], media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "Canonical network State", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/deep_learning/model/state.rs") }],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "State table lifecycle", location: "builtins::deep_learning::model::state::tests::network_state_table_is_canonical_and_validated" }],
        notes: &[],
    },
    introduced: None, status: Some(BuiltinDocumentationStatus::Partial),
};
const SET_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("set.State"), slug: Some("set.State"),
    summary: "Replace all admitted recurrent State rows atomically.",
    description: "The setter checks every typed Layer/Parameter identity, numeric class, geometry, and resident allocation before replacing the network-owned State table.",
    keywords: &["dlnetwork", "State", "recurrent"], related: &["dlnetwork", "get.State"],
    sections: &[BuiltinDocumentationSection { heading: "State replacement", paragraphs: &[
        "The table requires Layer, Parameter, and Value columns. Rows may be reordered if their identities are unique and complete. State is owned by the returned value network; a failed update leaves the input network unchanged.",
    ] }],
    examples: &[], example_exemption: Some("This is a dependent property accessor exercised by network state tests."),
    faqs: &[], links: &[], media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "Atomic network State update", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/deep_learning/model/state.rs") }],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "State setter rejection", location: "builtins::deep_learning::model::state::tests::network_state_table_is_canonical_and_validated" }],
        notes: &[],
    },
    introduced: None, status: Some(BuiltinDocumentationStatus::Partial),
};

fn receiver(request: &CallRequest, arity: usize) -> Vec<runmat_types::InferenceDiagnostic> {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != arity {
        diagnostics.push(crate::catalog::inference::argument_error(
            "RM-DL-STATE-ARITY",
            "incorrect number of State accessor arguments",
            arity,
        ));
    }
    if let Some(value) = request.arguments.first() {
        let valid = match &value.kind {
            ValueKindFact::Unknown => true,
            ValueKindFact::Object(object) => object.runtime_class.as_ref().is_none_or(|class| {
                [
                    standard::DL_NETWORK,
                    standard::SERIES_NETWORK,
                    standard::DAG_NETWORK,
                ]
                .iter()
                .any(|identity| class.is(*identity))
            }),
            _ => false,
        };
        if !valid {
            diagnostics.push(crate::catalog::inference::argument_error(
                "RM-DL-STATE-NETWORK",
                "expected a deep-learning network value",
                0,
            ));
        }
    }
    diagnostics
}

fn get_inference(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let diagnostics = receiver(request, 1);
    let result = match request.arguments.first().map(|value| &value.kind) {
        Some(ValueKindFact::Object(object)) => object
            .properties
            .get("State")
            .cloned()
            .unwrap_or_else(super::inference::network_rows_table_fact),
        _ => super::inference::network_rows_table_fact(),
    };
    crate::catalog::inference::finish_fixed(entry, request, result, diagnostics)
}

fn set_inference(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = receiver(request, 2);
    if let Some(value) = request.arguments.get(1) {
        let valid = match &value.kind {
            ValueKindFact::Unknown => true,
            ValueKindFact::Object(object) => object
                .runtime_class
                .as_ref()
                .is_none_or(|class| class.is(standard::TABLE)),
            _ => false,
        };
        if !valid {
            diagnostics.push(crate::catalog::inference::argument_error(
                "RM-DL-STATE-TABLE",
                "State must be a table",
                1,
            ));
        }
    }
    crate::catalog::inference::finish_fixed(
        entry,
        request,
        request
            .arguments
            .first()
            .cloned()
            .unwrap_or_else(|| ValueFact::unknown(DynamicReason::RuntimeValue)),
        diagnostics,
    )
}

macro_rules! state_entry {
    ($name:literal, $descriptor:ident, $documentation:ident, $inference:ident) => {
        BuiltinCatalogEntry {
            audience: crate::BuiltinCatalogAudience::Public,
            provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
            identity: BuiltinCatalogIdentity { name: $name },
            category: "deep_learning",
            documentation: $documentation,
            descriptor: &$descriptor,
            contract: BuiltinContractDeclaration {
                maturity: BuiltinContractMaturity::Incomplete,
                inference_rule: BuiltinInferenceRule::Identity(IdentityInferenceRule::new(
                    $inference as fn(&CallRequest, &BuiltinCatalogEntry) -> CallInference,
                )),
                compatibility: BuiltinCompatibility::Matlab,
                async_behavior: BuiltinAsyncBehavior::NeverSuspends,
                purity: BuiltinPurity::Pure,
                semantic_kind: BuiltinSemanticKind::General,
                workspace_effect: None,
                environment_effect: None,
                effects: &[EffectKind::MayThrow],
                capabilities: &[],
            },
            placement: BuiltinPlacementContract {
                preparation: BuiltinOperandPreparation::RuntimeOwned,
                portability: BuiltinPortability::NativeAndWasm,
                accelerator: BuiltinAcceleratorPolicy::Forbidden,
                residency: BuiltinResidencyPolicy::Host,
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
            extensions: &[],
            integer_capabilities: &[],
            integer_audit: None,
            suppress_auto_output: false,
        }
    };
}

pub(super) const GET_STATE_CATALOG_ENTRY: BuiltinCatalogEntry = state_entry!(
    "get.State",
    GET_DESCRIPTOR,
    GET_DOCUMENTATION,
    get_inference
);
pub(super) const SET_STATE_CATALOG_ENTRY: BuiltinCatalogEntry = state_entry!(
    "set.State",
    SET_DESCRIPTOR,
    SET_DOCUMENTATION,
    set_inference
);
