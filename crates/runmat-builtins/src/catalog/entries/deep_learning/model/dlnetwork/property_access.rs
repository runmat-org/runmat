//! Dependent network-parameter accessors share the canonical layer authority.
use crate::*;
use runmat_types::{
    standard, CallInference, CallRequest, DynamicReason, EffectKind, ExecutionStackRequirement,
    InferenceDiagnostic, ValueFact, ValueKindFact,
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
        name: "learnables",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "A Layer/Parameter/Value table matching the network parameter topology.",
    },
];
const GET_OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "learnables",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "A table derived from the network's canonical Layers.",
}];
const SET_OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "network",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "The updated value network.",
}];
const GET_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "learnables = get.Learnables(network)",
    inputs: &GET_INPUTS,
    outputs: &GET_OUTPUTS,
}];
const SET_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "network = set.Learnables(network, learnables)",
    inputs: &SET_INPUTS,
    outputs: &SET_OUTPUTS,
}];
const ACCESS_ERROR: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DEEP_LEARNING.INVALID_INPUT",
    identifier: Some("RunMat:deepLearning:InvalidInput"),
    when: "The network or Learnables table does not match the supported parameter topology.",
    message: "network Learnables access received invalid input",
};
const ACCESS_ERRORS: [BuiltinErrorDescriptor; 1] = [ACCESS_ERROR];
const GET_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &GET_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ACCESS_ERRORS,
};
const SET_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SET_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ACCESS_ERRORS,
};

const GET_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("get.Learnables"), slug: Some("get.Learnables"),
    summary: "Read the parameter table projected from a network's Layers.",
    description: "Network parameters are stored once in Layers. The Learnables table is derived when read; its resident values retain their GPU leases without a download.",
    keywords: &["dlnetwork", "Learnables", "parameters"], related: &["dlnetwork", "dlupdate"],
    sections: &[BuiltinDocumentationSection { heading: "Parameter ownership", paragraphs: &[
        "Layer and Parameter are string-array columns. Value is a cell column of dlarray values, with an empty value for an uninitialized parameter. Rows appear in layer order. Supported roles include fully connected and convolutional Weights/Bias, recurrent InputWeights/RecurrentWeights/Bias, and normalization Scale/Offset.",
        "Reading Learnables does not update the network. Assignment to the dependent property uses set.Learnables and validates the complete table before replacing any parameter.",
        "RunMat also exposes this projection on its SeriesNetwork and DAGNetwork values; this is a RunMat extension, not a claim that MATLAB publishes a Learnables property on those classes.",
        "This supported parameter subset is real numeric. Complex learnables are not currently admitted; they are not coerced to real values.",
    ] }],
    examples: &[], example_exemption: Some("This is a dependent-property accessor; executable network editing is covered by dlnetwork examples."),
    faqs: &[], links: &[], media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink {
            label: "Canonical network parameter projection",
            target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/deep_learning/model/parameters.rs"),
        }],
        verification: &[BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::UnitTest,
            label: "Network Learnables projection and update",
            location: "builtins::deep_learning::tests::network_learnables_setter_updates_only_canonical_layers",
        }], notes: &[],
    },
    introduced: None, status: Some(BuiltinDocumentationStatus::Partial),
};
const SET_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("set.Learnables"), slug: Some("set.Learnables"),
    summary: "Replace a network's validated parameter values.",
    description: "The setter validates every table row, identity, shape, and storage form before updating the canonical Layers. It does not gather resident values.",
    keywords: &["dlnetwork", "Learnables", "parameters"], related: &["dlnetwork", "dlupdate"],
    sections: &[BuiltinDocumentationSection { heading: "Atomic update", paragraphs: &[
        "The table must contain exactly Layer, Parameter, and Value columns corresponding to the existing network. Rows can be reordered when every typed Layer/Parameter identity remains unique and complete. Empty numeric values make the network uninitialized; valid replacements can initialize it. RunMat also accepts direct real numeric Value entries and cell-of-text identity columns as input extensions, while its public projection uses dlarray values and string columns.",
        "Invalid input returns an error without changing the value network or invoking a provider.",
    ] }],
    examples: &[], example_exemption: Some("This is a dependent-property setter; executable network editing is covered by dlnetwork examples."),
    faqs: &[], links: &[], media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink {
            label: "Canonical network parameter update",
            target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/deep_learning/model/parameters.rs"),
        }],
        verification: &[BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::UnitTest,
            label: "Atomic parameter replacement and rejection",
            location: "builtins::deep_learning::tests::network_learnables_reject_mismatched_identity_and_shape_before_publish",
        }], notes: &[],
    },
    introduced: None, status: Some(BuiltinDocumentationStatus::Partial),
};

fn get_inference(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    network_receiver(request, 1, &mut diagnostics);
    let result = match request.arguments.first().map(|argument| &argument.kind) {
        Some(ValueKindFact::Object(object)) => object
            .properties
            .get("Learnables")
            .cloned()
            .unwrap_or_else(super::inference::network_rows_table_fact),
        _ => super::inference::network_rows_table_fact(),
    };
    crate::catalog::inference::finish_fixed(entry, request, result, diagnostics)
}
fn set_inference(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    network_receiver(request, 2, &mut diagnostics);
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
                "RM-DL-LEARNABLES-TABLE",
                "Learnables must be a table",
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

fn network_receiver(
    request: &CallRequest,
    expected: usize,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) {
    if request.arguments.len() != expected {
        diagnostics.push(crate::catalog::inference::argument_error(
            "RM-DL-LEARNABLES-ARITY",
            "incorrect number of property-access arguments",
            expected,
        ));
    }
    if let Some(receiver) = request.arguments.first() {
        let valid = match &receiver.kind {
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
                "RM-DL-LEARNABLES-NETWORK",
                "expected a deep-learning network value",
                0,
            ));
        }
    }
}

pub(super) const GET_LEARNABLES_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    audience: crate::BuiltinCatalogAudience::Public,
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity {
        name: "get.Learnables",
    },
    category: "deep_learning",
    documentation: GET_DOCUMENTATION,
    descriptor: &GET_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Incomplete,
        inference_rule: BuiltinInferenceRule::Identity(IdentityInferenceRule::new(
            get_inference as fn(&CallRequest, &BuiltinCatalogEntry) -> CallInference,
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
};
pub(super) const SET_LEARNABLES_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    audience: crate::BuiltinCatalogAudience::Public,
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity {
        name: "set.Learnables",
    },
    category: "deep_learning",
    documentation: SET_DOCUMENTATION,
    descriptor: &SET_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Incomplete,
        inference_rule: BuiltinInferenceRule::Identity(IdentityInferenceRule::new(
            set_inference as fn(&CallRequest, &BuiltinCatalogEntry) -> CallInference,
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
};
