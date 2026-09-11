mod documentation;

use documentation::NOT_DOCUMENTATION;

pub const NOT_ERROR_INVALID_INPUT: crate::BuiltinErrorDescriptor = crate::BuiltinErrorDescriptor {
    code: "RM.NOT.INVALID_INPUT",
    identifier: Some("RunMat:not:InvalidInput"),
    when: "The input is not supported logical, numeric, character, complex, table, timetable, or resident numeric data.",
    message: "not: unsupported input type",
};
const ERRORS: [crate::BuiltinErrorDescriptor; 1] = [NOT_ERROR_INVALID_INPUT];
const OUTPUTS: [crate::BuiltinParamDescriptor; 1] = [crate::BuiltinParamDescriptor {
    name: "tf",
    ty: crate::BuiltinParamType::Any,
    arity: crate::BuiltinParamArity::Required,
    default: None,
    description: "Logical negation with the input array shape or tabular container.",
}];
const INPUTS: [crate::BuiltinParamDescriptor; 1] = [crate::BuiltinParamDescriptor {
    name: "A",
    ty: crate::BuiltinParamType::Any,
    arity: crate::BuiltinParamArity::Required,
    default: None,
    description: "Logical, numeric, character, complex, table, or timetable input.",
}];
const SIGNATURES: [crate::BuiltinSignatureDescriptor; 1] = [crate::BuiltinSignatureDescriptor {
    label: "tf = not(A)",
    inputs: &INPUTS,
    outputs: &OUTPUTS,
}];
pub const NOT_DESCRIPTOR: crate::BuiltinDescriptor = crate::BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: crate::BuiltinOutputMode::Fixed,
    completion_policy: crate::BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
const INTEGER_INPUTS: [crate::BuiltinIntegerInputCapability; 1] =
    [crate::BuiltinIntegerInputCapability {
        name: "A",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: crate::BuiltinIntegerInputAvailability::Documented,
        scalar_double: crate::BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Zero becomes true and every nonzero fixed-width integer becomes false without binary64 conversion.",
    }];
pub const NOT_INTEGER_CAPABILITIES: [crate::BuiltinIntegerCapabilityDescriptor; 1] =
    [crate::BuiltinIntegerCapabilityDescriptor {
        form: "tf = not(integer_A)",
        inputs: &INTEGER_INPUTS,
        computation_domain: crate::BuiltinIntegerComputationDomain::Predicate,
        output_class: crate::BuiltinIntegerOutputClassRule::Logical,
        overflow: crate::BuiltinIntegerOverflowRule::NotApplicable,
        backend: crate::BuiltinIntegerBackendRule::GatherFallback,
        overload: crate::BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
        notes: "Integer truth is evaluated from authoritative typed storage; explicitly resident fallback results return to the owning provider.",
    }];
const BINDINGS: [crate::BuiltinBindingDeclaration; 1] = crate::REQUIRED_DEFAULT_BINDING;
const EFFECTS: [runmat_types::EffectKind; 2] = [
    runmat_types::EffectKind::MaySuspend,
    runmat_types::EffectKind::MayThrow,
];
pub const NOT_CATALOG_ENTRY: crate::BuiltinCatalogEntry = crate::BuiltinCatalogEntry {
    provenance: crate::BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: crate::BuiltinCatalogIdentity { name: "not" },
    category: "logical/bit",
    documentation: NOT_DOCUMENTATION,
    descriptor: &NOT_DESCRIPTOR,
    contract: crate::BuiltinContractDeclaration {
        maturity: crate::BuiltinContractMaturity::Complete,
        inference_rule: crate::BuiltinInferenceRule::Logical(
            crate::LogicalInferenceRule::Elementwise(crate::LogicalElementwiseRule::Unary(
                crate::LogicalUnaryOperator::Not,
            )),
        ),
        compatibility: crate::BuiltinCompatibility::Matlab,
        async_behavior: crate::BuiltinAsyncBehavior::MaySuspend,
        purity: crate::BuiltinPurity::Pure,
        semantic_kind: crate::BuiltinSemanticKind::Elementwise,
        workspace_effect: None,
        environment_effect: None,
        effects: &EFFECTS,
        capabilities: &[],
    },
    placement: crate::BuiltinPlacementContract {
        portability: crate::BuiltinPortability::NativeAndWasm,
        accelerator: crate::BuiltinAcceleratorPolicy::Optional,
        residency: crate::BuiltinResidencyPolicy::PreserveInputs,
        fusion: crate::BuiltinFusionPolicy::Candidate,
        distributed: crate::BuiltinDistributedPolicy::MaterializeArguments,
    },
    link: crate::BuiltinLinkContract {
        reachability: crate::BuiltinReachability::Always,
        policy: crate::BuiltinLinkPolicy::PortableRuntime,
        execution_stack: runmat_types::ExecutionStackRequirement::Any,
        artifact_dependencies: &[],
    },
    bindings: &BINDINGS,
    extensions: &[],
    integer_capabilities: &NOT_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&NOT_CATALOG_ENTRY];
