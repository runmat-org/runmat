mod documentation;

use documentation::ALLFINITE_DOCUMENTATION;

const OUTPUTS: [crate::BuiltinParamDescriptor; 1] = [crate::BuiltinParamDescriptor {
    name: "tf",
    ty: crate::BuiltinParamType::LogicalArray,
    arity: crate::BuiltinParamArity::Required,
    default: None,
    description: "Logical scalar that is true when every input element is finite.",
}];
const INPUTS: [crate::BuiltinParamDescriptor; 1] = [crate::BuiltinParamDescriptor {
    name: "A",
    ty: crate::BuiltinParamType::Any,
    arity: crate::BuiltinParamArity::Required,
    default: None,
    description: "Numeric, logical, character, or supported string value to reduce.",
}];
const SIGNATURES: [crate::BuiltinSignatureDescriptor; 1] = [crate::BuiltinSignatureDescriptor {
    label: "tf = allfinite(A)",
    inputs: &INPUTS,
    outputs: &OUTPUTS,
}];

pub const ALLFINITE_ERROR_INVALID_INPUT: crate::BuiltinErrorDescriptor =
    crate::BuiltinErrorDescriptor {
        code: "RM.ALLFINITE.INVALID_INPUT",
        identifier: Some("RunMat:allfinite:InvalidInput"),
        when: "Input is not numeric, logical, character, or a supported string value.",
        message: "allfinite: expected numeric, logical, character, or supported string input",
    };
pub const ALLFINITE_ERROR_INTERNAL: crate::BuiltinErrorDescriptor = crate::BuiltinErrorDescriptor {
    code: "RM.ALLFINITE.INTERNAL",
    identifier: Some("RunMat:allfinite:InternalError"),
    when: "Provider execution, transfer, storage validation, or reduction fails.",
    message: "allfinite: internal error",
};
pub const ALLFINITE_ERROR_TOO_MANY_OUTPUTS: crate::BuiltinErrorDescriptor =
    crate::BuiltinErrorDescriptor {
        code: "RM.ALLFINITE.TOO_MANY_OUTPUTS",
        identifier: Some("RunMat:allfinite:TooManyOutputs"),
        when: "More than one output is requested.",
        message: "allfinite: too many output arguments",
    };
const ERRORS: [crate::BuiltinErrorDescriptor; 3] = [
    ALLFINITE_ERROR_INVALID_INPUT,
    ALLFINITE_ERROR_INTERNAL,
    ALLFINITE_ERROR_TOO_MANY_OUTPUTS,
];
pub const ALLFINITE_DESCRIPTOR: crate::BuiltinDescriptor = crate::BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: crate::BuiltinOutputMode::Fixed,
    completion_policy: crate::BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const ALLFINITE_STRING_INPUT_EXTENSION: crate::BuiltinExtensionDescriptor =
    crate::BuiltinExtensionDescriptor {
        id: "allfinite-string-input",
        mode: crate::BuiltinExtensionMode::RunMatOnly,
        description: "allfinite with string input is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:AllfiniteStringInputExtension"),
    };
const EXTENSIONS: [crate::BuiltinExtensionDescriptor; 1] = [ALLFINITE_STRING_INPUT_EXTENSION];

const INTEGER_INPUTS: [crate::BuiltinIntegerInputCapability; 1] =
    [crate::BuiltinIntegerInputCapability {
        name: "A",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: crate::BuiltinIntegerInputAvailability::Documented,
        scalar_double: crate::BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Every representable value in all eight fixed-width integer classes is finite.",
    }];
const INTEGER_CAPABILITIES: [crate::BuiltinIntegerCapabilityDescriptor; 1] =
    [crate::BuiltinIntegerCapabilityDescriptor {
        form: "tf = allfinite(integer_A)",
        inputs: &INTEGER_INPUTS,
        computation_domain: crate::BuiltinIntegerComputationDomain::Predicate,
        output_class: crate::BuiltinIntegerOutputClassRule::Logical,
        overflow: crate::BuiltinIntegerOverflowRule::NotApplicable,
        backend: crate::BuiltinIntegerBackendRule::HostAndGpu,
        overload: crate::BuiltinIntegerOverloadKind::Multiple,
        notes: "Returns logical true from exact class and shape metadata without converting or reading integer payload values, including empty arrays.",
    }];

const BINDINGS: [crate::BuiltinBindingDeclaration; 1] = crate::REQUIRED_DEFAULT_BINDING;
const EFFECTS: [runmat_types::EffectKind; 2] = [
    runmat_types::EffectKind::MaySuspend,
    runmat_types::EffectKind::MayThrow,
];
pub const ALLFINITE_CATALOG_ENTRY: crate::BuiltinCatalogEntry = crate::BuiltinCatalogEntry {
    identity: crate::BuiltinCatalogIdentity { name: "allfinite" },
    category: "logical/tests",
    documentation: ALLFINITE_DOCUMENTATION,
    descriptor: &ALLFINITE_DESCRIPTOR,
    contract: crate::BuiltinContractDeclaration {
        maturity: crate::BuiltinContractMaturity::Complete,
        inference_rule: crate::BuiltinInferenceRule::Logical(
            crate::LogicalInferenceRule::ScalarReduction(crate::ScalarLogicalReduction::AllFinite),
        ),
        compatibility: crate::BuiltinCompatibility::Matlab,
        async_behavior: crate::BuiltinAsyncBehavior::MaySuspend,
        purity: crate::BuiltinPurity::Pure,
        semantic_kind: crate::BuiltinSemanticKind::General,
        workspace_effect: None,
        environment_effect: None,
        effects: &EFFECTS,
        capabilities: &[],
    },
    placement: crate::BuiltinPlacementContract {
        portability: crate::BuiltinPortability::NativeAndWasm,
        accelerator: crate::BuiltinAcceleratorPolicy::Optional,
        residency: crate::BuiltinResidencyPolicy::GatherToHost,
        fusion: crate::BuiltinFusionPolicy::Boundary,
        distributed: crate::BuiltinDistributedPolicy::MaterializeArguments,
    },
    link: crate::BuiltinLinkContract {
        reachability: crate::BuiltinReachability::Always,
        policy: crate::BuiltinLinkPolicy::PortableRuntime,
        execution_stack: runmat_types::ExecutionStackRequirement::Any,
        artifact_dependencies: &[],
    },
    bindings: &BINDINGS,
    extensions: &EXTENSIONS,
    integer_capabilities: &INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&ALLFINITE_CATALOG_ENTRY];
