use crate::{
    BuiltinAcceleratorPolicy, BuiltinAsyncBehavior, BuiltinBindingDeclaration, BuiltinCatalogEntry,
    BuiltinCatalogIdentity, BuiltinCompatibility, BuiltinCompletionPolicy,
    BuiltinContractDeclaration, BuiltinContractMaturity, BuiltinDescriptor,
    BuiltinDistributedPolicy, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinFusionPolicy, BuiltinInferenceRule, BuiltinIntegerBackendRule,
    BuiltinIntegerCapabilityDescriptor, BuiltinIntegerComputationDomain,
    BuiltinIntegerInputAvailability, BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule,
    BuiltinIntegerOverflowRule, BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule,
    BuiltinLinkContract, BuiltinLinkPolicy, BuiltinOutputMode, BuiltinParamArity,
    BuiltinParamDescriptor, BuiltinParamType, BuiltinPlacementContract, BuiltinPortability,
    BuiltinPurity, BuiltinReachability, BuiltinResidencyPolicy, BuiltinSemanticKind,
    BuiltinSignatureDescriptor, StatsInferenceRule, StatsRandomInferenceRule, ALL_INTEGER_CLASSES,
};
use runmat_types::{EffectKind, ExecutionStackRequirement};

mod documentation;
use documentation::BINORND_DOCUMENTATION;

const OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "r",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Binomially distributed samples with the requested shape.",
}];
const TRIALS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "n",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Positive integer trial counts supplied as dense real single or double values.",
};
const PROBABILITY: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "p",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Success probabilities in the inclusive interval from zero to one.",
};
const PARAMETERS: [BuiltinParamDescriptor; 2] = [TRIALS, PROBABILITY];
const PARAMETERS_AND_SIZE: [BuiltinParamDescriptor; 3] = [
    TRIALS,
    PROBABILITY,
    BuiltinParamDescriptor {
        name: "sz",
        ty: BuiltinParamType::SizeArg,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Output size as one scalar or one row vector.",
    },
];
const PARAMETERS_AND_DIMENSIONS: [BuiltinParamDescriptor; 3] = [
    TRIALS,
    PROBABILITY,
    BuiltinParamDescriptor {
        name: "sz1, sz2, ...",
        ty: BuiltinParamType::SizeArg,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Output dimension sizes supplied as separate scalar values.",
    },
];
const SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "r = binornd(n, p)",
        inputs: &PARAMETERS,
        outputs: &OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "r = binornd(n, p, sz)",
        inputs: &PARAMETERS_AND_SIZE,
        outputs: &OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "r = binornd(n, p, sz1, sz2, ...)",
        inputs: &PARAMETERS_AND_DIMENSIONS,
        outputs: &OUTPUTS,
    },
];

pub const BINORND_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.BINORND.INVALID_ARGUMENT",
    identifier: Some("RunMat:binornd:InvalidArgument"),
    when: "Parameters or size controls are missing, unsupported, outside their domains, or shape-incompatible.",
    message: "binornd: invalid argument",
};
pub const BINORND_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.BINORND.INTERNAL",
    identifier: Some("RunMat:binornd:Internal"),
    when: "Random-state access, allocation, provider gather, or output restoration fails.",
    message: "binornd: internal error",
};
pub const BINORND_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.BINORND.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:binornd:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "binornd: too many output arguments",
};
const ERRORS: [BuiltinErrorDescriptor; 3] = [
    BINORND_ERROR_INVALID_ARGUMENT,
    BINORND_ERROR_INTERNAL,
    BINORND_ERROR_TOO_MANY_OUTPUTS,
];
pub const BINORND_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const BINORND_INTEGER_TRIALS_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "binornd-integer-trials",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "binornd with fixed-width integer trial counts is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:BinorndIntegerTrialsExtension"),
    };
pub const BINORND_INTEGER_PROBABILITY_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "binornd-integer-probability",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "binornd with fixed-width integer probabilities is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:BinorndIntegerProbabilityExtension"),
    };
pub const BINORND_INTEGER_SIZE_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "binornd-integer-size",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "binornd with fixed-width integer size controls is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:BinorndIntegerSizeExtension"),
};
pub const BINORND_LOGICAL_INPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "binornd-logical-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "binornd with logical parameters is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:BinorndLogicalInputExtension"),
    };
const EXTENSIONS: [BuiltinExtensionDescriptor; 4] = [
    BINORND_INTEGER_TRIALS_EXTENSION,
    BINORND_INTEGER_PROBABILITY_EXTENSION,
    BINORND_INTEGER_SIZE_EXTENSION,
    BINORND_LOGICAL_INPUT_EXTENSION,
];

const INTEGER_TRIALS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "n",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Values must be positive trial counts and exactly representable at the binary64 sampling boundary.",
}];
const INTEGER_PROBABILITY: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "p",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Values remain authoritative until their checked binary64 sampling boundary.",
}];
const INTEGER_SIZE: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "sz, sz1, ...",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
    notes: "Values are decoded exactly from authoritative storage into bounded dimensions.",
}];
pub const BINORND_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "r = binornd(integer_n, p, ___)",
        inputs: &INTEGER_TRIALS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Integer trial counts produce double unless the probability parameter is single.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "r = binornd(n, integer_p, ___)",
        inputs: &INTEGER_PROBABILITY,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Integer probabilities produce double unless the trial-count parameter is single.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "r = binornd(n, p, integer_sz)",
        inputs: &INTEGER_SIZE,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Integer sizes do not select output class or execution residency.",
    },
];

const BINDINGS: [BuiltinBindingDeclaration; 1] = crate::REQUIRED_DEFAULT_BINDING;
const EFFECTS: [EffectKind; 3] = [
    EffectKind::Randomness,
    EffectKind::MaySuspend,
    EffectKind::MayThrow,
];
pub const BINORND_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    provenance: crate::BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "binornd" },
    category: "stats/random",
    documentation: BINORND_DOCUMENTATION,
    descriptor: &BINORND_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Complete,
        inference_rule: BuiltinInferenceRule::Stats(StatsInferenceRule::Random(
            StatsRandomInferenceRule::Binomial,
        )),
        compatibility: BuiltinCompatibility::Matlab,
        async_behavior: BuiltinAsyncBehavior::MaySuspend,
        purity: BuiltinPurity::Impure,
        semantic_kind: BuiltinSemanticKind::General,
        workspace_effect: None,
        environment_effect: None,
        effects: &EFFECTS,
        capabilities: &[],
    },
    placement: BuiltinPlacementContract {
        portability: BuiltinPortability::NativeAndWasm,
        accelerator: BuiltinAcceleratorPolicy::Optional,
        residency: BuiltinResidencyPolicy::PreserveInputs,
        fusion: BuiltinFusionPolicy::Never,
        distributed: BuiltinDistributedPolicy::MaterializeArguments,
    },
    link: BuiltinLinkContract {
        reachability: BuiltinReachability::Always,
        policy: BuiltinLinkPolicy::PortableRuntime,
        execution_stack: ExecutionStackRequirement::Any,
        artifact_dependencies: &[],
    },
    bindings: &BINDINGS,
    extensions: &EXTENSIONS,
    integer_capabilities: &BINORND_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&BINORND_CATALOG_ENTRY];
