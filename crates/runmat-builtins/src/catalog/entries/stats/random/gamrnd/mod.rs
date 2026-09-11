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
use documentation::GAMRND_DOCUMENTATION;

const OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "r",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Nonnegative gamma-distributed samples with the requested shape.",
}];
const PARAMETER_A: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "a",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Nonnegative shape parameter; a scalar or dense real single/double array.",
};
const PARAMETER_B: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "b",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Positive scale parameter; a scalar or dense real single/double array.",
};
const PARAMETERS: [BuiltinParamDescriptor; 2] = [PARAMETER_A, PARAMETER_B];
const PARAMETERS_AND_SIZE: [BuiltinParamDescriptor; 3] = [
    PARAMETER_A,
    PARAMETER_B,
    BuiltinParamDescriptor {
        name: "sz",
        ty: BuiltinParamType::SizeArg,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Output size as one scalar or one row vector.",
    },
];
const PARAMETERS_AND_DIMENSIONS: [BuiltinParamDescriptor; 3] = [
    PARAMETER_A,
    PARAMETER_B,
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
        label: "r = gamrnd(a, b)",
        inputs: &PARAMETERS,
        outputs: &OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "r = gamrnd(a, b, sz)",
        inputs: &PARAMETERS_AND_SIZE,
        outputs: &OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "r = gamrnd(a, b, sz1, sz2, ...)",
        inputs: &PARAMETERS_AND_DIMENSIONS,
        outputs: &OUTPUTS,
    },
];

pub const GAMRND_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GAMRND.INVALID_ARGUMENT",
    identifier: Some("RunMat:gamrnd:InvalidArgument"),
    when: "Parameters or size arguments are missing, unsupported, outside their domains, or shape-incompatible.",
    message: "gamrnd: invalid argument",
};
pub const GAMRND_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GAMRND.INTERNAL",
    identifier: Some("RunMat:gamrnd:Internal"),
    when: "Random-state access, allocation, provider gather, or output restoration fails.",
    message: "gamrnd: internal error",
};
pub const GAMRND_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GAMRND.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:gamrnd:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "gamrnd: too many output arguments",
};
const ERRORS: [BuiltinErrorDescriptor; 3] = [
    GAMRND_ERROR_INVALID_ARGUMENT,
    GAMRND_ERROR_INTERNAL,
    GAMRND_ERROR_TOO_MANY_OUTPUTS,
];
pub const GAMRND_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const GAMRND_INTEGER_SHAPE_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "gamrnd-integer-shape-parameter",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "gamrnd with a fixed-width integer shape parameter is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:GamrndIntegerShapeParameterExtension"),
};
pub const GAMRND_INTEGER_SCALE_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "gamrnd-integer-scale-parameter",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "gamrnd with a fixed-width integer scale parameter is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:GamrndIntegerScaleParameterExtension"),
};
pub const GAMRND_INTEGER_SIZE_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "gamrnd-integer-size",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "gamrnd with fixed-width integer size arguments is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:GamrndIntegerSizeExtension"),
};
const EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    GAMRND_INTEGER_SHAPE_EXTENSION,
    GAMRND_INTEGER_SCALE_EXTENSION,
    GAMRND_INTEGER_SIZE_EXTENSION,
];

const INTEGER_SHAPE_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "a",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Every value must be exactly representable at the binary64 sampling boundary.",
}];
const INTEGER_SCALE_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "b",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Every value must be exactly representable at the binary64 sampling boundary.",
}];
const INTEGER_SIZE_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "sz, sz1, ...",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
    notes: "Values are decoded exactly from authoritative storage into bounded dimensions.",
}];
pub const GAMRND_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "r = gamrnd(integer_a, b, ___)",
        inputs: &INTEGER_SHAPE_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Integer parameters produce double unless the other documented parameter is single.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "r = gamrnd(a, integer_b, ___)",
        inputs: &INTEGER_SCALE_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Integer parameters produce double unless the other documented parameter is single.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "r = gamrnd(a, b, integer_sz)",
        inputs: &INTEGER_SIZE_INPUT,
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
pub const GAMRND_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    provenance: crate::BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "gamrnd" },
    category: "stats/random",
    documentation: GAMRND_DOCUMENTATION,
    descriptor: &GAMRND_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Complete,
        inference_rule: BuiltinInferenceRule::Stats(StatsInferenceRule::Random(
            StatsRandomInferenceRule::Gamma,
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
    integer_capabilities: &GAMRND_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&GAMRND_CATALOG_ENTRY];
