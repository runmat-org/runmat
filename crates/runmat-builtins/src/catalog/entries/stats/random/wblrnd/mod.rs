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
use documentation::WBLRND_DOCUMENTATION;

const OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "r",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Positive Weibull-distributed samples with the requested shape.",
}];
const SCALE: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "a",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Positive scale parameter supplied as a scalar or dense real array.",
};
const SHAPE: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "b",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Positive shape parameter supplied as a scalar or dense real array.",
};
const PARAMETERS: [BuiltinParamDescriptor; 2] = [SCALE, SHAPE];
const PARAMETERS_AND_SIZE: [BuiltinParamDescriptor; 3] = [
    SCALE,
    SHAPE,
    BuiltinParamDescriptor {
        name: "sz",
        ty: BuiltinParamType::SizeArg,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Output size as one scalar or one row vector.",
    },
];
const PARAMETERS_AND_DIMENSIONS: [BuiltinParamDescriptor; 3] = [
    SCALE,
    SHAPE,
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
        label: "r = wblrnd(a, b)",
        inputs: &PARAMETERS,
        outputs: &OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "r = wblrnd(a, b, sz)",
        inputs: &PARAMETERS_AND_SIZE,
        outputs: &OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "r = wblrnd(a, b, sz1, sz2, ...)",
        inputs: &PARAMETERS_AND_DIMENSIONS,
        outputs: &OUTPUTS,
    },
];

pub const WBLRND_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.WBLRND.INVALID_ARGUMENT",
    identifier: Some("RunMat:wblrnd:InvalidArgument"),
    when: "Parameters or size controls are missing, unsupported, outside their domains, or shape-incompatible.",
    message: "wblrnd: invalid argument",
};
pub const WBLRND_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.WBLRND.INTERNAL",
    identifier: Some("RunMat:wblrnd:Internal"),
    when: "Random-state access, allocation, provider gather, or output restoration fails.",
    message: "wblrnd: internal error",
};
pub const WBLRND_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.WBLRND.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:wblrnd:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "wblrnd: too many output arguments",
};
const ERRORS: [BuiltinErrorDescriptor; 3] = [
    WBLRND_ERROR_INVALID_ARGUMENT,
    WBLRND_ERROR_INTERNAL,
    WBLRND_ERROR_TOO_MANY_OUTPUTS,
];
pub const WBLRND_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const WBLRND_INTEGER_SCALE_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "wblrnd-integer-scale",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "wblrnd with a fixed-width integer scale parameter is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:WblrndIntegerScaleExtension"),
};
pub const WBLRND_INTEGER_SHAPE_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "wblrnd-integer-shape",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "wblrnd with a fixed-width integer shape parameter is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:WblrndIntegerShapeExtension"),
};
pub const WBLRND_INTEGER_SIZE_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "wblrnd-integer-size",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "wblrnd with fixed-width integer size controls is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:WblrndIntegerSizeExtension"),
};
pub const WBLRND_LOGICAL_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "wblrnd-logical-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "wblrnd with logical parameters or size controls is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:WblrndLogicalInputExtension"),
};
const EXTENSIONS: [BuiltinExtensionDescriptor; 4] = [
    WBLRND_INTEGER_SCALE_EXTENSION,
    WBLRND_INTEGER_SHAPE_EXTENSION,
    WBLRND_INTEGER_SIZE_EXTENSION,
    WBLRND_LOGICAL_INPUT_EXTENSION,
];

const INTEGER_SCALE: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "a",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Values must be positive and exactly representable at the binary64 sampling boundary.",
}];
const INTEGER_SHAPE: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "b",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Values must be positive and exactly representable at the binary64 sampling boundary.",
}];
const INTEGER_SIZE: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "sz, sz1, ...",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
    notes: "Values are decoded exactly from authoritative storage into bounded dimensions.",
}];
pub const WBLRND_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "r = wblrnd(integer_a, b, ___)",
        inputs: &INTEGER_SCALE,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Integer scale parameters produce double unless the shape parameter is single.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "r = wblrnd(a, integer_b, ___)",
        inputs: &INTEGER_SHAPE,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Integer shape parameters produce double unless the scale parameter is single.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "r = wblrnd(a, b, integer_sz)",
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
pub const WBLRND_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    provenance: crate::BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "wblrnd" },
    category: "stats/random",
    documentation: WBLRND_DOCUMENTATION,
    descriptor: &WBLRND_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Complete,
        inference_rule: BuiltinInferenceRule::Stats(StatsInferenceRule::Random(
            StatsRandomInferenceRule::Weibull,
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
    integer_capabilities: &WBLRND_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&WBLRND_CATALOG_ENTRY];
