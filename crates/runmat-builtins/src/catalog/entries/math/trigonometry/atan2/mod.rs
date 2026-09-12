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
    BuiltinSignatureDescriptor, MathInferenceRule, ALL_INTEGER_CLASSES,
};
use runmat_types::{EffectKind, ExecutionStackRequirement};

mod documentation;
use documentation::ATAN2_DOCUMENTATION;

const OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Z",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Four-quadrant angle in radians with the broadcasted input shape.",
}];
const INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Real single, double, table, or timetable y-coordinate; integer, logical, and character forms are RunMat extensions.",
    },
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Compatible real x-coordinate or scalar-expansion operand.",
    },
];
const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "Z = atan2(Y, X)",
    inputs: &INPUTS,
    outputs: &OUTPUTS,
}];

pub const ATAN2_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ATAN2.INVALID_INPUT",
    identifier: Some("RunMat:atan2:InvalidInput"),
    when: "An input is not a supported real numeric, logical, character, table, or timetable value, or paired tabular inputs do not have compatible containers and variables.",
    message: "atan2: invalid input",
};
pub const ATAN2_ERROR_COMPLEX_UNSUPPORTED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ATAN2.COMPLEX_UNSUPPORTED",
    identifier: Some("RunMat:atan2:ComplexUnsupported"),
    when: "At least one operand is complex.",
    message: "atan2: complex inputs are not supported",
};
pub const ATAN2_ERROR_SIZE_MISMATCH: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ATAN2.SIZE_MISMATCH",
    identifier: Some("RunMat:atan2:SizeMismatch"),
    when: "Numeric operands are not broadcast-compatible.",
    message: "atan2: input sizes are not compatible for implicit expansion",
};
pub const ATAN2_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ATAN2.INTERNAL",
    identifier: Some("RunMat:atan2:Internal"),
    when: "Internal conversion, allocation, provider execution, or residency restoration fails.",
    message: "atan2: internal error",
};
pub const ATAN2_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ATAN2.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:atan2:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "atan2: too many output arguments",
};
const ERRORS: [BuiltinErrorDescriptor; 5] = [
    ATAN2_ERROR_INVALID_INPUT,
    ATAN2_ERROR_COMPLEX_UNSUPPORTED,
    ATAN2_ERROR_SIZE_MISMATCH,
    ATAN2_ERROR_INTERNAL,
    ATAN2_ERROR_TOO_MANY_OUTPUTS,
];
pub const ATAN2_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const ATAN2_INTEGER_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "atan2-integer-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "atan2 with fixed-width integer input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:Atan2IntegerInputExtension"),
};
pub const ATAN2_LOGICAL_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "atan2-logical-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "atan2 with logical input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:Atan2LogicalInputExtension"),
};
pub const ATAN2_CHARACTER_INPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "atan2-character-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "atan2 with character input is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:Atan2CharacterInputExtension"),
    };
pub const ATAN2_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    ATAN2_INTEGER_INPUT_EXTENSION,
    ATAN2_LOGICAL_INPUT_EXTENSION,
    ATAN2_CHARACTER_INPUT_EXTENSION,
];

const INTEGER_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "Y",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Every real fixed-width integer class is admitted in RunMat mode and converts at the floating-point algorithm boundary.",
    },
    BuiltinIntegerInputCapability {
        name: "X",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Integer and floating operands may be combined because atan2 always produces a floating result.",
    },
];
pub const ATAN2_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "Z = atan2(Y, X) with fixed-width integer input",
        inputs: &INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::BroadcastCompatible,
        notes: "Native integer storage remains authoritative until the explicit binary64 angle calculation; a single-precision companion selects a single result.",
    }];

const BINDINGS: [BuiltinBindingDeclaration; 1] = crate::REQUIRED_DEFAULT_BINDING;
const EFFECTS: [EffectKind; 2] = [EffectKind::MaySuspend, EffectKind::MayThrow];
pub const ATAN2_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    provenance: crate::BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "atan2" },
    category: "math/trigonometry",
    documentation: ATAN2_DOCUMENTATION,
    descriptor: &ATAN2_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Complete,
        inference_rule: BuiltinInferenceRule::Math(MathInferenceRule::Atan2),
        compatibility: BuiltinCompatibility::Matlab,
        async_behavior: BuiltinAsyncBehavior::MaySuspend,
        purity: BuiltinPurity::Pure,
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
        fusion: BuiltinFusionPolicy::Candidate,
        distributed: BuiltinDistributedPolicy::MaterializeArguments,
    },
    link: BuiltinLinkContract {
        reachability: BuiltinReachability::Always,
        policy: BuiltinLinkPolicy::PortableRuntime,
        execution_stack: ExecutionStackRequirement::Any,
        artifact_dependencies: &[],
    },
    bindings: &BINDINGS,
    extensions: &ATAN2_EXTENSIONS,
    integer_capabilities: &ATAN2_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&ATAN2_CATALOG_ENTRY];
