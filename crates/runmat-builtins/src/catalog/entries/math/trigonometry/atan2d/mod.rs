use crate::*;
use runmat_types::{EffectKind, ExecutionStackRequirement};

mod documentation;
mod inference;

const INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Real y-coordinate, array, table, or timetable.",
    },
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Real x-coordinate, broadcast-compatible array, table, or timetable.",
    },
];
const OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "D",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Four-quadrant angle in degrees, preserving a tabular input container.",
}];
const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "D = atan2d(Y, X)",
    inputs: &INPUTS,
    outputs: &OUTPUTS,
}];

pub const ATAN2D_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ATAN2D.INVALID_INPUT", identifier: Some("RunMat:atan2d:InvalidInput"),
    when: "Either coordinate is not supported real numeric or tabular input, tabular variables are incompatible, or resident inputs have different owners.",
    message: "atan2d: invalid input",
};
pub const ATAN2D_ERROR_COMPLEX_UNSUPPORTED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ATAN2D.COMPLEX_UNSUPPORTED",
    identifier: Some("RunMat:atan2d:ComplexUnsupported"),
    when: "Either coordinate is complex.",
    message: "atan2d: complex inputs are not supported",
};
pub const ATAN2D_ERROR_SIZE_MISMATCH: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ATAN2D.SIZE_MISMATCH",
    identifier: Some("RunMat:atan2d:SizeMismatch"),
    when: "Coordinate shapes are not broadcast-compatible.",
    message: "atan2d: input sizes are not compatible for implicit expansion",
};
pub const ATAN2D_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ATAN2D.INTERNAL",
    identifier: Some("RunMat:atan2d:Internal"),
    when: "Gather, calculation, allocation or owner-preserving restoration fails.",
    message: "atan2d: internal error",
};
pub const ATAN2D_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ATAN2D.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:atan2d:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "atan2d: too many output arguments",
};
const ERRORS: [BuiltinErrorDescriptor; 5] = [
    ATAN2D_ERROR_INVALID_INPUT,
    ATAN2D_ERROR_COMPLEX_UNSUPPORTED,
    ATAN2D_ERROR_SIZE_MISMATCH,
    ATAN2D_ERROR_INTERNAL,
    ATAN2D_ERROR_TOO_MANY_OUTPUTS,
];
pub const ATAN2D_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
pub const ATAN2D_INTEGER_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "atan2d-integer-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "atan2d with integer coordinates is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:Atan2dIntegerInputExtension"),
};
pub const ATAN2D_LOGICAL_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "atan2d-logical-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "atan2d with logical coordinates is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:Atan2dLogicalInputExtension"),
};
pub const ATAN2D_CHARACTER_INPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "atan2d-character-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "atan2d with character coordinates is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:Atan2dCharacterInputExtension"),
    };
const EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    ATAN2D_INTEGER_INPUT_EXTENSION,
    ATAN2D_LOGICAL_INPUT_EXTENSION,
    ATAN2D_CHARACTER_INPUT_EXTENSION,
];
const INTEGER_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "Y",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Integer values are validated before the floating angle calculation.",
    },
    BuiltinIntegerInputCapability {
        name: "X",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Integer values are validated before the floating angle calculation.",
    },
];
const INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] = [BuiltinIntegerCapabilityDescriptor {
    form: "D = atan2d(integer_Y, integer_X)", inputs: &INTEGER_INPUTS,
    computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
    output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
    overflow: BuiltinIntegerOverflowRule::Error,
    backend: BuiltinIntegerBackendRule::GatherFallback,
    overload: BuiltinIntegerOverloadKind::BroadcastCompatible,
    notes: "Native integer storage remains authoritative until binary64 computation; a single-precision companion selects single output.",
}];
const EFFECTS: [EffectKind; 2] = [EffectKind::MaySuspend, EffectKind::MayThrow];

pub const ATAN2D_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    audience: BuiltinCatalogAudience::Public,
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "atan2d" },
    category: "math/trigonometry",
    documentation: documentation::DOCUMENTATION,
    descriptor: &ATAN2D_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Incomplete,
        inference_rule: BuiltinInferenceRule::Identity(IdentityInferenceRule::new(
            inference::infer,
        )),
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
        preparation: BuiltinOperandPreparation::RuntimeOwned,
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
    bindings: &REQUIRED_DEFAULT_BINDING,
    extensions: &EXTENSIONS,
    integer_capabilities: &INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&ATAN2D_CATALOG_ENTRY];
