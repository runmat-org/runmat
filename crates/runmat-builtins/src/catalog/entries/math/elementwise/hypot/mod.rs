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
use documentation::HYPOT_DOCUMENTATION;

const OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "C",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Nonnegative element-wise Euclidean norm with the broadcasted input shape.",
}];
const INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Single, double, or complex-floating first component; integer, logical, and character forms are RunMat extensions.",
    },
    BuiltinParamDescriptor {
        name: "B",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Second component with a size compatible for implicit expansion.",
    },
];
const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "C = hypot(A, B)",
    inputs: &INPUTS,
    outputs: &OUTPUTS,
}];

pub const HYPOT_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.HYPOT.INVALID_INPUT",
    identifier: Some("RunMat:hypot:InvalidInput"),
    when: "An operand cannot be converted to a supported numeric representation.",
    message: "hypot: invalid input",
};
pub const HYPOT_ERROR_SIZE_MISMATCH: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.HYPOT.SIZE_MISMATCH",
    identifier: Some("RunMat:hypot:SizeMismatch"),
    when: "Operands are not broadcast-compatible.",
    message: "hypot: size mismatch",
};
pub const HYPOT_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.HYPOT.INTERNAL",
    identifier: Some("RunMat:hypot:Internal"),
    when: "Internal conversion, allocation, provider execution, or residency restoration fails.",
    message: "hypot: internal error",
};
pub const HYPOT_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.HYPOT.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:hypot:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "hypot: too many output arguments",
};
const ERRORS: [BuiltinErrorDescriptor; 4] = [
    HYPOT_ERROR_INVALID_INPUT,
    HYPOT_ERROR_SIZE_MISMATCH,
    HYPOT_ERROR_INTERNAL,
    HYPOT_ERROR_TOO_MANY_OUTPUTS,
];
pub const HYPOT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const HYPOT_INTEGER_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "hypot-integer-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "hypot with fixed-width integer input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:HypotIntegerInputExtension"),
};
pub const HYPOT_LOGICAL_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "hypot-logical-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "hypot with logical input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:HypotLogicalInputExtension"),
};
pub const HYPOT_CHARACTER_INPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "hypot-character-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "hypot with character input is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:HypotCharacterInputExtension"),
    };
pub const HYPOT_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    HYPOT_INTEGER_INPUT_EXTENSION,
    HYPOT_LOGICAL_INPUT_EXTENSION,
    HYPOT_CHARACTER_INPUT_EXTENSION,
];

const INTEGER_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "A",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Every real fixed-width integer class is admitted in RunMat mode after exact binary64 validation.",
    },
    BuiltinIntegerInputCapability {
        name: "B",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Integer and floating operands may be combined because hypot produces a floating result.",
    },
];
pub const HYPOT_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "C = hypot(A, B) with fixed-width integer input",
        inputs: &INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::BroadcastCompatible,
        notes: "Native integer storage remains authoritative until exact validation and the stable binary64 norm calculation.",
    }];

const BINDINGS: [BuiltinBindingDeclaration; 1] = crate::REQUIRED_DEFAULT_BINDING;
const EFFECTS: [EffectKind; 2] = [EffectKind::MaySuspend, EffectKind::MayThrow];
pub const HYPOT_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    provenance: crate::BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "hypot" },
    category: "math/elementwise",
    documentation: HYPOT_DOCUMENTATION,
    descriptor: &HYPOT_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Complete,
        inference_rule: BuiltinInferenceRule::Math(MathInferenceRule::Hypot),
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
    extensions: &HYPOT_EXTENSIONS,
    integer_capabilities: &HYPOT_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&HYPOT_CATALOG_ENTRY];
