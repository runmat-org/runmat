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
    BuiltinSignatureDescriptor, DiscreteInferenceRule, MathInferenceRule, ALL_INTEGER_CLASSES,
};
use runmat_types::NumericClass;
use runmat_types::{EffectKind, ExecutionStackRequirement};

mod documentation;
use documentation::FACTORIAL_DOCUMENTATION;

const OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "F",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Elementwise factorial values with the input class and shape.",
}];
const INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "N",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Dense real, finite, nonnegative integer-valued input.",
}];
const LIKE_INPUTS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "N",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Dense real, finite, nonnegative integer-valued input.",
    },
    BuiltinParamDescriptor {
        name: "like",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Literal string \"like\".",
    },
    BuiltinParamDescriptor {
        name: "prototype",
        ty: BuiltinParamType::LikePrototype,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "RunMat output-residency prototype; the result class still follows N.",
    },
];
const SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "F = factorial(N)",
        inputs: &INPUTS,
        outputs: &OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "F = factorial(N, \"like\", prototype)",
        inputs: &LIKE_INPUTS,
        outputs: &OUTPUTS,
    },
];

pub const FACTORIAL_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FACTORIAL.INVALID_ARGUMENT",
    identifier: Some("RunMat:factorial:InvalidArgument"),
    when: "The invocation has an invalid arity or malformed optional arguments.",
    message: "factorial: invalid argument",
};
pub const FACTORIAL_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FACTORIAL.INVALID_INPUT",
    identifier: Some("RunMat:factorial:InvalidInput"),
    when: "Input is complex, sparse, nonnumeric, nonfinite, negative, or not integer-valued.",
    message: "factorial: invalid input",
};
pub const FACTORIAL_ERROR_GPU_UNSUPPORTED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FACTORIAL.GPU_UNSUPPORTED",
    identifier: Some("RunMat:factorial:GpuUnsupported"),
    when: "Resident output is requested but no compatible provider owns or can create it.",
    message: "factorial: gpu output not supported",
};
pub const FACTORIAL_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FACTORIAL.INTERNAL",
    identifier: Some("RunMat:factorial:Internal"),
    when: "Internal allocation, provider execution, gather, or restoration fails.",
    message: "factorial: internal error",
};
pub const FACTORIAL_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FACTORIAL.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:factorial:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "factorial: too many output arguments",
};
const ERRORS: [BuiltinErrorDescriptor; 5] = [
    FACTORIAL_ERROR_INVALID_ARGUMENT,
    FACTORIAL_ERROR_INVALID_INPUT,
    FACTORIAL_ERROR_GPU_UNSUPPORTED,
    FACTORIAL_ERROR_INTERNAL,
    FACTORIAL_ERROR_TOO_MANY_OUTPUTS,
];
pub const FACTORIAL_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const FACTORIAL_LIKE_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "factorial-like-residency",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "factorial(N, 'like', prototype) is a RunMat residency extension",
    error_identifier: Some("RunMat:compatibility:FactorialLikeResidencyExtension"),
};
pub const FACTORIAL_LOGICAL_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "factorial-logical-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "factorial with logical input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:FactorialLogicalInputExtension"),
};
const EXTENSIONS: [BuiltinExtensionDescriptor; 2] =
    [FACTORIAL_LIKE_EXTENSION, FACTORIAL_LOGICAL_EXTENSION];

const INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "N",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Signed values must be nonnegative; the output keeps the input integer class.",
}];
pub const FACTORIAL_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "F = factorial(integer_N)",
        inputs: &INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::Saturate,
        backend: BuiltinIntegerBackendRule::GpuRestricted,
        overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
        notes: "All eight host classes saturate at the class maximum; interactive GPU and distributed inputs exclude int64 and uint64 under the compatible contract.",
    }];

const BINDINGS: [BuiltinBindingDeclaration; 1] = crate::REQUIRED_DEFAULT_BINDING;
const EFFECTS: [EffectKind; 2] = [EffectKind::MaySuspend, EffectKind::MayThrow];
const DISTRIBUTED_NUMERIC_CLASSES: [NumericClass; 8] = [
    NumericClass::Double,
    NumericClass::Single,
    NumericClass::Int8,
    NumericClass::UInt8,
    NumericClass::Int16,
    NumericClass::UInt16,
    NumericClass::Int32,
    NumericClass::UInt32,
];
pub const FACTORIAL_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    identity: BuiltinCatalogIdentity { name: "factorial" },
    category: "math/discrete",
    documentation: FACTORIAL_DOCUMENTATION,
    descriptor: &FACTORIAL_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Complete,
        inference_rule: BuiltinInferenceRule::Math(MathInferenceRule::Discrete(
            DiscreteInferenceRule::Factorial,
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
        portability: BuiltinPortability::NativeAndWasm,
        accelerator: BuiltinAcceleratorPolicy::Optional,
        residency: BuiltinResidencyPolicy::Dynamic,
        fusion: BuiltinFusionPolicy::Never,
        distributed: BuiltinDistributedPolicy::MapUnaryConstrained(
            crate::BuiltinDistributedMapContract::numeric_classes(&DISTRIBUTED_NUMERIC_CLASSES),
        ),
    },
    link: BuiltinLinkContract {
        reachability: BuiltinReachability::Always,
        policy: BuiltinLinkPolicy::PortableRuntime,
        execution_stack: ExecutionStackRequirement::Any,
        artifact_dependencies: &[],
    },
    bindings: &BINDINGS,
    extensions: &EXTENSIONS,
    integer_capabilities: &FACTORIAL_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};
