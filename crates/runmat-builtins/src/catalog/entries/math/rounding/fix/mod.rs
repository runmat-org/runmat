use crate::{
    BuiltinAcceleratorPolicy, BuiltinAsyncBehavior, BuiltinBindingDeclaration, BuiltinCatalogEntry,
    BuiltinCatalogIdentity, BuiltinCompatibility, BuiltinCompletionPolicy,
    BuiltinContractDeclaration, BuiltinContractMaturity, BuiltinDescriptor,
    BuiltinDistributedPolicy, BuiltinErrorDescriptor, BuiltinFusionPolicy, BuiltinInferenceRule,
    BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor, BuiltinIntegerComputationDomain,
    BuiltinIntegerInputAvailability, BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule,
    BuiltinIntegerOverflowRule, BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule,
    BuiltinLinkContract, BuiltinLinkPolicy, BuiltinOutputMode, BuiltinParamArity,
    BuiltinParamDescriptor, BuiltinParamType, BuiltinPlacementContract, BuiltinPortability,
    BuiltinPurity, BuiltinReachability, BuiltinResidencyPolicy, BuiltinSemanticKind,
    BuiltinSignatureDescriptor, MathInferenceRule, RoundingFunction, ALL_INTEGER_CLASSES,
};
use runmat_types::{EffectKind, ExecutionStackRequirement};

mod documentation;
use documentation::FIX_DOCUMENTATION;

const OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Values rounded toward zero.",
}];
const INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Numeric, logical, character, table, or timetable input.",
}];
const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "Y = fix(X)",
    inputs: &INPUTS,
    outputs: &OUTPUTS,
}];

pub const FIX_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FIX.INVALID_INPUT",
    identifier: Some("RunMat:fix:InvalidInput"),
    when: "Input is not a supported numeric, logical, character, table, or timetable value.",
    message: "fix: invalid input",
};
pub const FIX_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FIX.INVALID_ARGUMENT",
    identifier: Some("RunMat:fix:InvalidArgument"),
    when: "The invocation does not contain exactly one input.",
    message: "fix: invalid argument",
};
pub const FIX_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FIX.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:fix:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "fix: too many output arguments",
};
pub const FIX_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FIX.INTERNAL",
    identifier: Some("RunMat:fix:Internal"),
    when: "Internal conversion, allocation, provider execution, or residency restoration fails.",
    message: "fix: internal error",
};
const ERRORS: [BuiltinErrorDescriptor; 4] = [
    FIX_ERROR_INVALID_INPUT,
    FIX_ERROR_INVALID_ARGUMENT,
    FIX_ERROR_TOO_MANY_OUTPUTS,
    FIX_ERROR_INTERNAL,
];
pub const FIX_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

const INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "X",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Every real integer value is already integral, so its class, shape, bits, and supported residency are preserved.",
}];
pub const FIX_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "Y = fix(X) with real integer X, including integer table or timetable variables",
        inputs: &INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
        notes: "Host integer storage is returned unchanged; resident integer input retains its owning-provider handle.",
    }];

const BINDINGS: [BuiltinBindingDeclaration; 1] = crate::REQUIRED_DEFAULT_BINDING;
const EFFECTS: [EffectKind; 2] = [EffectKind::MaySuspend, EffectKind::MayThrow];
pub const FIX_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    identity: BuiltinCatalogIdentity { name: "fix" },
    category: "math/rounding",
    documentation: FIX_DOCUMENTATION,
    descriptor: &FIX_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Complete,
        inference_rule: BuiltinInferenceRule::Math(MathInferenceRule::Rounding(
            RoundingFunction::Fix,
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
        residency: BuiltinResidencyPolicy::PreserveInputs,
        fusion: BuiltinFusionPolicy::Candidate,
        distributed: BuiltinDistributedPolicy::MapUnary,
    },
    link: BuiltinLinkContract {
        reachability: BuiltinReachability::Always,
        policy: BuiltinLinkPolicy::PortableRuntime,
        execution_stack: ExecutionStackRequirement::Any,
        artifact_dependencies: &[],
    },
    bindings: &BINDINGS,
    extensions: &[],
    integer_capabilities: &FIX_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};
