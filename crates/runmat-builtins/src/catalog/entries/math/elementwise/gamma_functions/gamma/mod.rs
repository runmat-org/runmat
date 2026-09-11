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
    BuiltinSignatureDescriptor, GammaFunctionInferenceRule, MathInferenceRule,
};
use runmat_types::{EffectKind, ExecutionStackRequirement};

mod documentation;
use documentation::GAMMA_DOCUMENTATION;

const OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Real gamma-function values with the input class and shape.",
}];
const INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Dense real single- or double-precision input.",
}];
const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "Y = gamma(X)",
    inputs: &INPUTS,
    outputs: &OUTPUTS,
}];

pub const GAMMA_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GAMMA.INVALID_ARGUMENT",
    identifier: Some("RunMat:gamma:InvalidArgument"),
    when: "The invocation does not have exactly one input.",
    message: "gamma: invalid argument",
};
pub const GAMMA_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GAMMA.INVALID_INPUT",
    identifier: Some("RunMat:gamma:InvalidInput"),
    when: "Input is not dense real single or double numeric data.",
    message: "gamma: invalid input",
};
pub const GAMMA_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GAMMA.INTERNAL",
    identifier: Some("RunMat:gamma:Internal"),
    when: "Internal allocation, provider execution, gather, or residency restoration fails.",
    message: "gamma: internal error",
};
pub const GAMMA_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GAMMA.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:gamma:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "gamma: too many output arguments",
};
const ERRORS: [BuiltinErrorDescriptor; 4] = [
    GAMMA_ERROR_INVALID_ARGUMENT,
    GAMMA_ERROR_INVALID_INPUT,
    GAMMA_ERROR_INTERNAL,
    GAMMA_ERROR_TOO_MANY_OUTPUTS,
];
pub const GAMMA_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

const INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "X",
    classes: &[],
    availability: BuiltinIntegerInputAvailability::Rejected,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Fixed-width integer input is rejected before host evaluation or provider dispatch.",
}];
pub const GAMMA_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "Y = gamma(X) with fixed-width integer X",
        inputs: &INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
        notes: "All eight fixed-width integer classes are unsupported in both compatibility modes; numeric literals remain double and are supported.",
    }];

const BINDINGS: [BuiltinBindingDeclaration; 1] = crate::REQUIRED_DEFAULT_BINDING;
const EFFECTS: [EffectKind; 2] = [EffectKind::MaySuspend, EffectKind::MayThrow];
pub const GAMMA_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    provenance: crate::BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "gamma" },
    category: "math/elementwise",
    documentation: GAMMA_DOCUMENTATION,
    descriptor: &GAMMA_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Complete,
        inference_rule: BuiltinInferenceRule::Math(MathInferenceRule::GammaFunction(
            GammaFunctionInferenceRule::Gamma,
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
        fusion: BuiltinFusionPolicy::Never,
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
    integer_capabilities: &GAMMA_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};
