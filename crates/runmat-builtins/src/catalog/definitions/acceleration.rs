use crate::{
    BuiltinAcceleratorPolicy, BuiltinAsyncBehavior, BuiltinBindingAvailability,
    BuiltinBindingDeclaration, BuiltinBindingIdentity, BuiltinCatalogEntry, BuiltinCatalogIdentity,
    BuiltinCompatibility, BuiltinCompletionPolicy, BuiltinContractDeclaration,
    BuiltinContractMaturity, BuiltinDescriptor, BuiltinDocumentation, BuiltinErrorDescriptor,
    BuiltinExtensionDescriptor, BuiltinExtensionMode, BuiltinFusionPolicy, BuiltinInferenceRuleId,
    BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor, BuiltinIntegerComputationDomain,
    BuiltinIntegerInputAvailability, BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule,
    BuiltinIntegerOverflowRule, BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule,
    BuiltinLinkContract, BuiltinLinkPolicy, BuiltinOutputMode, BuiltinParamArity,
    BuiltinParamDescriptor, BuiltinParamType, BuiltinPlacementContract, BuiltinPortability,
    BuiltinPurity, BuiltinReachability, BuiltinResidencyPolicy, BuiltinSemanticKind,
    BuiltinSignatureDescriptor, ALL_INTEGER_CLASSES,
};
use runmat_types::EffectKind;

pub const GATHER_CONTAINER_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "gather-recursive-container",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "recursively gathering gpuArray values nested in containers is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:GatherRecursiveContainerExtension"),
};
pub const GATHER_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [GATHER_CONTAINER_EXTENSION];
const GATHER_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "A",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes:
        "Host integer inputs pass through; resident integer inputs download exact class and shape.",
}];
pub const GATHER_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] = [BuiltinIntegerCapabilityDescriptor {
    form: "[X1, X2, ...] = gather(integer_A1, integer_A2, ...)", inputs: &GATHER_INTEGER_INPUTS,
    computation_domain: BuiltinIntegerComputationDomain::Structural,
    output_class: BuiltinIntegerOutputClassRule::PreserveInput,
    overflow: BuiltinIntegerOverflowRule::NotApplicable,
    backend: BuiltinIntegerBackendRule::GatherFallback,
    overload: BuiltinIntegerOverloadKind::Multiple,
    notes: "Each host result preserves its input's exact integer class and shape without invalidating the source handle.",
}];
const OUTPUT_SINGLE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Host-resident value gathered from input.",
}];
const OUTPUT_VARIADIC: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Host-resident outputs matching each input.",
}];
const INPUT_SINGLE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input value to gather from GPU to host.",
}];
const INPUT_VARIADIC: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "X1",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "First input value.",
    },
    BuiltinParamDescriptor {
        name: "Xn",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Additional input values.",
    },
];
const SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "X = gather(X)",
        inputs: &INPUT_SINGLE,
        outputs: &OUTPUT_SINGLE,
    },
    BuiltinSignatureDescriptor {
        label: "[X1, X2, ...] = gather(X1, X2, ...)",
        inputs: &INPUT_VARIADIC,
        outputs: &OUTPUT_VARIADIC,
    },
];
pub const GATHER_ERROR_NOT_ENOUGH_INPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GATHER.NOT_ENOUGH_INPUTS",
    identifier: Some("RunMat:gather:NotEnoughInputs"),
    when: "No input arguments were provided.",
    message: "gather: not enough input arguments",
};
pub const GATHER_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GATHER.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:gather:TooManyOutputs"),
    when: "Requested outputs exceed one for single-input gather.",
    message: "gather: too many output arguments",
};
pub const GATHER_ERROR_OUTPUT_COUNT_MISMATCH: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GATHER.OUTPUT_COUNT_MISMATCH",
    identifier: Some("RunMat:gather:OutputCountMismatch"),
    when: "Requested output count does not match number of inputs.",
    message: "gather: number of outputs must match number of inputs",
};
pub const GATHER_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GATHER.INTERNAL",
    identifier: Some("RunMat:gather:InternalError"),
    when: "Internal output construction failed.",
    message: "gather: internal error",
};
const ERRORS: [BuiltinErrorDescriptor; 4] = [
    GATHER_ERROR_NOT_ENOUGH_INPUTS,
    GATHER_ERROR_TOO_MANY_OUTPUTS,
    GATHER_ERROR_OUTPUT_COUNT_MISMATCH,
    GATHER_ERROR_INTERNAL,
];
pub const GATHER_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
const BINDINGS: [BuiltinBindingDeclaration; 1] = [BuiltinBindingDeclaration {
    identity: BuiltinBindingIdentity {
        builtin: BuiltinCatalogIdentity { name: "gather" },
        variant: "default",
    },
    availability: BuiltinBindingAvailability::Required,
}];
const EFFECTS: [EffectKind; 1] = [EffectKind::MayThrow];
pub const GATHER_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    identity: BuiltinCatalogIdentity { name: "gather" },
    category: "acceleration/gpu",
    documentation: BuiltinDocumentation {
        summary: "Gather gpuArray data back to host memory.",
        keywords: &["accelerate", "download", "gather", "gpuArray"],
        related: &[],
        introduced: None,
        status: None,
        examples: &[],
    },
    descriptor: &GATHER_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Complete,
        inference_rule: BuiltinInferenceRuleId("acceleration.gather"),
        compatibility: BuiltinCompatibility::Matlab,
        async_behavior: BuiltinAsyncBehavior::NeverSuspends,
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
        residency: BuiltinResidencyPolicy::GatherToHost,
        fusion: BuiltinFusionPolicy::Boundary,
    },
    link: BuiltinLinkContract {
        reachability: BuiltinReachability::Always,
        policy: BuiltinLinkPolicy::PortableRuntime,
        artifact_dependencies: &[],
    },
    bindings: &BINDINGS,
    extensions: &GATHER_EXTENSIONS,
    integer_capabilities: &GATHER_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};
