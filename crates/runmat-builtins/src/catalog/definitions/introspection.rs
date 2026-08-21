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

const FEVAL_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "varargout",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Function return value(s).",
}];
const FEVAL_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "f",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Function handle, handle text, closure, or object receiver.",
    },
    BuiltinParamDescriptor {
        name: "varargin",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Function call arguments.",
    },
];
const FEVAL_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "[varargout] = feval(f, varargin)",
    inputs: &FEVAL_INPUTS,
    outputs: &FEVAL_OUTPUT,
}];

pub const FEVAL_ERROR_HANDLE_NAME_INVALID: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FEVAL.HANDLE_NAME_INVALID",
    identifier: Some("RunMat:FevalHandleNameInvalid"),
    when: "A function or method handle name is empty.",
    message: "feval: function handle name must not be empty",
};
pub const FEVAL_ERROR_HANDLE_SHAPE_INVALID: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FEVAL.HANDLE_SHAPE_INVALID",
    identifier: Some("RunMat:FevalHandleShapeInvalid"),
    when: "Text handle input has invalid char/string array shape.",
    message: "feval: function handle text input must be scalar row text",
};
pub const FEVAL_ERROR_SEMANTIC_UNAVAILABLE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FEVAL.SEMANTIC_UNAVAILABLE",
    identifier: Some("RunMat:SemanticFunctionUnavailable"),
    when: "Semantic function identity cannot be invoked in current runtime state.",
    message: "feval: semantic function handle is unavailable",
};
pub const FEVAL_ERROR_FUNCTION_VALUE_UNSUPPORTED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FEVAL.FUNCTION_VALUE_UNSUPPORTED",
    identifier: Some("RunMat:FevalFunctionValueUnsupported"),
    when: "The first argument is not a supported callable value.",
    message: "feval: unsupported function value",
};
pub const FEVAL_ERRORS: [BuiltinErrorDescriptor; 4] = [
    FEVAL_ERROR_HANDLE_NAME_INVALID,
    FEVAL_ERROR_HANDLE_SHAPE_INVALID,
    FEVAL_ERROR_SEMANTIC_UNAVAILABLE,
    FEVAL_ERROR_FUNCTION_VALUE_UNSUPPORTED,
];
pub const FEVAL_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &FEVAL_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &FEVAL_ERRORS,
};

pub const FEVAL_AT_PREFIXED_TEXT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "feval-at-prefixed-text-target",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "feval with an @-prefixed text target is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:FevalAtPrefixedTextTargetExtension"),
    };
pub const FEVAL_OBJECT_RECEIVER_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "feval-object-receiver",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "feval with an object receiver is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:FevalObjectReceiverExtension"),
    };
pub const FEVAL_EXTENSIONS: [BuiltinExtensionDescriptor; 2] = [
    FEVAL_AT_PREFIXED_TEXT_EXTENSION,
    FEVAL_OBJECT_RECEIVER_EXTENSION,
];

const FEVAL_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "varargin",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "All eight integer classes are forwarded unchanged when they are valid inputs for the function selected by name or handle; scalar-double and mixed-class rules belong to that target.",
    }];
pub const FEVAL_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "[varargout] = feval(fun,integer_varargin) with integer target arguments or results",
        inputs: &FEVAL_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::FunctionSpecific,
        backend: BuiltinIntegerBackendRule::FunctionSpecific,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "feval performs no numeric conversion or arithmetic of its own; input admission, output count and class, overflow, shape, and host or provider behavior are inherited from the selected function.",
    }];

const FEVAL_BINDINGS: [BuiltinBindingDeclaration; 1] = [BuiltinBindingDeclaration {
    identity: BuiltinBindingIdentity {
        builtin: BuiltinCatalogIdentity { name: "feval" },
        variant: "default",
    },
    availability: BuiltinBindingAvailability::Required,
}];
const FEVAL_EFFECTS: [EffectKind; 4] = [
    EffectKind::HostCallback,
    EffectKind::MaySuspend,
    EffectKind::MayThrow,
    EffectKind::Unknown,
];
pub const FEVAL_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    identity: BuiltinCatalogIdentity { name: "feval" },
    category: "introspection",
    documentation: BuiltinDocumentation {
        summary: "Invoke a function selected by name, handle, closure, or object receiver.",
        keywords: &[
            "callback",
            "dispatch",
            "feval",
            "function handle",
            "varargout",
        ],
        related: &[],
        introduced: None,
        status: None,
        examples: &[],
    },
    descriptor: &FEVAL_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::DynamicByDesign,
        inference_rule: BuiltinInferenceRuleId("introspection.feval"),
        compatibility: BuiltinCompatibility::Matlab,
        async_behavior: BuiltinAsyncBehavior::MaySuspend,
        purity: BuiltinPurity::Impure,
        semantic_kind: BuiltinSemanticKind::General,
        workspace_effect: None,
        environment_effect: None,
        effects: &FEVAL_EFFECTS,
        capabilities: &[],
    },
    placement: BuiltinPlacementContract {
        portability: BuiltinPortability::NativeAndWasm,
        accelerator: BuiltinAcceleratorPolicy::Optional,
        residency: BuiltinResidencyPolicy::Dynamic,
        fusion: BuiltinFusionPolicy::Boundary,
    },
    link: BuiltinLinkContract {
        reachability: BuiltinReachability::Dynamic,
        policy: BuiltinLinkPolicy::PortableRuntime,
        artifact_dependencies: &[],
    },
    bindings: &FEVAL_BINDINGS,
    extensions: &FEVAL_EXTENSIONS,
    integer_capabilities: &FEVAL_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};
