use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinExtensionDescriptor, BuiltinExtensionMode,
    BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor, BuiltinIntegerComputationDomain,
    BuiltinIntegerInputAvailability, BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule,
    BuiltinIntegerOverflowRule, BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule,
    BuiltinOutputMode, BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType,
    BuiltinSignatureDescriptor, Value,
};

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

pub const FEVAL_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &FEVAL_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &crate::FEVAL_ERRORS,
};

pub(crate) const FEVAL_AT_PREFIXED_TEXT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "feval-at-prefixed-text-target",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "feval with an @-prefixed text target is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:FevalAtPrefixedTextTargetExtension"),
    };

pub(crate) const FEVAL_OBJECT_RECEIVER_EXTENSION: BuiltinExtensionDescriptor =
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
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
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

#[runmat_macros::runtime_builtin(
    name = "feval",
    descriptor(self::FEVAL_DESCRIPTOR),
    extensions(self::FEVAL_EXTENSIONS),
    integer_capabilities(self::FEVAL_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::introspection::feval"
)]
pub async fn feval_builtin_registered(f: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    crate::feval_builtin(f, rest).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn descriptor_declares_function_specific_integer_forwarding() {
        assert_eq!(FEVAL_INTEGER_CAPABILITIES.len(), 1);
        let capability = &FEVAL_INTEGER_CAPABILITIES[0];
        assert_eq!(capability.inputs[0].classes.len(), 8);
        assert_eq!(
            capability.computation_domain,
            BuiltinIntegerComputationDomain::FunctionSpecific
        );
        assert_eq!(
            capability.backend,
            BuiltinIntegerBackendRule::FunctionSpecific
        );
        assert_eq!(FEVAL_EXTENSIONS.len(), 2);
    }
}
