use crate::*;

pub const GET_INTEGER_HANDLE_ALIAS_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "get-integer-handle-alias",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "typed-integer numeric graphics aliases for get are a RunMat extension",
        error_identifier: Some("RunMat:compatibility:GetIntegerHandleAliasExtension"),
    };
pub const GET_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [GET_INTEGER_HANDLE_ALIAS_EXTENSION];

const INTEGER_HANDLE_INPUTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "h",
    classes: &crate::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
    notes: "RunMat resolves exactly representable typed integer aliases only under extension policy; graphics objects are the documented compatibility surface.",
}];
pub const GET_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "value = get(integer_graphics_alias, property?)",
        inputs: &INTEGER_HANDLE_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::ScalarOnly,
        notes: "Only handle resolution is integer-sensitive. Returned property payloads retain their authoritative classes and storage.",
    }];

const OUTPUT_VALUE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "value",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "A property value or the known-property struct.",
}];
const INPUT_HANDLE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "h",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "A live graphics handle.",
}];
const INPUT_HANDLE_PROPERTY: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "h",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "A live graphics handle.",
    },
    BuiltinParamDescriptor {
        name: "property",
        ty: BuiltinParamType::PropertyName,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Case-insensitive property name.",
    },
];
const SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "value = get(h)",
        inputs: &INPUT_HANDLE,
        outputs: &OUTPUT_VALUE,
    },
    BuiltinSignatureDescriptor {
        label: "value = get(h, property)",
        inputs: &INPUT_HANDLE_PROPERTY,
        outputs: &OUTPUT_VALUE,
    },
];
pub const GET_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GET.INVALID_ARGUMENT",
    identifier: Some("RunMat:get:InvalidArgument"),
    when: "The handle or property selector is invalid, or property validation fails.",
    message: "get: invalid argument",
};
pub const GET_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GET.INTERNAL",
    identifier: Some("RunMat:get:Internal"),
    when: "Graphics state or source-data readback fails after admission.",
    message: "get: internal operation failed",
};
const ERRORS: [BuiltinErrorDescriptor; 2] = [GET_ERROR_INVALID_ARGUMENT, GET_ERROR_INTERNAL];
pub const GET_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
