use crate::*;
const NEW_HANDLE_OBJECT_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "handle",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "New handle-object instance.",
}];
const NEW_HANDLE_OBJECT_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "class_name",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Class name for created object.",
}];
const NEW_HANDLE_OBJECT_SIGNATURES: [BuiltinSignatureDescriptor; 1] =
    [BuiltinSignatureDescriptor {
        label: "handle = new_handle_object(class_name)",
        inputs: &NEW_HANDLE_OBJECT_INPUTS,
        outputs: &NEW_HANDLE_OBJECT_OUTPUT,
    }];
pub const NEW_HANDLE_OBJECT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &NEW_HANDLE_OBJECT_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::HiddenInternal,
    errors: &[
        NEW_HANDLE_OBJECT_ERROR_ARITY,
        NEW_HANDLE_OBJECT_ERROR_OUTPUTS,
    ],
};
pub const NEW_HANDLE_OBJECT_ERROR_ARITY: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.NEW_HANDLE_OBJECT.ARITY",
    identifier: Some("RunMat:NEW_HANDLE_OBJECT:Arity"),
    when: "The number of inputs does not match the fixed signature.",
    message: "new_handle_object: expected 1 input arguments",
};
pub const NEW_HANDLE_OBJECT_ERROR_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.NEW_HANDLE_OBJECT.OUTPUTS",
    identifier: Some("RunMat:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "new_handle_object: at most one output is supported",
};
pub const NEW_HANDLE_OBJECT_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "scalar integer textual controls",
        inputs: &[
            BuiltinIntegerInputCapability {
                name: "textual controls",
                classes: &ALL_INTEGER_CLASSES,
                availability: BuiltinIntegerInputAvailability::RunMatOnly,
                scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
                notes: "Existing scalar Value::Int controls are formatted using exact decimal_string, without f64 conversion. Integer arrays and device storage are rejected."
            }
        ],
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::ScalarOnly,
        notes: "Preserves the legacy scalar-to-text conversion; no integer numerical result or gather fallback."
    }
];
