use crate::*;
const NEW_OBJECT_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "obj",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Created class object value.",
}];
const NEW_OBJECT_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "class_name",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Class name for the object to create.",
}];
const NEW_OBJECT_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "obj = new_object(class_name)",
    inputs: &NEW_OBJECT_INPUTS,
    outputs: &NEW_OBJECT_OUTPUT,
}];
pub const NEW_OBJECT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &NEW_OBJECT_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[NEW_OBJECT_ERROR_ARITY, NEW_OBJECT_ERROR_OUTPUTS],
};
pub const NEW_OBJECT_ERROR_ARITY: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.NEW_OBJECT.ARITY",
    identifier: Some("RunMat:NEW_OBJECT:Arity"),
    when: "The number of inputs does not match the fixed signature.",
    message: "new_object: expected 1 input arguments",
};
pub const NEW_OBJECT_ERROR_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.NEW_OBJECT.OUTPUTS",
    identifier: Some("RunMat:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "new_object: at most one output is supported",
};
pub const NEW_OBJECT_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] = [
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
