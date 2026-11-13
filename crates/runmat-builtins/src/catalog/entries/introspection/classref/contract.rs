use crate::*;
const CLASSREF_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "ref",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Class reference value.",
}];
const CLASSREF_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "class_name",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Class name.",
}];
const CLASSREF_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "ref = classref(class_name)",
    inputs: &CLASSREF_INPUTS,
    outputs: &CLASSREF_OUTPUT,
}];
pub const CLASSREF_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &CLASSREF_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::HiddenInternal,
    errors: &[CLASSREF_ERROR_ARITY, CLASSREF_ERROR_OUTPUTS],
};
pub const CLASSREF_ERROR_ARITY: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CLASSREF.ARITY",
    identifier: Some("RunMat:CLASSREF:Arity"),
    when: "The number of inputs does not match the fixed signature.",
    message: "classref: expected 1 input arguments",
};
pub const CLASSREF_ERROR_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CLASSREF.OUTPUTS",
    identifier: Some("RunMat:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "classref: at most one output is supported",
};
pub const CLASSREF_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] = [
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
