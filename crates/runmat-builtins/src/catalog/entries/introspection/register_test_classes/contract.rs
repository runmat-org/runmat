use crate::*;
const REGISTER_TEST_CLASSES_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "status",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Registration status value.",
}];
const REGISTER_TEST_CLASSES_INPUTS: [BuiltinParamDescriptor; 0] = [];
const REGISTER_TEST_CLASSES_SIGNATURES: [BuiltinSignatureDescriptor; 1] =
    [BuiltinSignatureDescriptor {
        label: "status = __register_test_classes()",
        inputs: &REGISTER_TEST_CLASSES_INPUTS,
        outputs: &REGISTER_TEST_CLASSES_OUTPUT,
    }];
pub const REGISTER_TEST_CLASSES_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &REGISTER_TEST_CLASSES_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::HiddenInternal,
    errors: &[
        REGISTER_TEST_CLASSES_ERROR_ARITY,
        REGISTER_TEST_CLASSES_ERROR_OUTPUTS,
    ],
};
pub const REGISTER_TEST_CLASSES_ERROR_ARITY: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.REGISTER_TEST_CLASSES.ARITY",
    identifier: Some("RunMat:REGISTER_TEST_CLASSES:Arity"),
    when: "The number of inputs does not match the fixed signature.",
    message: "__register_test_classes: expected 0 input arguments",
};
pub const REGISTER_TEST_CLASSES_ERROR_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.REGISTER_TEST_CLASSES.OUTPUTS",
    identifier: Some("RunMat:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "__register_test_classes: at most one output is supported",
};
pub const REGISTER_TEST_CLASSES_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "The fixture seeder accepts no operands; its scalar double status is not an integer arithmetic API."
};
