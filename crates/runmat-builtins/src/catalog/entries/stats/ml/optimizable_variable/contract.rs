use crate::*;

const NAME: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "Name",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Nonempty variable name.",
};
const RANGE: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "Range",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Two numeric bounds or nonempty categorical choices.",
};
const OPTIONS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "NameValue",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Type, Transform and Optimize name-value pairs.",
};
const OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "variable",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Host optimizableVariable metadata object.",
}];
const BASE: [BuiltinParamDescriptor; 2] = [NAME, RANGE];
const FULL: [BuiltinParamDescriptor; 3] = [NAME, RANGE, OPTIONS];
const SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "v = optimizableVariable(Name, Range)",
        inputs: &BASE,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "v = optimizableVariable(Name, Range, Name, Value)",
        inputs: &FULL,
        outputs: &OUTPUT,
    },
];
pub const ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.OPTIMIZABLE_VARIABLE.INVALID_ARGUMENT",
    identifier: Some("RunMat:optimizableVariable:InvalidArgument"),
    when: "The name or name-value argument structure is malformed.",
    message: "optimizableVariable: invalid argument",
};
pub const ERROR_INVALID_RANGE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.OPTIMIZABLE_VARIABLE.INVALID_RANGE",
    identifier: Some("RunMat:optimizableVariable:InvalidRange"),
    when: "Range is incompatible with the selected variable type.",
    message: "optimizableVariable: invalid range",
};
pub const ERROR_INVALID_OPTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.OPTIMIZABLE_VARIABLE.INVALID_OPTION",
    identifier: Some("RunMat:optimizableVariable:InvalidOption"),
    when: "An option name or value is unsupported.",
    message: "optimizableVariable: invalid option",
};
pub const ERROR_FLOW: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.OPTIMIZABLE_VARIABLE.FLOW",
    identifier: Some("RunMat:optimizableVariable:Flow"),
    when: "Materializing a resident metadata operand fails.",
    message: "optimizableVariable: flow failure",
};
const ERRORS: [BuiltinErrorDescriptor; 4] = [
    ERROR_INVALID_ARGUMENT,
    ERROR_INVALID_RANGE,
    ERROR_INVALID_OPTION,
    ERROR_FLOW,
];
pub const OPTIMIZABLE_VARIABLE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
