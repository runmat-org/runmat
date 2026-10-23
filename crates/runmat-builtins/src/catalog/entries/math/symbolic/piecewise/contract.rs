use crate::*;

pub const PIECEWISE_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "f",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Symbolic piecewise expression.",
}];

pub const PIECEWISE_INPUTS: [BuiltinParamDescriptor; 3] = [BuiltinParamDescriptor {
    name: "condition",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Host scalar symbolic or real numeric/logical condition expression; conditions are preserved, not evaluated.",
},BuiltinParamDescriptor {
    name:"value",ty:BuiltinParamType::Any,arity:BuiltinParamArity::Required,default:None,description:"First host scalar symbolic or real numeric/logical branch value.",
},BuiltinParamDescriptor {
    name:"arguments",ty:BuiltinParamType::Any,arity:BuiltinParamArity::Variadic,default:None,description:"Further scalar condition/value expressions and an optional final default; arguments are stored in order without branch evaluation.",
}];

pub const PIECEWISE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "f = piecewise(cond1, value1, ...)",
    inputs: &PIECEWISE_INPUTS,
    outputs: &PIECEWISE_OUTPUT,
}];

pub const PIECEWISE_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PIECEWISE.INVALID_INPUT",
    identifier: Some("RunMat:piecewise:InvalidInput"),
    when: "Arguments cannot be represented by the scalar symbolic core.",
    message: "piecewise: expected symbolic or scalar numeric arguments",
};

pub const PIECEWISE_ERRORS: [BuiltinErrorDescriptor; 2] = [
    PIECEWISE_ERROR_INVALID_INPUT,
    PIECEWISE_ERROR_TOO_MANY_OUTPUTS,
];

pub const PIECEWISE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &PIECEWISE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &PIECEWISE_ERRORS,
};

pub const PIECEWISE_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PIECEWISE.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:piecewise:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "piecewise: too many output arguments",
};

pub const PIECEWISE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] = [BuiltinIntegerCapabilityDescriptor { form: "piecewise(..., integer_value, ...)", inputs: &[BuiltinIntegerInputCapability { name: "scalar condition or value", classes: &crate::ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::Documented, scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable, notes: "Native integer scalars enter exact BigInt-backed symbolic rationals, including all int64/uint64 values." }], computation_domain: BuiltinIntegerComputationDomain::ExactInteger, output_class: BuiltinIntegerOutputClassRule::FunctionSpecific, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::HostOnly, overload: BuiltinIntegerOverloadKind::ScalarOnly, notes: "This constructs a scalar symbolic expression; it does not evaluate conditions or implement symbolic arrays." }];
