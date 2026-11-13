use crate::*;
const MAKE_ANON_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "handle_text",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Anonymous function text handle.",
}];
const MAKE_ANON_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "params",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Comma-separated parameter list.",
    },
    BuiltinParamDescriptor {
        name: "body",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Anonymous body expression text.",
    },
];
const MAKE_ANON_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "handle_text = make_anon(params, body)",
    inputs: &MAKE_ANON_INPUTS,
    outputs: &MAKE_ANON_OUTPUT,
}];
pub const MAKE_ANON_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &MAKE_ANON_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::HiddenInternal,
    errors: &[MAKE_ANON_ERROR_ARITY, MAKE_ANON_ERROR_OUTPUTS],
};
pub const MAKE_ANON_ERROR_ARITY: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MAKE_ANON.ARITY",
    identifier: Some("RunMat:MAKE_ANON:Arity"),
    when: "The number of inputs does not match the fixed signature.",
    message: "make_anon: expected 2 input arguments",
};
pub const MAKE_ANON_ERROR_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MAKE_ANON.OUTPUTS",
    identifier: Some("RunMat:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "make_anon: at most one output is supported",
};
pub const MAKE_ANON_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] = [
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
