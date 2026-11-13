use crate::*;

const POLE_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "p",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Poles of the SISO tf or ss model as a column vector.",
}];
const POLE_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "sys",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "SISO tf model or ss state-space model.",
}];
const POLE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "p = pole(sys)",
    inputs: &POLE_INPUTS,
    outputs: &POLE_OUTPUT,
}];
const POLE_ERRORS: [BuiltinErrorDescriptor; 4] = [
    BuiltinErrorDescriptor {
        code: "RM.POLE.INVALID_MODEL",
        identifier: Some("RunMat:pole:InvalidModel"),
        when: "Input system is not a valid SISO tf or ss object.",
        message: "pole: invalid model",
    },
    BuiltinErrorDescriptor {
        code: "RM.POLE.UNSUPPORTED_MODEL",
        identifier: Some("RunMat:pole:UnsupportedModel"),
        when: "Model form is unsupported.",
        message: "pole: unsupported model",
    },
    BuiltinErrorDescriptor {
        code: "RM.POLE.INVALID_ARGUMENT",
        identifier: Some("RunMat:pole:InvalidArgument"),
        when: "Model metadata or arguments are malformed.",
        message: "pole: invalid argument",
    },
    BuiltinErrorDescriptor {
        code: "RM.POLE.INTERNAL",
        identifier: Some("RunMat:pole:Internal"),
        when: "Root calculation or output construction failed.",
        message: "pole: internal error",
    },
];
pub const POLE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &POLE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &POLE_ERRORS,
};
const POLE_INTEGER_MODEL_INDEX_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "J1,...,JN model-array indices",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The compatibility target documents positive-integer model-array subscripts. They are structural selectors and do not enter pole computation.",
    }];
pub const POLE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "P = pole(sys,integer_J1,...,integer_JN)",
        inputs: &POLE_INTEGER_MODEL_INDEX_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The compatibility contract is exact one-based model selection. RunMat currently implements singular tf/ss objects only, so the broader model-array overload rejects before any integer conversion can occur.",
    }];
