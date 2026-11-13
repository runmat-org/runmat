use crate::*;
const HOLD_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "state",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
    notes: "Documented numeric scalar states accept every built-in integer class; all classes are inspected exactly and only values 0 and 1 are accepted.",
}];
pub const HOLD_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor { form: "hold([ax,] integer_state)", inputs: &HOLD_INTEGER_INPUTS, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::NotApplicable, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::HostOnly, overload: BuiltinIntegerOverloadKind::Multiple, notes: "The scalar state is read without floating conversion and mutates only the selected axes hold policy after validation." }];

const HOLD_INPUTS_NONE: [BuiltinParamDescriptor; 0] = [];

const HOLD_INPUTS_MODE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "mode",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Hold mode ('on'|'off'|'all') or numeric/logical scalar.",
}];

const HOLD_INPUTS_AX_MODE: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "ax",
        ty: BuiltinParamType::AxesHandle,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Target axes.",
    },
    BuiltinParamDescriptor {
        name: "mode",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Hold mode ('on'|'off'|'all') or numeric/logical scalar.",
    },
];

const HOLD_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "hold()",
        inputs: &HOLD_INPUTS_NONE,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "hold(mode)",
        inputs: &HOLD_INPUTS_MODE,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "hold(ax, mode)",
        inputs: &HOLD_INPUTS_AX_MODE,
        outputs: &[],
    },
];

const HOLD_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.HOLD.INVALID_ARGUMENT",
    identifier: Some("RunMat:hold:InvalidArgument"),
    when: "Hold mode argument is unsupported.",
    message: "hold: invalid argument",
};

const HOLD_ERRORS: [BuiltinErrorDescriptor; 2] =
    [HOLD_ERROR_INVALID_ARGUMENT, HOLD_ERROR_TOO_MANY_OUTPUTS];

pub const HOLD_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &HOLD_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &HOLD_ERRORS,
};

pub const HOLD_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.HOLD.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:hold:TooManyOutputs"),
    when: "Any output is explicitly requested.",
    message: "hold: too many output arguments",
};
