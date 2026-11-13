use super::super::debugging_contract::*;
use crate::*;
pub const DBSTACK_OUTPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "ST",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Call-stack entries in a structure array.",
    },
    BuiltinParamDescriptor {
        name: "I",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Current workspace index within the returned stack.",
    },
];

pub const DBSTACK_STACK_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "ST",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Call-stack entries in a structure array.",
}];

pub const DBSTACK_N_INPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "n",
    ty: BuiltinParamType::IntegerScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Number of leading stack entries to omit.",
}];

pub const DBSTACK_OPTION_INPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "option",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "`-completenames` requests full source paths.",
}];

pub const DBSTACK_OPTION_N_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "option",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "`-completenames` requests full source paths.",
    },
    BuiltinParamDescriptor {
        name: "n",
        ty: BuiltinParamType::IntegerScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Number of leading stack entries to omit.",
    },
];

pub const DBSTACK_N_OPTION_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "n",
        ty: BuiltinParamType::IntegerScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Number of leading stack entries to omit.",
    },
    BuiltinParamDescriptor {
        name: "option",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "`-completenames` requests full source paths.",
    },
];

pub const DBSTACK_SIGNATURES: [BuiltinSignatureDescriptor; 11] = [
    BuiltinSignatureDescriptor {
        label: "dbstack",
        inputs: &[],
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "ST = dbstack",
        inputs: &[],
        outputs: &DBSTACK_STACK_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "ST = dbstack(n)",
        inputs: &DBSTACK_N_INPUT,
        outputs: &DBSTACK_STACK_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "ST = dbstack('-completenames')",
        inputs: &DBSTACK_OPTION_INPUT,
        outputs: &DBSTACK_STACK_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "ST = dbstack('-completenames', n)",
        inputs: &DBSTACK_OPTION_N_INPUTS,
        outputs: &DBSTACK_STACK_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "ST = dbstack(n, '-completenames')",
        inputs: &DBSTACK_N_OPTION_INPUTS,
        outputs: &DBSTACK_STACK_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "[ST,I] = dbstack",
        inputs: &[],
        outputs: &DBSTACK_OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "[ST,I] = dbstack(n)",
        inputs: &DBSTACK_N_INPUT,
        outputs: &DBSTACK_OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "[ST,I] = dbstack('-completenames')",
        inputs: &DBSTACK_OPTION_INPUT,
        outputs: &DBSTACK_OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "[ST,I] = dbstack('-completenames', n)",
        inputs: &DBSTACK_OPTION_N_INPUTS,
        outputs: &DBSTACK_OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "[ST,I] = dbstack(n, '-completenames')",
        inputs: &DBSTACK_N_OPTION_INPUTS,
        outputs: &DBSTACK_OUTPUTS,
    },
];

pub const DEBUG_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DEBUG.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:dbstack:TooManyOutputs"),
    when: "More than two outputs are requested from dbstack.",
    message: "dbstack: too many output arguments",
};

pub const DBSTACK_ERRORS: [BuiltinErrorDescriptor; 6] = [
    DEBUG_ERROR_TOO_MANY_INPUTS,
    DEBUG_ERROR_TOO_MANY_OUTPUTS,
    DEBUG_ERROR_INVALID_INPUT,
    DEBUG_ERROR_NO_CURRENT_FILE,
    DEBUG_ERROR_FILE_NOT_FOUND,
    DEBUG_ERROR_FILE_READ,
];

pub const DBSTACK_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &DBSTACK_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &DBSTACK_ERRORS,
};

pub const DBSTACK_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "n",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "n is an exact nonnegative scalar count of leading stack entries to omit; values outside the host indexing domain reject.",
    }];

pub const DBSTACK_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "ST or [ST, I] = dbstack(n) or dbstack('-completenames', n)",
        inputs: &DBSTACK_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The skip count is parsed from exact scalar storage. Stack line and workspace-index fields follow the debugger's host metadata representation rather than n's input class.",
    }];
