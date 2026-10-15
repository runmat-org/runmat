use crate::*;

const TIC_OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "timerVal",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Opaque session-owned uint64 stopwatch token used by toc.",
}];
const TIC_INPUTS: [BuiltinParamDescriptor; 0] = [];
const TIC_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "timerVal = tic()",
    inputs: &TIC_INPUTS,
    outputs: &TIC_OUTPUTS,
}];
pub const TIC_ERROR_CONTEXT_UNAVAILABLE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TIC.CONTEXT_UNAVAILABLE",
    identifier: Some("RunMat:tic:ContextUnavailable"),
    when: "No invocation timing context is installed.",
    message: "tic: runtime context is unavailable",
};
pub const TIC_ERROR_TOKEN_EXHAUSTED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TIC.TOKEN_EXHAUSTED",
    identifier: Some("RunMat:tic:TokenExhausted"),
    when: "The process-local stopwatch token identity space is exhausted.",
    message: "tic: stopwatch token identity space exhausted",
};
const TIC_ERRORS: [BuiltinErrorDescriptor; 2] =
    [TIC_ERROR_CONTEXT_UNAVAILABLE, TIC_ERROR_TOKEN_EXHAUSTED];
pub const TIC_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &TIC_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &TIC_ERRORS,
};

const TOC_OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "elapsed",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Elapsed wall-clock seconds as a scalar double.",
}];
const TOC_INPUTS_NONE: [BuiltinParamDescriptor; 0] = [];
const TOC_INPUTS_HANDLE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "timerVal",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Opaque uint64 token returned by tic in this session.",
}];
const TOC_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "elapsed = toc()",
        inputs: &TOC_INPUTS_NONE,
        outputs: &TOC_OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "elapsed = toc(timerVal)",
        inputs: &TOC_INPUTS_HANDLE,
        outputs: &TOC_OUTPUTS,
    },
];
pub const TOC_ERROR_NO_MATCHING_TIC: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TOC.NO_MATCHING_TIC",
    identifier: Some("RunMat:toc:NoMatchingTic"),
    when: "toc() is called without a prior tic() in the current session.",
    message: "toc: no matching tic",
};
pub const TOC_ERROR_INVALID_HANDLE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TOC.INVALID_HANDLE",
    identifier: Some("RunMat:toc:InvalidTimerHandle"),
    when: "The token is malformed, stale, or belongs to another session.",
    message: "toc: invalid timer handle",
};
pub const TOC_ERROR_TOO_MANY_INPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TOC.TOO_MANY_INPUTS",
    identifier: Some("RunMat:toc:TooManyInputs"),
    when: "More than one input argument is supplied.",
    message: "toc: too many input arguments",
};
pub const TOC_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TOC.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:toc:TooManyOutputs"),
    when: "More than one output is requested from toc.",
    message: "toc: too many output arguments",
};
const TOC_ERRORS: [BuiltinErrorDescriptor; 4] = [
    TOC_ERROR_NO_MATCHING_TIC,
    TOC_ERROR_INVALID_HANDLE,
    TOC_ERROR_TOO_MANY_INPUTS,
    TOC_ERROR_TOO_MANY_OUTPUTS,
];
pub const TOC_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &TOC_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &TOC_ERRORS,
};

const UINT64_TIMER_CLASS: [BuiltinIntegerClass; 1] = [BuiltinIntegerClass::Uint64];
const TOC_TIMER_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "timerVal",
    classes: &UINT64_TIMER_CLASS,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::Rejected,
    notes:
        "tic returns an opaque uint64 token; toc validates its exact value in the active session.",
}];
pub const TOC_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "elapsed = toc(uint64_timerVal)",
        inputs: &TOC_TIMER_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The uint64 is an opaque session token, not a timestamp or elapsed-time count.",
    }];
