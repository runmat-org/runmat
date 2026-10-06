use crate::*;

const NO_OUTPUTS: [BuiltinParamDescriptor; 0] = [];
const MESSAGE_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "msg",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Legacy message text or structure.",
}];
const MIN_PARAM: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "minArgs",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Minimum allowed argument count.",
};
const MAX_PARAM: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "maxArgs",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Maximum allowed argument count.",
};
const NUM_PARAM: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "numArgs",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Legacy explicit output count to validate.",
};
const ARITY_CHECK_INPUTS: [BuiltinParamDescriptor; 2] = [MIN_PARAM, MAX_PARAM];
const NARGOUTCHK_LEGACY_INPUTS: [BuiltinParamDescriptor; 3] = [MIN_PARAM, MAX_PARAM, NUM_PARAM];
const NARGOUTCHK_LEGACY_STRUCT_INPUTS: [BuiltinParamDescriptor; 4] = [
    MIN_PARAM,
    MAX_PARAM,
    NUM_PARAM,
    BuiltinParamDescriptor {
        name: "outputType",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("struct"),
        description: "Literal struct output selector.",
    },
];
const NARGINCHK_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "narginchk(minArgs, maxArgs)",
    inputs: &ARITY_CHECK_INPUTS,
    outputs: &NO_OUTPUTS,
}];
const NARGOUTCHK_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "nargoutchk(minArgs, maxArgs)",
        inputs: &ARITY_CHECK_INPUTS,
        outputs: &NO_OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "msgText = nargoutchk(minArgs, maxArgs, numArgs)",
        inputs: &NARGOUTCHK_LEGACY_INPUTS,
        outputs: &MESSAGE_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "msgStruct = nargoutchk(minArgs, maxArgs, numArgs, 'struct')",
        inputs: &NARGOUTCHK_LEGACY_STRUCT_INPUTS,
        outputs: &MESSAGE_OUTPUT,
    },
];

const MIN_INTEGER: BuiltinIntegerInputCapability = BuiltinIntegerInputCapability { name: "minArgs", classes: &ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::Documented, scalar_double: BuiltinIntegerScalarDoubleRule::Allowed, notes: "All eight integer classes are explicitly documented and parsed exactly as nonnegative scalar bounds." };
const MAX_INTEGER: BuiltinIntegerInputCapability = BuiltinIntegerInputCapability { name: "maxArgs", classes: &ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::Documented, scalar_double: BuiltinIntegerScalarDoubleRule::Allowed, notes: "All eight integer classes are explicitly documented; positive floating infinity remains the documented unbounded maximum." };
const NARGINCHK_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 2] = [MIN_INTEGER, MAX_INTEGER];
const NARGOUTCHK_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 3] = [
    MIN_INTEGER, MAX_INTEGER,
    BuiltinIntegerInputCapability { name: "numArgs", classes: &ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::Documented, scalar_double: BuiltinIntegerScalarDoubleRule::Allowed, notes: "The legacy explicit output count accepts all eight documented integer classes and is evaluated without floating conversion." },
];
pub const NARGINCHK_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor { form: "narginchk(integer_minArgs, integer_maxArgs)", inputs: &NARGINCHK_INTEGER_INPUTS, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::NotApplicable, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::HostOnly, overload: BuiltinIntegerOverloadKind::ScalarOnly, notes: "Bounds are compared exactly with the current host call-context count; interactive resident bounds are not documented and reject without provider access." }];
pub const NARGOUTCHK_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor { form: "nargoutchk(integer_minArgs, integer_maxArgs)", inputs: &NARGINCHK_INTEGER_INPUTS, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::NotApplicable, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::HostOnly, overload: BuiltinIntegerOverloadKind::ScalarOnly, notes: "Bounds are compared exactly with the current host call-context output count; interactive resident bounds reject without provider access." },
    BuiltinIntegerCapabilityDescriptor { form: "msg = nargoutchk(integer_minArgs, integer_maxArgs, integer_numArgs, 'struct'?)", inputs: &NARGOUTCHK_INTEGER_INPUTS, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::FunctionSpecific, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::HostOnly, overload: BuiltinIntegerOverloadKind::ScalarOnly, notes: "The discouraged legacy form compares three authoritative integer scalars and returns message text, a message structure, or the documented empty result without changing integer payloads." },
];

pub const NARGINCHK_ERROR_NOT_ENOUGH_INPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.NARGINCHK.NOT_ENOUGH_INPUTS",
    identifier: Some("RunMat:NotEnoughInputs"),
    when: "The current function was called with fewer inputs than the lower bound.",
    message: "narginchk: not enough input arguments",
};
pub const NARGINCHK_ERROR_TOO_MANY_INPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.NARGINCHK.TOO_MANY_INPUTS",
    identifier: Some("RunMat:TooManyInputs"),
    when: "The current function was called with more inputs than the upper bound.",
    message: "narginchk: too many input arguments",
};
pub const NARGINCHK_ERROR_ARGUMENT_INVALID: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.NARGINCHK.ARGUMENT_INVALID",
    identifier: Some("RunMat:NarginchkArgumentInvalid"),
    when: "A bound argument is not a nonnegative integer scalar or valid Inf upper bound.",
    message: "narginchk: bounds must be nonnegative integer scalars",
};
pub const NARGINCHK_ERROR_BOUNDS_INVALID: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.NARGINCHK.BOUNDS_INVALID",
    identifier: Some("RunMat:NarginchkBoundsInvalid"),
    when: "The minimum bound is greater than the maximum bound.",
    message: "narginchk: minArgs must be less than or equal to maxArgs",
};
pub const NARGINCHK_ERROR_CONTEXT_UNAVAILABLE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.NARGINCHK.CONTEXT_UNAVAILABLE",
    identifier: Some("RunMat:NarginchkContextUnavailable"),
    when: "The runtime dispatcher is invoked without VM function-call context.",
    message: "narginchk: function call context is unavailable",
};
pub const NARGINCHK_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.NARGINCHK.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:TooManyOutputs"),
    when: "A caller requests a result from the zero-output narginchk form.",
    message: "narginchk: too many output arguments",
};
pub const NARGINCHK_ERRORS: [BuiltinErrorDescriptor; 6] = [
    NARGINCHK_ERROR_NOT_ENOUGH_INPUTS,
    NARGINCHK_ERROR_TOO_MANY_INPUTS,
    NARGINCHK_ERROR_ARGUMENT_INVALID,
    NARGINCHK_ERROR_BOUNDS_INVALID,
    NARGINCHK_ERROR_CONTEXT_UNAVAILABLE,
    NARGINCHK_ERROR_TOO_MANY_OUTPUTS,
];
pub const NARGINCHK_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &NARGINCHK_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &NARGINCHK_ERRORS,
};

pub const NARGOUTCHK_ERROR_NOT_ENOUGH_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.NARGOUTCHK.NOT_ENOUGH_OUTPUTS",
    identifier: Some("RunMat:NotEnoughOutputs"),
    when: "The current function was called with fewer requested outputs than the lower bound.",
    message: "nargoutchk: not enough output arguments",
};
pub const NARGOUTCHK_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.NARGOUTCHK.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:TooManyOutputs"),
    when: "The current function exceeds maxArgs or the caller requests more results than the selected form returns.",
    message: "nargoutchk: too many output arguments",
};
pub const NARGOUTCHK_ERROR_ARGUMENT_INVALID: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.NARGOUTCHK.ARGUMENT_INVALID",
    identifier: Some("RunMat:NargoutchkArgumentInvalid"),
    when: "A bound argument is not a nonnegative integer scalar or valid Inf upper bound.",
    message: "nargoutchk: bounds must be nonnegative integer scalars",
};
pub const NARGOUTCHK_ERROR_BOUNDS_INVALID: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.NARGOUTCHK.BOUNDS_INVALID",
    identifier: Some("RunMat:NargoutchkBoundsInvalid"),
    when: "The minimum bound is greater than the maximum bound.",
    message: "nargoutchk: minArgs must be less than or equal to maxArgs",
};
pub const NARGOUTCHK_ERROR_CONTEXT_UNAVAILABLE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.NARGOUTCHK.CONTEXT_UNAVAILABLE",
    identifier: Some("RunMat:NargoutchkContextUnavailable"),
    when: "The runtime dispatcher is invoked without VM function-call context.",
    message: "nargoutchk: function call context is unavailable",
};
pub const NARGOUTCHK_ERRORS: [BuiltinErrorDescriptor; 5] = [
    NARGOUTCHK_ERROR_NOT_ENOUGH_OUTPUTS,
    NARGOUTCHK_ERROR_TOO_MANY_OUTPUTS,
    NARGOUTCHK_ERROR_ARGUMENT_INVALID,
    NARGOUTCHK_ERROR_BOUNDS_INVALID,
    NARGOUTCHK_ERROR_CONTEXT_UNAVAILABLE,
];
pub const NARGOUTCHK_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &NARGOUTCHK_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &NARGOUTCHK_ERRORS,
};
