use crate::*;
const NUM_ARGUMENTS_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "n",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Number of expected subsref outputs or subsasgn inputs.",
}];

const NUM_ARGUMENTS_INPUTS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Target of the indexing expression.",
    },
    BuiltinParamDescriptor {
        name: "S",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Substruct-compatible indexing descriptor.",
    },
    BuiltinParamDescriptor {
        name: "indexingContext",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "IndexingContext enum object or compatible text token.",
    },
];

const NUM_ARGUMENTS_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "n = numArgumentsFromSubscript(A, S, indexingContext)",
    inputs: &NUM_ARGUMENTS_INPUTS,
    outputs: &NUM_ARGUMENTS_OUTPUT,
}];

pub const NUM_ARGUMENTS_ERROR_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.NUM_ARGUMENTS_FROM_SUBSCRIPT.INVALID_ARGUMENT",
    identifier: Some("RunMat:numArgumentsFromSubscript:InvalidArgument"),
    when: "Inputs do not match the documented target, substruct, and indexing-context form.",
    message: "numArgumentsFromSubscript: invalid argument",
};

pub const NUM_ARGUMENTS_ERROR_SUBSTRUCT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.NUM_ARGUMENTS_FROM_SUBSCRIPT.INVALID_SUBSTRUCT",
    identifier: Some("RunMat:numArgumentsFromSubscript:InvalidSubstruct"),
    when: "The substruct-compatible descriptor is missing type/subs fields or has invalid values.",
    message: "numArgumentsFromSubscript: invalid substruct",
};

pub const NUM_ARGUMENTS_ERROR_CONTEXT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.NUM_ARGUMENTS_FROM_SUBSCRIPT.INVALID_CONTEXT",
    identifier: Some("RunMat:numArgumentsFromSubscript:InvalidContext"),
    when: "The indexing context is not Statement, Expression, or Assignment.",
    message: "numArgumentsFromSubscript: invalid indexing context",
};

pub const NUM_ARGUMENTS_ERROR_COUNT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.NUM_ARGUMENTS_FROM_SUBSCRIPT.COUNT_NOT_EXACT_DOUBLE",
    identifier: Some("RunMat:numArgumentsFromSubscript:CountNotExactDouble"),
    when: "The computed argument count cannot be represented exactly by the documented double scalar output.",
    message: "numArgumentsFromSubscript: result exceeds exact double range",
};

pub const NUM_ARGUMENTS_ERRORS: [BuiltinErrorDescriptor; 5] = [
    NUM_ARGUMENTS_FROM_SUBSCRIPT_ERROR_OUTPUTS,
    NUM_ARGUMENTS_ERROR_ARGUMENT,
    NUM_ARGUMENTS_ERROR_SUBSTRUCT,
    NUM_ARGUMENTS_ERROR_CONTEXT,
    NUM_ARGUMENTS_ERROR_COUNT,
];

pub const NUM_ARGUMENTS_FROM_SUBSCRIPT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &NUM_ARGUMENTS_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &NUM_ARGUMENTS_ERRORS,
};

const NUM_ARGUMENTS_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "A",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The indexing target may be an array of any integer class; ordinary array cases use only exact shape and container metadata.",
    },
    BuiltinIntegerInputCapability {
        name: "S.subs",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Integer scalar and vector subscripts are validated directly from authoritative storage, including full-width positive uint64 selectors, without binary64 conversion.",
    },
];

pub const NUM_ARGUMENTS_FROM_SUBSCRIPT_INTEGER_CAPABILITIES:
    [BuiltinIntegerCapabilityDescriptor; 1] = [BuiltinIntegerCapabilityDescriptor {
    form: "n = numArgumentsFromSubscript(integer_A, S_with_integer_subscripts, indexingContext)",
    inputs: &NUM_ARGUMENTS_INTEGER_INPUTS,
    computation_domain: BuiltinIntegerComputationDomain::Structural,
    output_class: BuiltinIntegerOutputClassRule::Double,
    overflow: BuiltinIntegerOverflowRule::Error,
    backend: BuiltinIntegerBackendRule::HostAndGpu,
    overload: BuiltinIntegerOverloadKind::FunctionSpecific,
    notes: "Metadata-decidable target extents do not gather resident payloads. The structural count crosses to double only after exact-range validation.",
}];

pub const NUM_ARGUMENTS_FROM_SUBSCRIPT_ERROR_OUTPUTS: BuiltinErrorDescriptor =
    BuiltinErrorDescriptor {
        code: "RM.NUM_ARGUMENTS_FROM_SUBSCRIPT.TOO_MANY_OUTPUTS",
        identifier: Some("RunMat:numArgumentsFromSubscript:TooManyOutputs"),
        when: "More than one output is requested.",
        message: "numArgumentsFromSubscript: expected at most one output",
    };
