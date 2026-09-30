use super::super::*;

pub(in crate::builtins::missing::domain) const MISSING_TEXT: &str = "<missing>";

const VALUE_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "B",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Result value.",
}];
const VALUE_AND_MASK_OUTPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "B",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Result value.",
    },
    BuiltinParamDescriptor {
        name: "TF",
        ty: BuiltinParamType::LogicalArray,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Logical mask of entries, rows, or columns that were filled or removed.",
    },
];
const VALUE_AND_ARGS_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input value.",
    },
    BuiltinParamDescriptor {
        name: "args",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Method, dimension, or option arguments.",
    },
];

pub(in crate::builtins::missing::domain) const FILLMISSING_SIGNATURES:
    [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "B = fillmissing(A, method, ...)",
    inputs: &VALUE_AND_ARGS_INPUTS,
    outputs: &VALUE_AND_MASK_OUTPUTS,
}];
pub(in crate::builtins::missing::domain) const RMMISSING_SIGNATURES: [BuiltinSignatureDescriptor;
    1] = [BuiltinSignatureDescriptor {
    label: "B = rmmissing(A, ...)",
    inputs: &VALUE_AND_ARGS_INPUTS,
    outputs: &VALUE_AND_MASK_OUTPUTS,
}];
pub(in crate::builtins::missing::domain) const NANAWARE_SIGNATURES: [BuiltinSignatureDescriptor;
    1] = [BuiltinSignatureDescriptor {
    label: "B = nanmean(A, ...)",
    inputs: &VALUE_AND_ARGS_INPUTS,
    outputs: &VALUE_OUTPUT,
}];

pub(in crate::builtins::missing::domain) const MISSING_ERROR_INVALID_ARGUMENT:
    BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MISSING.INVALID_ARGUMENT",
    identifier: Some("RunMat:missing:InvalidArgument"),
    when: "Arguments do not match a supported missing-value syntax.",
    message: "missing-value builtin: invalid argument",
};
pub(in crate::builtins::missing::domain) const MISSING_ERROR_UNSUPPORTED_TYPE:
    BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MISSING.UNSUPPORTED_TYPE",
    identifier: Some("RunMat:missing:UnsupportedType"),
    when: "The input type has no missing-value representation in RunMat.",
    message: "missing-value builtin: unsupported input type",
};
pub(in crate::builtins::missing::domain) const MISSING_ERROR_INTERNAL: BuiltinErrorDescriptor =
    BuiltinErrorDescriptor {
        code: "RM.MISSING.INTERNAL",
        identifier: Some("RunMat:missing:InternalError"),
        when: "Internal shape or table materialization fails.",
        message: "missing-value builtin: internal error",
    };
pub(in crate::builtins::missing::domain) const MISSING_ERRORS: [BuiltinErrorDescriptor; 3] = [
    MISSING_ERROR_INVALID_ARGUMENT,
    MISSING_ERROR_UNSUPPORTED_TYPE,
    MISSING_ERROR_INTERNAL,
];

pub(in crate::builtins::missing::domain) fn any_type(
    _args: &[Type],
    _ctx: &ResolveContext,
) -> Type {
    Type::Unknown
}
