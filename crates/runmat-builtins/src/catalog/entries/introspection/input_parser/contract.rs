use crate::*;

const INPUT_PARSER_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "p",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "inputParser handle object.",
}];

const INPUT_PARSER_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "p = inputParser()",
    inputs: &[],
    outputs: &INPUT_PARSER_OUTPUT,
}];

const ADD_PARAMETER_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "p",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Updated inputParser handle object.",
}];

const ADD_PARAMETER_INPUTS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "p",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "inputParser handle object.",
    },
    BuiltinParamDescriptor {
        name: "name",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Parameter name.",
    },
    BuiltinParamDescriptor {
        name: "default",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Default value for the parameter.",
    },
];

const ADD_PARAMETER_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "p = addParameter(p, name, default)",
    inputs: &ADD_PARAMETER_INPUTS,
    outputs: &ADD_PARAMETER_OUTPUT,
}];

const PARSE_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "p",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Updated inputParser handle object.",
}];

const PARSE_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "p",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "inputParser handle object.",
    },
    BuiltinParamDescriptor {
        name: "nameValue",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Name-value arguments to parse.",
    },
];

const PARSE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "p = parse(p, Name, Value, ...)",
    inputs: &PARSE_INPUTS,
    outputs: &PARSE_OUTPUT,
}];

pub const INPUT_PARSER_ERROR_INVALID_PARSER: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.INPUTPARSER.INVALID_PARSER",
    identifier: Some("RunMat:inputParser:InvalidParser"),
    when: "The parser argument is not a valid inputParser handle.",
    message: "inputParser: expected a valid inputParser handle",
};

pub const INPUT_PARSER_ERROR_PARAMETER_NAME: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.INPUTPARSER.PARAMETER_NAME",
    identifier: Some("RunMat:inputParser:InvalidParameterName"),
    when: "A parameter name is not a character vector or string scalar.",
    message: "inputParser: parameter names must be text scalars",
};

pub const INPUT_PARSER_ERROR_DUPLICATE_PARAMETER: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.INPUTPARSER.DUPLICATE_PARAMETER",
    identifier: Some("RunMat:inputParser:DuplicateParameter"),
    when: "The same parameter name is registered more than once.",
    message: "inputParser: duplicate parameter name",
};

pub const INPUT_PARSER_ERROR_NAME_VALUE_PAIRS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.INPUTPARSER.NAME_VALUE_PAIRS",
    identifier: Some("RunMat:inputParser:NameValuePairs"),
    when: "The parse argument list does not contain complete name-value pairs.",
    message: "inputParser: expected name-value pairs",
};

pub const INPUT_PARSER_ERROR_UNKNOWN_PARAMETER: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.INPUTPARSER.UNKNOWN_PARAMETER",
    identifier: Some("RunMat:inputParser:UnknownParameter"),
    when: "A parsed name does not match any registered parameter.",
    message: "inputParser: unknown parameter name",
};

const INPUT_PARSER_ERRORS: [BuiltinErrorDescriptor; 0] = [];
const ADD_PARAMETER_ERRORS: [BuiltinErrorDescriptor; 3] = [
    INPUT_PARSER_ERROR_INVALID_PARSER,
    INPUT_PARSER_ERROR_PARAMETER_NAME,
    INPUT_PARSER_ERROR_DUPLICATE_PARAMETER,
];
const PARSE_ERRORS: [BuiltinErrorDescriptor; 4] = [
    INPUT_PARSER_ERROR_INVALID_PARSER,
    INPUT_PARSER_ERROR_PARAMETER_NAME,
    INPUT_PARSER_ERROR_NAME_VALUE_PAIRS,
    INPUT_PARSER_ERROR_UNKNOWN_PARAMETER,
];

pub const INPUT_PARSER_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &INPUT_PARSER_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &INPUT_PARSER_ERRORS,
};

pub const ADD_PARAMETER_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &ADD_PARAMETER_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ADD_PARAMETER_ERRORS,
};

pub const PARSE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &PARSE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &PARSE_ERRORS,
};
