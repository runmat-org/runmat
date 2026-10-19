use crate::*;

pub const INPUT_NO_PROMPT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "input-no-prompt",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "input() with RunMat's default prompt is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:InputNoPromptExtension"),
};
pub const INPUT_SWAPPED_ARGUMENTS_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "input-swapped-arguments",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "input('s', prompt) argument order is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:InputSwappedArgumentsExtension"),
    };
pub const INPUT_EXTENSIONS: [BuiltinExtensionDescriptor; 2] =
    [INPUT_NO_PROMPT_EXTENSION, INPUT_SWAPPED_ARGUMENTS_EXTENSION];
pub const INPUT_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "input accepts only a text prompt and the text flag 's'. In expression mode the entered expression is evaluated normally, so any integer result retains the class and shape produced by that expression rather than serving as an integer input role of input itself.",
};

const INPUT_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "value",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Parsed scalar/matrix value, or raw text when using string mode.",
}];
const INPUT_INPUTS_NONE: [BuiltinParamDescriptor; 0] = [];
const INPUT_INPUTS_PROMPT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "prompt",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Prompt text shown to the user.",
}];
const INPUT_INPUTS_PROMPT_FLAG: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "prompt",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Prompt text shown to the user.",
    },
    BuiltinParamDescriptor {
        name: "stringFlag",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Set to 's' to return the raw input text.",
    },
];
const INPUT_INPUTS_FLAG_PROMPT: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "stringFlag",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Set to 's' to return the raw input text.",
    },
    BuiltinParamDescriptor {
        name: "prompt",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Prompt text shown to the user.",
    },
];
const INPUT_SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "value = input()",
        inputs: &INPUT_INPUTS_NONE,
        outputs: &INPUT_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "value = input(prompt)",
        inputs: &INPUT_INPUTS_PROMPT,
        outputs: &INPUT_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "value = input(prompt, stringFlag)",
        inputs: &INPUT_INPUTS_PROMPT_FLAG,
        outputs: &INPUT_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "value = input(stringFlag, prompt)",
        inputs: &INPUT_INPUTS_FLAG_PROMPT,
        outputs: &INPUT_OUTPUT,
    },
];
pub const INPUT_ERROR_TOO_MANY_INPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.INPUT.TOO_MANY_INPUTS",
    identifier: Some("RunMat:input:TooManyInputs"),
    when: "More than two input arguments are passed to input.",
    message: "input: too many inputs",
};
pub const INPUT_ERROR_INVALID_STRING_FLAG: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.INPUT.INVALID_STRING_FLAG",
    identifier: Some("RunMat:input:InvalidStringFlag"),
    when: "The string mode flag is not a scalar string/char 's'.",
    message: "input: invalid string flag",
};
pub const INPUT_ERROR_PROMPT_ROW_VECTOR: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.INPUT.PROMPT_ROW_VECTOR",
    identifier: Some("RunMat:input:PromptMustBeRowVector"),
    when: "Prompt char array is not 1-by-N.",
    message: "input: prompt must be a row vector",
};
pub const INPUT_ERROR_PROMPT_SCALAR_STRING: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.INPUT.PROMPT_SCALAR_STRING",
    identifier: Some("RunMat:input:PromptMustBeScalarString"),
    when: "Prompt string array is not scalar.",
    message: "input: prompt must be a scalar string",
};
pub const INPUT_ERROR_INVALID_PROMPT_TYPE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.INPUT.INVALID_PROMPT_TYPE",
    identifier: Some("RunMat:input:InvalidPromptType"),
    when: "Prompt is not a string scalar or row char vector.",
    message: "input: invalid prompt type",
};
pub const INPUT_ERROR_INTERACTION_FAILED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.INPUT.INTERACTION_FAILED",
    identifier: Some("RunMat:input:InteractionFailed"),
    when: "Interactive prompt callback fails.",
    message: "input: interaction failed",
};
pub const INPUT_ERROR_EVAL_FAILED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.INPUT.EVAL_FAILED",
    identifier: Some("RunMat:input:EvalFailed"),
    when: "Expression evaluation hook rejects the input expression.",
    message: "input: invalid expression",
};
pub const INPUT_ERROR_INVALID_NUMERIC_EXPRESSION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.INPUT.INVALID_NUMERIC_EXPRESSION",
    identifier: Some("RunMat:input:InvalidNumericExpression"),
    when: "Numeric fallback parser rejects the input expression.",
    message: "input: invalid numeric expression",
};
const INPUT_ERRORS: [BuiltinErrorDescriptor; 9] = [
    INPUT_ERROR_TOO_MANY_INPUTS,
    INPUT_ERROR_TOO_MANY_OUTPUTS,
    INPUT_ERROR_INVALID_STRING_FLAG,
    INPUT_ERROR_PROMPT_ROW_VECTOR,
    INPUT_ERROR_PROMPT_SCALAR_STRING,
    INPUT_ERROR_INVALID_PROMPT_TYPE,
    INPUT_ERROR_INTERACTION_FAILED,
    INPUT_ERROR_EVAL_FAILED,
    INPUT_ERROR_INVALID_NUMERIC_EXPRESSION,
];
pub const INPUT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &INPUT_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &INPUT_ERRORS,
};
pub const INPUT_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.INPUT.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:input:TooManyOutputs"),
    when: "More than one output argument is requested from input.",
    message: "input: expected at most one output argument",
};
