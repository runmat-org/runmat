use crate::*;
pub const VALIDATE_FUNCTION_SIGNATURES_JSON_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor { kind: BuiltinIntegerAuditKind::NotApplicable, canonical_builtin: None, notes: "validateFunctionSignaturesJSON accepts text paths in the compatibility target and a text JSON payload in the current RunMat implementation; integer and resident numeric values are not valid in either form and reject without conversion or provider access." };
pub const VALIDATE_FUNCTION_SIGNATURES_JSON_ERROR_OUTPUTS: BuiltinErrorDescriptor =
    BuiltinErrorDescriptor {
        code: "RM.VALIDATE_FUNCTION_SIGNATURES_JSON.TOO_MANY_OUTPUTS",
        identifier: Some("RunMat:validateFunctionSignaturesJSON:TooManyOutputs"),
        when: "Any output is explicitly requested from this zero-output validator.",
        message: "validateFunctionSignaturesJSON: too many output arguments",
    };
pub const VALIDATE_FUNCTION_SIGNATURES_JSON_ERROR_INVALID: BuiltinErrorDescriptor =
    BuiltinErrorDescriptor {
        code: "RM.VALIDATE_FUNCTION_SIGNATURES_JSON.INVALID_ARGUMENT",
        identifier: Some("RunMat:validateFunctionSignaturesJSON:InvalidArgument"),
        when: "The supported input count, control type, or option contract is violated.",
        message: "validateFunctionSignaturesJSON: invalid argument",
    };
pub const VALIDATE_FUNCTION_SIGNATURES_JSON_ERROR_FAILED: BuiltinErrorDescriptor =
    BuiltinErrorDescriptor {
        code: "RM.VALIDATE_FUNCTION_SIGNATURES_JSON.VALIDATION_FAILED",
        identifier: Some("RunMat:validateFunctionSignaturesJSON:ValidationFailed"),
        when: "The supplied value does not satisfy the validator.",
        message: "validateFunctionSignaturesJSON: value does not satisfy validator",
    };
pub const VALIDATE_FUNCTION_SIGNATURES_JSON_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[BuiltinSignatureDescriptor {
        label: "validateFunctionSignaturesJSON(A)",
        inputs: &[super::super::support::VALUE],
        outputs: &[],
    }],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[
        VALIDATE_FUNCTION_SIGNATURES_JSON_ERROR_OUTPUTS,
        VALIDATE_FUNCTION_SIGNATURES_JSON_ERROR_INVALID,
        super::super::support::VALIDATION_ERROR_TEXT_INPUT,
        VALIDATE_FUNCTION_SIGNATURES_JSON_ERROR_FAILED,
    ],
};
pub const VALIDATE_FUNCTION_SIGNATURES_JSON_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
 id: "signature-json-payload", mode: BuiltinExtensionMode::RunMatOnly,
 description: "JSON payload syntax validation; MATLAB filename and no-input forms are not implemented.",
 error_identifier: Some("RunMat:compatibility:ValidateFunctionSignaturesJSONPayloadExtension"),
};
