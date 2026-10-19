use crate::*;

const STATUS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "status",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Zero on success and one when the target cannot be resolved or written.",
};
const MESSAGE: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "message",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "RunMat diagnostic text, or an empty character row on success.",
};
const MESSAGE_ID: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "message_id",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "RunMat diagnostic identifier, or an empty character row on success.",
};
const STATUS_OUTPUT: [BuiltinParamDescriptor; 1] = [STATUS];
const DIAGNOSTIC_OUTPUTS: [BuiltinParamDescriptor; 3] = [STATUS, MESSAGE, MESSAGE_ID];
const FILENAME: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "filename",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Target pathdef file as a character vector or string scalar.",
}];
const SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "status = savepath()",
        inputs: &[],
        outputs: &STATUS_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "status = savepath(filename)",
        inputs: &FILENAME,
        outputs: &STATUS_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "[status, message, message_id] = savepath()",
        inputs: &[],
        outputs: &DIAGNOSTIC_OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "[status, message, message_id] = savepath(filename)",
        inputs: &FILENAME,
        outputs: &DIAGNOSTIC_OUTPUTS,
    },
];

macro_rules! error {
    ($name:ident, $code:literal, $identifier:literal, $when:literal, $message:literal) => {
        pub const $name: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
            code: $code,
            identifier: Some($identifier),
            when: $when,
            message: $message,
        };
    };
}

error!(
    SAVEPATH_ERROR_ARGUMENT_TYPE,
    "RM.SAVEPATH.ARGUMENT_TYPE",
    "RunMat:savepath:InvalidFilename",
    "The filename is not a character row, string scalar, or admitted numeric character-code row.",
    "savepath: filename must be a character vector or string scalar"
);
error!(
    SAVEPATH_ERROR_EMPTY_FILENAME,
    "RM.SAVEPATH.EMPTY_FILENAME",
    "RunMat:savepath:EmptyFilename",
    "The explicit filename is empty.",
    "savepath: filename must not be empty"
);
error!(
    SAVEPATH_ERROR_TOO_MANY_INPUTS,
    "RM.SAVEPATH.TOO_MANY_INPUTS",
    "RunMat:savepath:TooManyInputs",
    "More than one input is provided.",
    "savepath: too many input arguments"
);
error!(
    SAVEPATH_ERROR_TOO_MANY_OUTPUTS,
    "RM.SAVEPATH.TOO_MANY_OUTPUTS",
    "RunMat:savepath:TooManyOutputs",
    "More than three outputs are requested.",
    "savepath: too many output arguments"
);
error!(
    SAVEPATH_ERROR_CANNOT_WRITE,
    "RM.SAVEPATH.CANNOT_WRITE",
    "RunMat:savepath:cannotWriteFile",
    "The target file cannot be written.",
    "savepath: unable to write file"
);
error!(
    SAVEPATH_ERROR_CANNOT_RESOLVE,
    "RM.SAVEPATH.CANNOT_RESOLVE",
    "RunMat:savepath:cannotResolveFile",
    "The output path cannot be resolved.",
    "savepath: unable to resolve output path"
);
error!(
    SAVEPATH_ERROR_PROVIDER,
    "RM.SAVEPATH.PROVIDER_FAILED",
    "RunMat:savepath:ProviderFailed",
    "An admitted resident numeric character-code row cannot be gathered.",
    "savepath: unable to read resident character codes"
);

const ERRORS: [BuiltinErrorDescriptor; 7] = [
    SAVEPATH_ERROR_ARGUMENT_TYPE,
    SAVEPATH_ERROR_EMPTY_FILENAME,
    SAVEPATH_ERROR_TOO_MANY_INPUTS,
    SAVEPATH_ERROR_TOO_MANY_OUTPUTS,
    SAVEPATH_ERROR_CANNOT_WRITE,
    SAVEPATH_ERROR_CANNOT_RESOLVE,
    SAVEPATH_ERROR_PROVIDER,
];
pub const SAVEPATH_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

macro_rules! extension {
    ($name:ident, $id:literal, $description:literal, $error:literal) => {
        pub const $name: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
            id: $id,
            mode: BuiltinExtensionMode::RunMatOnly,
            description: $description,
            error_identifier: Some($error),
        };
    };
}

extension!(
    SAVEPATH_DIAGNOSTIC_OUTPUTS_EXTENSION,
    "savepath-diagnostic-outputs",
    "savepath diagnostic message outputs are a RunMat extension",
    "RunMat:compatibility:SavepathDiagnosticOutputsExtension"
);
extension!(
    SAVEPATH_DIRECTORY_TARGET_EXTENSION,
    "savepath-directory-target",
    "savepath directory target shorthand is a RunMat extension",
    "RunMat:compatibility:SavepathDirectoryTargetExtension"
);
extension!(
    SAVEPATH_NUMERIC_CHARACTER_CODES_EXTENSION,
    "savepath-numeric-character-codes",
    "savepath with numeric character-code rows is a RunMat extension",
    "RunMat:compatibility:SavepathNumericCharacterCodesExtension"
);

pub(super) const EXTENSIONS: &[BuiltinExtensionDescriptor] = &[
    SAVEPATH_DIAGNOSTIC_OUTPUTS_EXTENSION,
    SAVEPATH_DIRECTORY_TARGET_EXTENSION,
    SAVEPATH_NUMERIC_CHARACTER_CODES_EXTENSION,
];

const INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "filename",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "A dense real integer row may encode the target path exactly.",
}];
pub(super) const INTEGER_CAPABILITIES: &[BuiltinIntegerCapabilityDescriptor] =
    &[BuiltinIntegerCapabilityDescriptor {
        form: "status = savepath(integer_character_codes)",
        inputs: &INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Code points decode exactly before the host filesystem boundary.",
    }];
