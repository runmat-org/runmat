use crate::*;

macro_rules! fclose_extension {
    ($name:ident, $id:literal, $description:literal, $error:literal) => {
        pub const $name: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
            id: $id,
            mode: BuiltinExtensionMode::RunMatOnly,
            description: $description,
            error_identifier: Some($error),
        };
    };
}
fclose_extension!(
    FCLOSE_NO_INPUT_EXTENSION,
    "fclose-no-input-close-all",
    "fclose() close-all syntax is a RunMat extension",
    "RunMat:compatibility:FcloseNoInputExtension"
);
fclose_extension!(
    FCLOSE_INTEGER_ID_EXTENSION,
    "fclose-integer-fileid",
    "integer-class fclose file identifiers are a RunMat extension",
    "RunMat:compatibility:FcloseIntegerIdExtension"
);
fclose_extension!(
    FCLOSE_VECTOR_ID_EXTENSION,
    "fclose-fileid-vector",
    "fclose file-identifier arrays are a RunMat extension",
    "RunMat:compatibility:FcloseVectorIdExtension"
);
fclose_extension!(
    FCLOSE_LOGICAL_ID_EXTENSION,
    "fclose-logical-fileid",
    "logical fclose file identifiers are a RunMat extension",
    "RunMat:compatibility:FcloseLogicalIdExtension"
);
fclose_extension!(
    FCLOSE_CELL_ID_EXTENSION,
    "fclose-cell-fileids",
    "cell fclose file-identifier collections are a RunMat extension",
    "RunMat:compatibility:FcloseCellIdExtension"
);
fclose_extension!(
    FCLOSE_STRING_ALL_EXTENSION,
    "fclose-string-all",
    "string-scalar fclose all syntax is a RunMat extension",
    "RunMat:compatibility:FcloseStringAllExtension"
);
fclose_extension!(
    FCLOSE_MESSAGE_OUTPUT_EXTENSION,
    "fclose-message-output",
    "a second fclose diagnostic output is a RunMat extension",
    "RunMat:compatibility:FcloseMessageOutputExtension"
);
fclose_extension!(
    FCLOSE_RESIDENT_ID_EXTENSION,
    "fclose-resident-fileid",
    "resident fclose file identifiers are a RunMat extension",
    "RunMat:compatibility:FcloseResidentIdExtension"
);
fclose_extension!(
    FCLOSE_SINGLE_ID_EXTENSION,
    "fclose-single-fileid",
    "single-precision fclose file identifiers are a RunMat extension",
    "RunMat:compatibility:FcloseSingleIdExtension"
);
pub const FCLOSE_EXTENSIONS: [BuiltinExtensionDescriptor; 9] = [
    FCLOSE_NO_INPUT_EXTENSION,
    FCLOSE_INTEGER_ID_EXTENSION,
    FCLOSE_VECTOR_ID_EXTENSION,
    FCLOSE_LOGICAL_ID_EXTENSION,
    FCLOSE_CELL_ID_EXTENSION,
    FCLOSE_STRING_ALL_EXTENSION,
    FCLOSE_MESSAGE_OUTPUT_EXTENSION,
    FCLOSE_RESIDENT_ID_EXTENSION,
    FCLOSE_SINGLE_ID_EXTENSION,
];

pub const FCLOSE_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "fileID",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "All eight integer classes are parsed exactly and must fit the runtime file-identifier domain before registry access.",
    }];
pub const FCLOSE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "status = fclose(integer_fileID)",
        inputs: &FCLOSE_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::ScalarOnly,
        notes: "This RunMat-only scalar form returns double status and never converts the identifier through binary64.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "status = fclose(integer_fileIDs)",
        inputs: &FCLOSE_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The collection and resident forms are independently gated RunMat extensions; every identifier validates before any close side effect.",
    },
];

pub const FCLOSE_OUTPUT_STATUS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "status",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "0 on success, -1 on failure.",
}];

pub const FCLOSE_OUTPUT_MESSAGE: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "status",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "0 on success, -1 on failure.",
    },
    BuiltinParamDescriptor {
        name: "msg",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"\""),
        description: "Failure message when status is -1.",
    },
];

pub const FCLOSE_INPUTS_NONE: [BuiltinParamDescriptor; 0] = [];
pub const FCLOSE_INPUTS_FID: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "fid",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "File identifier scalar, vector, cell list, or keyword \"all\".",
}];
pub const FCLOSE_INPUTS_ALL: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "mode",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: Some("\"all\""),
    description: "Close all open user file identifiers.",
}];

pub const FCLOSE_SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "status = fclose()",
        inputs: &FCLOSE_INPUTS_NONE,
        outputs: &FCLOSE_OUTPUT_STATUS,
    },
    BuiltinSignatureDescriptor {
        label: "status = fclose(fid)",
        inputs: &FCLOSE_INPUTS_FID,
        outputs: &FCLOSE_OUTPUT_STATUS,
    },
    BuiltinSignatureDescriptor {
        label: "status = fclose(\"all\")",
        inputs: &FCLOSE_INPUTS_ALL,
        outputs: &FCLOSE_OUTPUT_STATUS,
    },
    BuiltinSignatureDescriptor {
        label: "[status, msg] = fclose(...)",
        inputs: &FCLOSE_INPUTS_FID,
        outputs: &FCLOSE_OUTPUT_MESSAGE,
    },
];

pub const FCLOSE_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FCLOSE.INVALID_INPUT",
    identifier: Some("RunMat:fclose:InvalidInput"),
    when: "Input argument count or file identifier value/type is invalid.",
    message: "fclose: invalid input arguments",
};

pub const FCLOSE_ERROR_INVALID_IDENTIFIER: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FCLOSE.INVALID_IDENTIFIER",
    identifier: Some("RunMat:fclose:InvalidIdentifier"),
    when: "The provided file identifier does not refer to an open file.",
    message: "fclose: invalid file identifier. Use fopen to generate a valid file ID.",
};

pub const FCLOSE_ERROR_IO: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FCLOSE.IO",
    identifier: Some("RunMat:fclose:IoFailure"),
    when: "Closing an open file failed due to I/O error.",
    message: "fclose: failed to close file",
};

pub const FCLOSE_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FCLOSE.INTERNAL",
    identifier: None,
    when: "Internal runtime control-flow or conversion failed.",
    message: "fclose: internal error",
};

pub const FCLOSE_ERRORS: [BuiltinErrorDescriptor; 4] = [
    FCLOSE_ERROR_INVALID_INPUT,
    FCLOSE_ERROR_INVALID_IDENTIFIER,
    FCLOSE_ERROR_IO,
    FCLOSE_ERROR_INTERNAL,
];

pub const FCLOSE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &FCLOSE_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &FCLOSE_ERRORS,
};
