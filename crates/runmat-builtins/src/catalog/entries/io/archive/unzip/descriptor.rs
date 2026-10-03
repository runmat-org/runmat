use crate::*;

const UNZIP_OUTPUT_FILES: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "filenames",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Cell array of extracted file paths.",
}];
const UNZIP_INPUT_ARCHIVE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "zipfilename",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "ZIP archive path or HTTP/HTTPS URL.",
}];
const UNZIP_INPUT_ARCHIVE_OUTPUT: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "zipfilename",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "ZIP archive path or HTTP/HTTPS URL.",
    },
    BuiltinParamDescriptor {
        name: "outputfolder",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Destination folder for extracted files.",
    },
];
const UNZIP_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "unzip(zipfilename)",
        inputs: &UNZIP_INPUT_ARCHIVE,
        outputs: &UNZIP_OUTPUT_FILES,
    },
    BuiltinSignatureDescriptor {
        label: "unzip(zipfilename, outputfolder)",
        inputs: &UNZIP_INPUT_ARCHIVE_OUTPUT,
        outputs: &UNZIP_OUTPUT_FILES,
    },
    BuiltinSignatureDescriptor {
        label: "filenames = unzip(___)",
        inputs: &UNZIP_INPUT_ARCHIVE_OUTPUT,
        outputs: &UNZIP_OUTPUT_FILES,
    },
];

pub const UNZIP_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.UNZIP.INVALID_ARGUMENT",
    identifier: Some("RunMat:unzip:InvalidArgument"),
    when: "Arguments do not match supported unzip invocation forms.",
    message: "unzip: invalid argument",
};
pub const UNZIP_ERROR_FILENAME: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.UNZIP.FILENAME",
    identifier: Some("RunMat:unzip:Filename"),
    when: "ZIP filename or URL is invalid.",
    message: "unzip: invalid ZIP filename",
};
pub const UNZIP_ERROR_OUTPUT_FOLDER: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.UNZIP.OUTPUT_FOLDER",
    identifier: Some("RunMat:unzip:OutputFolder"),
    when: "Output folder is invalid or cannot be created.",
    message: "unzip: invalid output folder",
};
pub const UNZIP_ERROR_HTTP: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.UNZIP.HTTP",
    identifier: Some("RunMat:unzip:Http"),
    when: "Remote ZIP archive cannot be downloaded.",
    message: "unzip: download failed",
};
pub const UNZIP_ERROR_ZIP: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.UNZIP.ZIP",
    identifier: Some("RunMat:unzip:Zip"),
    when: "ZIP archive cannot be decoded or contains unsupported entries.",
    message: "unzip: invalid ZIP archive",
};
pub const UNZIP_ERROR_SECURITY: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.UNZIP.SECURITY",
    identifier: Some("RunMat:unzip:UnsafePath"),
    when: "ZIP entry would escape the destination folder.",
    message: "unzip: unsafe archive path",
};
pub const UNZIP_ERROR_IO: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.UNZIP.IO",
    identifier: Some("RunMat:unzip:Io"),
    when: "Extraction fails due to filesystem I/O.",
    message: "unzip: extraction failed",
};
pub const UNZIP_ERROR_OUTPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.UNZIP.OUTPUT",
    identifier: Some("RunMat:unzip:Output"),
    when: "Returned filename cell array cannot be materialized.",
    message: "unzip: output materialization failed",
};
pub const UNZIP_ERROR_OUTPUT_COUNT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.UNZIP.OUTPUT_COUNT",
    identifier: Some("RunMat:unzip:OutputCount"),
    when: "Caller requests more outputs than unzip supports.",
    message: "unzip: unsupported output count",
};
const UNZIP_ERRORS: [BuiltinErrorDescriptor; 9] = [
    UNZIP_ERROR_INVALID_ARGUMENT,
    UNZIP_ERROR_FILENAME,
    UNZIP_ERROR_OUTPUT_FOLDER,
    UNZIP_ERROR_HTTP,
    UNZIP_ERROR_ZIP,
    UNZIP_ERROR_SECURITY,
    UNZIP_ERROR_IO,
    UNZIP_ERROR_OUTPUT,
    UNZIP_ERROR_OUTPUT_COUNT,
];

pub const UNZIP_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &UNZIP_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &UNZIP_ERRORS,
};
