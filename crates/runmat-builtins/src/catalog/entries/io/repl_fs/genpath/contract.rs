use crate::*;

const OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "pathstr",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Generated path-list character row.",
}];
const FOLDER: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "folder",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Root folder to traverse.",
}];
const FOLDER_EXCLUDES: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "folder",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Root folder to traverse.",
    },
    BuiltinParamDescriptor {
        name: "excludes",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "RunMat path-list extension naming folders to omit.",
    },
];
const SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "pathstr = genpath()",
        inputs: &[],
        outputs: &OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "pathstr = genpath(folder)",
        inputs: &FOLDER,
        outputs: &OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "pathstr = genpath(folder, excludes)",
        inputs: &FOLDER_EXCLUDES,
        outputs: &OUTPUTS,
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

error!(GENPATH_ERROR_FOLDER_TYPE, "RM.GENPATH.FOLDER_TYPE", "RunMat:genpath:InvalidFolder", "The folder argument is not an admitted character row, string scalar, or numeric character-code row.", "genpath: folder must be a character vector or string scalar");
error!(GENPATH_ERROR_EXCLUDES_TYPE, "RM.GENPATH.EXCLUDES_TYPE", "RunMat:genpath:InvalidExcludes", "The excludes argument is not an admitted character row, string scalar, or numeric character-code row.", "genpath: excludes must be a character vector or string scalar");
error!(
    GENPATH_ERROR_TOO_MANY_INPUTS,
    "RM.GENPATH.TOO_MANY_INPUTS",
    "RunMat:genpath:TooManyInputs",
    "More than two inputs are provided.",
    "genpath: too many input arguments"
);
error!(
    GENPATH_ERROR_CURRENT_FOLDER,
    "RM.GENPATH.CURRENT_FOLDER",
    "RunMat:genpath:CurrentFolderUnavailable",
    "The current folder cannot be resolved.",
    "genpath: unable to resolve current directory"
);
error!(
    GENPATH_ERROR_FOLDER_NOT_FOUND,
    "RM.GENPATH.FOLDER_NOT_FOUND",
    "RunMat:genpath:FolderNotFound",
    "The requested root folder does not exist.",
    "genpath: folder not found"
);
error!(
    GENPATH_ERROR_NOT_FOLDER,
    "RM.GENPATH.NOT_FOLDER",
    "RunMat:genpath:NotFolder",
    "The requested root exists but is not a folder.",
    "genpath: path is not a folder"
);
error!(
    GENPATH_ERROR_PROVIDER,
    "RM.GENPATH.PROVIDER_FAILED",
    "RunMat:genpath:ProviderFailed",
    "An admitted resident numeric character-code row cannot be gathered from its owner.",
    "genpath: unable to read resident character codes"
);

const ERRORS: [BuiltinErrorDescriptor; 7] = [
    GENPATH_ERROR_FOLDER_TYPE,
    GENPATH_ERROR_EXCLUDES_TYPE,
    GENPATH_ERROR_TOO_MANY_INPUTS,
    GENPATH_ERROR_CURRENT_FOLDER,
    GENPATH_ERROR_FOLDER_NOT_FOUND,
    GENPATH_ERROR_NOT_FOLDER,
    GENPATH_ERROR_PROVIDER,
];
pub const GENPATH_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const GENPATH_EXCLUDES_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "genpath-excludes",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "genpath(folder, excludes) is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:GenpathExcludesExtension"),
};
pub const GENPATH_NUMERIC_CHARACTER_CODES_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "genpath-numeric-character-codes",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "genpath with numeric character-code rows is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:GenpathNumericCharacterCodesExtension"),
    };
pub(super) const EXTENSIONS: &[BuiltinExtensionDescriptor] = &[
    GENPATH_EXCLUDES_EXTENSION,
    GENPATH_NUMERIC_CHARACTER_CODES_EXTENSION,
];

const INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "folder",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "A dense real integer row may encode one folder or exclusion path-list.",
}];
pub(super) const INTEGER_CAPABILITIES: &[BuiltinIntegerCapabilityDescriptor] = &[
    BuiltinIntegerCapabilityDescriptor {
        form: "pathstr = genpath(integer_character_codes)",
        inputs: &INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes:
            "Code points decode exactly, invalid Unicode rejects, and traversal remains host-owned.",
    },
];
