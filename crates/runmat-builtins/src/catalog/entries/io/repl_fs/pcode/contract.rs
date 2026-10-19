use crate::*;
const INPUT_ITEM: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "item",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "MATLAB source file, source name, folder, or wildcard pattern to encode.",
}];

const INPUT_ITEM_VERSION: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "item",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "MATLAB source file, source name, folder, or wildcard pattern to encode.",
    },
    BuiltinParamDescriptor {
        name: "version",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("-R2007b"),
        description: "Obfuscation compatibility selector, either -R2007b or -R2022a.",
    },
];

const INPUT_ITEMS_INPLACE: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "item1",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "First MATLAB source file, source name, folder, or wildcard pattern.",
    },
    BuiltinParamDescriptor {
        name: "itemN",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Additional source files, source names, folders, or wildcard patterns.",
    },
    BuiltinParamDescriptor {
        name: "inplace",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Use -inplace to write each P-code file beside its source file.",
    },
];

const SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "pcode(item)",
        inputs: &INPUT_ITEM,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "pcode(item, version)",
        inputs: &INPUT_ITEM_VERSION,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "pcode(item1, item2, ..., itemN, \"-inplace\")",
        inputs: &INPUT_ITEMS_INPLACE,
        outputs: &[],
    },
];

pub const PCODE_ERROR_NOT_ENOUGH_INPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PCODE.NOT_ENOUGH_INPUTS",
    identifier: Some("RunMat:pcode:NotEnoughInputs"),
    when: "No source item is provided.",
    message: "pcode: not enough input arguments",
};
pub const PCODE_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PCODE.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:pcode:TooManyOutputs"),
    when: "`pcode` is called with requested output arguments.",
    message: "pcode: too many output arguments",
};
pub const PCODE_ERROR_ARG_TYPE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PCODE.ARG_TYPE",
    identifier: Some("RunMat:pcode:InvalidArgument"),
    when: "An argument is not a character row, string scalar, or scalar string array.",
    message: "pcode: arguments must be character vectors or string scalars",
};
pub const PCODE_ERROR_EMPTY_ITEM: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PCODE.EMPTY_ITEM",
    identifier: Some("RunMat:pcode:EmptyItem"),
    when: "A source item is empty.",
    message: "pcode: source item must not be empty",
};
pub const PCODE_ERROR_INVALID_OPTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PCODE.INVALID_OPTION",
    identifier: Some("RunMat:pcode:InvalidOption"),
    when: "An option is not -inplace, -R2007b, or -R2022a.",
    message: "pcode: invalid option",
};
pub const PCODE_ERROR_PATH_RESOLVE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PCODE.PATH_RESOLVE",
    identifier: Some("RunMat:pcode:PathResolveFailed"),
    when: "RunMat cannot resolve a source item or destination path.",
    message: "pcode: failed to resolve path",
};
pub const PCODE_ERROR_FILE_NOT_FOUND: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PCODE.FILE_NOT_FOUND",
    identifier: Some("RunMat:pcode:FileNotFound"),
    when: "No MATLAB `.m` source file matches the item.",
    message: "pcode: source file not found",
};
pub const PCODE_ERROR_UNSUPPORTED_EXTENSION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PCODE.UNSUPPORTED_EXTENSION",
    identifier: Some("RunMat:pcode:UnsupportedExtension"),
    when: "The source item names an unsupported file type, including `.mlx` live scripts.",
    message: "pcode: only MATLAB .m source files are supported",
};
pub const PCODE_ERROR_FILE_READ: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PCODE.FILE_READ",
    identifier: Some("RunMat:pcode:FileReadFailed"),
    when: "A source file cannot be read as text.",
    message: "pcode: failed to read source file",
};
pub const PCODE_ERROR_FILE_WRITE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PCODE.FILE_WRITE",
    identifier: Some("RunMat:pcode:FileWriteFailed"),
    when: "The destination `.p` file cannot be written.",
    message: "pcode: failed to write P-code file",
};
pub const PCODE_ERROR_INVALID_PCODE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PCODE.INVALID_PCODE",
    identifier: Some("RunMat:pcode:InvalidPcode"),
    when: "A `.p` file is not a RunMat-generated P-code container or fails integrity checks.",
    message: "pcode: invalid or unsupported P-code file",
};

const ERRORS: [BuiltinErrorDescriptor; 11] = [
    PCODE_ERROR_NOT_ENOUGH_INPUTS,
    PCODE_ERROR_TOO_MANY_OUTPUTS,
    PCODE_ERROR_ARG_TYPE,
    PCODE_ERROR_EMPTY_ITEM,
    PCODE_ERROR_INVALID_OPTION,
    PCODE_ERROR_PATH_RESOLVE,
    PCODE_ERROR_FILE_NOT_FOUND,
    PCODE_ERROR_UNSUPPORTED_EXTENSION,
    PCODE_ERROR_FILE_READ,
    PCODE_ERROR_FILE_WRITE,
    PCODE_ERROR_INVALID_PCODE,
];

pub const PCODE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
