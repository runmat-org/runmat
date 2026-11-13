use crate::*;

pub const DLMREAD_NUMERIC_DELIMITER_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "dlmread-numeric-delimiter",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "numeric delimiter character codes for dlmread are a RunMat extension",
        error_identifier: Some("RunMat:compatibility:DlmreadNumericDelimiterExtension"),
    };
pub const DLMREAD_RESIDENT_ARGUMENT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "dlmread-resident-argument",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "resident dlmread arguments are gathered as a RunMat extension",
        error_identifier: Some("RunMat:compatibility:DlmreadResidentArgumentExtension"),
    };
pub const DLMREAD_COMPOSED_RANGE_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "dlmread-composed-range",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "omitted-delimiter and offset-plus-range dlmread forms are RunMat extensions",
        error_identifier: Some("RunMat:compatibility:DlmreadComposedRangeExtension"),
    };
pub const DLMREAD_COLON_RANGE_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "dlmread-colon-spreadsheet-range",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "colon-delimited spreadsheet ranges for dlmread are a RunMat extension",
    error_identifier: Some("RunMat:compatibility:DlmreadColonSpreadsheetRangeExtension"),
};
pub const DLMREAD_EXTENSIONS: [BuiltinExtensionDescriptor; 4] = [
    DLMREAD_NUMERIC_DELIMITER_EXTENSION,
    DLMREAD_RESIDENT_ARGUMENT_EXTENSION,
    DLMREAD_COMPOSED_RANGE_EXTENSION,
    DLMREAD_COLON_RANGE_EXTENSION,
];

const DLMREAD_INTEGER_OFFSET_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "R1/R2",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The compatibility target documents all eight integer classes for zero-based row offsets.",
    },
    BuiltinIntegerInputCapability {
        name: "C1/C2",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The compatibility target documents all eight integer classes for zero-based column offsets.",
    },
];
pub const DLMREAD_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "M = dlmread(filename,delimiter,integer_offsets_or_range)",
        inputs: &DLMREAD_INTEGER_OFFSET_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Typed offsets and numeric range vectors are decoded from authoritative integer storage before host file parsing; output is always a host double matrix.",
    }];

const DLMREAD_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "M",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Numeric matrix parsed from the delimited text file.",
}];
const DLMREAD_INPUTS_FILENAME: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "filename",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Delimited text file path.",
}];
const DLMREAD_INPUTS_FILENAME_DELIMITER: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "filename",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Delimited text file path.",
    },
    BuiltinParamDescriptor {
        name: "delimiter",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Delimiter as string/char or numeric code.",
    },
];
const DLMREAD_INPUTS_FILENAME_ROW_COL: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "filename",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Delimited text file path.",
    },
    BuiltinParamDescriptor {
        name: "row",
        ty: BuiltinParamType::IntegerScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Zero-based starting row offset.",
    },
    BuiltinParamDescriptor {
        name: "col",
        ty: BuiltinParamType::IntegerScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Zero-based starting column offset.",
    },
];
const DLMREAD_INPUTS_FILENAME_DELIMITER_RANGE: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "filename",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Delimited text file path.",
    },
    BuiltinParamDescriptor {
        name: "delimiter",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Delimiter as string/char or numeric code.",
    },
    BuiltinParamDescriptor {
        name: "range",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Range as A1-style string or numeric range vector.",
    },
];
const DLMREAD_INPUTS_FILENAME_DELIMITER_ROW_COL: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "filename",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Delimited text file path.",
    },
    BuiltinParamDescriptor {
        name: "delimiter",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Delimiter as string/char or numeric code.",
    },
    BuiltinParamDescriptor {
        name: "row",
        ty: BuiltinParamType::IntegerScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Zero-based starting row offset.",
    },
    BuiltinParamDescriptor {
        name: "col",
        ty: BuiltinParamType::IntegerScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Zero-based starting column offset.",
    },
];
const DLMREAD_INPUTS_FILENAME_DELIMITER_ROW_COL_RANGE: [BuiltinParamDescriptor; 5] = [
    BuiltinParamDescriptor {
        name: "filename",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Delimited text file path.",
    },
    BuiltinParamDescriptor {
        name: "delimiter",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Delimiter as string/char or numeric code.",
    },
    BuiltinParamDescriptor {
        name: "row",
        ty: BuiltinParamType::IntegerScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Zero-based starting row offset.",
    },
    BuiltinParamDescriptor {
        name: "col",
        ty: BuiltinParamType::IntegerScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Zero-based starting column offset.",
    },
    BuiltinParamDescriptor {
        name: "range",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Range as A1-style string or numeric range vector.",
    },
];
const DLMREAD_SIGNATURES: [BuiltinSignatureDescriptor; 6] = [
    BuiltinSignatureDescriptor {
        label: "M = dlmread(filename)",
        inputs: &DLMREAD_INPUTS_FILENAME,
        outputs: &DLMREAD_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "M = dlmread(filename, delimiter)",
        inputs: &DLMREAD_INPUTS_FILENAME_DELIMITER,
        outputs: &DLMREAD_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "M = dlmread(filename, row, col)",
        inputs: &DLMREAD_INPUTS_FILENAME_ROW_COL,
        outputs: &DLMREAD_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "M = dlmread(filename, delimiter, range)",
        inputs: &DLMREAD_INPUTS_FILENAME_DELIMITER_RANGE,
        outputs: &DLMREAD_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "M = dlmread(filename, delimiter, row, col)",
        inputs: &DLMREAD_INPUTS_FILENAME_DELIMITER_ROW_COL,
        outputs: &DLMREAD_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "M = dlmread(filename, delimiter, row, col, range)",
        inputs: &DLMREAD_INPUTS_FILENAME_DELIMITER_ROW_COL_RANGE,
        outputs: &DLMREAD_OUTPUT,
    },
];
pub const DLMREAD_ERROR_ARG_CONFIG: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DLMREAD.ARG_CONFIG",
    identifier: None,
    when: "Argument list does not match supported dlmread call forms.",
    message: "dlmread: invalid argument configuration",
};
pub const DLMREAD_ERROR_DELIMITER: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DLMREAD.DELIMITER",
    identifier: None,
    when: "Delimiter value is invalid.",
    message: "dlmread: invalid delimiter",
};
pub const DLMREAD_ERROR_INDEX: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DLMREAD.INDEX",
    identifier: None,
    when: "Row/column indices are invalid.",
    message: "dlmread: invalid row/column index",
};
pub const DLMREAD_ERROR_RANGE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DLMREAD.RANGE",
    identifier: None,
    when: "Range specification is malformed or semantically invalid.",
    message: "dlmread: invalid range",
};
pub const DLMREAD_ERROR_FILENAME: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DLMREAD.FILENAME",
    identifier: None,
    when: "Filename argument is not a scalar string/char vector.",
    message: "dlmread: invalid filename input",
};
pub const DLMREAD_ERROR_FILENAME_EMPTY: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DLMREAD.FILENAME_EMPTY",
    identifier: None,
    when: "Filename resolves to an empty path.",
    message: "dlmread: filename must not be empty",
};
pub const DLMREAD_ERROR_IO_OPEN: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DLMREAD.IO_OPEN",
    identifier: None,
    when: "Input file cannot be opened.",
    message: "dlmread: unable to open file",
};
pub const DLMREAD_ERROR_IO_READ: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DLMREAD.IO_READ",
    identifier: None,
    when: "Input file cannot be read.",
    message: "dlmread: failed to read file",
};
pub const DLMREAD_ERROR_NON_NUMERIC_TOKEN: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DLMREAD.NON_NUMERIC_TOKEN",
    identifier: None,
    when: "A token cannot be parsed as numeric in the selected import region.",
    message: "dlmread: nonnumeric token encountered",
};
pub const DLMREAD_ERROR_TENSOR_BUILD: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DLMREAD.TENSOR_BUILD",
    identifier: None,
    when: "Internal tensor output materialization fails.",
    message: "dlmread: unable to construct output matrix",
};
pub const DLMREAD_ERRORS: [BuiltinErrorDescriptor; 10] = [
    DLMREAD_ERROR_ARG_CONFIG,
    DLMREAD_ERROR_DELIMITER,
    DLMREAD_ERROR_INDEX,
    DLMREAD_ERROR_RANGE,
    DLMREAD_ERROR_FILENAME,
    DLMREAD_ERROR_FILENAME_EMPTY,
    DLMREAD_ERROR_IO_OPEN,
    DLMREAD_ERROR_IO_READ,
    DLMREAD_ERROR_NON_NUMERIC_TOKEN,
    DLMREAD_ERROR_TENSOR_BUILD,
];
pub const DLMREAD_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &DLMREAD_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &DLMREAD_ERRORS,
};
