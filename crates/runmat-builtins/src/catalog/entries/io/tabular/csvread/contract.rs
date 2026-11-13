use crate::*;

pub const CSVREAD_COLON_RANGE_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "csvread-colon-range",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "csvread A1 ranges separated by ':' are a RunMat extension",
    error_identifier: Some("RunMat:compatibility:CsvreadColonRangeExtension"),
};
pub const CSVREAD_TWO_VECTOR_RANGE_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "csvread-two-vector-range",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "csvread two-element numeric ranges are a RunMat extension",
        error_identifier: Some("RunMat:compatibility:CsvreadTwoVectorRangeExtension"),
    };
pub const CSVREAD_RESIDENT_CONTROL_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "csvread-resident-control-inputs",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "csvread resident filename, offset, or range controls are a RunMat extension",
        error_identifier: Some("RunMat:compatibility:CsvreadResidentControlExtension"),
    };
pub const CSVREAD_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    CSVREAD_COLON_RANGE_EXTENSION,
    CSVREAD_TWO_VECTOR_RANGE_EXTENSION,
    CSVREAD_RESIDENT_CONTROL_EXTENSION,
];

const CSVREAD_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 3] = [
    BuiltinIntegerInputCapability { name: "row", classes: &crate::ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::Documented, scalar_double: BuiltinIntegerScalarDoubleRule::Allowed, notes: "All eight integer classes are documented for the zero-based row control and are validated from authoritative storage." },
    BuiltinIntegerInputCapability { name: "col", classes: &crate::ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::Documented, scalar_double: BuiltinIntegerScalarDoubleRule::Allowed, notes: "All eight integer classes are documented for the zero-based column control and are validated from authoritative storage." },
    BuiltinIntegerInputCapability { name: "range", classes: &crate::ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::Documented, scalar_double: BuiltinIntegerScalarDoubleRule::Allowed, notes: "The documented four-element numeric range accepts all eight integer classes and decodes every offset exactly." },
];
pub const CSVREAD_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] = [BuiltinIntegerCapabilityDescriptor { form: "M = csvread(filename, integer_row, integer_col, integer_range?)", inputs: &CSVREAD_INTEGER_INPUTS, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::Double, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::HostOnly, overload: BuiltinIntegerOverloadKind::StructuralParameter, notes: "Integer inputs control file selection only. Values are read exactly from authoritative integer storage; the CSV result remains double or complex double." }];

const CSVREAD_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "M",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Numeric matrix read from the CSV file.",
}];
const CSVREAD_INPUTS_FILENAME: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "filename",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "CSV file path.",
}];
const CSVREAD_INPUTS_FILENAME_ROW_COL: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "filename",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "CSV file path.",
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
const CSVREAD_INPUTS_FILENAME_ROW_COL_RANGE: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "filename",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "CSV file path.",
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
        description: "A1-style range string or numeric range vector.",
    },
];
const CSVREAD_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "M = csvread(filename)",
        inputs: &CSVREAD_INPUTS_FILENAME,
        outputs: &CSVREAD_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "M = csvread(filename, row, col)",
        inputs: &CSVREAD_INPUTS_FILENAME_ROW_COL,
        outputs: &CSVREAD_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "M = csvread(filename, row, col, range)",
        inputs: &CSVREAD_INPUTS_FILENAME_ROW_COL_RANGE,
        outputs: &CSVREAD_OUTPUT,
    },
];
pub const CSVREAD_ERROR_ARG_CONFIG: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CSVREAD.ARG_CONFIG",
    identifier: None,
    when: "Argument list does not match supported csvread call forms.",
    message: "csvread: invalid argument configuration",
};
pub const CSVREAD_ERROR_INDEX: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CSVREAD.INDEX",
    identifier: None,
    when: "Row/column offset arguments are invalid.",
    message: "csvread: invalid row/column index",
};
pub const CSVREAD_ERROR_RANGE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CSVREAD.RANGE",
    identifier: None,
    when: "Range argument is malformed or semantically invalid.",
    message: "csvread: invalid range",
};
pub const CSVREAD_ERROR_FILENAME: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CSVREAD.FILENAME",
    identifier: None,
    when: "Filename argument is not a scalar string/char vector.",
    message: "csvread: invalid filename input",
};
pub const CSVREAD_ERROR_FILENAME_EMPTY: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CSVREAD.FILENAME_EMPTY",
    identifier: None,
    when: "Filename resolves to an empty string.",
    message: "csvread: filename must not be empty",
};
pub const CSVREAD_ERROR_IO_OPEN: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CSVREAD.IO_OPEN",
    identifier: None,
    when: "Input file cannot be opened.",
    message: "csvread: unable to open file",
};
pub const CSVREAD_ERROR_IO_READ: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CSVREAD.IO_READ",
    identifier: None,
    when: "Input file cannot be read.",
    message: "csvread: failed to read file",
};
pub const CSVREAD_ERROR_NON_NUMERIC_TOKEN: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CSVREAD.NON_NUMERIC_TOKEN",
    identifier: None,
    when: "A CSV field cannot be parsed as numeric.",
    message: "csvread: nonnumeric token encountered",
};
pub const CSVREAD_ERROR_TENSOR_BUILD: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CSVREAD.TENSOR_BUILD",
    identifier: None,
    when: "Internal tensor materialization for csvread output fails.",
    message: "csvread: unable to construct output matrix",
};
const CSVREAD_ERRORS: [BuiltinErrorDescriptor; 9] = [
    CSVREAD_ERROR_ARG_CONFIG,
    CSVREAD_ERROR_INDEX,
    CSVREAD_ERROR_RANGE,
    CSVREAD_ERROR_FILENAME,
    CSVREAD_ERROR_FILENAME_EMPTY,
    CSVREAD_ERROR_IO_OPEN,
    CSVREAD_ERROR_IO_READ,
    CSVREAD_ERROR_NON_NUMERIC_TOKEN,
    CSVREAD_ERROR_TENSOR_BUILD,
];
pub const CSVREAD_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &CSVREAD_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &CSVREAD_ERRORS,
};
