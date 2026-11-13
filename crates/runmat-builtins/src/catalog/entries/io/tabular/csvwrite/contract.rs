use crate::*;

pub const CSVWRITE_BYTES_OUTPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "csvwrite-bytes-written-output",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "requesting a bytes-written output from csvwrite is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:CsvwriteBytesOutputExtension"),
    };
pub const CSVWRITE_RESIDENT_INPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "csvwrite-resident-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "direct csvwrite of resident data or controls is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:CsvwriteResidentInputExtension"),
    };
pub const CSVWRITE_EXTENSIONS: [BuiltinExtensionDescriptor; 2] = [
    CSVWRITE_BYTES_OUTPUT_EXTENSION,
    CSVWRITE_RESIDENT_INPUT_EXTENSION,
];
const CSVWRITE_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 3] = [
    BuiltinIntegerInputCapability { name: "M", classes: &crate::ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::Documented, scalar_double: BuiltinIntegerScalarDoubleRule::Allowed, notes: "All eight real integer classes are documented and serialize directly from authoritative storage." },
    BuiltinIntegerInputCapability { name: "row", classes: &crate::ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::Documented, scalar_double: BuiltinIntegerScalarDoubleRule::Allowed, notes: "The zero-based row offset accepts all eight integer classes and is checked exactly." },
    BuiltinIntegerInputCapability { name: "col", classes: &crate::ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::Documented, scalar_double: BuiltinIntegerScalarDoubleRule::Allowed, notes: "The zero-based column offset accepts all eight integer classes and is checked exactly." },
];
pub const CSVWRITE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] = [BuiltinIntegerCapabilityDescriptor { form: "csvwrite(filename, integer_M, integer_row?, integer_col?)", inputs: &CSVWRITE_INTEGER_INPUTS, computation_domain: BuiltinIntegerComputationDomain::ExactInteger, output_class: BuiltinIntegerOutputClassRule::FunctionSpecific, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::Multiple, notes: "Integer matrix elements serialize with their exact values. Resident input is independently gated and gathered through the handle owner." }];

const CSVWRITE_INPUTS_FILENAME_DATA: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "filename",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "CSV output path.",
    },
    BuiltinParamDescriptor {
        name: "M",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Numeric/logical matrix data to write.",
    },
];
const CSVWRITE_INPUTS_FILENAME_DATA_ROW_COL: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "filename",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "CSV output path.",
    },
    BuiltinParamDescriptor {
        name: "M",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Numeric/logical matrix data to write.",
    },
    BuiltinParamDescriptor {
        name: "row",
        ty: BuiltinParamType::IntegerScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Zero-based row offset before writing values.",
    },
    BuiltinParamDescriptor {
        name: "col",
        ty: BuiltinParamType::IntegerScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Zero-based column offset before writing values.",
    },
];
const CSVWRITE_NO_OUTPUT: [BuiltinParamDescriptor; 0] = [];
const CSVWRITE_BYTES_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "bytes",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Bytes written, available only with the RunMat output extension.",
}];
const CSVWRITE_SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "csvwrite(filename, M)",
        inputs: &CSVWRITE_INPUTS_FILENAME_DATA,
        outputs: &CSVWRITE_NO_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "csvwrite(filename, M, row, col)",
        inputs: &CSVWRITE_INPUTS_FILENAME_DATA_ROW_COL,
        outputs: &CSVWRITE_NO_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "bytes = csvwrite(filename, M)",
        inputs: &CSVWRITE_INPUTS_FILENAME_DATA,
        outputs: &CSVWRITE_BYTES_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "bytes = csvwrite(filename, M, row, col)",
        inputs: &CSVWRITE_INPUTS_FILENAME_DATA_ROW_COL,
        outputs: &CSVWRITE_BYTES_OUTPUT,
    },
];
pub const CSVWRITE_ERROR_FILENAME: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CSVWRITE.FILENAME",
    identifier: None,
    when: "Filename argument is not a scalar string/char vector.",
    message: "csvwrite: invalid filename input",
};
pub const CSVWRITE_ERROR_FILENAME_EMPTY: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CSVWRITE.FILENAME_EMPTY",
    identifier: None,
    when: "Filename resolves to an empty string.",
    message: "csvwrite: filename must not be empty",
};
pub const CSVWRITE_ERROR_OFFSETS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CSVWRITE.OFFSETS",
    identifier: None,
    when: "Offset arguments are missing, malformed, or out of bounds.",
    message: "csvwrite: invalid row/column offsets",
};
pub const CSVWRITE_ERROR_DATA_SHAPE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CSVWRITE.DATA_SHAPE",
    identifier: None,
    when: "Input data is not a 2-D matrix.",
    message: "csvwrite: input must be 2-D",
};
pub const CSVWRITE_ERROR_DATA_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CSVWRITE.DATA_INPUT",
    identifier: None,
    when: "Input data cannot be converted to a numeric/logical tensor.",
    message: "csvwrite: input must be numeric or logical",
};
pub const CSVWRITE_ERROR_IO_OPEN: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CSVWRITE.IO_OPEN",
    identifier: None,
    when: "Output file cannot be opened.",
    message: "csvwrite: unable to open file for writing",
};
pub const CSVWRITE_ERROR_IO_WRITE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CSVWRITE.IO_WRITE",
    identifier: None,
    when: "Output file write/flush fails.",
    message: "csvwrite: write failed",
};
const CSVWRITE_ERRORS: [BuiltinErrorDescriptor; 7] = [
    CSVWRITE_ERROR_FILENAME,
    CSVWRITE_ERROR_FILENAME_EMPTY,
    CSVWRITE_ERROR_OFFSETS,
    CSVWRITE_ERROR_DATA_INPUT,
    CSVWRITE_ERROR_DATA_SHAPE,
    CSVWRITE_ERROR_IO_OPEN,
    CSVWRITE_ERROR_IO_WRITE,
];
pub const CSVWRITE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &CSVWRITE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &CSVWRITE_ERRORS,
};
