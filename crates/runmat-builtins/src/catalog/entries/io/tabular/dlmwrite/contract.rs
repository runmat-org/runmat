use crate::*;

pub const DLMWRITE_RESIDENT_INPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "dlmwrite-resident-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "dlmwrite with resident gpuArray data is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:DlmwriteResidentInputExtension"),
    };
pub const DLMWRITE_BYTE_COUNT_OUTPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "dlmwrite-byte-count-output",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "requesting a dlmwrite byte-count output is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:DlmwriteByteCountOutputExtension"),
    };
pub const DLMWRITE_EXTENSIONS: [BuiltinExtensionDescriptor; 2] = [
    DLMWRITE_RESIDENT_INPUT_EXTENSION,
    DLMWRITE_BYTE_COUNT_OUTPUT_EXTENSION,
];
const DLMWRITE_INTEGER_DATA_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "M",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The compatibility target documents all eight integer matrix classes; default formatting serializes authoritative values exactly.",
    }];
const DLMWRITE_INTEGER_PRECISION_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "precision",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes:
            "The compatibility target documents all eight integer classes for the positive significant-digit count.",
    }];
const DLMWRITE_INTEGER_OFFSET_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "row/col/roffset/coffset",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Typed nonnegative scalar offsets are decoded exactly and must produce a checked, bounded materializable file layout.",
    }];
pub const DLMWRITE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "dlmwrite(filename,integer_M,___)",
        inputs: &DLMWRITE_INTEGER_DATA_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Without explicit precision, integer text is emitted directly from native storage, including uint64 values above flintmax.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "dlmwrite(___,precision=integer_N)",
        inputs: &DLMWRITE_INTEGER_PRECISION_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The precision count must fit a positive u32 and controls floating text formatting.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "dlmwrite(___,integer_offsets)",
        inputs: &DLMWRITE_INTEGER_OFFSET_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Offsets must fit usize and produce a checked layout of at most 50,000,000 fields; resident structural arguments gather before exact decoding.",
    },
];

const DLMWRITE_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "bytesWritten",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Number of bytes written to the output file.",
}];
const DLMWRITE_INPUTS_FILENAME_DATA: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "filename",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Output file path.",
    },
    BuiltinParamDescriptor {
        name: "M",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Numeric/logical matrix to write.",
    },
];
const DLMWRITE_INPUTS_FILENAME_DATA_DELIM_ROW_COL: [BuiltinParamDescriptor; 5] = [
    BuiltinParamDescriptor {
        name: "filename",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Output file path.",
    },
    BuiltinParamDescriptor {
        name: "M",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Numeric/logical matrix to write.",
    },
    BuiltinParamDescriptor {
        name: "delimiter",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("\",\""),
        description: "Delimiter token.",
    },
    BuiltinParamDescriptor {
        name: "row",
        ty: BuiltinParamType::IntegerScalar,
        arity: BuiltinParamArity::Required,
        default: Some("0"),
        description: "Row offset before writing matrix rows.",
    },
    BuiltinParamDescriptor {
        name: "col",
        ty: BuiltinParamType::IntegerScalar,
        arity: BuiltinParamArity::Required,
        default: Some("0"),
        description: "Column offset before writing matrix columns.",
    },
];
const DLMWRITE_INPUTS_FILENAME_DATA_NAME_VALUE: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "filename",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Output file path.",
    },
    BuiltinParamDescriptor {
        name: "M",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Numeric/logical matrix to write.",
    },
    BuiltinParamDescriptor {
        name: "name",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Option name.",
    },
    BuiltinParamDescriptor {
        name: "optionValue",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Option value.",
    },
];
const DLMWRITE_INPUTS_FILENAME_DATA_NAME_VALUE_PAIRS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "filename",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Output file path.",
    },
    BuiltinParamDescriptor {
        name: "M",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Numeric/logical matrix to write.",
    },
    BuiltinParamDescriptor {
        name: "args...",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Optional delimiter/offset/append and name-value options.",
    },
];
const DLMWRITE_NO_OUTPUT: [BuiltinParamDescriptor; 0] = [];
const DLMWRITE_SIGNATURES: [BuiltinSignatureDescriptor; 8] = [
    BuiltinSignatureDescriptor {
        label: "dlmwrite(filename, M)",
        inputs: &DLMWRITE_INPUTS_FILENAME_DATA,
        outputs: &DLMWRITE_NO_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "dlmwrite(filename, M, delimiter, row, col)",
        inputs: &DLMWRITE_INPUTS_FILENAME_DATA_DELIM_ROW_COL,
        outputs: &DLMWRITE_NO_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "dlmwrite(filename, M, name, optionValue)",
        inputs: &DLMWRITE_INPUTS_FILENAME_DATA_NAME_VALUE,
        outputs: &DLMWRITE_NO_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "dlmwrite(filename, M, args...)",
        inputs: &DLMWRITE_INPUTS_FILENAME_DATA_NAME_VALUE_PAIRS,
        outputs: &DLMWRITE_NO_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "bytesWritten = dlmwrite(filename, M)",
        inputs: &DLMWRITE_INPUTS_FILENAME_DATA,
        outputs: &DLMWRITE_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "bytesWritten = dlmwrite(filename, M, delimiter, row, col)",
        inputs: &DLMWRITE_INPUTS_FILENAME_DATA_DELIM_ROW_COL,
        outputs: &DLMWRITE_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "bytesWritten = dlmwrite(filename, M, name, optionValue)",
        inputs: &DLMWRITE_INPUTS_FILENAME_DATA_NAME_VALUE,
        outputs: &DLMWRITE_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "bytesWritten = dlmwrite(filename, M, args...)",
        inputs: &DLMWRITE_INPUTS_FILENAME_DATA_NAME_VALUE_PAIRS,
        outputs: &DLMWRITE_OUTPUT,
    },
];

pub const DLMWRITE_ERROR_ARG_CONFIG: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DLMWRITE.ARG_CONFIG",
    identifier: None,
    when: "Argument sequence or name-value grammar is malformed.",
    message: "dlmwrite: invalid argument configuration",
};
pub const DLMWRITE_ERROR_FILENAME: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DLMWRITE.FILENAME",
    identifier: None,
    when: "Filename is missing, empty, or not a scalar string/char vector.",
    message: "dlmwrite: invalid filename input",
};
pub const DLMWRITE_ERROR_OPTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DLMWRITE.OPTION",
    identifier: None,
    when: "Option name or value is invalid.",
    message: "dlmwrite: invalid option value",
};
pub const DLMWRITE_ERROR_DATA: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DLMWRITE.DATA",
    identifier: None,
    when: "Input data cannot be converted to a supported numeric/logical matrix.",
    message: "dlmwrite: invalid input data",
};
pub const DLMWRITE_ERROR_DATA_SHAPE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DLMWRITE.DATA_SHAPE",
    identifier: None,
    when: "Input matrix is not 2-D.",
    message: "dlmwrite: input must be 2-D",
};
pub const DLMWRITE_ERROR_FORMAT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DLMWRITE.FORMAT",
    identifier: None,
    when: "Numeric precision format is invalid or unsupported.",
    message: "dlmwrite: invalid precision format",
};
pub const DLMWRITE_ERROR_IO: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DLMWRITE.IO",
    identifier: None,
    when: "File inspection/open/write/flush operations fail.",
    message: "dlmwrite: file write failed",
};
pub const DLMWRITE_ERROR_SIZE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DLMWRITE.SIZE",
    identifier: None,
    when: "Row or column offsets overflow or request an impractically large file layout.",
    message: "dlmwrite: requested output layout is too large",
};
pub const DLMWRITE_ERRORS: [BuiltinErrorDescriptor; 8] = [
    DLMWRITE_ERROR_ARG_CONFIG,
    DLMWRITE_ERROR_FILENAME,
    DLMWRITE_ERROR_OPTION,
    DLMWRITE_ERROR_DATA,
    DLMWRITE_ERROR_DATA_SHAPE,
    DLMWRITE_ERROR_FORMAT,
    DLMWRITE_ERROR_IO,
    DLMWRITE_ERROR_SIZE,
];
pub const DLMWRITE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &DLMWRITE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &DLMWRITE_ERRORS,
};
