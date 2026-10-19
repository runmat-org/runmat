use crate::*;

pub const XLSWRITE_NUMERIC_RANGE_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "xlswrite-numeric-range",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "Use a numeric row and column vector as an xlswrite range",
        error_identifier: Some("RunMat:compatibility:XlswriteNumericRangeExtension"),
    };
pub const XLSWRITE_EXPLICIT_GPU_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "xlswrite-explicit-gpu-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "Pass explicit gpuArray input to xlswrite",
        error_identifier: Some("RunMat:compatibility:XlswriteExplicitGpuInputExtension"),
    };
pub const XLSWRITE_EXTENSIONS: [BuiltinExtensionDescriptor; 2] = [
    XLSWRITE_NUMERIC_RANGE_EXTENSION,
    XLSWRITE_EXPLICIT_GPU_EXTENSION,
];
const XLSWRITE_INTEGER_DATA_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "A",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "All eight integer classes are documented for spreadsheet data and serialize from authoritative storage.",
    }];
const XLSWRITE_INTEGER_SHEET_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "sheet",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "All eight integer classes are documented for the one-based worksheet selector and are decoded exactly.",
    }];
const XLSWRITE_INTEGER_RANGE_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "numeric range",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "RunMat mode accepts an integer row and column vector in place of the documented textual range.",
    }];
pub const XLSWRITE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    BuiltinIntegerCapabilityDescriptor { form: "status = xlswrite(filename, integer_A, ___)", inputs: &XLSWRITE_INTEGER_DATA_INPUT, computation_domain: BuiltinIntegerComputationDomain::ExactInteger, output_class: BuiltinIntegerOutputClassRule::FunctionSpecific, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::FunctionSpecific, notes: "Dense, sparse, and scalar cell integer values retain their exact signed or unsigned values through spreadsheet serialization." },
    BuiltinIntegerCapabilityDescriptor { form: "status = xlswrite(filename, A, integer_sheet, ___)", inputs: &XLSWRITE_INTEGER_SHEET_INPUT, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::FunctionSpecific, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::HostOnly, overload: BuiltinIntegerOverloadKind::StructuralParameter, notes: "The exact positive selector chooses the worksheet without a floating conversion." },
    BuiltinIntegerCapabilityDescriptor { form: "status = xlswrite(filename, A, sheet, integer_range)", inputs: &XLSWRITE_INTEGER_RANGE_INPUT, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::FunctionSpecific, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::StructuralParameter, notes: "Strict mode rejects the numeric-range extension before file access; RunMat mode parses the native integer vector exactly." },
];

const XLSWRITE_OUTPUT_STATUS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "status",
    ty: BuiltinParamType::LogicalArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "True when the spreadsheet was written successfully.",
};
const XLSWRITE_OUTPUT_MESSAGE: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "message",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Diagnostic struct with message and identifier fields.",
};
const XLSWRITE_OUTPUTS_STATUS: [BuiltinParamDescriptor; 1] = [XLSWRITE_OUTPUT_STATUS];
const XLSWRITE_OUTPUTS_STATUS_MESSAGE: [BuiltinParamDescriptor; 2] =
    [XLSWRITE_OUTPUT_STATUS, XLSWRITE_OUTPUT_MESSAGE];

const XLSWRITE_INPUTS_FILENAME_A: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "filename",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Spreadsheet file path.",
    },
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Numeric, logical, string, character, or cell data to write.",
    },
];
const XLSWRITE_INPUTS_FILENAME_A_SELECTOR: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "filename",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Spreadsheet file path.",
    },
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Numeric, logical, string, character, or cell data to write.",
    },
    BuiltinParamDescriptor {
        name: "sheetOrRange",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Worksheet name/index or Excel A1 range start.",
    },
];
const XLSWRITE_INPUTS_FILENAME_A_SHEET_RANGE: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "filename",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Spreadsheet file path.",
    },
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Numeric, logical, string, character, or cell data to write.",
    },
    BuiltinParamDescriptor {
        name: "sheet",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Worksheet name or one-based worksheet index.",
    },
    BuiltinParamDescriptor {
        name: "range",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Excel A1 range start or range.",
    },
];
const XLSWRITE_INPUTS_FILENAME_A_SHEET_RANGE_BASIC: [BuiltinParamDescriptor; 5] = [
    BuiltinParamDescriptor {
        name: "filename",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Spreadsheet file path.",
    },
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Numeric, logical, string, character, or cell data to write.",
    },
    BuiltinParamDescriptor {
        name: "sheet",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Worksheet name or one-based worksheet index.",
    },
    BuiltinParamDescriptor {
        name: "range",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Excel A1 range start or range.",
    },
    BuiltinParamDescriptor {
        name: "mode",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"basic\""),
        description: "Legacy '-basic' compatibility flag.",
    },
];

const XLSWRITE_SIGNATURES: [BuiltinSignatureDescriptor; 8] = [
    BuiltinSignatureDescriptor {
        label: "status = xlswrite(filename, A)",
        inputs: &XLSWRITE_INPUTS_FILENAME_A,
        outputs: &XLSWRITE_OUTPUTS_STATUS,
    },
    BuiltinSignatureDescriptor {
        label: "status = xlswrite(filename, A, sheetOrRange)",
        inputs: &XLSWRITE_INPUTS_FILENAME_A_SELECTOR,
        outputs: &XLSWRITE_OUTPUTS_STATUS,
    },
    BuiltinSignatureDescriptor {
        label: "status = xlswrite(filename, A, sheet, range)",
        inputs: &XLSWRITE_INPUTS_FILENAME_A_SHEET_RANGE,
        outputs: &XLSWRITE_OUTPUTS_STATUS,
    },
    BuiltinSignatureDescriptor {
        label: "status = xlswrite(filename, A, sheet, range, '-basic')",
        inputs: &XLSWRITE_INPUTS_FILENAME_A_SHEET_RANGE_BASIC,
        outputs: &XLSWRITE_OUTPUTS_STATUS,
    },
    BuiltinSignatureDescriptor {
        label: "[status,message] = xlswrite(filename, A)",
        inputs: &XLSWRITE_INPUTS_FILENAME_A,
        outputs: &XLSWRITE_OUTPUTS_STATUS_MESSAGE,
    },
    BuiltinSignatureDescriptor {
        label: "[status,message] = xlswrite(filename, A, sheetOrRange)",
        inputs: &XLSWRITE_INPUTS_FILENAME_A_SELECTOR,
        outputs: &XLSWRITE_OUTPUTS_STATUS_MESSAGE,
    },
    BuiltinSignatureDescriptor {
        label: "[status,message] = xlswrite(filename, A, sheet, range)",
        inputs: &XLSWRITE_INPUTS_FILENAME_A_SHEET_RANGE,
        outputs: &XLSWRITE_OUTPUTS_STATUS_MESSAGE,
    },
    BuiltinSignatureDescriptor {
        label: "[status,message] = xlswrite(filename, A, sheet, range, '-basic')",
        inputs: &XLSWRITE_INPUTS_FILENAME_A_SHEET_RANGE_BASIC,
        outputs: &XLSWRITE_OUTPUTS_STATUS_MESSAGE,
    },
];

pub const XLSWRITE_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.XLSWRITE.INVALID_ARGUMENT",
    identifier: Some("RunMat:xlswrite:InvalidArgument"),
    when: "Argument list does not match supported xlswrite call forms.",
    message: "xlswrite: invalid argument",
};
pub const XLSWRITE_ERROR_FILENAME: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.XLSWRITE.FILENAME",
    identifier: Some("RunMat:xlswrite:Filename"),
    when: "Filename is invalid or cannot be normalized.",
    message: "xlswrite: invalid filename",
};
pub const XLSWRITE_ERROR_RANGE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.XLSWRITE.RANGE",
    identifier: Some("RunMat:xlswrite:Range"),
    when: "Range specification is malformed or exceeds worksheet limits.",
    message: "xlswrite: invalid range",
};
pub const XLSWRITE_ERROR_SHEET: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.XLSWRITE.SHEET",
    identifier: Some("RunMat:xlswrite:Sheet"),
    when: "Worksheet selector is invalid.",
    message: "xlswrite: invalid sheet",
};
pub const XLSWRITE_ERROR_DATA: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.XLSWRITE.DATA",
    identifier: Some("RunMat:xlswrite:Data"),
    when: "Input data cannot be represented as supported spreadsheet cells.",
    message: "xlswrite: invalid data",
};
pub const XLSWRITE_ERROR_IO: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.XLSWRITE.IO",
    identifier: Some("RunMat:xlswrite:Io"),
    when: "Spreadsheet cannot be written.",
    message: "xlswrite: unable to write spreadsheet",
};
pub const XLSWRITE_ERROR_OUTPUT_COUNT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.XLSWRITE.OUTPUT_COUNT",
    identifier: Some("RunMat:xlswrite:OutputCount"),
    when: "Caller requests more outputs than xlswrite supports.",
    message: "xlswrite: unsupported output count",
};
const XLSWRITE_ERRORS: [BuiltinErrorDescriptor; 7] = [
    XLSWRITE_ERROR_INVALID_ARGUMENT,
    XLSWRITE_ERROR_FILENAME,
    XLSWRITE_ERROR_RANGE,
    XLSWRITE_ERROR_SHEET,
    XLSWRITE_ERROR_DATA,
    XLSWRITE_ERROR_IO,
    XLSWRITE_ERROR_OUTPUT_COUNT,
];

pub const XLSWRITE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &XLSWRITE_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &XLSWRITE_ERRORS,
};
