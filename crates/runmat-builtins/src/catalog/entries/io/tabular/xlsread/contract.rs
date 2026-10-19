use crate::*;

pub const XLSREAD_NUMERIC_RANGE_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "xlsread-numeric-range",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "Use a numeric row and column vector as an xlsread range",
        error_identifier: Some("RunMat:compatibility:XlsreadNumericRangeExtension"),
    };
pub const XLSREAD_EXPLICIT_GPU_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "xlsread-explicit-gpu-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "Pass explicit gpuArray selectors to xlsread",
    error_identifier: Some("RunMat:compatibility:XlsreadExplicitGpuInputExtension"),
};
pub const XLSREAD_EXTENSIONS: [BuiltinExtensionDescriptor; 2] = [
    XLSREAD_NUMERIC_RANGE_EXTENSION,
    XLSREAD_EXPLICIT_GPU_EXTENSION,
];
const XLSREAD_INTEGER_SHEET_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "sheet",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "All eight integer classes are documented for the one-based worksheet selector and are decoded exactly.",
    }];
const XLSREAD_INTEGER_RANGE_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "numeric range",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "RunMat mode accepts an integer row and column vector in place of the documented textual range.",
    }];
pub const XLSREAD_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor { form: "num = xlsread(filename, integer_sheet, ___)", inputs: &XLSREAD_INTEGER_SHEET_INPUT, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::Double, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::HostOnly, overload: BuiltinIntegerOverloadKind::StructuralParameter, notes: "The exact positive selector chooses a worksheet; numeric spreadsheet output remains double." },
    BuiltinIntegerCapabilityDescriptor { form: "num = xlsread(filename, sheet, integer_range)", inputs: &XLSREAD_INTEGER_RANGE_INPUT, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::Double, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::StructuralParameter, notes: "Strict mode rejects the numeric-range extension before file access; RunMat mode parses the native integer vector exactly." },
];

const XLSREAD_OUTPUT_NUM: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "num",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Numeric worksheet data.",
};
const XLSREAD_OUTPUT_TXT: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "txt",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Text worksheet cells.",
};
const XLSREAD_OUTPUT_RAW: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "raw",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Raw worksheet cells.",
};
const XLSREAD_OUTPUTS_NUM: [BuiltinParamDescriptor; 1] = [XLSREAD_OUTPUT_NUM];
const XLSREAD_OUTPUTS_ALL: [BuiltinParamDescriptor; 3] =
    [XLSREAD_OUTPUT_NUM, XLSREAD_OUTPUT_TXT, XLSREAD_OUTPUT_RAW];

const XLSREAD_INPUTS_FILENAME: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "filename",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Spreadsheet file path.",
}];
const XLSREAD_INPUTS_FILENAME_SELECTOR: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "filename",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Spreadsheet file path.",
    },
    BuiltinParamDescriptor {
        name: "sheetOrRange",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Worksheet name/index or Excel range.",
    },
];
const XLSREAD_INPUTS_FILENAME_SHEET_RANGE: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "filename",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Spreadsheet file path.",
    },
    BuiltinParamDescriptor {
        name: "sheet",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Worksheet name or one-based index.",
    },
    BuiltinParamDescriptor {
        name: "range",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Excel A1 range.",
    },
];
const XLSREAD_INPUTS_FILENAME_SHEET_RANGE_BASIC: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "filename",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Spreadsheet file path.",
    },
    BuiltinParamDescriptor {
        name: "sheet",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Worksheet name or one-based index.",
    },
    BuiltinParamDescriptor {
        name: "range",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Excel A1 range.",
    },
    BuiltinParamDescriptor {
        name: "mode",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"basic\""),
        description: "Legacy '-basic' compatibility flag.",
    },
];

const XLSREAD_SIGNATURES: [BuiltinSignatureDescriptor; 5] = [
    BuiltinSignatureDescriptor {
        label: "num = xlsread(filename)",
        inputs: &XLSREAD_INPUTS_FILENAME,
        outputs: &XLSREAD_OUTPUTS_NUM,
    },
    BuiltinSignatureDescriptor {
        label: "num = xlsread(filename, sheetOrRange)",
        inputs: &XLSREAD_INPUTS_FILENAME_SELECTOR,
        outputs: &XLSREAD_OUTPUTS_NUM,
    },
    BuiltinSignatureDescriptor {
        label: "num = xlsread(filename, sheet, range)",
        inputs: &XLSREAD_INPUTS_FILENAME_SHEET_RANGE,
        outputs: &XLSREAD_OUTPUTS_NUM,
    },
    BuiltinSignatureDescriptor {
        label: "num = xlsread(filename, sheet, range, '-basic')",
        inputs: &XLSREAD_INPUTS_FILENAME_SHEET_RANGE_BASIC,
        outputs: &XLSREAD_OUTPUTS_NUM,
    },
    BuiltinSignatureDescriptor {
        label: "[num, txt, raw] = xlsread(___)",
        inputs: &XLSREAD_INPUTS_FILENAME_SHEET_RANGE_BASIC,
        outputs: &XLSREAD_OUTPUTS_ALL,
    },
];

pub const XLSREAD_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.XLSREAD.INVALID_ARGUMENT",
    identifier: Some("RunMat:xlsread:InvalidArgument"),
    when: "Argument list does not match supported xlsread call forms.",
    message: "xlsread: invalid argument",
};
pub const XLSREAD_ERROR_FILENAME: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.XLSREAD.FILENAME",
    identifier: Some("RunMat:xlsread:Filename"),
    when: "Filename is invalid or cannot be normalized.",
    message: "xlsread: invalid filename",
};
pub const XLSREAD_ERROR_RANGE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.XLSREAD.RANGE",
    identifier: Some("RunMat:xlsread:Range"),
    when: "Range specification is malformed or semantically invalid.",
    message: "xlsread: invalid range",
};
pub const XLSREAD_ERROR_SHEET: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.XLSREAD.SHEET",
    identifier: Some("RunMat:xlsread:Sheet"),
    when: "Requested worksheet is missing or invalid.",
    message: "xlsread: invalid sheet",
};
pub const XLSREAD_ERROR_IO: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.XLSREAD.IO",
    identifier: Some("RunMat:xlsread:Io"),
    when: "Spreadsheet cannot be opened or read.",
    message: "xlsread: unable to read spreadsheet",
};
pub const XLSREAD_ERROR_OUTPUT_COUNT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.XLSREAD.OUTPUT_COUNT",
    identifier: Some("RunMat:xlsread:OutputCount"),
    when: "Caller requests more outputs than xlsread supports.",
    message: "xlsread: unsupported output count",
};
pub const XLSREAD_ERROR_TENSOR_BUILD: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.XLSREAD.TENSOR_BUILD",
    identifier: Some("RunMat:xlsread:TensorBuild"),
    when: "Output arrays cannot be materialized.",
    message: "xlsread: unable to construct output",
};
const XLSREAD_ERRORS: [BuiltinErrorDescriptor; 7] = [
    XLSREAD_ERROR_INVALID_ARGUMENT,
    XLSREAD_ERROR_FILENAME,
    XLSREAD_ERROR_RANGE,
    XLSREAD_ERROR_SHEET,
    XLSREAD_ERROR_IO,
    XLSREAD_ERROR_OUTPUT_COUNT,
    XLSREAD_ERROR_TENSOR_BUILD,
];

pub const XLSREAD_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &XLSREAD_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &XLSREAD_ERRORS,
};
