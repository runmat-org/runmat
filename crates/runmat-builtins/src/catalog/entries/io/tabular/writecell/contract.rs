use crate::*;

const WRITECELL_NO_OUTPUT: [BuiltinParamDescriptor; 0] = [];
const WRITECELL_BYTES_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "bytes",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "RunMat-only bytes-written output.",
}];
const WRITECELL_INPUTS_CELL_FILENAME: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "C",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Cell array to write.",
    },
    BuiltinParamDescriptor {
        name: "filename",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Output file path.",
    },
];
const WRITECELL_INPUTS_NAME_VALUE: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "C",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Cell array to write.",
    },
    BuiltinParamDescriptor {
        name: "filename",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Output file path.",
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
        description: "Value for the preceding option name.",
    },
];
const WRITECELL_INPUTS_NAME_VALUE_PAIRS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "C",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Cell array to write.",
    },
    BuiltinParamDescriptor {
        name: "filename",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Output file path.",
    },
    BuiltinParamDescriptor {
        name: "nameValuePairs...",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Name-value option pairs.",
    },
];
const WRITECELL_SIGNATURES: [BuiltinSignatureDescriptor; 6] = [
    BuiltinSignatureDescriptor {
        label: "writecell(C, filename)",
        inputs: &WRITECELL_INPUTS_CELL_FILENAME,
        outputs: &WRITECELL_NO_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "writecell(C, filename, name, optionValue)",
        inputs: &WRITECELL_INPUTS_NAME_VALUE,
        outputs: &WRITECELL_NO_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "writecell(C, filename, nameValuePairs...)",
        inputs: &WRITECELL_INPUTS_NAME_VALUE_PAIRS,
        outputs: &WRITECELL_NO_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "bytes = writecell(C, filename)",
        inputs: &WRITECELL_INPUTS_CELL_FILENAME,
        outputs: &WRITECELL_BYTES_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "bytes = writecell(C, filename, name, optionValue)",
        inputs: &WRITECELL_INPUTS_NAME_VALUE,
        outputs: &WRITECELL_BYTES_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "bytes = writecell(C, filename, nameValuePairs...)",
        inputs: &WRITECELL_INPUTS_NAME_VALUE_PAIRS,
        outputs: &WRITECELL_BYTES_OUTPUT,
    },
];

pub const WRITECELL_ERROR_ARG_CONFIG: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.WRITECELL.ARG_CONFIG",
    identifier: None,
    when: "Filename argument is missing or name-value options are malformed.",
    message: "writecell: invalid argument configuration",
};
pub const WRITECELL_ERROR_FILENAME: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.WRITECELL.FILENAME",
    identifier: None,
    when: "Filename is not a valid scalar path string.",
    message: "writecell: invalid filename input",
};
pub const WRITECELL_ERROR_OPTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.WRITECELL.OPTION",
    identifier: None,
    when: "A provided option value is invalid.",
    message: "writecell: invalid option value",
};
pub const WRITECELL_ERROR_DATA: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.WRITECELL.DATA",
    identifier: None,
    when: "Input data cannot be converted into supported cell export rows.",
    message: "writecell: invalid input data",
};
pub const WRITECELL_ERROR_DATA_SHAPE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.WRITECELL.DATA_SHAPE",
    identifier: None,
    when: "Input cell array has unsupported dimensionality.",
    message: "writecell: input must be 2-D",
};
pub const WRITECELL_ERROR_IO: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.WRITECELL.IO",
    identifier: None,
    when: "The destination file cannot be opened or written.",
    message: "writecell: file write failed",
};
const WRITECELL_ERRORS: [BuiltinErrorDescriptor; 6] = [
    WRITECELL_ERROR_ARG_CONFIG,
    WRITECELL_ERROR_FILENAME,
    WRITECELL_ERROR_OPTION,
    WRITECELL_ERROR_DATA,
    WRITECELL_ERROR_DATA_SHAPE,
    WRITECELL_ERROR_IO,
];

pub const WRITECELL_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &WRITECELL_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &WRITECELL_ERRORS,
};

pub const WRITECELL_EXPLICIT_GPU_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "writecell-explicit-gpu-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "writecell with explicit gpuArray input is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:WritecellExplicitGpuInputExtension"),
    };
pub const WRITECELL_BYTES_OUTPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "writecell-bytes-written-output",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "Request a bytes-written output from writecell",
        error_identifier: Some("RunMat:compatibility:WritecellBytesOutputExtension"),
    };
pub const WRITECELL_EXTENSIONS: [BuiltinExtensionDescriptor; 2] = [
    WRITECELL_EXPLICIT_GPU_EXTENSION,
    WRITECELL_BYTES_OUTPUT_EXTENSION,
];

const WRITECELL_INTEGER_CONTENT_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "integer scalar cell contents",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Cell arrays may contain scalar values from every native integer class. RunMat reads authoritative integer storage when formatting delimited text or spreadsheet cells.",
    }];
const WRITECELL_INTEGER_SHEET_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "Sheet",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "A numeric sheet selector may use any native integer class. RunMat decodes it exactly as a positive one-based structural index.",
    }];
pub const WRITECELL_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "writecell(cell_with_integer_scalars, filename, ___)",
        inputs: &WRITECELL_INTEGER_CONTENT_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Integer cells retain their decimal value through text and spreadsheet serialization without an intermediate floating conversion.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "writecell(C, filename, 'Sheet', integer_index)",
        inputs: &WRITECELL_INTEGER_SHEET_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "The exact selector is range-checked before conversion to the host index used by the spreadsheet writer.",
    },
];
