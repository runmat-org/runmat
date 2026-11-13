use crate::*;

pub const WRITEMATRIX_BYTES_OUTPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "writematrix-bytes-written-output",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "Request a bytes-written output from writematrix",
        error_identifier: Some("RunMat:compatibility:WritematrixBytesOutputExtension"),
    };
pub const WRITEMATRIX_EXPLICIT_GPU_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "writematrix-explicit-gpu-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "Pass explicit gpuArray input to writematrix",
        error_identifier: Some("RunMat:compatibility:WritematrixExplicitGpuInputExtension"),
    };
pub const WRITEMATRIX_QUOTE_ALIAS_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "writematrix-quote-strings-logical-alias",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "Use logical or numeric aliases for the QuoteStrings option",
        error_identifier: Some("RunMat:compatibility:WritematrixQuoteStringsAliasExtension"),
    };
pub const WRITEMATRIX_TEXT_OPTIONS_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "writematrix-text-format-options",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "Use the DecimalSeparator or LineEnding text-format option",
        error_identifier: Some("RunMat:compatibility:WritematrixTextFormatOptionsExtension"),
    };
pub const WRITEMATRIX_EXTENSIONS: [BuiltinExtensionDescriptor; 4] = [
    WRITEMATRIX_BYTES_OUTPUT_EXTENSION,
    WRITEMATRIX_EXPLICIT_GPU_EXTENSION,
    WRITEMATRIX_QUOTE_ALIAS_EXTENSION,
    WRITEMATRIX_TEXT_OPTIONS_EXTENSION,
];
pub const WRITEMATRIX_INTEGER_DATA_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "A",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Numeric matrices from all eight integer classes serialize from authoritative integer storage.",
    }];
pub const WRITEMATRIX_INTEGER_QUOTE_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "QuoteStrings logical alias",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "RunMat mode accepts exact integer zero and one as aliases for the text-valued QuoteStrings option.",
    }];
pub const WRITEMATRIX_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor { form: "writematrix(integer_A, filename, ___)", inputs: &WRITEMATRIX_INTEGER_DATA_INPUT, computation_domain: BuiltinIntegerComputationDomain::ExactInteger, output_class: BuiltinIntegerOutputClassRule::NotApplicable, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::FunctionSpecific, notes: "Every integer element is rendered from its exact signed or unsigned decimal value. Automatically resident data gathers transparently; explicit device input is separately gated." },
    BuiltinIntegerCapabilityDescriptor { form: "writematrix(A, filename, 'QuoteStrings', integer_alias)", inputs: &WRITEMATRIX_INTEGER_QUOTE_INPUT, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::NotApplicable, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::StructuralParameter, notes: "Strict mode accepts the documented text values; RunMat mode additionally maps exact zero to none and exact one to all." },
];

pub const WRITEMATRIX_NO_OUTPUT: [BuiltinParamDescriptor; 0] = [];
const WRITEMATRIX_BYTES_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "bytes",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "RunMat-only bytes-written output.",
}];
pub const WRITEMATRIX_INPUTS_DATA_FILENAME: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "data",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Numeric or string matrix to write.",
    },
    BuiltinParamDescriptor {
        name: "filename",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Output file path.",
    },
];
pub const WRITEMATRIX_INPUTS_DATA_FILENAME_NAME_VALUE: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "data",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Numeric or string matrix to write.",
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
pub const WRITEMATRIX_INPUTS_DATA_FILENAME_NAME_VALUE_PAIRS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "data",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Numeric or string matrix to write.",
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
pub const WRITEMATRIX_SIGNATURES: [BuiltinSignatureDescriptor; 6] = [
    BuiltinSignatureDescriptor {
        label: "writematrix(data, filename)",
        inputs: &WRITEMATRIX_INPUTS_DATA_FILENAME,
        outputs: &WRITEMATRIX_NO_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "writematrix(data, filename, name, optionValue)",
        inputs: &WRITEMATRIX_INPUTS_DATA_FILENAME_NAME_VALUE,
        outputs: &WRITEMATRIX_NO_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "writematrix(data, filename, nameValuePairs...)",
        inputs: &WRITEMATRIX_INPUTS_DATA_FILENAME_NAME_VALUE_PAIRS,
        outputs: &WRITEMATRIX_NO_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "bytes = writematrix(data, filename)",
        inputs: &WRITEMATRIX_INPUTS_DATA_FILENAME,
        outputs: &WRITEMATRIX_BYTES_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "bytes = writematrix(data, filename, name, optionValue)",
        inputs: &WRITEMATRIX_INPUTS_DATA_FILENAME_NAME_VALUE,
        outputs: &WRITEMATRIX_BYTES_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "bytes = writematrix(data, filename, nameValuePairs...)",
        inputs: &WRITEMATRIX_INPUTS_DATA_FILENAME_NAME_VALUE_PAIRS,
        outputs: &WRITEMATRIX_BYTES_OUTPUT,
    },
];
pub const WRITEMATRIX_ERROR_ARG_CONFIG: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.WRITEMATRIX.ARG_CONFIG",
    identifier: None,
    when: "Filename argument is missing or name-value options are malformed.",
    message: "writematrix: invalid argument configuration",
};
pub const WRITEMATRIX_ERROR_FILENAME: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.WRITEMATRIX.FILENAME",
    identifier: None,
    when: "Filename is not a valid scalar path string.",
    message: "writematrix: invalid filename input",
};
pub const WRITEMATRIX_ERROR_OPTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.WRITEMATRIX.OPTION",
    identifier: None,
    when: "A provided option value is invalid.",
    message: "writematrix: invalid option value",
};
pub const WRITEMATRIX_ERROR_DATA: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.WRITEMATRIX.DATA",
    identifier: None,
    when: "Input data cannot be converted into a supported writematrix matrix form.",
    message: "writematrix: invalid input data",
};
pub const WRITEMATRIX_ERROR_DATA_SHAPE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.WRITEMATRIX.DATA_SHAPE",
    identifier: None,
    when: "Input data has unsupported dimensionality.",
    message: "writematrix: input must be 2-D",
};
pub const WRITEMATRIX_ERROR_IO: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.WRITEMATRIX.IO",
    identifier: None,
    when: "The destination file cannot be opened or written.",
    message: "writematrix: file write failed",
};
pub const WRITEMATRIX_ERRORS: [BuiltinErrorDescriptor; 6] = [
    WRITEMATRIX_ERROR_ARG_CONFIG,
    WRITEMATRIX_ERROR_FILENAME,
    WRITEMATRIX_ERROR_OPTION,
    WRITEMATRIX_ERROR_DATA,
    WRITEMATRIX_ERROR_DATA_SHAPE,
    WRITEMATRIX_ERROR_IO,
];
pub const WRITEMATRIX_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &WRITEMATRIX_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &WRITEMATRIX_ERRORS,
};
