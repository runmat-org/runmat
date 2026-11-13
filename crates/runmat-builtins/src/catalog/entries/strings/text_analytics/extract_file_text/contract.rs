use crate::*;

pub const EXTRACT_FILE_TEXT_RESIDENT_PAGES_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "extractfiletext-resident-pages",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "extractFileText with a resident Pages control is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:ExtractFileTextResidentPagesExtension"),
    };

pub const EXTRACT_FILE_TEXT_BROAD_FILE_CELL_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "extractfiletext-broad-file-cell",
        mode: BuiltinExtensionMode::RunMatOnly,
        description:
            "extractFileText with a string-valued or nested filename cell is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:ExtractFileTextBroadFileCellExtension"),
    };

pub const EXTRACT_FILE_TEXT_EXTENSIONS: [BuiltinExtensionDescriptor; 2] = [
    EXTRACT_FILE_TEXT_RESIDENT_PAGES_EXTENSION,
    EXTRACT_FILE_TEXT_BROAD_FILE_CELL_EXTENSION,
];

const INTEGER_PAGES_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "Pages",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Pages is a nonempty vector of positive page numbers and accepts every built-in integer class.",
    }];

const INTEGER_FILENAME_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "filename",
        classes: &[],
        availability: BuiltinIntegerInputAvailability::Rejected,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes:
            "The filename or URL is text; integer data rejects before file IO or provider access.",
    }];

pub const EXTRACT_FILE_TEXT_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "str = extractFileText(filename, 'Pages', integer_pages)",
        inputs: &INTEGER_PAGES_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Pages values are parsed exactly without floating materialization. PDF extraction itself remains an explicitly unsupported format in RunMat.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "str = extractFileText(integer_filename)",
        inputs: &INTEGER_FILENAME_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "Integer filenames are outside the public text domain and reject without conversion.",
    },
];

const OUT_TEXT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "str",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Extracted text.",
}];

const IN_FILE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "filename",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Path to a text, HTML, or DOCX file.",
}];

const IN_FILE_REST: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "filename",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Path to a text, HTML, or DOCX file.",
    },
    BuiltinParamDescriptor {
        name: "NameValue",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Name-value options: Encoding, ExtractionMethod, Password, and Pages.",
    },
];

pub const EXTRACT_FILE_TEXT_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.EXTRACT_FILE_TEXT.INVALID_INPUT",
    identifier: Some("RunMat:extractFileText:InvalidInput"),
    when: "Inputs do not match a supported extractFileText form.",
    message: "extractFileText: invalid input",
};

pub const EXTRACT_FILE_TEXT_ERROR_IO: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.EXTRACT_FILE_TEXT.IO",
    identifier: Some("RunMat:extractFileText:IOError"),
    when: "The requested file cannot be read.",
    message: "extractFileText: file read failed",
};

pub const EXTRACT_FILE_TEXT_ERROR_UNSUPPORTED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.EXTRACT_FILE_TEXT.UNSUPPORTED",
    identifier: Some("RunMat:extractFileText:UnsupportedFormat"),
    when: "The requested file type or option requires unsupported extraction infrastructure.",
    message: "extractFileText: unsupported file type or option",
};

const ERRORS: [BuiltinErrorDescriptor; 4] = [
    EXTRACT_FILE_TEXT_ERROR_INVALID_INPUT,
    EXTRACT_FILE_TEXT_ERROR_IO,
    EXTRACT_FILE_TEXT_ERROR_UNSUPPORTED,
    EXTRACT_FILE_TEXT_ERROR_TOO_MANY_OUTPUTS,
];

pub const EXTRACT_FILE_TEXT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[
        BuiltinSignatureDescriptor {
            label: "str = extractFileText(filename)",
            inputs: &IN_FILE,
            outputs: &OUT_TEXT,
        },
        BuiltinSignatureDescriptor {
            label: "str = extractFileText(filename,Name,Value)",
            inputs: &IN_FILE_REST,
            outputs: &OUT_TEXT,
        },
    ],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const EXTRACT_FILE_TEXT_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor =
    BuiltinErrorDescriptor {
        code: "RM.EXTRACT_FILE_TEXT.TOO_MANY_OUTPUTS",
        identifier: Some("RunMat:extractFileText:TooManyOutputs"),
        when: "More than one output is requested.",
        message: "extractFileText: too many output arguments",
    };
