use crate::*;

#[cfg(test)]
mod tests;

pub const READTABLE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &READTABLE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &TABLE_ERRORS,
};

pub const READTABLE_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "T = readtable(filename)",
        inputs: &READTABLE_INPUTS_FILENAME,
        outputs: &TABLE_IMPORT_ANY_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "T = readtable(filename, nameValuePairs...)",
        inputs: &READTABLE_INPUTS_NAME_VALUE,
        outputs: &TABLE_IMPORT_ANY_OUTPUT,
    },
];

pub const READTABLE_INPUTS_FILENAME: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "filename",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Text or spreadsheet file path.",
}];

pub const TABLE_IMPORT_ANY_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "out",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Result value.",
}];

pub const READTABLE_INPUTS_NAME_VALUE: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "filename",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Text or spreadsheet file path.",
    },
    BuiltinParamDescriptor {
        name: "nameValuePairs",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Name-value import options.",
    },
];

pub const SPREADSHEET_IMPORT_OPTIONS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SPREADSHEET_IMPORT_OPTIONS_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &TABLE_ERRORS,
};

pub const SPREADSHEET_IMPORT_OPTIONS_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "opts = spreadsheetImportOptions()",
        inputs: &[],
        outputs: &SPREADSHEET_IMPORT_OPTIONS_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "opts = spreadsheetImportOptions(nameValuePairs...)",
        inputs: &SPREADSHEET_IMPORT_OPTIONS_INPUTS_NAME_VALUE,
        outputs: &SPREADSHEET_IMPORT_OPTIONS_OUTPUT,
    },
];

pub const SPREADSHEET_IMPORT_OPTIONS_OUTPUT: [BuiltinParamDescriptor; 1] =
    [BuiltinParamDescriptor {
        name: "opts",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Typed spreadsheet import-options object.",
    }];

pub const SPREADSHEET_IMPORT_OPTIONS_INPUTS_NAME_VALUE: [BuiltinParamDescriptor; 1] =
    [BuiltinParamDescriptor {
        name: "nameValuePairs",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Name-value option pairs.",
    }];

pub const DETECT_IMPORT_OPTIONS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &DETECT_IMPORT_OPTIONS_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &TABLE_ERRORS,
};

pub const DETECT_IMPORT_OPTIONS_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "opts = detectImportOptions(filename)",
        inputs: &DETECT_IMPORT_OPTIONS_INPUTS_FILENAME,
        outputs: &DETECT_IMPORT_OPTIONS_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "opts = detectImportOptions(filename, nameValuePairs...)",
        inputs: &DETECT_IMPORT_OPTIONS_INPUTS_NAME_VALUE,
        outputs: &DETECT_IMPORT_OPTIONS_OUTPUT,
    },
];

pub const DETECT_IMPORT_OPTIONS_INPUTS_FILENAME: [BuiltinParamDescriptor; 1] =
    [BuiltinParamDescriptor {
        name: "filename",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Text or spreadsheet file path to inspect.",
    }];

pub const DETECT_IMPORT_OPTIONS_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "opts",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Typed text or spreadsheet import-options object for tabular readers.",
}];

pub const DETECT_IMPORT_OPTIONS_INPUTS_NAME_VALUE: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "filename",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Text or spreadsheet file path to inspect.",
    },
    BuiltinParamDescriptor {
        name: "nameValuePairs",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Detection overrides such as Delimiter, Range, Sheet, Encoding, or TextType.",
    },
];

pub const DETECT_IMPORT_OPTIONS_EXTENSIONS: [BuiltinExtensionDescriptor; 1] =
    [DETECT_IMPORT_OPTIONS_INTEGER_NUM_HEADER_LINES_EXTENSION];

pub const DETECT_IMPORT_OPTIONS_INTEGER_NUM_HEADER_LINES_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "detectimportoptions-integer-num-header-lines",
        mode: BuiltinExtensionMode::RunMatOnly,
        description:
            "detectImportOptions with a typed-integer NumHeaderLines value is a RunMat extension",
        error_identifier: Some(
            "RunMat:compatibility:DetectImportOptionsIntegerNumHeaderLinesExtension",
        ),
    };

pub const DETECT_IMPORT_OPTIONS_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "opts = detectImportOptions(filename, integer_location_controls...)",
        inputs: &DETECT_IMPORT_OPTIONS_INTEGER_LOCATION_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The implemented text/spreadsheet subset preserves exact integer controls and returns host import metadata; JSON/XML/Word/HTML/archive families and concrete MATLAB options classes remain explicit general gaps.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "opts = detectImportOptions(filename, \"NumHeaderLines\", typed_integer)",
        inputs: &DETECT_IMPORT_OPTIONS_INTEGER_HEADER_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "RunMat mode parses this pre-existing extension exactly; compatibility mode rejects before file or provider access.",
    },
];

pub const DETECT_IMPORT_OPTIONS_INTEGER_LOCATION_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "Range, DataRange, Sheet, ExpectedNumVariables, or VariableNamesLine",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Supported text/spreadsheet location and size controls are parsed exactly from authoritative integer storage and validated before conversion to host indices; numeric DataRange supports scalar rows and one 1-by-2 interval, textual cell references remain available for detected-option replay, and multiple numeric Nx2 intervals remain a general gap.",
    }];

pub const DETECT_IMPORT_OPTIONS_INTEGER_HEADER_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "NumHeaderLines",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The compatibility target's public datatype list is single/double; typed-integer values are admitted only in RunMat mode after an explicit extension gate.",
    }];

pub const READTABLE_EXTENSIONS: [BuiltinExtensionDescriptor; 1] =
    [READTABLE_TYPED_INTEGER_CONTROL_EXTENSION];

pub const READTABLE_TYPED_INTEGER_CONTROL_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "readtable-typed-integer-control",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "readtable accepts typed-integer controls whose public datatype tables are floating-only as a RunMat extension",
        error_identifier: Some("RunMat:compatibility:ReadtableTypedIntegerControlExtension"),
    };

pub const READTABLE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "T = readtable(filename, import_options_with_integer_VariableTypes)",
        inputs: &TABLE_IMPORT_INTEGER_VARIABLE_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::OptionDependent,
        overflow: BuiltinIntegerOverflowRule::Saturate,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "Each imported integer table variable retains its documented class and exact values inside the table.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "T = readtable(filename, 'VariableNamesLine', integer_line)",
        inputs: &TABLE_IMPORT_VARIABLE_NAMES_LINE_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The documented integer line control affects only host import structure and never selects output class or residency.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "T = readtable(filename, typed_integer_extension_controls...)",
        inputs: &TABLE_IMPORT_EXTENSION_CONTROL_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The pre-existing typed-control extension is independently gated; ordinary resident arguments may still gather transparently for host I/O.",
    },
];

pub const TABLE_IMPORT_INTEGER_VARIABLE_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "VariableTypes integer class",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Documented integer variable types parse decimal text directly into private native storage, including exact int64 and uint64 endpoints and saturating conversion of missing, infinite, or out-of-range values.",
    }];

pub const TABLE_IMPORT_VARIABLE_NAMES_LINE_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "VariableNamesLine",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The public datatype table explicitly includes all eight integer classes. RunMat reads the authoritative scalar exactly and bounds-checks it before deriving the zero-based host header offset.",
    }];

pub const TABLE_IMPORT_EXTENSION_CONTROL_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "Range, Sheet, NumHeaderLines, or another implemented control whose public datatype list excludes native integers",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The compatibility target's public datatype tables restrict these implemented controls to floating or nonnumeric classes. RunMat mode retains exact typed-integer admission; compatibility mode rejects before gather or file access.",
    }];

pub const READCELL_EXTENSIONS: [BuiltinExtensionDescriptor; 1] =
    [READCELL_TYPED_INTEGER_CONTROL_EXTENSION];

pub const READCELL_TYPED_INTEGER_CONTROL_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "readcell-typed-integer-control",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "readcell accepts typed-integer controls whose public datatype tables are floating-only as a RunMat extension",
        error_identifier: Some("RunMat:compatibility:ReadcellTypedIntegerControlExtension"),
    };

pub const READCELL_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "C = readcell(filename, 'NumHeaderLines', integer_count)",
        inputs: &READCELL_NUM_HEADER_LINES_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Imported numeric cells retain the documented double scalar representation; the integer header count affects only import structure.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "C = readcell(filename, typed_integer_extension_controls...)",
        inputs: &TABLE_IMPORT_EXTENSION_CONTROL_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Compatibility mode rejects typed floating-only controls before file access; RunMat mode validates them exactly.",
    },
];

pub const READCELL_NUM_HEADER_LINES_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "NumHeaderLines",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The public contract is a positive integer. RunMat accepts every native integer class, reads the authoritative scalar exactly, and bounds-checks it before skipping host input records.",
    }];

pub const SPREADSHEET_IMPORT_OPTIONS_EXTENSIONS: [BuiltinExtensionDescriptor; 1] =
    [SPREADSHEET_OPTIONS_TYPED_INTEGER_CONTROL_EXTENSION];

pub const SPREADSHEET_OPTIONS_TYPED_INTEGER_CONTROL_EXTENSION:
    BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "spreadsheetimportoptions-typed-integer-location-control",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "spreadsheetImportOptions accepts typed-integer location controls outside documented NumVariables as a RunMat extension",
    error_identifier: Some(
        "RunMat:compatibility:SpreadsheetImportOptionsTypedIntegerControlExtension",
    ),
};

pub const SPREADSHEET_IMPORT_OPTIONS_INTEGER_CAPABILITIES:
    [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "opts = spreadsheetImportOptions('NumVariables', integer_num_vars, ___)",
        inputs: &SPREADSHEET_NUM_VARIABLES_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "VariableTypes may name all eight native integer output classes for later readtable use; the options structure itself is host metadata.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "opts = spreadsheetImportOptions(typed_integer_location_controls...)",
        inputs: &TABLE_IMPORT_EXTENSION_CONTROL_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Typed location controls beyond NumVariables are a gated RunMat extension and are never rounded through binary64.",
    },
];

pub const SPREADSHEET_NUM_VARIABLES_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "NumVariables",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The documented positive scalar integer is read exactly from every integer class and must fit usize before import-option metadata is allocated.",
    }];

pub const READCELL_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[BuiltinSignatureDescriptor {
        label: "C = readcell(filename, options...)",
        inputs: &READTABLE_INPUTS_NAME_VALUE,
        outputs: &TABLE_IMPORT_ANY_OUTPUT,
    }],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &TABLE_ERRORS,
};
