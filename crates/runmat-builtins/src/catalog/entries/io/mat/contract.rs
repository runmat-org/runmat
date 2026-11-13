use crate::*;

pub const MATFILE_CLASS: runmat_types::StaticClassIdentity =
    runmat_types::StaticClassIdentity::new("matlab.io.MatFile");

const fn param(
    name: &'static str,
    ty: BuiltinParamType,
    arity: BuiltinParamArity,
    default: Option<&'static str>,
    description: &'static str,
) -> BuiltinParamDescriptor {
    BuiltinParamDescriptor {
        name,
        ty,
        arity,
        default,
        description,
    }
}

const PATH: BuiltinParamDescriptor = param(
    "filename",
    BuiltinParamType::StringScalar,
    BuiltinParamArity::Required,
    Some("\"matlab.mat\""),
    "Level-5 MAT-file path.",
);
const VARIABLE: BuiltinParamDescriptor = param(
    "varName",
    BuiltinParamType::StringScalar,
    BuiltinParamArity::Variadic,
    None,
    "Whole variable names.",
);
const OPTION: BuiltinParamDescriptor = param(
    "option",
    BuiltinParamType::StringScalar,
    BuiltinParamArity::Required,
    None,
    "Supported MAT selection or save option.",
);
const PATTERN: BuiltinParamDescriptor = param(
    "pattern",
    BuiltinParamType::StringScalar,
    BuiltinParamArity::Variadic,
    None,
    "Regular-expression patterns.",
);
const LOAD_OUTPUT: [BuiltinParamDescriptor; 1] = [param(
    "S",
    BuiltinParamType::Any,
    BuiltinParamArity::Required,
    None,
    "Scalar struct of loaded variables; zero-output calls assign workspace bindings.",
)];
const SAVE_OUTPUT: [BuiltinParamDescriptor; 1] = [param(
    "status",
    BuiltinParamType::NumericScalar,
    BuiltinParamArity::Required,
    None,
    "Double zero on success.",
)];
const MATFILE_OUTPUT: [BuiltinParamDescriptor; 1] = [param(
    "m",
    BuiltinParamType::Any,
    BuiltinParamArity::Required,
    None,
    "Session-bound matlab.io.MatFile compatibility object.",
)];
const LOAD_SIGNATURES: [BuiltinSignatureDescriptor; 5] = [
    BuiltinSignatureDescriptor {
        label: "S = load()",
        inputs: &[],
        outputs: &LOAD_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "S = load(filename)",
        inputs: &[PATH],
        outputs: &LOAD_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "S = load(filename, varName1, varName2, ...)",
        inputs: &[PATH, VARIABLE],
        outputs: &LOAD_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "S = load(filename, '-regexp', pattern1, ...)",
        inputs: &[PATH, OPTION, PATTERN],
        outputs: &LOAD_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "S = load(option, value, ...)",
        inputs: &[OPTION, VARIABLE],
        outputs: &LOAD_OUTPUT,
    },
];
const SAVE_SIGNATURES: [BuiltinSignatureDescriptor; 6] = [
    BuiltinSignatureDescriptor {
        label: "status = save()",
        inputs: &[],
        outputs: &SAVE_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "status = save(filename)",
        inputs: &[PATH],
        outputs: &SAVE_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "status = save(filename, varName1, varName2, ...)",
        inputs: &[PATH, VARIABLE],
        outputs: &SAVE_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "status = save(filename, '-struct', structVar, field1, ...)",
        inputs: &[PATH, OPTION, VARIABLE],
        outputs: &SAVE_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "status = save(filename, '-regexp', pattern1, ...)",
        inputs: &[PATH, OPTION, PATTERN],
        outputs: &SAVE_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "save(option, value, ...)",
        inputs: &[OPTION, VARIABLE],
        outputs: &SAVE_OUTPUT,
    },
];
const MATFILE_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "m = matfile(filename)",
        inputs: &[PATH],
        outputs: &MATFILE_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "m = matfile(filename, 'Writable', tf)",
        inputs: &[
            PATH,
            OPTION,
            param(
                "tf",
                BuiltinParamType::Any,
                BuiltinParamArity::Required,
                None,
                "Scalar logical or numeric writable flag.",
            ),
        ],
        outputs: &MATFILE_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "m = matfile(filename, 'Writable', tf, ...)",
        inputs: &[
            PATH,
            OPTION,
            param(
                "tf",
                BuiltinParamType::Any,
                BuiltinParamArity::Required,
                None,
                "Scalar writable flag.",
            ),
            param(
                "repeatedNameValue",
                BuiltinParamType::Any,
                BuiltinParamArity::Variadic,
                None,
                "Further complete Writable name-value pairs; the last value wins.",
            ),
        ],
        outputs: &MATFILE_OUTPUT,
    },
];
const SUBSCRIPT: BuiltinParamDescriptor = param(
    "S",
    BuiltinParamType::Any,
    BuiltinParamArity::Required,
    None,
    "Standard substruct-compatible whole-member path.",
);
const RECEIVER: BuiltinParamDescriptor = param(
    "m",
    BuiltinParamType::Any,
    BuiltinParamArity::Required,
    None,
    "Live matlab.io.MatFile object.",
);
const SUBSREF_OUTPUT: [BuiltinParamDescriptor; 1] = [param(
    "value",
    BuiltinParamType::Any,
    BuiltinParamArity::Required,
    None,
    "Property or whole MAT variable.",
)];
const SUBSREF_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "value = matlab.io.MatFile.subsref(m, S)",
    inputs: &[RECEIVER, SUBSCRIPT],
    outputs: &SUBSREF_OUTPUT,
}];
const SUBSASGN_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "m = matlab.io.MatFile.subsasgn(m, S, rhs)",
    inputs: &[
        RECEIVER,
        SUBSCRIPT,
        param(
            "rhs",
            BuiltinParamType::Any,
            BuiltinParamArity::Required,
            None,
            "Whole variable value.",
        ),
    ],
    outputs: &MATFILE_OUTPUT,
}];

const fn error(
    code: &'static str,
    identifier: &'static str,
    when: &'static str,
    message: &'static str,
) -> BuiltinErrorDescriptor {
    BuiltinErrorDescriptor {
        code,
        identifier: Some(identifier),
        when,
        message,
    }
}
pub const LOAD_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = error(
    "RM.LOAD.INVALID_ARGUMENT",
    "RunMat:load:InvalidArgument",
    "Inputs are not a supported load form.",
    "load: invalid argument",
);
pub const LOAD_ERROR_INVALID_OPTION: BuiltinErrorDescriptor = error(
    "RM.LOAD.INVALID_OPTION",
    "RunMat:load:InvalidOption",
    "A load option or selector is invalid.",
    "load: invalid option",
);
pub const LOAD_ERROR_FILENAME: BuiltinErrorDescriptor = error(
    "RM.LOAD.FILENAME",
    "RunMat:load:Filename",
    "The MAT path is invalid.",
    "load: invalid filename",
);
pub const LOAD_ERROR_SELECTION: BuiltinErrorDescriptor = error(
    "RM.LOAD.SELECTION",
    "RunMat:load:Selection",
    "No requested variable can be selected.",
    "load: variable selection failed",
);
pub const LOAD_ERROR_IO: BuiltinErrorDescriptor = error(
    "RM.LOAD.IO",
    "RunMat:load:Io",
    "The MAT file cannot be read or decoded.",
    "load: MAT-file I/O failure",
);
pub const LOAD_ERROR_OUTPUT_COUNT: BuiltinErrorDescriptor = error(
    "RM.LOAD.OUTPUT_COUNT",
    "RunMat:load:OutputCount",
    "More than one output was requested.",
    "load: unsupported output count",
);
pub const LOAD_ERROR_WORKSPACE: BuiltinErrorDescriptor = error(
    "RM.LOAD.WORKSPACE",
    "RunMat:load:Workspace",
    "Statement-form binding assignment failed.",
    "load: workspace assignment failed",
);
const LOAD_ERRORS: [BuiltinErrorDescriptor; 7] = [
    LOAD_ERROR_INVALID_ARGUMENT,
    LOAD_ERROR_INVALID_OPTION,
    LOAD_ERROR_FILENAME,
    LOAD_ERROR_SELECTION,
    LOAD_ERROR_IO,
    LOAD_ERROR_OUTPUT_COUNT,
    LOAD_ERROR_WORKSPACE,
];
pub const LOAD_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &LOAD_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &LOAD_ERRORS,
};

pub const SAVE_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = error(
    "RM.SAVE.INVALID_ARGUMENT",
    "RunMat:save:InvalidArgument",
    "Inputs are not a supported save form.",
    "save: invalid argument",
);
pub const SAVE_ERROR_INVALID_OPTION: BuiltinErrorDescriptor = error(
    "RM.SAVE.INVALID_OPTION",
    "RunMat:save:InvalidOption",
    "A save option or value is invalid.",
    "save: invalid option",
);
pub const SAVE_ERROR_SELECTION: BuiltinErrorDescriptor = error(
    "RM.SAVE.SELECTION",
    "RunMat:save:Selection",
    "A workspace variable or field cannot be resolved.",
    "save: variable selection failed",
);
pub const SAVE_ERROR_FILENAME: BuiltinErrorDescriptor = error(
    "RM.SAVE.FILENAME",
    "RunMat:save:Filename",
    "The MAT path is invalid.",
    "save: invalid filename",
);
pub const SAVE_ERROR_IO: BuiltinErrorDescriptor = error(
    "RM.SAVE.IO",
    "RunMat:save:Io",
    "MAT output cannot be written.",
    "save: MAT-file I/O failure",
);
pub const SAVE_ERROR_UNSUPPORTED: BuiltinErrorDescriptor = error(
    "RM.SAVE.UNSUPPORTED",
    "RunMat:save:Unsupported",
    "The mode or value cannot be saved.",
    "save: unsupported operation",
);
pub const SAVE_ERROR_WORKSPACE: BuiltinErrorDescriptor = error(
    "RM.SAVE.WORKSPACE",
    "RunMat:save:Workspace",
    "Workspace state is unavailable.",
    "save: workspace state unavailable",
);
const SAVE_ERRORS: [BuiltinErrorDescriptor; 7] = [
    SAVE_ERROR_INVALID_ARGUMENT,
    SAVE_ERROR_INVALID_OPTION,
    SAVE_ERROR_SELECTION,
    SAVE_ERROR_FILENAME,
    SAVE_ERROR_IO,
    SAVE_ERROR_UNSUPPORTED,
    SAVE_ERROR_WORKSPACE,
];
pub const SAVE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SAVE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &SAVE_ERRORS,
};

pub const SAVE_STATUS_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "save-status-output",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "Capturing the numeric zero success status from save is a RunMat extension; MATLAB-compatible save is a zero-output statement.",
    error_identifier: Some("RunMat:compatibility:SaveStatusOutputExtension"),
};

pub const MATFILE_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = error(
    "RM.MATFILE.INVALID_ARGUMENT",
    "RunMat:matfile:InvalidArgument",
    "Arguments are not a supported matfile form or the receiver is invalid.",
    "matfile: invalid argument",
);
pub const MATFILE_ERROR_INVALID_OPTION: BuiltinErrorDescriptor = error(
    "RM.MATFILE.INVALID_OPTION",
    "RunMat:matfile:InvalidOption",
    "Writable name-value controls are invalid.",
    "matfile: invalid option",
);
pub const MATFILE_ERROR_IO: BuiltinErrorDescriptor = error(
    "RM.MATFILE.IO",
    "RunMat:matfile:Io",
    "MAT-file data cannot be read or written.",
    "matfile: MAT-file I/O failure",
);
pub const MATFILE_ERROR_SELECTION: BuiltinErrorDescriptor = error(
    "RM.MATFILE.SELECTION",
    "RunMat:matfile:Selection",
    "The whole variable is missing or invalid.",
    "matfile: variable selection failed",
);
pub const MATFILE_ERROR_UNSUPPORTED: BuiltinErrorDescriptor = error(
    "RM.MATFILE.UNSUPPORTED",
    "RunMat:matfile:Unsupported",
    "Partial indexing or a read-only assignment was requested.",
    "matfile: unsupported operation",
);
const MATFILE_ERRORS: [BuiltinErrorDescriptor; 5] = [
    MATFILE_ERROR_INVALID_ARGUMENT,
    MATFILE_ERROR_INVALID_OPTION,
    MATFILE_ERROR_IO,
    MATFILE_ERROR_SELECTION,
    MATFILE_ERROR_UNSUPPORTED,
];
pub const MATFILE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &MATFILE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &MATFILE_ERRORS,
};
pub const MATFILE_SUBSREF_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SUBSREF_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &MATFILE_ERRORS,
};
pub const MATFILE_SUBSASGN_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SUBSASGN_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &MATFILE_ERRORS,
};

pub const LOAD_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] = [BuiltinIntegerCapabilityDescriptor {
    form: "integer_value = load(...) or S.integer_field = load(...)" , inputs: &[],
    computation_domain: BuiltinIntegerComputationDomain::Structural,
    output_class: BuiltinIntegerOutputClassRule::PreserveInput,
    overflow: BuiltinIntegerOverflowRule::Error,
    backend: BuiltinIntegerBackendRule::HostOnly,
    overload: BuiltinIntegerOverloadKind::FunctionSpecific,
    notes: "Level-5 integer tags reconstruct all eight signed/unsigned classes exactly, including nested cells, structures, sparse values, complex components, and typed object records.",
}];

const WRITABLE_INTEGER_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "Writable",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Exact integer zero is false; every nonzero exact integer is true.",
    }];
pub const MATFILE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "m = matfile(filename, 'Writable', integer_flag)",
        inputs: &WRITABLE_INTEGER_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Predicate,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::ScalarOnly,
        notes: "Writable is converted to a Boolean control before the sealed MAT object is issued.",
    }];

pub const MATFILE_SUBSREF_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "integer_value = m.integer_variable",
        inputs: &[],
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "Complete Level-5 reads retain exact integer class, shape, and payload.",
    }];

const SUBSASGN_INTEGER_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "rhs",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "All eight signed/unsigned classes encode without floating-point conversion.",
    }];
pub const MATFILE_SUBSASGN_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "m.integer_variable = integer_rhs",
        inputs: &SUBSASGN_INTEGER_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Whole-variable assignment preserves exact integer storage; accepted resident rhs values materialize through their actual owner.",
    }];
