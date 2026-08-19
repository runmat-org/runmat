use super::*;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinExtensionDescriptor, BuiltinExtensionMode,
    BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor, BuiltinIntegerComputationDomain,
    BuiltinIntegerInputAvailability, BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule,
    BuiltinIntegerOverflowRule, BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule,
    BuiltinOutputMode, BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType,
    BuiltinSignatureDescriptor,
};
use runmat_macros::runtime_builtin;

const ARRAY2TIMETABLE_BUILTIN_NAME: &str = "array2timetable";

const TIMETABLE_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "TT",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Output timetable.",
}];
const TIMETABLE_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "rowTimes or timing options",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Datetime/duration row times, SampleRate, TimeStep, or preallocation options.",
    },
    BuiltinParamDescriptor {
        name: "variables and Name,Value options",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Same-height data variables plus VariableNames, Size, VariableTypes, and timing options.",
    },
];
const TIMETABLE_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "TT = timetable()",
        inputs: &[],
        outputs: &TIMETABLE_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "TT = timetable(rowTimes, variables..., Name=Value...)",
        inputs: &TIMETABLE_INPUTS,
        outputs: &TIMETABLE_OUTPUT,
    },
];
pub const TIMETABLE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &TIMETABLE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[],
};

const TIMETABLE2TABLE_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "T",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Output table.",
}];
const TIMETABLE2TABLE_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "TT",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input timetable.",
    },
    BuiltinParamDescriptor {
        name: "ConvertRowTimes",
        ty: BuiltinParamType::PropertyValue,
        arity: BuiltinParamArity::Optional,
        default: Some("true"),
        description:
            "Logical flag that controls whether row times become the first table variable.",
    },
];
const TIMETABLE2TABLE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "T = timetable2table(TT, ConvertRowTimes=value)",
    inputs: &TIMETABLE2TABLE_INPUTS,
    outputs: &TIMETABLE2TABLE_OUTPUT,
}];
pub const TIMETABLE2TABLE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &TIMETABLE2TABLE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[],
};

pub(in crate::builtins::table) const TIMETABLE2TABLE_INTEGER_OPTION_EXTENSION:
    BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "timetable2table-typed-integer-convert-row-times",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "timetable2table with a typed-integer ConvertRowTimes value is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:Timetable2TableIntegerConvertRowTimesExtension"),
};
const TIMETABLE2TABLE_EXPLICIT_GPU_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "timetable2table-explicit-gpu-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "timetable2table with explicitly GPU-resident input is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:Timetable2TableExplicitGpuInputExtension"),
    };
pub const TIMETABLE2TABLE_EXTENSIONS: [BuiltinExtensionDescriptor; 2] = [
    TIMETABLE2TABLE_INTEGER_OPTION_EXTENSION,
    TIMETABLE2TABLE_EXPLICIT_GPU_EXTENSION,
];
const TIMETABLE2TABLE_INTEGER_VARIABLE_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "integer timetable variables",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Every data variable retains its native integer class, shape, and payload when timetable metadata is converted to table metadata.",
    }];
const TIMETABLE2TABLE_INTEGER_OPTION_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "ConvertRowTimes",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The public option is logical. RunMat mode additionally accepts exact integer zero or one.",
    }];
pub const TIMETABLE2TABLE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "T = timetable2table(TT_with_integer_variables, Name=Value...)",
        inputs: &TIMETABLE2TABLE_INTEGER_VARIABLE_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "Contained integer variables retain their exact classes and values during conversion. By default, row times become the first table variable.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "T = timetable2table(TT, ConvertRowTimes=typed_integer_flag)",
        inputs: &TIMETABLE2TABLE_INTEGER_OPTION_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The mode gate runs before residency access; admitted values are decoded exactly and never pass through f64.",
    },
];

pub(in crate::builtins::table) const TIMETABLE_NUMERIC_ROW_TIMES_EXTENSION:
    BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "timetable-numeric-row-times",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "timetable with numeric row times is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:TimetableNumericRowTimesExtension"),
};
const TIMETABLE_GENERATED_ROW_TIMES_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "timetable-generated-row-times",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "timetable with data variables but no timing source uses generated row times as a RunMat extension",
        error_identifier: Some("RunMat:compatibility:TimetableGeneratedRowTimesExtension"),
    };
const TIMETABLE_EXPLICIT_GPU_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "timetable-explicit-gpu-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "timetable with explicitly GPU-resident input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:TimetableExplicitGpuInputExtension"),
};
pub const TIMETABLE_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    TIMETABLE_NUMERIC_ROW_TIMES_EXTENSION,
    TIMETABLE_GENERATED_ROW_TIMES_EXTENSION,
    TIMETABLE_EXPLICIT_GPU_EXTENSION,
];
const TIMETABLE_INTEGER_VARIABLE_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "data variables or VariableTypes",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Timetable data variables may use all eight integer classes and retain authoritative native storage.",
    }];
const TIMETABLE_INTEGER_SIZE_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "Size",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The two-element preallocation size is decoded exactly as nonnegative platform dimensions.",
    }];
const TIMETABLE_INTEGER_SAMPLE_RATE_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "SampleRate",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "A positive numeric sample rate crosses the explicit floating seconds-to-days timing boundary after exact integer validation.",
    }];
const TIMETABLE_INTEGER_ROW_TIMES_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "numeric RowTimes",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Compatibility mode requires datetime or duration row times. RunMat mode retains exact numeric row-time storage for numeric timetable workflows.",
    }];
pub const TIMETABLE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 4] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "TT = timetable(rowTimes, integer_var1, ..., integer_varN, Name=Value...)",
        inputs: &TIMETABLE_INTEGER_VARIABLE_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Each data variable remains independent of row-time metadata and retains its exact class and values.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "TT = timetable(Size=integer_sz, VariableTypes=types, timingName=timingValue)",
        inputs: &TIMETABLE_INTEGER_SIZE_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::OptionDependent,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Integer VariableTypes allocate zero-filled native columns directly; a timing source is required for nonempty preallocation.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "TT = timetable(..., SampleRate=integer_Fs, ...)",
        inputs: &TIMETABLE_INTEGER_SAMPLE_RATE_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The sample-rate control determines duration or datetime row times and does not change any data-variable class.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "TT = timetable(integer_row_times, variables...)",
        inputs: &TIMETABLE_INTEGER_ROW_TIMES_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "This mode-gated numeric extension preserves class, shape, and values and supports exact timerange selection.",
    },
];

const TABLE2TIMETABLE_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "TT",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Output timetable.",
}];
const TABLE2TIMETABLE_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "T",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input table.",
    },
    BuiltinParamDescriptor {
        name: "Name,Value",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "RowTimes, SampleRate, TimeStep, and StartTime options.",
    },
];
const TABLE2TIMETABLE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "TT = table2timetable(T, Name=Value...)",
    inputs: &TABLE2TIMETABLE_INPUTS,
    outputs: &TABLE2TIMETABLE_OUTPUT,
}];
pub const TABLE2TIMETABLE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &TABLE2TIMETABLE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[],
};
pub(crate) const TABLE2TIMETABLE_GPU_INPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "table2timetable-gpu-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "table2timetable with an explicit resident GPU payload is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:Table2TimetableGpuInputExtension"),
    };
pub(crate) const TABLE2TIMETABLE_GENERATED_TIMES_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "table2timetable-generated-row-times",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "table2timetable without a time variable or timing option uses generated row times as a RunMat extension",
        error_identifier: Some("RunMat:compatibility:Table2TimetableGeneratedTimesExtension"),
    };
pub(crate) const TABLE2TIMETABLE_NUMERIC_ROW_TIMES_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "table2timetable-numeric-row-times",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "table2timetable with a numeric row-time vector is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:Table2TimetableNumericRowTimesExtension"),
    };
pub const TABLE2TIMETABLE_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    TABLE2TIMETABLE_GPU_INPUT_EXTENSION,
    TABLE2TIMETABLE_GENERATED_TIMES_EXTENSION,
    TABLE2TIMETABLE_NUMERIC_ROW_TIMES_EXTENSION,
];
const TABLE2TIMETABLE_INTEGER_VARIABLES: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "integer variables contained in T",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Non-time table variables retain their exact integer class and payload in the output timetable.",
    }];
const TABLE2TIMETABLE_INTEGER_INDEX: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "RowTimes variable index",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "A positive one-based scalar index selects a datetime or duration table variable and is decoded exactly from authoritative storage.",
    }];
const TABLE2TIMETABLE_INTEGER_SAMPLE_RATE: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "Fs",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The positive numeric SampleRate scalar enters the deliberate floating seconds-to-days timing boundary after exact sign validation.",
    }];
const TABLE2TIMETABLE_INTEGER_ROW_TIMES: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "numeric RowTimes vector",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "RunMat mode preserves the numeric vector as exact row-time metadata; MATLAB-compatible mode requires datetime or duration row times.",
    }];
pub const TABLE2TIMETABLE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 4] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "TT = table2timetable(T_with_integer_variables, Name=Value...)",
        inputs: &TABLE2TIMETABLE_INTEGER_VARIABLES,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "Timetable construction changes container metadata only; unrelated integer variables retain their exact classes and values.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "TT = table2timetable(T, RowTimes=integer_time_variable_index)",
        inputs: &TABLE2TIMETABLE_INTEGER_INDEX,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The selected time variable becomes row-time metadata and is removed from the data-variable list; the selector never passes through f64.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "TT = table2timetable(T, SampleRate=integer_Fs, StartTime=t0)",
        inputs: &TABLE2TIMETABLE_INTEGER_SAMPLE_RATE,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Fs determines double duration/datetime row times; all table data variables retain their independent classes.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "TT = table2timetable(T, RowTimes=integer_vector)",
        inputs: &TABLE2TIMETABLE_INTEGER_ROW_TIMES,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "The RunMat-only numeric row-time vector retains authoritative integer storage and must be a height-by-one column.",
    },
];

pub(crate) const ARRAY2TIMETABLE_GPU_INPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "array2timetable-gpu-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description:
            "array2timetable with an interactive resident GPU argument is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:Array2TimetableGpuInputExtension"),
    };

pub const ARRAY2TIMETABLE_EXTENSIONS: [BuiltinExtensionDescriptor; 1] =
    [ARRAY2TIMETABLE_GPU_INPUT_EXTENSION];

pub(crate) const READTIMETABLE_TYPED_INTEGER_CONTROL_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "readtimetable-typed-integer-control",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "readtimetable accepts typed-integer controls whose public datatype tables are floating-only as a RunMat extension",
        error_identifier: Some("RunMat:compatibility:ReadtimetableTypedIntegerControlExtension"),
    };
pub const READTIMETABLE_EXTENSIONS: [BuiltinExtensionDescriptor; 2] = [
    READTIMETABLE_TYPED_INTEGER_CONTROL_EXTENSION,
    TABLE2TIMETABLE_NUMERIC_ROW_TIMES_EXTENSION,
];

fn ensure_numeric_row_times_extension(args: &[Value], builtin_name: &str) -> BuiltinResult<()> {
    let has_numeric_vector = args.chunks_exact(2).any(|pair| {
        scalar_text(&pair[0], "RowTimes option")
            .is_ok_and(|name| name.eq_ignore_ascii_case("RowTimes"))
            && match &pair[1] {
                Value::Tensor(tensor) => tensor.len() != 1,
                Value::GpuTensor(handle) => handle.shape.iter().product::<usize>() != 1,
                _ => false,
            }
    });
    if has_numeric_vector {
        crate::compatibility::ensure_builtin_extension_enabled(
            &TABLE2TIMETABLE_NUMERIC_ROW_TIMES_EXTENSION,
            builtin_name,
        )?;
    }
    Ok(())
}
const READTIMETABLE_INTEGER_VARIABLE_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "VariableTypes integer class",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Documented integer variable types are imported directly into exact native table-variable storage before timetable construction.",
    }];
const READTIMETABLE_INTEGER_LOCATION_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "VariableNamesLine",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The public datatype table explicitly includes all eight integer classes. RunMat reads the authoritative scalar exactly and bounds-checks it before deriving the host header offset.",
    }];
const READTIMETABLE_INTEGER_EXTENSION_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "Range, Sheet, NumHeaderLines, ExpectedNumVariables, boolean numeric flags, or related floating-only control",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Typed forms outside the public datatype tables remain a gated RunMat extension and are rejected in compatibility mode before gather or file access.",
    }];
pub const READTIMETABLE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "TT = readtimetable(filename, import_options_with_integer_VariableTypes)",
        inputs: &READTIMETABLE_INTEGER_VARIABLE_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::OptionDependent,
        overflow: BuiltinIntegerOverflowRule::Saturate,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "Timetable containment preserves the exact integer class and payload of imported variables; row-time inference does not coerce unrelated data variables.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "TT = readtimetable(filename, 'VariableNamesLine', integer_line)",
        inputs: &READTIMETABLE_INTEGER_LOCATION_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The documented line control is an exact host index and does not select output residency.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "TT = readtimetable(filename, typed_integer_extension_controls...)",
        inputs: &READTIMETABLE_INTEGER_EXTENSION_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Automatic residency may gather transparently for host I/O, but unsupported typed controls cannot bypass the compatibility gate.",
    },
];

const ARRAY2TIMETABLE_INTEGER_DATA_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "X",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The documented homogeneous array domain includes all eight real integer classes.",
    }];

const ARRAY2TIMETABLE_INTEGER_SAMPLE_RATE_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "Fs",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The documented positive numeric scalar SampleRate control accepts every real integer class as well as floating scalars.",
    }];

pub const ARRAY2TIMETABLE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "TT = array2timetable(integer_X, timingName,timingValue, Name,Value...)",
        inputs: &ARRAY2TIMETABLE_INTEGER_DATA_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Each X column becomes a timetable variable with X's exact authoritative integer storage and class; row times are stored separately. Interactive resident GPU arguments are mode-gated RunMat extensions that gather before timetable construction.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "TT = array2timetable(X, \"SampleRate\", integer_Fs, Name,Value...)",
        inputs: &ARRAY2TIMETABLE_INTEGER_SAMPLE_RATE_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Fs is decoded as an exact positive scalar before the reciprocal seconds-to-days timing boundary; timetable variable classes are determined independently by X.",
    },
];

#[runtime_builtin(
    name = "timetable",
    category = "table",
    summary = "Create a timetable from row times and variables.",
    keywords = "timetable,table,RowTimes,TimeStep,VariableNames",
    accel = "cpu",
    descriptor(crate::builtins::table::builtins::timetable::TIMETABLE_DESCRIPTOR),
    extensions(crate::builtins::table::builtins::timetable::TIMETABLE_EXTENSIONS),
    integer_capabilities(
        crate::builtins::table::builtins::timetable::TIMETABLE_INTEGER_CAPABILITIES
    ),
    builtin_path = "crate::builtins::table::builtins"
)]
pub(crate) async fn timetable_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    ensure_table_class_registered();
    if args
        .iter()
        .any(crate::builtins::common::validation::value_contains_explicit_gpu)
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &TIMETABLE_EXPLICIT_GPU_EXTENSION,
            "timetable",
        )?;
    }
    let args = gather_values(&args).await?;
    let (explicit_row_times, variables, options) = split_timetable_constructor_args(args)?;
    let timing_forms = usize::from(explicit_row_times.is_some())
        + usize::from(options.sample_rate.is_some())
        + usize::from(options.time_step.is_some());
    if timing_forms > 1 {
        return Err(invalid_argument(
            "timetable: specify exactly one of RowTimes, SampleRate, or TimeStep",
        ));
    }
    if options.start_time.is_some() && options.sample_rate.is_none() && options.time_step.is_none()
    {
        return Err(invalid_argument(
            "timetable: StartTime requires SampleRate or TimeStep",
        ));
    }

    let preallocated = options.table.size.is_some() || options.table.variable_types.is_some();
    let (names, variables, row_names, height) = if preallocated {
        preallocated_table_columns(variables, options.table, "timetable")?
    } else {
        let names = options
            .table
            .variable_names
            .unwrap_or_else(|| generated_variable_names(variables.len()));
        let height = variables
            .first()
            .map(value_row_count)
            .transpose()?
            .unwrap_or(0);
        (names, variables, options.table.row_names, height)
    };

    let row_times = if let Some(row_times) = explicit_row_times {
        Some(row_times)
    } else if options.sample_rate.is_some() || options.time_step.is_some() {
        Some(array2timetable_row_times(
            &Array2TimetableOptions {
                row_times: None,
                sample_rate: options.sample_rate,
                time_step: options.time_step,
                start_time: options.start_time,
                variable_names: None,
                dimension_names: None,
            },
            height,
        )?)
    } else {
        None
    };

    if let Some(row_times) = row_times.as_ref() {
        if !is_time_like_value(row_times) {
            crate::compatibility::ensure_builtin_extension_enabled(
                &TIMETABLE_NUMERIC_ROW_TIMES_EXTENSION,
                "timetable",
            )?;
            if !matches!(row_times, Value::Num(_) | Value::Int(_) | Value::Tensor(_)) {
                return Err(invalid_argument(
                    "timetable: RowTimes must contain datetime or duration values",
                ));
            }
        }
    } else if height > 0 {
        crate::compatibility::ensure_builtin_extension_enabled(
            &TIMETABLE_GENERATED_ROW_TIMES_EXTENSION,
            "timetable",
        )?;
    }

    let mut value = table_from_columns_with_class(TIMETABLE_CLASS, names, variables, row_names)?;
    if let Value::Object(object) = &mut value {
        set_timetable_row_times(object, row_times)?;
    }
    Ok(value)
}

#[runtime_builtin(
    name = "array2timetable",
    category = "table",
    summary = "Convert an array into a timetable.",
    keywords = "array2timetable,timetable,RowTimes,SampleRate,TimeStep,StartTime,VariableNames,DimensionNames",
    accel = "gather",
    descriptor(crate::builtins::table::ARRAY2TIMETABLE_DESCRIPTOR),
    extensions(crate::builtins::table::builtins::timetable::ARRAY2TIMETABLE_EXTENSIONS),
    integer_capabilities(
        crate::builtins::table::builtins::timetable::ARRAY2TIMETABLE_INTEGER_CAPABILITIES
    ),
    builtin_path = "crate::builtins::table::builtins"
)]
pub(crate) async fn array2timetable_builtin(
    value: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
    ensure_table_class_registered();
    if matches!(value, Value::GpuTensor(_))
        || rest
            .iter()
            .any(|value| matches!(value, Value::GpuTensor(_)))
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &ARRAY2TIMETABLE_GPU_INPUT_EXTENSION,
            ARRAY2TIMETABLE_BUILTIN_NAME,
        )?;
    }
    let value = gather_if_needed_async(&value)
        .await
        .map_err(map_control_flow)?;
    let rest = gather_values(&rest).await?;
    let options = parse_array2timetable_options(&rest)?;
    let columns = split_value_columns(value)?;
    let names = options
        .variable_names
        .clone()
        .unwrap_or_else(|| generated_variable_names(columns.len()));
    validate_array2timetable_names(&names, options.dimension_names.as_deref())?;
    let height = columns
        .first()
        .map(value_row_count)
        .transpose()?
        .unwrap_or(0);
    let row_times = array2timetable_row_times(&options, height)?;
    let mut out = table_from_columns_with_class(TIMETABLE_CLASS, names, columns, None)?;
    if let Value::Object(object) = &mut out {
        set_timetable_row_times(object, Some(row_times))?;
        if let Some(dimension_names) = options.dimension_names {
            set_table_dimension_names(object, dimension_names, ARRAY2TIMETABLE_BUILTIN_NAME)?;
        }
    }
    Ok(out)
}

#[runtime_builtin(
    name = "table2timetable",
    category = "table",
    summary = "Convert a table into a timetable.",
    keywords = "table2timetable,timetable,RowTimes",
    accel = "cpu",
    descriptor(crate::builtins::table::builtins::timetable::TABLE2TIMETABLE_DESCRIPTOR),
    extensions(crate::builtins::table::builtins::timetable::TABLE2TIMETABLE_EXTENSIONS),
    integer_capabilities(
        crate::builtins::table::builtins::timetable::TABLE2TIMETABLE_INTEGER_CAPABILITIES
    ),
    builtin_path = "crate::builtins::table::builtins"
)]
pub(crate) async fn table2timetable_builtin(
    value: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
    ensure_numeric_row_times_extension(&rest, "table2timetable")?;
    if crate::builtins::common::validation::value_contains_explicit_gpu(&value)
        || rest
            .iter()
            .any(crate::builtins::common::validation::value_contains_explicit_gpu)
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &TABLE2TIMETABLE_GPU_INPUT_EXTENSION,
            "table2timetable",
        )?;
    }
    let host = gather_if_needed_async(&value)
        .await
        .map_err(map_control_flow)?;
    let rest = gather_values(&rest).await?;
    let options = parse_table2timetable_options(&rest)?;
    let object = into_table_object(host, "table2timetable")?;
    let names = table_variable_names_from_object(&object)?;
    let variables = table_variables(&object)?;
    let row_names = selected_row_names(&object, &(0..table_height(&object)?).collect::<Vec<_>>())?;
    let height = table_height(&object)?;
    let (times, out_names) = if let Some(row_times) = options.row_times.as_ref() {
        if is_time_like_value(row_times) {
            validate_explicit_row_times(row_times, height)?;
            (Some(row_times.clone()), names)
        } else if matches!(row_times, Value::Tensor(tensor) if tensor.len() != 1) {
            crate::compatibility::ensure_builtin_extension_enabled(
                &TABLE2TIMETABLE_NUMERIC_ROW_TIMES_EXTENSION,
                "table2timetable",
            )?;
            let Value::Tensor(tensor) = row_times else {
                unreachable!("numeric row-time extension requires a tensor")
            };
            if tensor.len() != height || tensor.cols() != 1 {
                return Err(invalid_argument(format!(
                    "table2timetable: numeric RowTimes must be a {height}-by-1 vector"
                )));
            }
            (Some(row_times.clone()), names)
        } else {
            let selected = table2timetable_time_variable_name(row_times, &names)?;
            let selected_value = variables.fields.get(&selected).cloned().ok_or_else(|| {
                invalid_variable(format!("table2timetable: missing variable '{selected}'"))
            })?;
            if !is_time_like_value(&selected_value) {
                return Err(invalid_argument(
                    "table2timetable: RowTimes variable must contain datetime or duration values",
                ));
            }
            let out_names = names.into_iter().filter(|name| name != &selected).collect();
            (Some(selected_value), out_names)
        }
    } else if let Some(generated) = table2timetable_generated_row_times(&options, height)? {
        (Some(generated), names)
    } else if let Some(first) = names.first() {
        let first_value = variables.fields.get(first).cloned();
        if first_value
            .as_ref()
            .map(is_time_like_value)
            .unwrap_or(false)
        {
            (first_value, names[1..].to_vec())
        } else {
            (None, names)
        }
    } else {
        (None, names)
    };
    if times.is_none() {
        crate::compatibility::ensure_builtin_extension_enabled(
            &TABLE2TIMETABLE_GENERATED_TIMES_EXTENSION,
            "table2timetable",
        )?;
    }
    let mut out_columns = Vec::with_capacity(out_names.len());
    for name in &out_names {
        out_columns.push(variables.fields.get(name).cloned().ok_or_else(|| {
            invalid_variable(format!("table2timetable: missing variable '{name}'"))
        })?);
    }
    let mut out =
        table_from_columns_with_class(TIMETABLE_CLASS, out_names, out_columns, row_names)?;
    if let Value::Object(object) = &mut out {
        set_timetable_row_times(object, times)?;
    }
    Ok(out)
}

fn table2timetable_time_variable_name(selector: &Value, names: &[String]) -> BuiltinResult<String> {
    if let Ok(name) = scalar_text(selector, "RowTimes variable") {
        if names.iter().any(|candidate| candidate == &name) {
            return Ok(name);
        }
        return Err(invalid_variable(format!(
            "table2timetable: unknown RowTimes variable '{name}'"
        )));
    }
    let index = positive_usize(selector, "RowTimes variable index")?;
    names.get(index - 1).cloned().ok_or_else(|| {
        invalid_variable(format!(
            "table2timetable: RowTimes variable index {index} is out of range"
        ))
    })
}

#[runtime_builtin(
    name = "timetable2table",
    category = "table",
    summary = "Convert a timetable into a table.",
    keywords = "timetable2table,timetable,table,ConvertRowTimes",
    accel = "cpu",
    descriptor(crate::builtins::table::builtins::timetable::TIMETABLE2TABLE_DESCRIPTOR),
    extensions(crate::builtins::table::builtins::timetable::TIMETABLE2TABLE_EXTENSIONS),
    integer_capabilities(
        crate::builtins::table::builtins::timetable::TIMETABLE2TABLE_INTEGER_CAPABILITIES
    ),
    builtin_path = "crate::builtins::table::builtins"
)]
pub(crate) async fn timetable2table_builtin(
    value: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
    if rest.chunks_exact(2).any(|pair| {
        scalar_text(&pair[0], "timetable2table option").is_ok_and(|name| {
            name.eq_ignore_ascii_case("ConvertRowTimes")
                && crate::builtins::common::validation::value_contains_native_integer_class(
                    &pair[1],
                )
        })
    }) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &TIMETABLE2TABLE_INTEGER_OPTION_EXTENSION,
            "timetable2table",
        )?;
    }
    if crate::builtins::common::validation::value_contains_explicit_gpu(&value)
        || rest
            .iter()
            .any(crate::builtins::common::validation::value_contains_explicit_gpu)
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &TIMETABLE2TABLE_EXPLICIT_GPU_EXTENSION,
            "timetable2table",
        )?;
    }
    let host = gather_if_needed_async(&value)
        .await
        .map_err(map_control_flow)?;
    let rest = gather_values(&rest).await?;
    let convert_row_times = parse_timetable2table_convert_row_times(&rest)?;
    let object = into_timetable_object(host, "timetable2table")?;
    let mut names = table_variable_names_from_object(&object)?;
    let variables = table_variables(&object)?;
    let mut columns = Vec::with_capacity(names.len() + usize::from(convert_row_times));
    if convert_row_times {
        if let Some(row_times) = timetable_row_times(&object)? {
            columns.push(row_times);
            names.insert(0, "Time".to_string());
        }
    }
    for name in table_variable_names_from_object(&object)? {
        columns.push(variables.fields.get(&name).cloned().ok_or_else(|| {
            invalid_variable(format!("timetable2table: missing variable '{name}'"))
        })?);
    }
    let row_names = selected_row_names(&object, &(0..table_height(&object)?).collect::<Vec<_>>())?;
    table_from_columns_with_properties(names, columns, row_names)
}

fn parse_timetable2table_convert_row_times(args: &[Value]) -> BuiltinResult<bool> {
    if args.is_empty() {
        return Ok(true);
    }
    if args.len() != 2 {
        return Err(invalid_argument(
            "timetable2table: ConvertRowTimes must be supplied as one name-value pair",
        ));
    }
    let name = scalar_text(&args[0], "timetable2table option")?;
    if !name.eq_ignore_ascii_case("ConvertRowTimes") {
        return Err(invalid_argument(format!(
            "timetable2table: unsupported option '{name}'"
        )));
    }
    zero_one_bool_scalar(&args[1], "ConvertRowTimes")
}

#[runtime_builtin(
    name = "readtimetable",
    category = "io/tabular",
    summary = "Read tabular data into a timetable.",
    keywords = "readtimetable,timetable,readtable,RowTimes",
    accel = "cpu",
    descriptor(crate::builtins::table::TABLE_COMPAT_DESCRIPTOR),
    extensions(crate::builtins::table::builtins::timetable::READTIMETABLE_EXTENSIONS),
    integer_capabilities(
        crate::builtins::table::builtins::timetable::READTIMETABLE_INTEGER_CAPABILITIES
    ),
    builtin_path = "crate::builtins::table::builtins"
)]
pub(crate) async fn readtimetable_builtin(path: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    ensure_numeric_row_times_extension(&rest, "readtimetable")?;
    super::io::enforce_table_import_integer_control_gate(
        &rest,
        &READTIMETABLE_TYPED_INTEGER_CONTROL_EXTENSION,
        "readtimetable",
        &["VariableNamesLine"],
    )?;
    let path = gather_if_needed_async(&path)
        .await
        .map_err(map_control_flow)?;
    let rest = gather_values(&rest).await?;
    let (readtable_args, timetable_args) = split_readtimetable_options(&rest)?;
    let table = super::io::readtable_builtin(path, readtable_args).await?;
    table2timetable_builtin(table, timetable_args).await
}
