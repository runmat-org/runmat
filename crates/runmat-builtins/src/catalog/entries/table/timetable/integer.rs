use crate::*;
const TIMETABLE2TABLE_INTEGER_VARIABLE_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "integer timetable variables",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Every data variable retains its native integer class, shape, and payload when timetable metadata is converted to table metadata.",
    }];
const TIMETABLE2TABLE_INTEGER_OPTION_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "ConvertRowTimes",
        classes: &crate::ALL_INTEGER_CLASSES,
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

const TIMETABLE_INTEGER_VARIABLE_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "data variables or VariableTypes",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Timetable data variables may use all eight integer classes and retain authoritative native storage.",
    }];
const TIMETABLE_INTEGER_SIZE_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "Size",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The two-element preallocation size is decoded exactly as nonnegative platform dimensions.",
    }];
const TIMETABLE_INTEGER_SAMPLE_RATE_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "SampleRate",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "A positive numeric sample rate crosses the explicit floating seconds-to-days timing boundary after exact integer validation.",
    }];
const TIMETABLE_INTEGER_ROW_TIMES_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "numeric RowTimes",
        classes: &crate::ALL_INTEGER_CLASSES,
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

const TABLE2TIMETABLE_INTEGER_VARIABLES: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "integer variables contained in T",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Non-time table variables retain their exact integer class and payload in the output timetable.",
    }];
const TABLE2TIMETABLE_INTEGER_INDEX: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "RowTimes variable index",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "A positive one-based scalar index selects a datetime or duration table variable and is decoded exactly from authoritative storage.",
    }];
const TABLE2TIMETABLE_INTEGER_SAMPLE_RATE: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "Fs",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The positive numeric SampleRate scalar enters the deliberate floating seconds-to-days timing boundary after exact sign validation.",
    }];
const TABLE2TIMETABLE_INTEGER_ROW_TIMES: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "numeric RowTimes vector",
        classes: &crate::ALL_INTEGER_CLASSES,
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

const READTIMETABLE_INTEGER_VARIABLE_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "VariableTypes integer class",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Documented integer variable types are imported directly into exact native table-variable storage before timetable construction.",
    }];
const READTIMETABLE_INTEGER_LOCATION_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "VariableNamesLine",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The public datatype table explicitly includes all eight integer classes. RunMat reads the authoritative scalar exactly and bounds-checks it before deriving the host header offset.",
    }];
const READTIMETABLE_INTEGER_EXTENSION_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "Range, Sheet, NumHeaderLines, ExpectedNumVariables, boolean numeric flags, or related floating-only control",
        classes: &crate::ALL_INTEGER_CLASSES,
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
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The documented homogeneous array domain includes all eight real integer classes.",
    }];

const ARRAY2TIMETABLE_INTEGER_SAMPLE_RATE_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "Fs",
        classes: &crate::ALL_INTEGER_CLASSES,
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
