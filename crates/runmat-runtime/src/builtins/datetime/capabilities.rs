use runmat_builtins::{
    BuiltinExtensionDescriptor, BuiltinExtensionMode, BuiltinIntegerBackendRule,
    BuiltinIntegerCapabilityDescriptor, BuiltinIntegerComputationDomain,
    BuiltinIntegerInputAvailability, BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule,
    BuiltinIntegerOverflowRule, BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule,
};

pub(super) const BUILTIN_NAME: &str = "datetime";
pub(super) const DATETIME_CLASS: runmat_types::StaticClassIdentity =
    runmat_types::standard::DATETIME;
pub(super) const CALENDAR_DURATION_CLASS: runmat_types::StaticClassIdentity =
    runmat_types::standard::CALENDAR_DURATION;
pub(super) const SERIAL_FIELD: &str = "__serial";
pub(super) const CALENDAR_MONTHS_FIELD: &str = "__months";
pub(super) const CALENDAR_DAYS_FIELD: &str = "__days";
pub(super) const FORMAT_FIELD: &str = "Format";
pub(super) const DEFAULT_DATE_FORMAT: &str = "dd-MMM-yyyy";
pub(super) const DEFAULT_DATETIME_FORMAT: &str = "dd-MMM-yyyy HH:mm:ss";
pub(super) const UNIX_DATENUM: f64 = 719_529.0;
pub(super) const SECONDS_PER_DAY: f64 = 86_400.0;
pub(super) const MAX_HOLIDAY_YEAR_SPAN: i32 = 1_000;
pub(super) const MAX_BUSDAYS_OUTPUT_LEN: i64 = 1_000_000;
// This exceeds the number of any supported target weekdays across Chrono's
// complete NaiveDate range, while keeping the O(1) whole-week offset safely
// representable as a TimeDelta. Larger controls cannot produce a valid date.
pub(super) const MAX_DATESHIFT_DAY_OCCURRENCE: u64 = 200_000_000;

pub(super) const DATETIME_RAW_DATENUM_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "datetime-implicit-datenum",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "A one-argument numeric value that is not an m-by-3 or m-by-6 date vector is interpreted as a serial date number only in RunMat compatibility-extension mode",
    error_identifier: Some("RunMat:compatibility:DatetimeImplicitDatenumExtension"),
};
pub(super) const DATETIME_LEGACY_COMPONENT_ARITY_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "datetime-four-five-components",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "Four- and five-component datetime constructor forms are retained only in RunMat compatibility-extension mode",
        error_identifier: Some("RunMat:compatibility:DatetimeLegacyComponentArityExtension"),
    };
pub(super) const DATETIME_LOGICAL_INPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "datetime-logical-numeric-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description:
            "Logical values in datetime numeric positions are a RunMat-only compatibility extension",
        error_identifier: Some("RunMat:compatibility:DatetimeLogicalInputExtension"),
    };
pub(super) const DATETIME_GPU_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "datetime-resident-numeric-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description:
        "Gathering resident numeric input into a host datetime object is a RunMat-only extension",
    error_identifier: Some("RunMat:compatibility:DatetimeGpuInputExtension"),
};
pub(super) const HOUR_TYPED_LEGACY_NUMERIC_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "hour-typed-legacy-serial-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "hour with single-precision or typed-integer legacy serial-date input is a RunMat extension because the public legacy documentation does not enumerate those storage classes",
        error_identifier: Some("RunMat:compatibility:HourTypedLegacySerialExtension"),
    };
pub(super) const YEAR_TYPED_LEGACY_NUMERIC_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "year-typed-legacy-serial-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "year with single-precision or typed-integer legacy serial-date input is a RunMat extension because the public legacy documentation does not enumerate those storage classes",
        error_identifier: Some("RunMat:compatibility:YearTypedLegacySerialExtension"),
    };
pub(super) const MINUTE_TYPED_LEGACY_NUMERIC_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "minute-typed-legacy-serial-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "minute with single-precision or typed-integer legacy serial-date input is a RunMat extension because the public legacy documentation does not enumerate those storage classes",
        error_identifier: Some("RunMat:compatibility:MinuteTypedLegacySerialExtension"),
    };
pub(super) const MONTH_TYPED_LEGACY_NUMERIC_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "month-typed-legacy-serial-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "month with single-precision or typed-integer legacy serial-date input is a RunMat extension because the public legacy documentation does not enumerate those storage classes",
        error_identifier: Some("RunMat:compatibility:MonthTypedLegacySerialExtension"),
    };
pub const DATETIME_EXTENSIONS: [BuiltinExtensionDescriptor; 4] = [
    DATETIME_RAW_DATENUM_EXTENSION,
    DATETIME_LEGACY_COMPONENT_ARITY_EXTENSION,
    DATETIME_LOGICAL_INPUT_EXTENSION,
    DATETIME_GPU_INPUT_EXTENSION,
];
pub const DAY_EXTENSIONS: [BuiltinExtensionDescriptor; 2] = [
    DATETIME_LOGICAL_INPUT_EXTENSION,
    DATETIME_GPU_INPUT_EXTENSION,
];
pub const HOUR_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    HOUR_TYPED_LEGACY_NUMERIC_EXTENSION,
    DATETIME_LOGICAL_INPUT_EXTENSION,
    DATETIME_GPU_INPUT_EXTENSION,
];
pub const YEAR_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    YEAR_TYPED_LEGACY_NUMERIC_EXTENSION,
    DATETIME_LOGICAL_INPUT_EXTENSION,
    DATETIME_GPU_INPUT_EXTENSION,
];
pub const MINUTE_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    MINUTE_TYPED_LEGACY_NUMERIC_EXTENSION,
    DATETIME_LOGICAL_INPUT_EXTENSION,
    DATETIME_GPU_INPUT_EXTENSION,
];
pub const MONTH_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    MONTH_TYPED_LEGACY_NUMERIC_EXTENSION,
    DATETIME_LOGICAL_INPUT_EXTENSION,
    DATETIME_GPU_INPUT_EXTENSION,
];
pub const DATESHIFT_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [DATETIME_GPU_INPUT_EXTENSION];

pub(super) const DATETIME_INTEGER_COMPONENT_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "date vectors and Y/M/D/H/M/S/MS components",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "All eight integer classes are read from authoritative storage and validated before conversion at the internal serial-date boundary.",
    }];
pub(super) const DATETIME_INTEGER_CONVERT_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "X with ConvertFrom",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Integer conversion input is documented; RunMat currently implements datenum only and reports other conversion epochs explicitly.",
    }];
pub(super) const DATETIME_INTEGER_RESIDENT_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "resident numeric input",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Resident input is gated before provider access and gathered only when RunMat extensions are enabled.",
    }];
pub const DATETIME_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    BuiltinIntegerCapabilityDescriptor { form: "datetime(integer_date_vector_or_components)", inputs: &DATETIME_INTEGER_COMPONENT_INPUTS, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::FunctionSpecific, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::HostOnly, overload: BuiltinIntegerOverloadKind::Multiple, notes: "Calendar structure is resolved exactly before the representable instant is stored at RunMat's current floating serial-date boundary." },
    BuiltinIntegerCapabilityDescriptor { form: "datetime(integer_X, 'ConvertFrom', dateType)", inputs: &DATETIME_INTEGER_CONVERT_INPUTS, computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific, output_class: BuiltinIntegerOutputClassRule::FunctionSpecific, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::HostOnly, overload: BuiltinIntegerOverloadKind::Multiple, notes: "The output is a host datetime object. TT2000 and other conversion epochs remain explicit implementation gaps; no false nanosecond-precision claim is made." },
    BuiltinIntegerCapabilityDescriptor { form: "datetime(resident_integer, ...)", inputs: &DATETIME_INTEGER_RESIDENT_INPUTS, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::FunctionSpecific, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::GpuRestricted, overload: BuiltinIntegerOverloadKind::Multiple, notes: "MATLAB-compatible mode rejects resident numeric input before provider lookup; extension mode gathers and returns a host datetime object." },
];
