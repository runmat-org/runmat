use runmat_builtins::{
    BuiltinExtensionDescriptor, BuiltinExtensionMode, BuiltinIntegerBackendRule,
    BuiltinIntegerCapabilityDescriptor, BuiltinIntegerComputationDomain,
    BuiltinIntegerInputAvailability, BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule,
    BuiltinIntegerOverflowRule, BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule,
};
pub(super) use runmat_builtins::{
    DATETIME_GPU_INPUT_EXTENSION, DATETIME_LEGACY_COMPONENT_ARITY_EXTENSION,
    DATETIME_LOGICAL_INPUT_EXTENSION, DATETIME_RAW_DATENUM_EXTENSION,
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
