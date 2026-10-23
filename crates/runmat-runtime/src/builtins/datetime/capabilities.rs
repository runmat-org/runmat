pub(super) use runmat_builtins::{
    DATETIME_GPU_INPUT_EXTENSION, DATETIME_LEGACY_COMPONENT_ARITY_EXTENSION,
    DATETIME_LOGICAL_INPUT_EXTENSION, DATETIME_RAW_DATENUM_EXTENSION,
};

pub(super) const BUILTIN_NAME: &str = "datetime";
pub(super) const DATETIME_CLASS: runmat_types::StaticClassIdentity =
    runmat_types::standard::DATETIME;
pub(super) const CALENDAR_DURATION_CLASS: runmat_types::StaticClassIdentity =
    runmat_types::standard::CALENDAR_DURATION;
pub(super) const SERIAL_FIELD: &str = runmat_builtins::DATETIME_SERIAL_PROPERTY;
pub(super) const CALENDAR_MONTHS_FIELD: &str = runmat_builtins::CALENDAR_MONTHS_PROPERTY;
pub(super) const CALENDAR_DAYS_FIELD: &str = runmat_builtins::CALENDAR_DAYS_PROPERTY;
pub(super) const FORMAT_FIELD: &str = "Format";
pub(super) const DEFAULT_DATE_FORMAT: &str = "dd-MMM-yyyy";
pub(super) const DEFAULT_DATETIME_FORMAT: &str = "dd-MMM-yyyy HH:mm:ss";
pub(super) const UNIX_DATENUM: f64 = 719_529.0;
pub(super) const SECONDS_PER_DAY: f64 = 86_400.0;
pub(super) const MAX_HOLIDAY_YEAR_SPAN: i32 = 1_000;
pub(super) const MAX_BUSDAYS_OUTPUT_LEN: i64 = 1_000_000;
pub use runmat_builtins::{
    DAY_EXTENSIONS, HOUR_EXTENSIONS, HOUR_TYPED_LEGACY_NUMERIC_EXTENSION, MINUTE_EXTENSIONS,
    MINUTE_TYPED_LEGACY_NUMERIC_EXTENSION, MONTH_EXTENSIONS, MONTH_TYPED_LEGACY_NUMERIC_EXTENSION,
    YEAR_EXTENSIONS, YEAR_TYPED_LEGACY_NUMERIC_EXTENSION,
};

pub use runmat_builtins::DATESHIFT_EXTENSIONS;
pub(super) use runmat_builtins::MAX_DATESHIFT_DAY_OCCURRENCE;
