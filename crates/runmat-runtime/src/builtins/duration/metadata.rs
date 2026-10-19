//! Runtime class identity and storage constants. Public contracts and
//! documentation belong to the `runmat-builtins` catalog.

pub(super) const BUILTIN_NAME: &str = "duration";
pub(super) const DURATION_CLASS: runmat_types::StaticClassIdentity =
    runmat_types::standard::DURATION;
pub(super) const DAYS_FIELD: &str = runmat_builtins::DURATION_DAYS_PROPERTY;
pub(super) const FORMAT_FIELD: &str = "Format";
pub(crate) const DEFAULT_DURATION_FORMAT: &str = "hh:mm:ss";
pub(super) const SECONDS_PER_DAY: f64 = 86_400.0;

pub(super) static DURATION_CLASS_REGISTERED: crate::class_registry::ClassRegistration =
    crate::class_registry::ClassRegistration::new(DURATION_CLASS);
