use super::capabilities::{CALENDAR_DURATION_CLASS, DATETIME_CLASS};

pub(super) type Broadcast3 = (Vec<f64>, Vec<f64>, Vec<f64>, Vec<usize>);

pub(super) static DATETIME_CLASS_REGISTERED: crate::class_registry::ClassRegistration =
    crate::class_registry::ClassRegistration::new(DATETIME_CLASS);
pub(super) static CALENDAR_DURATION_CLASS_REGISTERED: crate::class_registry::ClassRegistration =
    crate::class_registry::ClassRegistration::new(CALENDAR_DURATION_CLASS);

// Shared constructor-domain kernel errors retain their canonical source identity.
pub(super) use runmat_builtins::{DATETIME_ERROR_INVALID_ARGUMENT, DATETIME_ERROR_INVALID_INPUT};
