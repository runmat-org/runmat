//! Catalog authority for timer construction, discovery, methods, and setters.

mod contract;
mod documentation;
mod inference;
mod property;
#[macro_use]
mod identity;
mod constructors;
mod discovery;
mod methods;
mod setters;

#[cfg(test)]
mod tests;

pub use contract::{
    TIMERFIND_EXPLICIT_GPU_NUMERIC_FILTER_EXTENSION, TIMER_ERROR_CALLBACK, TIMER_ERROR_GC,
    TIMER_ERROR_INVALID_HANDLE, TIMER_ERROR_INVALID_INPUT, TIMER_ERROR_INVALID_PROPERTY,
    TIMER_EXPLICIT_GPU_NUMERIC_PROPERTY_EXTENSION, TIMER_METHOD_STATUS_EXTENSION,
    TIMER_STARTAT_RELATIVE_EXTENSION,
};
pub use property::{TimerBusyMode, TimerExecutionMode, TimerProperty, TimerVisibility};

use crate::BuiltinCatalogEntry;
use constructors::TIMER;
use discovery::{TIMERFIND, TIMERFINDALL};
use methods::{DELETE, START, STARTAT, STOP, WAIT};
use setters::{
    SET_BUSY_MODE, SET_ERROR_FCN, SET_EXECUTION_MODE, SET_NAME, SET_OBJECT_VISIBILITY, SET_PERIOD,
    SET_START_DELAY, SET_START_FCN, SET_STOP_FCN, SET_TAG, SET_TASKS_TO_EXECUTE, SET_TIMER_FCN,
    SET_USER_DATA,
};

pub(crate) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &TIMER,
    &TIMERFIND,
    &TIMERFINDALL,
    &START,
    &STARTAT,
    &STOP,
    &WAIT,
    &DELETE,
    &SET_TIMER_FCN,
    &SET_START_FCN,
    &SET_STOP_FCN,
    &SET_ERROR_FCN,
    &SET_PERIOD,
    &SET_START_DELAY,
    &SET_TASKS_TO_EXECUTE,
    &SET_BUSY_MODE,
    &SET_EXECUTION_MODE,
    &SET_NAME,
    &SET_TAG,
    &SET_OBJECT_VISIBILITY,
    &SET_USER_DATA,
];
