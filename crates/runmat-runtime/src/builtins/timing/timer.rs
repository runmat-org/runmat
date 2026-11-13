//! MATLAB-compatible session-owned timer handle objects.
//!
//! The host local-task driver polls scheduled callbacks after `start` returns.

mod access;
mod api;
mod arguments;
mod callback;
mod class;
mod discovery;
mod dispatch;
mod error;
mod lifecycle;
mod normalization;
pub(crate) mod property;
mod setter;
mod setter_bindings;
mod startat;

pub use api::*;
pub use setter_bindings::*;

use access::*;
use arguments::{apply_name_value_pairs, prepare_timer_property_pairs, PreparedTimerPair};
use callback::{
    callback_is_empty, run_callback, run_error_callback, text_callback_request,
    update_period_properties, TimerCallback,
};
use class::*;
use discovery::find_timers;
use dispatch::{delete_timer_values, start_timer_values, stop_timer_values, wait_timer_values};
use error::*;
use lifecycle::{start_one_timer, stop_one_timer, wait_one_timer};
use normalization::*;
use property::{TimerBusyMode, TimerExecutionMode, TimerProperty, TimerVisibility};
use runmat_builtins::{
    TIMERFIND_EXPLICIT_GPU_NUMERIC_FILTER_EXTENSION, TIMER_ERROR_CALLBACK, TIMER_ERROR_GC,
    TIMER_ERROR_INVALID_HANDLE, TIMER_ERROR_INVALID_INPUT, TIMER_ERROR_INVALID_PROPERTY,
    TIMER_EXPLICIT_GPU_NUMERIC_PROPERTY_EXTENSION, TIMER_METHOD_STATUS_EXTENSION,
    TIMER_STARTAT_RELATIVE_EXTENSION,
};
use startat::{is_legacy_relative_scalar, start_at_absolute};

#[cfg(all(test, not(target_arch = "wasm32")))]
use once_cell::sync::Lazy;
#[cfg(all(test, not(target_arch = "wasm32")))]
use std::sync::Mutex;

use runmat_builtins::BuiltinErrorDescriptor;
use runmat_macros::runtime_builtin;
use runmat_value::{HandleRef, IntValue, ObjectInstance, Value};

use crate::builtins::common::tensor;
use crate::context::RuntimeContext;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

#[cfg(all(test, not(target_arch = "wasm32")))]
static TIMER_TEST_LOCK: Lazy<Mutex<()>> = Lazy::new(|| Mutex::new(()));

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;

#[cfg(all(test, target_arch = "wasm32"))]
mod browser_tests;
