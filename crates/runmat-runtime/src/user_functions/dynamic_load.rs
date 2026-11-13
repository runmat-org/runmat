//! Preserve the original pre-selection argument owner at an admitted port.

use std::ops::ControlFlow;

use super::{DynamicFunctionLoadAttempt, DynamicFunctionLoadFuture};
use crate::RuntimeError;

/// Admission and continuation failures are terminal selected-port failures,
/// never a spelling miss that can replay resolution with another route.
pub async fn await_dynamic_function_load(
    future: Result<DynamicFunctionLoadFuture, RuntimeError>,
) -> DynamicFunctionLoadAttempt {
    match future {
        Err(error) => ControlFlow::Break(Err(error)),
        Ok(future) => match future.await {
            Err(error) => ControlFlow::Break(Err(error)),
            Ok(ControlFlow::Break(outputs)) => ControlFlow::Break(Ok(outputs)),
            Ok(ControlFlow::Continue(arguments)) => ControlFlow::Continue(arguments),
        },
    }
}
