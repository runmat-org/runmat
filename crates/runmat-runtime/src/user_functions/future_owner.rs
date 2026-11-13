//! Concrete selected callable futures at their original allocation boundary.

use std::future::Future;

use crate::context::RuntimeContext;
use crate::{execution, RuntimeError};

/// Admit the actual selected semantic/external future before constructing it.
/// A context-free callback is lawful only under the original unbounded policy.
#[inline(never)]
pub fn admit_user_function_with<F>(
    runtime: Option<&RuntimeContext>,
    factory: impl FnOnce() -> F,
) -> Result<super::UserFunctionFuture, RuntimeError>
where
    F: Future<Output = Result<execution::RootedValueSequence, RuntimeError>> + 'static,
{
    match runtime {
        Some(runtime) => execution::admit_executable_sequence_with(runtime, factory),
        None => execution::rooted_continuation::admit_unbounded_callable_with(
            "unscoped callable",
            factory,
        ),
    }
}

/// Same concrete owner for the lexical result's canonical rooted carrier.
#[inline(never)]
pub fn admit_lexical_function_with<F>(
    runtime: Option<&RuntimeContext>,
    factory: impl FnOnce() -> F,
) -> Result<super::LexicalFunctionFuture, RuntimeError>
where
    F: Future<Output = Result<crate::call::lexical::RootedLexicalCallResult, RuntimeError>>
        + 'static,
{
    match runtime {
        Some(runtime) => execution::admit_rooted_result_with(runtime, factory),
        None => execution::rooted_continuation::admit_unbounded_callable_with(
            "unscoped lexical callable",
            factory,
        ),
    }
}
