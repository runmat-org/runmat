use super::*;
use crate::interaction::{replace_async_handler, replace_eval_hook, EvalHookOutcome};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

fn runtime() -> crate::context::RuntimeContext {
    crate::context::RuntimeContext::new(std::rc::Rc::new(
        crate::execution::RuntimeExecutionService::new(),
    ))
}

fn responses(count: Arc<AtomicUsize>) -> crate::interaction::AsyncHandlerGuard {
    replace_async_handler(Some(Arc::new(move |_| {
        let attempt = count.fetch_add(1, Ordering::SeqCst);
        Box::pin(async move {
            let line = match attempt {
                0 => "missing_expression",
                1 => "41",
                _ => return Err("unexpected additional prompt".into()),
            };
            Ok(InteractionResponse::Line(line.into()))
        })
    })))
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn invalid_expression_emits_one_diagnostic_then_requests_a_new_response() {
    let runtime = runtime();
    let _scope = runtime.enter();
    let prompts = Arc::new(AtomicUsize::new(0));
    let _handler = responses(prompts.clone());
    let _eval = replace_eval_hook(Some(Arc::new(|_| {
        Box::pin(async {
            Ok(EvalHookOutcome::InvalidExpression(
                build_runtime_error("undefined expression")
                    .with_failure_origin(
                        crate::runtime_error::RuntimeFailureOrigin::LanguageEvaluation,
                    )
                    .build(),
            ))
        })
    })));
    let capture = crate::console::begin_capture().expect("console capture");
    let value = input_with_empty_prompt().expect("second response succeeds");
    let entries = capture.finish_entries();
    assert_eq!(value, Value::Num(41.0));
    assert_eq!(prompts.load(Ordering::SeqCst), 2);
    let diagnostics: Vec<_> = entries
        .iter()
        .filter(|entry| entry.stream == crate::console::ConsoleStream::Stderr)
        .collect();
    assert_eq!(diagnostics.len(), 1);
    assert!(diagnostics[0].text.contains("undefined expression"));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn cancellation_propagates_its_source_without_prompt_replay() {
    let runtime = runtime();
    let _scope = runtime.enter();
    let prompts = Arc::new(AtomicUsize::new(0));
    let _handler = responses(prompts.clone());
    let _eval = replace_eval_hook(Some(Arc::new(|_| {
        Box::pin(async {
            Err(build_runtime_error("cancelled")
                .with_source(crate::execution::ExecutionServiceError::Cancelled)
                .build())
        })
    })));
    let error = input_with_empty_prompt().expect_err("cancellation propagates");
    assert_eq!(prompts.load(Ordering::SeqCst), 1);
    let source = std::error::Error::source(&error)
        .expect("input evaluation source")
        .downcast_ref::<RuntimeError>()
        .expect("original evaluator error");
    assert!(std::error::Error::source(source)
        .expect("typed cancellation")
        .downcast_ref::<crate::execution::ExecutionServiceError>()
        .is_some());
}
