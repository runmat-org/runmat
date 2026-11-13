use crate::source_context;
use runmat_thread_local::runmat_thread_local;
use runmat_types::{SourceId, Span};
use std::cell::RefCell;
use std::future::Future;

#[path = "callsite/spans.rs"]
mod spans;
pub use spans::CallsiteSpans;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CallsiteInfo {
    pub source_id: Option<SourceId>,
    pub arg_spans: CallsiteSpans,
}

impl CallsiteInfo {
    pub fn from_projected_in_context<T>(
        source_id: Option<SourceId>,
        arguments: &[T],
        context: &crate::context::RuntimeContext,
        project: impl FnMut(&T) -> Span,
    ) -> Result<Self, crate::RuntimeError> {
        Ok(Self {
            source_id,
            arg_spans: CallsiteSpans::from_projected_in_context(arguments, context, project)
                .map_err(|source| {
                    crate::context::host_allocation::host_allocation_runtime_error(
                        Some(context),
                        "RunMat:Runtime:CallsiteProjection",
                        "callsite metadata admission failed",
                        source,
                    )
                })?,
        })
    }
    pub fn from_borrowed_in_context(
        source_id: Option<SourceId>,
        arg_spans: &[Span],
        context: &crate::context::RuntimeContext,
    ) -> Result<Self, crate::RuntimeError> {
        Ok(Self {
            source_id,
            arg_spans: CallsiteSpans::from_borrowed_in_context(arg_spans, context).map_err(
                |source| {
                    crate::context::host_allocation::host_allocation_runtime_error(
                        Some(context),
                        "RunMat:Runtime:CallsiteProjection",
                        "callsite metadata admission failed",
                        source,
                    )
                },
            )?,
        })
    }

    pub(crate) fn retained_quote(
        &self,
    ) -> Result<(u64, usize), crate::context::host_allocation::HostAllocationError> {
        self.arg_spans.retained_quote()
    }

    pub(crate) fn admit(
        self,
        context: &crate::context::RuntimeContext,
    ) -> Result<Self, crate::context::host_allocation::HostAllocationError> {
        Ok(Self {
            source_id: self.source_id,
            arg_spans: self.arg_spans.admit(context)?,
        })
    }

    pub(crate) fn admitted_clone_with_permit(
        &self,
        permit: crate::context::host_allocation::HostOutputPermit,
    ) -> Result<Self, crate::context::host_allocation::HostAllocationError> {
        Ok(Self {
            source_id: self.source_id,
            arg_spans: self.arg_spans.admitted_clone_with_permit(permit)?,
        })
    }

    pub fn without_first_argument(mut self) -> Self {
        self.arg_spans = self.arg_spans.without_first();
        self
    }
}

#[path = "callsite/poll.rs"]
mod poll;

/// Keep ordinary source metadata active only during each child poll.
/// Admission remains lazy and this wrapper introduces no allocation.
pub fn with_callsite<F: Future>(
    info: Option<CallsiteInfo>,
    future: F,
) -> impl Future<Output = Result<F::Output, crate::RuntimeError>> {
    poll::scope(info, future, poll::ScopeKind::Ordinary)
}

/// Scope caller-expression provenance to each poll, not a suspended task.
pub fn with_function_input_callsite<F: Future>(
    info: Option<CallsiteInfo>,
    future: F,
) -> impl Future<Output = Result<F::Output, crate::RuntimeError>> {
    poll::scope(info, future, poll::ScopeKind::FunctionInput)
}

fn admit_callsite(info: Option<CallsiteInfo>) -> Result<Option<CallsiteInfo>, crate::RuntimeError> {
    let Some(runtime) = crate::context::legacy::active() else {
        return Ok(info);
    };
    info.map(|info| {
        info.admit(&runtime).map_err(|source| {
            crate::context::host_allocation::host_allocation_runtime_error(
                Some(&runtime),
                "RunMat:Runtime:CallsiteProjection",
                "callsite metadata admission failed",
                source,
            )
        })
    })
    .transpose()
}

runmat_thread_local! {
    static CALLSITE_STACK: RefCell<Vec<CallsiteInfo>> = const { RefCell::new(Vec::new()) };
    static FUNCTION_INPUT_CALLSITE_STACK: RefCell<Vec<CallsiteInfo>> = const { RefCell::new(Vec::new()) };
}

pub struct CallsiteGuard {
    did_push: bool,
    projection: Option<crate::context::call_projection::CallProjectionGuard>,
    state: Option<std::rc::Rc<crate::context::RuntimeContextState>>,
}

impl Drop for CallsiteGuard {
    fn drop(&mut self) {
        if self.projection.take().is_some() {
            return;
        }
        if !self.did_push {
            return;
        }
        if let Some(state) = &self.state {
            state.call.borrow_mut().callsites.pop();
        } else {
            CALLSITE_STACK.with(|stack| {
                let mut stack = stack.borrow_mut();
                let _ = stack.pop();
            });
        }
    }
}

pub struct FunctionInputCallsiteGuard {
    did_push: bool,
    projection: Option<crate::context::call_projection::CallProjectionGuard>,
    state: Option<std::rc::Rc<crate::context::RuntimeContextState>>,
}

impl Drop for FunctionInputCallsiteGuard {
    fn drop(&mut self) {
        if self.projection.take().is_some() {
            return;
        }
        if !self.did_push {
            return;
        }
        if let Some(state) = &self.state {
            state.call.borrow_mut().function_input_callsites.pop();
        } else {
            FUNCTION_INPUT_CALLSITE_STACK.with(|stack| {
                let mut stack = stack.borrow_mut();
                let _ = stack.pop();
            });
        }
    }
}

pub fn push_callsite(
    source_id: Option<SourceId>,
    arg_spans: Option<Vec<Span>>,
) -> Result<CallsiteGuard, crate::RuntimeError> {
    push_callsite_spans(source_id, arg_spans.map(CallsiteSpans::from))
}

pub fn push_callsite_spans(
    source_id: Option<SourceId>,
    arg_spans: Option<CallsiteSpans>,
) -> Result<CallsiteGuard, crate::RuntimeError> {
    let Some(arg_spans) = arg_spans else {
        return Ok(CallsiteGuard {
            did_push: false,
            projection: None,
            state: None,
        });
    };
    let info = CallsiteInfo {
        source_id,
        arg_spans,
    };
    let info = admit_callsite(Some(info))?.expect("callsite is present");
    push_callsite_info(info)
}

fn push_callsite_info(info: CallsiteInfo) -> Result<CallsiteGuard, crate::RuntimeError> {
    if let Some(runtime) = crate::context::legacy::active() {
        return Ok(CallsiteGuard {
            did_push: true,
            projection: Some(crate::context::call_projection::push_callsite(
                runtime.call_projection(),
                info,
            )?),
            state: None,
        });
    }
    if let Some(state) = active_state() {
        state.call.borrow_mut().callsites.push(info);
        Ok(CallsiteGuard {
            did_push: true,
            projection: None,
            state: Some(state),
        })
    } else {
        CALLSITE_STACK.with(|stack| stack.borrow_mut().push(info));
        Ok(CallsiteGuard {
            did_push: true,
            projection: None,
            state: None,
        })
    }
}

pub fn push_function_input_callsite(
    source_id: Option<SourceId>,
    arg_spans: Option<Vec<Span>>,
) -> Result<FunctionInputCallsiteGuard, crate::RuntimeError> {
    push_function_input_callsite_spans(source_id, arg_spans.map(CallsiteSpans::from))
}

pub fn push_function_input_callsite_spans(
    source_id: Option<SourceId>,
    arg_spans: Option<CallsiteSpans>,
) -> Result<FunctionInputCallsiteGuard, crate::RuntimeError> {
    let Some(arg_spans) = arg_spans else {
        return Ok(FunctionInputCallsiteGuard {
            did_push: false,
            projection: None,
            state: None,
        });
    };
    let info = CallsiteInfo {
        source_id,
        arg_spans,
    };
    let info = admit_callsite(Some(info))?.expect("callsite is present");
    push_function_input_callsite_info(info)
}

fn push_function_input_callsite_info(
    info: CallsiteInfo,
) -> Result<FunctionInputCallsiteGuard, crate::RuntimeError> {
    if let Some(runtime) = crate::context::legacy::active() {
        return Ok(FunctionInputCallsiteGuard {
            did_push: true,
            projection: Some(
                crate::context::call_projection::push_function_input_callsite(
                    runtime.call_projection(),
                    info,
                )?,
            ),
            state: None,
        });
    }
    if let Some(state) = active_state() {
        state.call.borrow_mut().function_input_callsites.push(info);
        Ok(FunctionInputCallsiteGuard {
            did_push: true,
            projection: None,
            state: Some(state),
        })
    } else {
        FUNCTION_INPUT_CALLSITE_STACK.with(|stack| stack.borrow_mut().push(info));
        Ok(FunctionInputCallsiteGuard {
            did_push: true,
            projection: None,
            state: None,
        })
    }
}

pub fn current_callsite() -> Option<CallsiteInfo> {
    if let Some(runtime) = crate::context::legacy::active() {
        return runtime
            .call_projection()
            .borrow()
            .callsite()
            .or_else(|| runtime.state().call.borrow().callsites.last().cloned());
    }
    CALLSITE_STACK.with(|stack| stack.borrow().last().cloned())
}

pub fn current_function_input_callsite() -> Option<CallsiteInfo> {
    if let Some(runtime) = crate::context::legacy::active() {
        return runtime
            .call_projection()
            .borrow()
            .function_input_callsite()
            .or_else(|| {
                runtime
                    .state()
                    .call
                    .borrow()
                    .function_input_callsites
                    .last()
                    .cloned()
            });
    }
    FUNCTION_INPUT_CALLSITE_STACK.with(|stack| stack.borrow().last().cloned())
}

fn active_state() -> Option<std::rc::Rc<crate::context::RuntimeContextState>> {
    crate::context::legacy::active().map(|context| std::rc::Rc::clone(context.state()))
}

fn clamp_to_char_boundary(s: &str, mut idx: usize) -> usize {
    if idx > s.len() {
        idx = s.len();
    }
    while idx > 0 && !s.is_char_boundary(idx) {
        idx -= 1;
    }
    idx
}

fn normalize_label_text(raw: &str) -> String {
    let collapsed = raw.split_whitespace().collect::<Vec<_>>().join(" ");
    let trimmed = collapsed.trim();
    const MAX_LEN: usize = 80;
    if trimmed.len() <= MAX_LEN {
        return trimmed.to_string();
    }
    // Conservative truncation: keep valid UTF-8 boundaries.
    let mut end = MAX_LEN;
    while end > 0 && !trimmed.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}…", &trimmed[..end])
}

pub fn arg_text(arg_index: usize) -> Option<String> {
    let callsite = current_callsite()?;
    arg_text_for_callsite(&callsite, arg_index)
}

pub fn function_input_arg_text(arg_index: usize) -> Option<String> {
    let callsite = current_function_input_callsite()?;
    arg_text_for_callsite(&callsite, arg_index)
}

fn source_for_callsite(callsite: &CallsiteInfo) -> Option<std::sync::Arc<str>> {
    callsite
        .source_id
        .and_then(source_context::source_info)
        .map(|source| source.text)
        .or_else(source_context::current_source)
}

fn arg_text_for_callsite(callsite: &CallsiteInfo, arg_index: usize) -> Option<String> {
    let source = source_for_callsite(callsite)?;
    let span = callsite.arg_spans.get(arg_index)?;

    let start = clamp_to_char_boundary(&source, span.start);
    let end = clamp_to_char_boundary(&source, span.end);
    if end <= start {
        return None;
    }
    let raw = &source[start..end];
    let text = normalize_label_text(raw);
    if text.is_empty() {
        None
    } else {
        Some(text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::RuntimeContext;
    use crate::execution::RuntimeExecutionService;
    use std::rc::Rc;
    use std::task::{Context, Poll};

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn interleaved_pending_input_callsites_restore_each_session() {
        let first = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        let second = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        let future = |source: usize| {
            with_function_input_callsite(
                Some(CallsiteInfo {
                    source_id: Some(SourceId(source)),
                    arg_spans: vec![Span {
                        start: source,
                        end: source + 1,
                    }]
                    .into(),
                }),
                std::future::poll_fn(move |_| {
                    let current = current_function_input_callsite()
                        .expect("callsite active only during poll");
                    assert_eq!(current.source_id, Some(SourceId(source)));
                    Poll::<()>::Pending
                }),
            )
        };
        let mut first = Box::pin(first.scope(future(1)));
        let mut second = Box::pin(second.scope(future(2)));
        let waker = futures::task::noop_waker();
        let mut context = Context::from_waker(&waker);
        assert!(first.as_mut().poll(&mut context).is_pending());
        assert!(current_function_input_callsite().is_none());
        assert!(second.as_mut().poll(&mut context).is_pending());
        assert!(current_function_input_callsite().is_none());
        drop(first);
        assert!(second.as_mut().poll(&mut context).is_pending());
        drop(second);
        assert!(current_function_input_callsite().is_none());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn ordinary_callsite_is_not_ambient_while_builtin_future_is_pending() {
        let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        let info = CallsiteInfo {
            source_id: Some(SourceId(9)),
            arg_spans: vec![Span { start: 1, end: 2 }].into(),
        };
        let mut pending = Box::pin(runtime.scope(with_callsite(
            Some(info.clone()),
            std::future::poll_fn(move |_| {
                assert_eq!(current_callsite(), Some(info.clone()));
                Poll::<()>::Pending
            }),
        )));
        let waker = futures::task::noop_waker();
        let mut context = Context::from_waker(&waker);
        assert!(pending.as_mut().poll(&mut context).is_pending());
        assert!(current_callsite().is_none());
        drop(pending);
        assert!(current_callsite().is_none());
    }
}
