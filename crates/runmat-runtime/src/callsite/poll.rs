//! Poll-local provenance stored inline in the caller's existing future owner.
use super::{admit_callsite, push_callsite_info, push_function_input_callsite_info, CallsiteInfo};
use crate::RuntimeError;
use std::{
    future::Future,
    pin::Pin,
    task::{Context, Poll},
};

pub(super) enum ScopeKind {
    Ordinary,
    FunctionInput,
}

pin_project_lite::pin_project! {
    struct Scoped<F> {
        #[pin]
        future: Option<F>,
        info: Option<CallsiteInfo>,
        projection: ScopeKind,
        admitted: bool,
    }
}

#[inline(always)]
pub(super) fn scope<F: Future>(
    info: Option<CallsiteInfo>,
    future: F,
    projection: ScopeKind,
) -> impl Future<Output = Result<F::Output, RuntimeError>> {
    Scoped {
        future: Some(future),
        info,
        projection,
        admitted: false,
    }
}

impl<F: Future> Future for Scoped<F> {
    type Output = Result<F::Output, RuntimeError>;
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let mut this = self.project();
        assert!(
            this.future.as_ref().get_ref().is_some(),
            "callsite future polled after completion"
        );
        if !*this.admitted {
            match admit_callsite(this.info.take()) {
                Ok(info) => {
                    *this.info = info;
                    *this.admitted = true;
                }
                Err(error) => {
                    this.future.set(None);
                    return Poll::Ready(Err(error));
                }
            }
        }
        let result = {
            let guard = this
                .info
                .as_ref()
                .map(|info| match this.projection {
                    ScopeKind::Ordinary => {
                        push_callsite_info(info.clone()).map(futures::future::Either::Left)
                    }
                    ScopeKind::FunctionInput => push_function_input_callsite_info(info.clone())
                        .map(futures::future::Either::Right),
                })
                .transpose();
            match guard {
                Err(error) => Poll::Ready(Err(error)),
                Ok(_guard) => this
                    .future
                    .as_mut()
                    .as_pin_mut()
                    .expect("live callsite child")
                    .poll(cx)
                    .map(Ok),
            }
        };
        // Retire the child after its poll guard, before returning its output.
        // The inline field remains pinned across every Pending/abandonment path.
        if result.is_ready() {
            this.future.set(None);
            this.info.take();
        }
        result
    }
}
