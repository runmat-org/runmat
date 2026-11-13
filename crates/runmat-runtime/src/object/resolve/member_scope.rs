//! Concrete scope selection for the shared member-assignment evaluator.
//! Select the mode before its existing admitted continuation erases the future,
//! so an ordinary unboxed child cannot enlarge a contextual child's state.
use crate::context::{ContextFuture, RuntimeContext};
use std::future::Future;

pub(crate) trait MemberStoreScope<'a>: Copy {
    type Scoped<F: Future + 'a>: Future<Output = F::Output> + 'a;

    fn runtime(self) -> Option<&'a RuntimeContext>;
    fn scope<F: Future + 'a>(self, future: F) -> Self::Scoped<F>;
}

#[derive(Clone, Copy)]
pub(crate) struct OrdinaryMemberScope;

impl<'a> MemberStoreScope<'a> for OrdinaryMemberScope {
    type Scoped<F: Future + 'a> = F;

    fn runtime(self) -> Option<&'a RuntimeContext> {
        None
    }

    fn scope<F: Future + 'a>(self, future: F) -> F {
        future
    }
}

#[derive(Clone, Copy)]
pub(crate) struct ContextualMemberScope<'a>(pub(crate) &'a RuntimeContext);

impl<'a> MemberStoreScope<'a> for ContextualMemberScope<'a> {
    type Scoped<F: Future + 'a> = ContextFuture<F>;

    fn runtime(self) -> Option<&'a RuntimeContext> {
        Some(self.0)
    }

    fn scope<F: Future + 'a>(self, future: F) -> ContextFuture<F> {
        self.0.scope(future)
    }
}
