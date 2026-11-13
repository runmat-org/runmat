//! Current-project class selection for data restoration.
//!
//! A serialized class spelling is only a lookup request. The selected result
//! must come from the current source product and its exact constructor lease;
//! a separately registered builtin class carries its static registration.

use std::rc::Rc;

use runmat_types::ClassIdentity;
use runmat_value::SelectedObjectClassFacts;

use super::RuntimeContext;
use crate::{build_runtime_error, RuntimeError};

#[derive(Clone, Debug)]
pub enum CurrentClassProductResolution {
    Selected(Rc<SelectedObjectClassFacts>),
    RegisteredBuiltin(crate::class_registry::BuiltinClassIssuance),
}

impl runmat_gc::Trace for CurrentClassProductResolution {
    fn trace(&self, _: &mut dyn runmat_gc::Tracer) {
        match self {
            // SelectedObjectClassFacts contains port-free declarations and
            // counted definition bindings, not managed Values. Those bindings
            // retain their own original definition authority.
            Self::Selected(_) => {}
            // Builtin issuance contains only class identity, registry revision
            // and session identity, with no language-value payload.
            Self::RegisteredBuiltin(_) => {}
        }
    }
}

pub type CurrentClassProductFuture = crate::execution::RuntimeContinuationFuture<
    'static,
    crate::execution::RootedResult<CurrentClassProductResolution>,
>;

pub type CurrentClassProductResolver =
    dyn Fn(RuntimeContext, ClassIdentity) -> Result<CurrentClassProductFuture, RuntimeError>;

/// Retain one exact concrete resolver future, not a second box around a
/// previously erased callback. Class selection is root-free typed metadata;
/// its exhaustive Trace adds no completion-root allocation.
pub fn admit_current_class_product_with<F>(
    runtime: &RuntimeContext,
    make_future: impl FnOnce() -> F,
) -> Result<CurrentClassProductFuture, RuntimeError>
where
    F: std::future::Future<Output = Result<CurrentClassProductResolution, RuntimeError>> + 'static,
{
    use futures::FutureExt;
    crate::execution::admit_rooted_result_with(runtime, move || {
        make_future().map(|result| {
            result.and_then(|resolution| {
                crate::execution::RootedResult::new(resolution)
                    .map_err(|error| error.into_runtime_error())
            })
        })
    })
}

impl RuntimeContext {
    /// Install the selected session's current-source resolver. Invocation and
    /// background projections share this session-owned port through CallState.
    pub fn set_current_class_product_resolver(
        &self,
        resolver: Option<Rc<CurrentClassProductResolver>>,
    ) {
        self.state()
            .call
            .borrow_mut()
            .current_class_product_resolver = resolver;
    }

    pub async fn resolve_current_object_class(
        &self,
        class: &ClassIdentity,
    ) -> Result<CurrentClassProductResolution, RuntimeError> {
        let resolver = self
            .state()
            .call
            .borrow()
            .current_class_product_resolver
            .clone();
        let Some(resolver) = resolver else {
            if let Some(issuance) =
                crate::class_registry::issue_builtin_class_in_context(self, class)
            {
                return Ok(CurrentClassProductResolution::RegisteredBuiltin(issuance));
            }
            return Err(unavailable("current class-product resolver is absent"));
        };
        resolver(self.clone(), class.clone())?.await?.with_owned(Ok)
    }
}

pub(crate) fn unavailable(message: &'static str) -> RuntimeError {
    build_runtime_error(message)
        .with_identifier("RunMat:ClassProduct:Unavailable")
        .build()
}
