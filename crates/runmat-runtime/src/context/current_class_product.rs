//! Current-project class selection for data restoration.
//!
//! A serialized class spelling is only a lookup request. The selected result
//! must come from the current source product and its exact constructor lease;
//! a separately registered builtin class carries its static registration.

use std::rc::Rc;

use runmat_types::ClassIdentity;
use runmat_value::SelectedObjectClassFacts;

use super::{RuntimeContext, RuntimeServiceFuture};
use crate::{build_runtime_error, RuntimeError};

#[derive(Clone, Debug)]
pub enum CurrentClassProductResolution {
    Selected(Rc<SelectedObjectClassFacts>),
    RegisteredBuiltin(crate::class_registry::BuiltinClassIssuance),
}

pub type CurrentClassProductResolver =
    dyn Fn(
        RuntimeContext,
        ClassIdentity,
    ) -> RuntimeServiceFuture<Result<CurrentClassProductResolution, RuntimeError>>;

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
        resolver(self.clone(), class.clone()).await
    }
}

pub(crate) fn unavailable(message: &'static str) -> RuntimeError {
    build_runtime_error(message)
        .with_identifier("RunMat:ClassProduct:Unavailable")
        .build()
}
