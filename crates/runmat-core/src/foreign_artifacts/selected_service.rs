//! A selected foreign port retains its exact private bytes through pending
//! calls without retaining a session, selected code product, or context.

use std::rc::Rc;

use runmat_runtime::context::{
    ForeignCall, ForeignCallCompletion, RuntimeContext, RuntimeForeignService, RuntimeServiceFuture,
};

use super::SelectedForeignMaterialization;
use crate::ForeignUnitHost;

struct SelectedForeignService {
    materialization: Rc<SelectedForeignMaterialization>,
    inner: Rc<dyn RuntimeForeignService>,
}

impl RuntimeForeignService for SelectedForeignService {
    fn visit_retained_values(
        &self,
        visitor: &mut dyn FnMut(&runmat_value::Value) -> Option<()>,
    ) -> Option<()> {
        self.inner.visit_retained_values(visitor)
    }

    fn execution_stack_requirement(&self) -> runmat_types::ExecutionStackRequirement {
        self.inner.execution_stack_requirement()
    }

    fn invoke(
        &self,
        context: RuntimeContext,
        call: ForeignCall,
    ) -> RuntimeServiceFuture<ForeignCallCompletion> {
        let materialization = Rc::clone(&self.materialization);
        let invocation = self.inner.invoke(context, call);
        Box::pin(async move {
            let _materialization = materialization;
            invocation.await
        })
    }
}

/// Both interactive Core and the isolated worker bind this port after exact
/// selected-unit admission. The returned service cannot lose installed paths
/// while an invocation context or adapter future is still live.
pub fn selected_foreign_service(
    host: &ForeignUnitHost,
    materialization: Rc<SelectedForeignMaterialization>,
) -> Rc<dyn RuntimeForeignService> {
    Rc::new(SelectedForeignService {
        materialization,
        inner: host.foreign_service(),
    })
}
