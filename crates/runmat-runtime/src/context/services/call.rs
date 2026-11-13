//! One rooted callable request across foreign callbacks and executor routing.

use runmat_types::SourceId;
use runmat_value::Value;

use super::{RuntimeContext, RuntimeServiceFuture};
use crate::RuntimeError;

#[derive(Debug, Clone)]
pub struct RuntimeCallRequest {
    /// One selected callable and its supplied arguments. Closure captures and
    /// lexical ancestry remain in the descriptor's retained target owner.
    descriptor: crate::call::descriptor::CallableDescriptor,
    // This is the same physical callable generation as the descriptor's
    // retained target, with its GC roots kept through an unpolled call future.
    source_roots: Option<crate::call::RetainedCallableTarget>,
    // The descriptor owns the original supplied Vec; this lease protects its
    // GC children until the selected call finishes or the request is dropped.
    supplied_roots: crate::execution::RootedValueLease,
}

impl RuntimeCallRequest {
    pub(crate) fn new(
        descriptor: crate::call::descriptor::CallableDescriptor,
        source_roots: Option<crate::call::RetainedCallableTarget>,
        supplied_roots: crate::execution::RootedValueLease,
    ) -> Result<Self, RuntimeError> {
        use crate::call::descriptor::CallableTarget;

        let target_retained = matches!(
            &descriptor.target,
            CallableTarget::BoundLexical { .. }
                | CallableTarget::CapturedResolved { .. }
                | CallableTarget::RetainedFevalForward(_)
                | CallableTarget::RetainedBuiltin(_)
        );
        let captured_matches = match &descriptor.target {
            CallableTarget::BoundLexical { closure, .. }
            | CallableTarget::CapturedResolved { closure, .. } => source_roots
                .as_ref()
                .map(|owner| closure.matches_callable_owner(owner)),
            CallableTarget::RetainedFevalForward(value)
            | CallableTarget::RetainedBuiltin(value) => source_roots
                .as_ref()
                .map(|owner| value.matches_callable_owner(owner)),
            _ => None,
        };
        if target_retained && captured_matches != Some(true) {
            return Err(crate::runtime_error::semantic_error(
                "RuntimeCallOwner",
                "callback target and retained source owner disagree",
            ));
        }
        Ok(Self {
            descriptor,
            source_roots,
            supplied_roots,
        })
    }

    pub fn descriptor(&self) -> &crate::call::descriptor::CallableDescriptor {
        &self.descriptor
    }

    pub(crate) fn into_parts(
        self,
    ) -> (
        crate::call::descriptor::CallableDescriptor,
        Option<crate::call::RetainedCallableTarget>,
        crate::execution::RootedValueLease,
    ) {
        (self.descriptor, self.source_roots, self.supplied_roots)
    }
}

pub trait RuntimeCallService {
    /// Complete implementation-owned Values; unavailable is not an empty set.
    fn visit_retained_values(&self, _visitor: &mut dyn FnMut(&Value) -> Option<()>) -> Option<()> {
        None
    }
    fn resolve(&self, name: &str) -> Option<usize>;

    fn invoke(
        &self,
        runtime: RuntimeContext,
        request: RuntimeCallRequest,
    ) -> RuntimeServiceFuture<Result<crate::execution::RootedValueSequence, RuntimeError>>;

    fn source_functions(&self, _source_id: SourceId) -> Vec<(String, usize)> {
        Vec::new()
    }
}

/// Canonical call router for callbacks that re-enter the active RunMat
/// executor. The executor installs its semantic and external invokers in the
/// invocation context; this service keeps foreign adapters independent of VM,
/// JIT, and AOT implementation details.
#[derive(Debug, Default)]
pub struct RuntimeCallRouter;

impl RuntimeCallService for RuntimeCallRouter {
    fn visit_retained_values(&self, _visitor: &mut dyn FnMut(&Value) -> Option<()>) -> Option<()> {
        // This router has no state; pending requests belong to caller futures.
        Some(())
    }
    fn resolve(&self, name: &str) -> Option<usize> {
        crate::user_functions::resolve_semantic_function_by_name(name)
    }

    fn invoke(
        &self,
        runtime: RuntimeContext,
        request: RuntimeCallRequest,
    ) -> RuntimeServiceFuture<Result<crate::execution::RootedValueSequence, RuntimeError>> {
        let (descriptor, source_roots, supplied_roots) = request.into_parts();
        Box::pin(async move {
            let _source_roots = source_roots;
            let _supplied_roots = supplied_roots;
            crate::call::descriptor::execute_callable_descriptor_in_context(&runtime, descriptor)
                .await
        })
    }

    fn source_functions(&self, source_id: SourceId) -> Vec<(String, usize)> {
        crate::user_functions::source_functions_for(source_id)
            .into_iter()
            .map(|function| (function.name, function.function))
            .collect()
    }
}
