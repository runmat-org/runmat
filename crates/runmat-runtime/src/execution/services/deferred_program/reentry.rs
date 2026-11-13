//! An admitted definition/source receipt, not a callback override. The real
//! restored snapshot additionally checks its V3/static/interop owner before
//! requesting this receipt; the supplied descriptor is never rewritten.
use crate::{
    call::descriptor::CallableTarget,
    context::RuntimeContext,
    source_context::{RuntimeSourceRef, SelectedProgramSourceOwner},
    RuntimeError,
};
use runmat_execution::FunctionBindingOrigin;
use runmat_value::{BoundFunctionBinding, FunctionDefinitionOwner, Value};
use std::sync::Arc;
#[derive(Clone)]
pub struct AdmittedCallableReentry {
    source: SelectedProgramSourceOwner,
    origin: FunctionBindingOrigin,
    definition: Arc<FunctionDefinitionOwner>,
}
impl AdmittedCallableReentry {
    pub fn same_selected_definition(&self, other: &Self) -> bool {
        self.source.same_issuance(&other.source)
            && self.origin == other.origin
            && Arc::ptr_eq(&self.definition, &other.definition)
    }

    /// Host-side admitted producer only: all arguments come from its exact
    /// selected body, never serialized SourceIds, names or claimed digests.
    pub fn from_selected_definition(
        runtime: &RuntimeContext,
        source: SelectedProgramSourceOwner,
        binding: &BoundFunctionBinding,
        origin: FunctionBindingOrigin,
        definition: Arc<FunctionDefinitionOwner>,
        issued_source: RuntimeSourceRef,
    ) -> Result<Self, RuntimeError> {
        if !source.is_issued_in(runtime)
            || !source.owns_source(issued_source)
            || binding.authority.export_origin(&source.session_lease()) != Some(origin)
            || binding
                .authority
                .local_lease(binding.function)
                .is_none_or(|lease| !lease.matches(&definition))
        {
            return Err(unavailable(
                "retained callable differs from its admitted definition/source",
            ));
        }
        Ok(Self {
            source,
            origin,
            definition,
        })
    }
    pub(crate) fn validate(
        &self,
        runtime: &RuntimeContext,
        target: &CallableTarget,
    ) -> Result<(), RuntimeError> {
        let binding = match target {
            CallableTarget::Bound(binding) => binding,
            CallableTarget::BoundLexical { binding, closure } => {
                if !matches!(closure.as_value(), Value::Closure(value) if value.binding.as_ref() == Some(binding))
                {
                    return Err(unavailable(
                        "retained closure changed its exact bound definition",
                    ));
                }
                binding
            }
            _ => {
                return Err(unavailable(
                    "retained reentry requires an exact selected bound target",
                ))
            }
        };
        if !self.source.is_issued_in(runtime)
            || binding
                .authority
                .export_origin(&self.source.session_lease())
                != Some(self.origin)
            || binding
                .authority
                .local_lease(binding.function)
                .is_none_or(|lease| !lease.matches(&self.definition))
        {
            return Err(unavailable(
                "retained callable no longer matches its admitted source/definition",
            ));
        }
        Ok(())
    }
}
pub(super) fn unavailable(message: &str) -> RuntimeError {
    crate::build_runtime_error(message)
        .with_identifier("RunMat:Execution:SelectedCallableUnavailable")
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .build()
}
