//! Repeated/owned callback execution uses the canonical descriptor selector.
use super::{CallableDescriptor, CallableTarget, FunctionNameResolver};
use crate::call::{BuiltinArguments, RetainedCallableTarget};
use crate::context::RuntimeContext;
use crate::execution::{RetainedValueReadLease, RootedValue, RootedValueSequence, RootedValues};
use crate::RuntimeError;
use runmat_value::Value;

#[derive(Clone)]
pub struct RepeatableCallable {
    selected: CallableDescriptor,
    // The descriptor has root-free aliases; this source owns actual roots and
    // drops those registrations before the final retained generation.
    _target_owner: RetainedValueReadLease,
    reentry: Option<(
        crate::execution::DeferredGraphProgram,
        crate::execution::AdmittedCallableReentry,
    )>,
}
impl RepeatableCallable {
    pub(crate) fn admitted_reentry_program(
        &self,
    ) -> Option<&crate::execution::DeferredGraphProgram> {
        self.reentry.as_ref().map(|(program, _)| program)
    }

    /// Restore only a compiled catalog identity, never resolve the spelling
    /// through a user's current function namespace or source generation.
    pub(crate) fn from_compiled_builtin(
        runtime: &RuntimeContext,
        builtin: runmat_types::BuiltinId,
    ) -> Result<Self, RuntimeError> {
        if runmat_builtins::builtin_catalog_entry_by_name(&builtin.0)
            .is_none_or(|entry| entry.identity.name != builtin.0)
        {
            return Err(crate::build_runtime_error(
                "saved builtin identity is absent from this runtime",
            )
            .with_identifier("RunMat:Execution:SelectedCallableUnavailable")
            .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
            .build());
        }
        let target = RootedValue::in_context(
            Value::String(crate::call::argument_text::copy_str(&builtin.0, Some(runtime))?.into()),
            runtime,
        )?;
        let owner = RetainedValueReadLease::prepare_callable_rooted(target, Some(runtime))?;
        let retained = super::RetainedCallableValue::from_read_lease(&owner);
        Ok(Self {
            selected: CallableDescriptor {
                target: CallableTarget::RetainedBuiltin(retained),
                args: Vec::new(),
                requested_outputs: 1,
                metadata: super::CallableMetadata::feval(None),
            },
            _target_owner: owner,
            reentry: None,
        })
    }

    pub(crate) fn select(
        runtime: &RuntimeContext,
        target: Value,
        resolver: &impl FunctionNameResolver,
    ) -> Result<Self, RuntimeError> {
        Self::select_rooted(
            Some(runtime),
            RootedValue::in_context(target, runtime)?,
            resolver,
        )
    }
    /// Reuse feval's deferred-name selection where no selected unit resolver
    /// is present. Builtin and already-bound identities still freeze normally.
    pub(crate) fn select_deferred_rooted(
        runtime: Option<&RuntimeContext>,
        target: RootedValue,
    ) -> Result<Self, RuntimeError> {
        Self::select_rooted(runtime, target, &DeferredResolver)
    }
    pub fn select_rooted(
        runtime: Option<&RuntimeContext>,
        target: RootedValue,
        resolver: &impl FunctionNameResolver,
    ) -> Result<Self, RuntimeError> {
        let target = crate::call::feval::refine_target(target, runtime)?;
        let owner = RetainedValueReadLease::prepare_callable_rooted(target, runtime)?;
        let selected = CallableDescriptor::from_retained_feval_value(&owner, 1, resolver)?;
        Ok(Self {
            selected,
            _target_owner: owner,
            reentry: None,
        })
    }
    pub fn with_callsite(mut self, callsite: Option<crate::callsite::CallsiteInfo>) -> Self {
        self.selected.metadata.function_input_callsite = callsite;
        self
    }
    pub(crate) fn with_admitted_graph_reentry(
        mut self,
        program: crate::execution::DeferredGraphProgram,
        receipt: crate::execution::AdmittedCallableReentry,
    ) -> Self {
        self.reentry = Some((program, receipt));
        self
    }
    /// Borrow the root-free selected descriptor while this callable owns its source lease.
    pub fn selected_descriptor(&self) -> &CallableDescriptor {
        &self.selected
    }

    pub fn selected_target(&self) -> &CallableTarget {
        &self.selected.target
    }

    /// Prepare the actual selected callee's ownership only. Backend adapters
    /// must validate its existing binding against their unit and pass the
    /// original BoundLexical environment along with these capture owners.
    pub fn prepare_arguments(
        &self,
        runtime: Option<&RuntimeContext>,
        arguments: BuiltinArguments,
    ) -> Result<crate::call::RootedFunctionArguments, RuntimeError> {
        let captures = match &self.selected.target {
            CallableTarget::BoundLexical { .. } | CallableTarget::CapturedResolved { .. } => true,
            CallableTarget::RetainedFevalForward(value) => {
                matches!(value.value(), Value::Closure(_))
            }
            _ => false,
        };
        let callable = captures.then(|| RetainedCallableTarget::Direct(self._target_owner.clone()));
        arguments
            .rebind_callee(callable, runtime)
            .map(|arguments| arguments.into_function_arguments())
    }

    pub(crate) async fn invoke_rooted(
        &self,
        runtime: &RuntimeContext,
        arguments: RootedValues,
        requested_outputs: usize,
    ) -> Result<RootedValueSequence, RuntimeError> {
        self.invoke_retained(Some(runtime), arguments.into(), requested_outputs)
            .await
    }
    pub fn invoke_retained<'a>(
        &'a self,
        runtime: Option<&'a RuntimeContext>,
        arguments: BuiltinArguments,
        requested_outputs: usize,
    ) -> impl std::future::Future<Output = Result<RootedValueSequence, RuntimeError>> + 'a {
        invocation::run(self, runtime, arguments, requested_outputs)
    }
}
#[path = "repeatable/invocation.rs"]
mod invocation;
pub(super) fn retained_target_name(value: &Value) -> Option<&str> {
    match value {
        Value::String(text) => text
            .as_str()
            .and_then(crate::call::admission::text_target_name),
        Value::StringArray(array) if array.data().len() == 1 => array.data()[0]
            .as_str()
            .and_then(crate::call::admission::text_target_name),
        Value::FunctionHandle(name)
        | Value::ExternalFunctionHandle(name)
        | Value::MethodFunctionHandle(name) => Some(name.name().trim()),
        Value::Closure(closure) => Some(&closure.function_name),
        _ => None,
    }
}

struct DeferredResolver;
impl FunctionNameResolver for DeferredResolver {
    fn resolve_function(&self, _: &str) -> Option<runmat_types::FunctionId> {
        None
    }
    fn name_resolution_completeness(&self) -> super::NameResolutionCompleteness {
        super::NameResolutionCompleteness::Deferred
    }
}
