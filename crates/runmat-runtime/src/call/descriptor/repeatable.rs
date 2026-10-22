//! Repeated/owned callback execution uses the canonical descriptor selector.
use super::{CallableDescriptor, CallableTarget, FunctionNameResolver};
use crate::call::{BuiltinArguments, RetainedCallableTarget};
use crate::context::RuntimeContext;
use crate::execution::{RetainedValueReadLease, RootedValue, RootedValueSequence, RootedValues};
use crate::RuntimeError;
use runmat_value::Value;

pub struct RepeatableCallable {
    selected: CallableDescriptor,
    // The descriptor has root-free aliases; this source owns actual roots and
    // drops those registrations before the final retained generation.
    _target_owner: RetainedValueReadLease,
}
impl RepeatableCallable {
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
        })
    }
    pub fn with_callsite(mut self, callsite: Option<crate::callsite::CallsiteInfo>) -> Self {
        self.selected.metadata.function_input_callsite = callsite;
        self
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
    pub async fn invoke_retained(
        &self,
        runtime: Option<&RuntimeContext>,
        arguments: BuiltinArguments,
        requested_outputs: usize,
    ) -> Result<RootedValueSequence, RuntimeError> {
        let mut selected = self.selected.clone();
        selected.requested_outputs = requested_outputs;
        let selected = selected.with_retained_arguments(arguments)?;
        super::execute_retained_callable_descriptor(runtime, selected).await
    }
}
pub(super) fn retained_target_name(value: &Value) -> Option<&str> {
    match value {
        Value::String(text) => crate::call::admission::text_target_name(text),
        Value::StringArray(array) if array.data().len() == 1 => {
            crate::call::admission::text_target_name(&array.data()[0])
        }
        Value::FunctionHandle(name)
        | Value::ExternalFunctionHandle(name)
        | Value::MethodFunctionHandle(name) => Some(name.name().trim()),
        Value::Closure(closure) => Some(&closure.function_name),
        _ => None,
    }
}
