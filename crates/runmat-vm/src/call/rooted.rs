//! Managed roots follow call-owned values after they leave the VM frame.
use crate::runtime::gc::ManagedRoots;
use runmat_runtime::call::descriptor::{CallableDescriptor, CallableTarget};
use runmat_runtime::RuntimeError;
use runmat_value::Value;

pub(crate) fn root_values<'a>(
    values: impl IntoIterator<Item = &'a Value>,
) -> Result<ManagedRoots, RuntimeError> {
    ManagedRoots::from_values(values).map_err(RuntimeError::new)
}

struct CallableDescriptorRoots {
    _callable: Option<runmat_runtime::execution::RootedValueLease>,
    _arguments: ManagedRoots,
}

impl CallableDescriptorRoots {
    fn new(descriptor: &CallableDescriptor) -> Result<Self, RuntimeError> {
        let callable = match &descriptor.target {
            CallableTarget::FevalForward(value) => Some(value),
            CallableTarget::RetainedFevalForward(value)
            | CallableTarget::RetainedBuiltin(value) => Some(value.value()),
            CallableTarget::BoundLexical { closure, .. }
            | CallableTarget::CapturedResolved { closure, .. } => Some(closure.as_value()),
            CallableTarget::Bound(_)
            | CallableTarget::Resolved { .. }
            | CallableTarget::Scoped(_) => None,
        };
        let callable = callable
            .map(runmat_runtime::execution::RootedValueLease::for_traced)
            .transpose()
            .map_err(runmat_runtime::execution::ExecutionServiceError::into_runtime_error)?;
        Ok(Self {
            _callable: callable,
            _arguments: root_values(descriptor.args.iter())?,
        })
    }
}

pub(crate) fn execute_callable_descriptor<'a>(
    runtime: &'a runmat_runtime::context::RuntimeContext,
    descriptor: CallableDescriptor,
) -> impl std::future::Future<
    Output = Result<runmat_runtime::execution::RootedValueSequence, RuntimeError>,
> + 'a {
    let roots = CallableDescriptorRoots::new(&descriptor);
    async move {
        let _roots = roots?;
        runmat_runtime::call::descriptor::execute_callable_descriptor_in_context(
            runtime, descriptor,
        )
        .await
    }
}

/// Own the selected context and acquire the descriptor's managed roots before
/// returning the future. The caller may enqueue this future without leaving
/// the target or arguments unrooted until its first poll.
pub(crate) fn execute_callable_descriptor_owned(
    runtime: runmat_runtime::context::RuntimeContext,
    descriptor: CallableDescriptor,
) -> impl std::future::Future<
    Output = Result<runmat_runtime::execution::RootedValueSequence, RuntimeError>,
> {
    let roots = CallableDescriptorRoots::new(&descriptor);
    async move {
        let _roots = roots?;
        runmat_runtime::call::descriptor::execute_callable_descriptor_in_context(
            &runtime, descriptor,
        )
        .await
    }
}

pub(crate) async fn call_builtin_async(
    runtime: &runmat_runtime::context::RuntimeContext,
    name: &str,
    args: &[Value],
) -> Result<runmat_runtime::execution::RootedValue, RuntimeError> {
    let _roots = root_values(args)?;
    use runmat_runtime::sequence::ResolveValueSequence;
    let sequence =
        runmat_runtime::call_builtin_sequence_async_in_context(runtime, name, args).await?;
    sequence.try_map_owned(
        |sequence| {
            let values = sequence.resolve(
                runmat_types::SequenceUse::RequireSingle,
                runmat_runtime::sequence::SequenceResolutionContext::default(),
            )?;
            Ok(runmat_runtime::sequence::into_first_output(values)?
                .expect("one required call output was resolved"))
        },
        |error| error.into_runtime_error(),
    )
}

pub(crate) async fn call_builtin_async_with_outputs(
    runtime: &runmat_runtime::context::RuntimeContext,
    name: &str,
    args: &[Value],
    outputs: usize,
) -> Result<runmat_runtime::execution::RootedValueSequence, RuntimeError> {
    let _roots = root_values(args)?;
    runmat_runtime::call_builtin_sequence_async_with_outputs_in_context(
        runtime, name, args, outputs,
    )
    .await
}

pub(crate) async fn try_call_builtin_async_with_outputs(
    runtime: &runmat_runtime::context::RuntimeContext,
    name: &str,
    args: &[Value],
    outputs: usize,
) -> Result<Option<runmat_runtime::execution::RootedValueSequence>, RuntimeError> {
    let _roots = root_values(args)?;
    runmat_runtime::dispatcher::try_call_builtin_sequence_async_with_outputs_in_context(
        runtime, name, args, outputs,
    )
    .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_runtime::call::descriptor::CallableCallKind;
    use runmat_value::HandleRef;

    #[test]
    fn descriptor_roots_exist_before_a_queued_future_is_polled() {
        let initial = runmat_gc::gc_allocate_rooted(Value::String("descriptor input".into()))
            .expect("allocate managed input");
        let handle = initial.handle();
        let descriptor = CallableDescriptor::resolved(
            runmat_types::CallableIdentity::Builtin(runmat_types::BuiltinId("zeros".into())),
            vec![Value::HandleObject(HandleRef {
                class_name: "DescriptorInput".into(),
                target: handle,
                valid: true,
            })],
            1,
            runmat_types::CallableFallbackPolicy::None,
            CallableCallKind::Direct,
        );
        let runtime = runmat_runtime::context::RuntimeContext::new(std::rc::Rc::new(
            runmat_runtime::execution::RuntimeExecutionService::new(),
        ));
        let pending = execute_callable_descriptor_owned(runtime, descriptor);
        initial.unroot().expect("transfer input root");
        runmat_gc::gc_collect_major().expect("collect before child poll");
        assert_eq!(
            runmat_gc::gc_clone_value(&handle).expect("descriptor owns its input"),
            Value::String("descriptor input".into())
        );
        drop(pending);
        runmat_gc::gc_collect_major().expect("collect after queue retirement");
        assert!(runmat_gc::gc_clone_value(&handle).is_err());
    }
}
