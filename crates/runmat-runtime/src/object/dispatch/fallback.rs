//! Semantic fallback uses the canonical callable owner and execution ports.

use super::*;

fn descriptor(
    identity: CallableIdentityOwner,
    name: Utf8Text,
    args: crate::call::BuiltinArguments,
    requested_outputs: usize,
    policy: CallableFallbackPolicy,
) -> Result<CallableDescriptor<crate::call::BuiltinArguments>, RuntimeError> {
    let mut descriptor = CallableDescriptor::resolved_with_identity_owner(
        identity,
        Vec::new(),
        requested_outputs,
        policy,
        CallableCallKind::Direct,
    );
    descriptor.metadata.display_name = Some(name);
    descriptor.with_retained_arguments(args)
}

pub(super) async fn call(
    runtime: Option<&crate::context::RuntimeContext>,
    identity: CallableIdentityOwner,
    name: Utf8Text,
    args: crate::call::BuiltinArguments,
    requested_outputs: usize,
    policy: CallableFallbackPolicy,
) -> Result<RootedValueSequence, RuntimeError> {
    let descriptor = descriptor(identity, name, args, requested_outputs, policy)?;
    match runtime {
        Some(runtime) => {
            crate::call::descriptor::execute_retained_callable_descriptor(Some(runtime), descriptor)
                .await
        }
        None => {
            crate::call::descriptor::execute_retained_callable_descriptor(None, descriptor).await
        }
    }
}

pub(super) async fn try_call(
    runtime: Option<&crate::context::RuntimeContext>,
    identity: CallableIdentityOwner,
    name: Utf8Text,
    args: crate::call::BuiltinArguments,
    requested_outputs: usize,
    policy: CallableFallbackPolicy,
) -> Result<Option<RootedValueSequence>, RuntimeError> {
    let descriptor = descriptor(identity, name, args, requested_outputs, policy)?;
    match runtime {
        Some(runtime) => {
            crate::call::descriptor::try_execute_retained_callable_descriptor(
                Some(runtime),
                descriptor,
            )
            .await
        }
        None => {
            crate::call::descriptor::try_execute_retained_callable_descriptor(None, descriptor)
                .await
        }
    }
}
