use super::*;

pub(super) enum ConstructorInvocationTarget {
    Registered,
    ExactBound {
        owner: runmat_types::ClassMethodOwner,
        binding: runmat_value::BoundFunctionBinding,
    },
    ExactPrepared {
        owner: runmat_types::ClassMethodOwner,
        invoke: PreparedConstructorInvoke,
    },
}

/// The callback is constructed only after the exact receiver is rooted, and
/// receives that invocation's selected context under its receiver poll guard.
pub type PreparedConstructorInvoke = Box<
    dyn FnOnce(
        crate::context::RuntimeContext,
    ) -> std::pin::Pin<
        Box<
            dyn std::future::Future<
                Output = Result<crate::execution::RootedValueSequence, RuntimeError>,
            >,
        >,
    >,
>;

/// Core's exact staged body enters the ordinary constructor policy after its
/// selected class projection is installed, without re-resolving a class name.
pub async fn call_prepared_constructor_sequence_in_context(
    context: &crate::context::RuntimeContext,
    owner: runmat_types::ClassMethodOwner,
    binding: runmat_value::BoundFunctionBinding,
    args: &[Value],
    requested_outputs: usize,
    invoke: PreparedConstructorInvoke,
) -> Result<crate::execution::RootedValueSequence, RuntimeError> {
    if !binding.authority.admits_local(&context.session_lease())
        || binding.authority.local_lease(binding.function).is_none()
    {
        return Err(build_runtime_error(
            "prepared constructor lacks its selected definition lease",
        )
        .with_identifier("RunMat:ConstructorOwnerInvalid")
        .build());
    }
    call_admitted_prepared_constructor_sequence_in_context(
        context,
        owner,
        args,
        requested_outputs,
        invoke,
    )
    .await
}

/// Trusted VM admission must select the exact portable body, match its origin
/// and source owner, and verify its typed class product before calling this
/// policy. This port alone does not authenticate a portable artifact: the
/// portable-origin check below only prevents treating a local lease as one.
/// Both admitted routes share class creation, access, receiver, and output
/// normalization without pretending a portable binding has a local lease.
pub async fn call_portable_prepared_constructor_sequence_in_context(
    context: &crate::context::RuntimeContext,
    owner: runmat_types::ClassMethodOwner,
    binding: runmat_value::BoundFunctionBinding,
    args: &[Value],
    requested_outputs: usize,
    invoke: PreparedConstructorInvoke,
) -> Result<crate::execution::RootedValueSequence, RuntimeError> {
    if binding.authority.portable_origin().is_none() {
        return Err(
            build_runtime_error("portable constructor lacks its admitted binding origin")
                .with_identifier("RunMat:ConstructorOwnerInvalid")
                .build(),
        );
    }
    call_admitted_prepared_constructor_sequence_in_context(
        context,
        owner,
        args,
        requested_outputs,
        invoke,
    )
    .await
}

async fn call_admitted_prepared_constructor_sequence_in_context(
    context: &crate::context::RuntimeContext,
    owner: runmat_types::ClassMethodOwner,
    args: &[Value],
    requested_outputs: usize,
    invoke: PreparedConstructorInvoke,
) -> Result<crate::execution::RootedValueSequence, RuntimeError> {
    let class_identity = owner.declaring_class.clone();
    let selected = context.clone();
    selected
        .clone()
        .scope(async move {
            super::call_class_constructor(
                class_identity,
                args,
                Some(requested_outputs),
                Some(&selected),
                ConstructorInvocationTarget::ExactPrepared { owner, invoke },
            )
            .await?
            .try_map_owned(
                crate::call::arguments::adapt_legacy_builtin_result,
                |error| error.into_runtime_error(),
            )
        })
        .await
}

/// Invoke an already-admitted constructor definition while preserving the
/// ordinary class creation, access, receiver, and result policies. The caller
/// must prove the binding's exact admitted origin and method owner; this
/// boundary checks the runtime issuer and registered class method again.
pub async fn call_bound_constructor_sequence_in_context(
    context: &crate::context::RuntimeContext,
    owner: runmat_types::ClassMethodOwner,
    binding: runmat_value::BoundFunctionBinding,
    args: &[Value],
    requested_outputs: usize,
) -> Result<crate::execution::RootedValueSequence, RuntimeError> {
    if !binding.authority.admits_local(&context.session_lease()) {
        return Err(
            build_runtime_error("bound constructor belongs to another or ended session")
                .with_identifier("RunMat:BoundFunction:UnavailableInvoker")
                .build(),
        );
    }
    let class_identity = owner.declaring_class.clone();
    let selected = context.clone();
    selected
        .clone()
        .scope(async move {
            super::call_class_constructor(
                class_identity,
                args,
                Some(requested_outputs),
                Some(&selected),
                ConstructorInvocationTarget::ExactBound { owner, binding },
            )
            .await?
            .try_map_owned(
                crate::call::arguments::adapt_legacy_builtin_result,
                |error| error.into_runtime_error(),
            )
        })
        .await
}

#[cfg(test)]
#[path = "constructor/tests.rs"]
mod tests;
