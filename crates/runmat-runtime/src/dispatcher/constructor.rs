use super::*;

#[path = "constructor/invocation.rs"]
mod invocation;
pub use invocation::PreparedConstructorInvoke;
use invocation::{PreparedConstructorBody, PreparedConstructorGuard};

pub(super) enum ConstructorInvocationTarget<'a> {
    Registered,
    ExactBound {
        owner: runmat_types::ClassMethodOwner,
        binding: runmat_value::BoundFunctionBinding,
    },
    ExactPrepared {
        owner: runmat_types::ClassMethodOwner,
        invoke: PreparedConstructorBody,
        guard: &'a PreparedConstructorGuard,
    },
}

/// Prepared exact bodies own their segmented arguments inside `invoke`.
/// Constructor policy only needs their logical arity; registered and legacy
/// bound dispatch still borrow their actual contiguous arguments.
pub(super) enum ConstructorArguments<'a> {
    Borrowed(&'a [Value]),
    Retained(crate::call::BuiltinArguments),
    PreparedCount(usize),
}

impl ConstructorArguments<'_> {
    pub(super) fn is_empty(&self) -> bool {
        match self {
            Self::Borrowed(values) => values.is_empty(),
            Self::Retained(values) => values.is_empty(),
            Self::PreparedCount(count) => *count == 0,
        }
    }

    pub(super) fn retained(
        &self,
        context: Option<&crate::context::RuntimeContext>,
    ) -> Result<crate::call::BuiltinArguments, RuntimeError> {
        match self {
            Self::Borrowed(values) => crate::call::BuiltinArguments::from_borrowed(values, context),
            Self::Retained(values) => Ok(values.fork()),
            Self::PreparedCount(_) => Err(build_runtime_error(
                "prepared constructor has no borrowed argument slice",
            )
            .with_identifier("RunMat:ConstructorOwnerInvalid")
            .build()),
        }
    }
}

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
        ConstructorArguments::Borrowed(args),
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
        ConstructorArguments::Borrowed(args),
        requested_outputs,
        invoke,
    )
    .await
}

async fn call_admitted_prepared_constructor_sequence_in_context(
    context: &crate::context::RuntimeContext,
    owner: runmat_types::ClassMethodOwner,
    args: ConstructorArguments<'_>,
    requested_outputs: usize,
    invoke: PreparedConstructorInvoke,
) -> Result<crate::execution::RootedValueSequence, RuntimeError> {
    let class_identity = owner.declaring_class.clone();
    let selected = context.clone();
    selected
        .clone()
        .scope(async move {
            // The body is consumed by the borrowed target. Keep its original
            // external funding alive through the whole constructor child.
            let (guard, invoke) = invoke.into_parts();
            super::call_class_constructor(
                class_identity,
                args,
                Some(requested_outputs),
                Some(&selected),
                ConstructorInvocationTarget::ExactPrepared {
                    owner,
                    invoke,
                    guard: &guard,
                },
            )
            .await?
            .try_map_owned(
                crate::call::arguments::adapt_legacy_builtin_result,
                |error| error.into_runtime_error(),
            )
        })
        .await
}

pub async fn call_prepared_constructor_sequence_with_argument_count_in_context(
    context: &crate::context::RuntimeContext,
    owner: runmat_types::ClassMethodOwner,
    binding: runmat_value::BoundFunctionBinding,
    argument_count: usize,
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
        ConstructorArguments::PreparedCount(argument_count),
        requested_outputs,
        invoke,
    )
    .await
}

pub async fn call_portable_prepared_constructor_sequence_with_argument_count_in_context(
    context: &crate::context::RuntimeContext,
    owner: runmat_types::ClassMethodOwner,
    binding: runmat_value::BoundFunctionBinding,
    argument_count: usize,
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
        ConstructorArguments::PreparedCount(argument_count),
        requested_outputs,
        invoke,
    )
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
                ConstructorArguments::Borrowed(args),
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

/// Rooted counterpart used by dynamic source selection. The constructor
/// policy sees only the logical arity; the exact selected body retains the
/// segmented callable/capture owners until its terminal invocation.
pub async fn call_rooted_bound_constructor_sequence_in_context(
    context: &crate::context::RuntimeContext,
    owner: runmat_types::ClassMethodOwner,
    binding: runmat_value::BoundFunctionBinding,
    arguments: crate::call::RootedFunctionArguments,
    requested_outputs: usize,
) -> Result<crate::execution::RootedValueSequence, RuntimeError> {
    if !binding.authority.admits_local(&context.session_lease()) {
        return Err(
            build_runtime_error("bound constructor belongs to another or ended session")
                .with_identifier("RunMat:BoundFunction:UnavailableInvoker")
                .build(),
        );
    }
    let argument_count = arguments.len();
    let selected = context.clone();
    let invoke = PreparedConstructorInvoke::new(context, move |runtime| async move {
        crate::user_functions::try_call_rooted_bound_function_with_environment_in_context(
            &runtime,
            binding,
            arguments,
            requested_outputs,
            None,
        )
        .await
        .ok_or_else(|| {
            build_runtime_error("selected runtime has no exact bound invoker")
                .with_identifier("RunMat:BoundFunction:UnavailableInvoker")
                .build()
        })?
    })?;
    selected
        .clone()
        .scope(async move {
            call_admitted_prepared_constructor_sequence_in_context(
                &selected,
                owner,
                ConstructorArguments::PreparedCount(argument_count),
                requested_outputs,
                invoke,
            )
            .await
        })
        .await
}

#[cfg(test)]
#[path = "constructor/tests.rs"]
mod tests;
