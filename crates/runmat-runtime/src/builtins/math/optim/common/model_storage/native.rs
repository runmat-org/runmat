//! Original rooted children and exact native/object metadata publication.
use super::super::model_policy::Policy;
use super::*;
use crate::context::host_allocation::{
    HostAllocationPlan, HostAllocationPolicy, HostAllocationReceipt,
};
use crate::execution::{RootedValue, RootedValueHandles};
use runmat_types::StaticClassIdentity;
use runmat_value::optimization_model::{
    OptimizationNativeAdmission, OptimizationNativeInput, OptimizationNativeSource,
    PreparedOptimizationNative,
};
use runmat_value::{
    ExactArrayClaimPromotionError, ExactObjectBuilder, ObjectInstance, PreparedObjectClassName,
    SharedStorageIdentity, Value,
};
mod cursor;
mod roots;
mod traversal;

pub(in crate::builtins::math::optim) enum NativeGcRoot {
    Managed(runmat_gc::PreparedGcAllocation),
    Ordinary(runmat_gc::ExplicitRoot),
}
pub(in crate::builtins::math::optim) enum NativeHandleRoot {
    Managed { _roots: RootedValueHandles },
    Ordinary { _root: runmat_gc::ExplicitRoot },
}
pub(in crate::builtins::math::optim) fn retain_handle(
    handle: runmat_gc::GcHandle,
    policy: Policy,
    cooperation: &ScalarCooperation,
) -> BuiltinResult<NativeHandleRoot> {
    cooperation.check_allocation_epoch()?;
    let context = cooperation.runtime();
    let result = match HostAllocationReceipt::policy(context)
        .map_err(|source| allocation(context, source))?
    {
        HostAllocationPolicy::Managed => {
            let runtime = context
                .ok_or_else(|| allocation(context, HostAllocationError::MissingManagedOwner))?;
            RootedValueHandles::for_values_and_handles_iterator_in_context(
                std::iter::empty(),
                std::iter::once(handle),
                runtime,
            )
            .map(|_roots| NativeHandleRoot::Managed { _roots })
            .map_err(|source| roots::root_error(source, policy, context))
        }
        HostAllocationPolicy::Unbounded => runmat_gc::gc_root(handle)
            .map(|_root| NativeHandleRoot::Ordinary { _root })
            .map_err(|source| policy.gc("failed to retain original model handle", source)),
    };
    cooperation.check_allocation_epoch()?;
    result
}
impl NativeGcRoot {
    pub fn handle(&self) -> runmat_gc::GcHandle {
        match self {
            Self::Managed(root) => root.handle(),
            Self::Ordinary(root) => root.handle(),
        }
    }
}
pub(in crate::builtins::math::optim) fn allocate_gc(
    source: &RootedValue,
    policy: Policy,
    cooperation: &ScalarCooperation,
) -> BuiltinResult<NativeGcRoot> {
    // Source is already a fully owned/rooted native object. Never clone a
    // generic Value aggregate or hold a GC value borrow across root admission.
    let Value::Object(object) = source.value() else {
        return Err(policy.invalid("private model allocation requires its native object"));
    };
    let private = matches!(
        object.optimization_variable_payload(),
        Some(runmat_value::optimization_variable::OptimizationVariablePayload::Definition(_))
    );
    if !private || object.selected_class().is_some() {
        return Err(policy.invalid("private model allocation requires an original definition"));
    }
    cooperation.check_allocation_epoch()?;
    let context = cooperation.runtime();
    let result = match HostAllocationReceipt::policy(context)
        .map_err(|source| allocation(context, source))?
    {
        HostAllocationPolicy::Unbounded => {
            runmat_gc::gc_allocate_rooted(Value::Object(object.clone()))
                .map(NativeGcRoot::Ordinary)
                .map_err(|source| policy.gc("private model allocation failed", source))
        }
        HostAllocationPolicy::Managed => {
            let context = context
                .ok_or_else(|| allocation(context, HostAllocationError::MissingManagedOwner))?;
            runmat_gc::gc_allocate_prepared_rooted(Value::Object(object.clone()), |handle| {
                cooperation.check_allocation_epoch()?;
                let roots = RootedValueHandles::for_values_and_handles_iterator_in_context(
                    std::iter::empty(),
                    std::iter::once(handle),
                    context,
                )
                .map_err(|source| roots::root_error(source, policy, Some(context)))?;
                cooperation.check_allocation_epoch()?;
                Ok(roots.into_prepared_lease())
            })
            .map(NativeGcRoot::Managed)
            .map_err(|source| match source {
                runmat_gc::GcRootedAllocationError::Gc(source) => {
                    policy.gc("private model allocation failed", source)
                }
                runmat_gc::GcRootedAllocationError::Admission(source) => source,
            })
        }
    };
    cooperation.check_allocation_epoch()?;
    result
}

pub(in crate::builtins::math::optim) async fn publish(
    input: OptimizationNativeInput,
    class: StaticClassIdentity,
    policy: Policy,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<RootedValue> {
    cooperation.check_allocation_epoch()?;
    let runtime = cooperation.runtime().cloned();
    let context = runtime.as_ref();
    let _children = roots::retain(&input, policy, context)?;
    cooperation.check_allocation_epoch()?;
    let mode =
        HostAllocationReceipt::policy(context).map_err(|source| allocation(context, source))?;
    if mode == HostAllocationPolicy::Unbounded {
        let object = ObjectInstance::new(class.owned())
            .with_optimization_variable_payload(input.into_payload());
        let result = RootedValue::in_optional_context(Value::Object(object), context)?;
        cooperation.check_allocation_epoch()?;
        return Ok(result);
    }
    let context =
        context.ok_or_else(|| allocation(None, HostAllocationError::MissingManagedOwner))?;
    let native_bytes = OptimizationNativeInput::native_header_requested_bytes();
    let storage = Collections::new(cooperation)?;
    let mut seen: Map<SharedStorageIdentity, traversal::Seen> = storage.map();
    let mut publication = input.begin_admission();
    let mut cursors = storage.vector();
    let mut edges = storage.vector();
    traversal::append_children(publication.input(), &mut edges, &storage, cooperation).await?;
    let mut position = 0;
    loop {
        while let Some(edge) = edges.pop() {
            let object = traversal::read(edge, policy)?;
            traversal::insert_object(&object, &mut cursors, &mut seen, &storage, policy)?;
            cooperation.completed(1).await?;
            storage.check()?;
        }
        if position == cursors.len() {
            break;
        }
        traversal::append_children(cursors[position].input(), &mut edges, &storage, cooperation)
            .await?;
        position += 1;
    }
    drop(edges);
    let object_quote = ExactObjectBuilder::quote(class.display_name(), 0)
        .map_err(|source| allocation(Some(context), source.into()))?;
    let bytes = object_quote
        .requested_bytes
        .checked_add(native_bytes)
        .ok_or_else(|| allocation(Some(context), HostAllocationError::Overflow))?;
    let claims = object_quote
        .retained_claims
        .checked_add(1)
        .ok_or_else(|| allocation(Some(context), HostAllocationError::Overflow))?;
    let mut quote = cursor::Quote {
        storage: &storage,
        seen: &mut seen,
        policy,
        context,
        bytes,
        claims,
    };
    quote.cursor(&mut publication, cooperation).await?;
    for cursor in &mut cursors {
        quote.cursor(cursor, cooperation).await?;
    }
    let (bytes, claims) = quote.totals();
    let retained = HostAllocationPlan::retained_exact_output_bytes(
        u64::try_from(bytes)
            .map_err(|_| allocation(Some(context), HostAllocationError::Overflow))?,
        claims,
    )
    .map_err(|source| allocation(Some(context), source))?;
    cooperation.check_allocation_epoch()?;
    let receipt = HostAllocationReceipt::admit(
        Some(context),
        HostAllocationPlan::new(0, retained).map_err(|source| allocation(Some(context), source))?,
    )
    .map_err(|source| allocation(Some(context), source))?
    .ok_or_else(|| allocation(Some(context), HostAllocationError::StaleAuthority))?;
    let promotion = cursor::Promotion {
        storage: &storage,
        policy,
        context,
        receipt: &receipt,
    };
    promotion.cursor(&mut publication, cooperation).await?;
    for cursor in &mut cursors {
        promotion.cursor(cursor, cooperation).await?;
    }
    // Alias cursor backing retires only after its original owners have their
    // claims. The group's real root lease still protects every GC child.
    drop(cursors);
    let prepared: PreparedOptimizationNative = publication
        .finish()
        .map_err(|_| policy.invalid("native metadata admission is incomplete"))?;
    receipt
        .checkpoint(context)
        .map_err(|source| allocation(Some(context), source))?;
    cooperation.check_allocation_epoch()?;
    let object = receipt
        .with_output_construction(context, |permit| -> BuiltinResult<ObjectInstance> {
            let class = PreparedObjectClassName::prepare(class.display_name(), permit).map_err(
                |source| {
                    crate::build_runtime_error("native model class-name admission failed")
                        .with_source(source)
                        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
                        .build()
                },
            )?;
            let mut builder = ExactObjectBuilder::try_new(class, 0, permit)
                .map_err(|source| allocation(Some(context), source.into()))?;
            if !builder
                .try_set_optimization_native(prepared, permit)
                .map_err(|source| allocation(Some(context), source.into()))?
            {
                return Err(policy.invalid("native model header already exists"));
            }
            builder
                .finish()
                .map_err(|source| allocation(Some(context), source.into()))
        })
        .map_err(|source| allocation(Some(context), source))??;
    cooperation.check_allocation_epoch()?;
    let result = RootedValue::in_optional_context(Value::Object(object), Some(context))?;
    cooperation.check_allocation_epoch()?;
    Ok(result)
}
fn metadata(
    source: ExactArrayClaimPromotionError,
    policy: Policy,
    context: Option<&RuntimeContext>,
) -> RuntimeError {
    match source {
        ExactArrayClaimPromotionError::Allocation(source) => allocation(context, source.into()),
        source @ ExactArrayClaimPromotionError::UnsharedBacking => {
            crate::build_runtime_error(format!(
                "{}: model metadata backing is not shared",
                policy.identity.name
            ))
            .with_source(source)
            .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
            .build()
        }
    }
}
