//! One managed borrowed/owned gather walk. Private canonical aggregate slots
//! retain original immutable children; only downloaded leaves replace them.
use crate::builtins::common::gpu_helpers;
use crate::context::host_allocation::{
    host_allocation_runtime_error, HostAllocationError, HostAllocationPlan, HostAllocationPolicy,
    HostAllocationReceipt,
};
use crate::context::RuntimeContext;
use crate::execution::{CpuCooperationBudget, RootedValues};
use crate::RuntimeError;
use runmat_value::{Closure, ClosureHeaderError, Value};

pub(super) fn is_managed(runtime: Option<&RuntimeContext>) -> Result<bool, RuntimeError> {
    HostAllocationReceipt::policy(runtime)
        .map(|policy| policy == HostAllocationPolicy::Managed)
        .map_err(|source| allocation(runtime, source))
}

fn allocation(runtime: Option<&RuntimeContext>, source: HostAllocationError) -> RuntimeError {
    host_allocation_runtime_error(
        runtime,
        "RunMat:gather:Ownership",
        "managed gather admission failed",
        source,
    )
}

fn control(
    runtime: &RuntimeContext,
    source: crate::execution::ExecutionServiceError,
) -> RuntimeError {
    source.into_runtime_error_in_context(Some(runtime))
}

pub(super) async fn borrowed(
    source: &Value,
    runtime: &RuntimeContext,
) -> Result<Value, RuntimeError> {
    let mut budget = runtime.cpu_cooperation();
    budget
        .checkpoint(0)
        .await
        .map_err(|source| control(runtime, source))?;
    let owner = HostAllocationReceipt::capture_owner(Some(runtime))
        .map_err(|source| allocation(Some(runtime), source))?;
    runmat_value::validate_no_transient_sequence(source).map_err(|_| transient())?;
    let _roots = super::structure_alias::lease(source, Some(runtime))?;
    let mut value = retain_alias(source)?;
    walk_if_needed(&mut value, runtime, &mut budget).await?;
    budget
        .checkpoint(0)
        .await
        .map_err(|source| control(runtime, source))?;
    if let Some(owner) = owner {
        owner
            .check(runtime)
            .map_err(|source| allocation(Some(runtime), source))?;
    }
    Ok(value)
}

pub(super) async fn owned(
    mut value: Value,
    runtime: &RuntimeContext,
    budget: Option<&mut CpuCooperationBudget>,
) -> Result<Value, RuntimeError> {
    let mut local = runtime.cpu_cooperation();
    let budget = budget.unwrap_or(&mut local);
    budget
        .checkpoint(0)
        .await
        .map_err(|source| control(runtime, source))?;
    let owner = HostAllocationReceipt::capture_owner(Some(runtime))
        .map_err(|source| allocation(Some(runtime), source))?;
    runmat_value::validate_no_transient_sequence(&value).map_err(|_| transient())?;
    let _roots = super::structure_alias::lease(&value, Some(runtime))?;
    walk_if_needed(&mut value, runtime, budget).await?;
    budget
        .checkpoint(0)
        .await
        .map_err(|source| control(runtime, source))?;
    if let Some(owner) = owner {
        owner
            .check(runtime)
            .map_err(|source| allocation(Some(runtime), source))?;
    }
    Ok(value)
}

fn transient() -> RuntimeError {
    crate::runtime_error::semantic_error(
        "TransientSequenceBelowDispatch",
        "legacy output-list carrier reached scalar gather dispatch",
    )
}

/// Retain each original immutable physical owner. The walk prepares private
/// slots before changing a descendant, avoiding a redundant first slot copy.
fn retain_alias(source: &Value) -> Result<Value, RuntimeError> {
    let value = match source {
        Value::Cell(source) => Value::Cell(source.clone()),
        Value::Struct(source) => Value::Struct(source.clone()),
        Value::StructArray(source) => Value::StructArray(source.clone()),
        Value::Object(source) => Value::Object(source.clone()),
        Value::ObjectArray(source) => Value::ObjectArray(source.clone()),
        // Closures preserve the original callable instance unless captures
        // actually change; their transformation below funds a new header.
        Value::Closure(source) => Value::Closure(source.clone()),
        Value::OutputList(_) => return Err(transient()),
        leaf => leaf.clone(),
    };
    Ok(value)
}

/// Unchanged children retain their original owners without an erased future
/// or receipt per host leaf. They still participate in CPU cooperation and
/// cancellation; this inline gate allocates no continuation of its own.
async fn walk_if_needed(
    value: &mut Value,
    runtime: &RuntimeContext,
    budget: &mut CpuCooperationBudget,
) -> Result<(), RuntimeError> {
    if super::owned_gpu::has_transformable_gpu(value) {
        walk(value, runtime, budget)?.await
    } else {
        budget
            .checkpoint(1)
            .await
            .map_err(|source| control(runtime, source))
    }
}

/// Every transforming recursive suspension is erased at the existing exact admitted
/// continuation owner before allocation or first poll. No raw recursive Box.
fn walk<'a>(
    value: &'a mut Value,
    runtime: &'a RuntimeContext,
    budget: &'a mut CpuCooperationBudget,
) -> Result<crate::execution::rooted_continuation::RuntimeContinuationFuture<'a, ()>, RuntimeError>
{
    crate::execution::rooted_continuation::admit_domain_continuation_with(runtime, move || {
        super::poll_gather_in_context(runtime, async move {
            budget
                .checkpoint(1)
                .await
                .map_err(|source| control(runtime, source))?;
            match value {
                Value::GpuTensor(handle) => {
                    let readback = runmat_accelerate_api::numeric_readback_for_handle(handle)
                        .map_err(|source| {
                            crate::runtime_error::with_provider_execution_source(
                                crate::build_runtime_error(
                                    "gather: exact source provider is unavailable or not authorized",
                                )
                                .with_identifier("RunMat:gather:ProviderUnavailable")
                                .build(),
                                source.into(),
                            )
                        })?;
                    let quote = gpu_helpers::quote_exact_native_readback(&readback)?;
                    let receipt = HostAllocationReceipt::admit(
                        Some(runtime),
                        quote
                            .single_source_plan()
                            .map_err(|source| allocation(Some(runtime), source))?,
                    )
                    .map_err(|source| allocation(Some(runtime), source))?
                    .ok_or_else(|| {
                        allocation(Some(runtime), HostAllocationError::MissingManagedOwner)
                    })?;
                    let gathered = budget
                        .await_cancellable(
                            gpu_helpers::download_exact_native_readback_with_diagnostics(
                                runtime,
                                &readback,
                                &receipt,
                                quote,
                                gpu_helpers::NumericDownloadDiagnostic::Gather,
                            ),
                        )
                        .await
                        .map_err(|source| control(runtime, source))?
                        .map_err(|failure| failure.error)?;
                    receipt
                        .checkpoint(runtime)
                        .map_err(|source| allocation(Some(runtime), source))?;
                    *value = gathered;
                }
                Value::Cell(cell)
                    if cell
                        .data()
                        .iter()
                        .any(super::owned_gpu::has_transformable_gpu) =>
                {
                    let mut _roots = None;
                    if cell.try_data_mut_unique().is_none() {
                        let prepared = super::structure_alias::cell(cell, Some(runtime))?;
                        *cell = prepared.0;
                        _roots = Some(prepared.1);
                    }
                    for child in cell.try_data_mut_unique().expect("private Cell slots") {
                        walk_if_needed(child, runtime, budget).await?;
                    }
                }
                Value::Struct(record)
                    if record
                        .fields
                        .values()
                        .any(super::owned_gpu::has_transformable_gpu) =>
                {
                    let mut _roots = None;
                    if !record.fields.is_unique() {
                        let prepared = super::structure_alias::record(record, Some(runtime))?;
                        *record = prepared.0;
                        _roots = Some(prepared.1);
                    }
                    for child in record
                        .fields
                        .try_values_mut_unique()
                        .expect("private struct slots")
                    {
                        walk_if_needed(child, runtime, budget).await?;
                    }
                }
                Value::StructArray(array)
                    if array.column_owners().any(|(_, values)| {
                        values.iter().any(super::owned_gpu::has_transformable_gpu)
                    }) =>
                {
                    let source = array.clone(); // Original immutable map/shape owners only.
                    let (mut replacement, _roots) =
                        super::structure_alias::array(&source, Some(runtime))?;
                    for name in source.field_names() {
                        for child in replacement
                            .try_column_mut_unique(name)
                            .expect("private column slots")
                        {
                            walk_if_needed(child, runtime, budget).await?;
                        }
                    }
                    *array = replacement;
                }
                Value::Object(object)
                    if object
                        .properties()
                        .values()
                        .any(super::owned_gpu::has_transformable_gpu) =>
                {
                    let source = object.clone();
                    let (mut replacement, _roots) =
                        crate::execution::object::replacement::copy_generation(
                            &source,
                            Some(runtime),
                        )?;
                    for name in source.properties().keys() {
                        let child = replacement
                            .try_property_mut_unique(name)
                            .expect("private object property slot");
                        walk_if_needed(child, runtime, budget).await?;
                    }
                    *object = replacement;
                }
                Value::ObjectArray(array)
                    if array
                        .data()
                        .iter()
                        .any(super::owned_gpu::has_transformable_gpu) =>
                {
                    let (replacement, _roots) =
                        crate::execution::object_array::copy_source(array, Some(runtime))?
                            .into_value_and_lease();
                    let Value::ObjectArray(mut replacement) = replacement else {
                        unreachable!("object array constructor")
                    };
                    for index in 0..replacement.len() {
                        let mut child = replacement.data()[index].clone();
                        walk_if_needed(&mut child, runtime, budget).await?;
                        replacement
                            .try_replace_linear_unique(index, child)
                            .map_err(|_| {
                                crate::runtime_error::semantic_error(
                                    "ObjectArrayConstruction",
                                    "gather changed object-array class or storage kind",
                                )
                            })?;
                    }
                    *array = replacement;
                }
                Value::Closure(closure) => {
                    if closure
                        .captures
                        .iter()
                        .any(super::owned_gpu::has_transformable_gpu)
                    {
                        *closure = gather_closure(closure, runtime, budget).await?;
                    }
                }
                Value::OutputList(_) => return Err(transient()),
                // Canonical immutable leaves and shared managed referents retain
                // identity. Do not dereference handle/listener/foreign resources.
                _ => {}
            }
            Ok(())
        })
    })
}

async fn gather_closure(
    source: &Closure,
    runtime: &RuntimeContext,
    budget: &mut CpuCooperationBudget,
) -> Result<Closure, RuntimeError> {
    let slots = RootedValues::try_initialize_in_context(source.captures.len(), runtime, |index| {
        Ok::<_, RuntimeError>(source.captures[index].clone())
    })?;
    slots
        .with_owned_output_storage_async(async move |mut captures| {
            for child in captures
                .try_as_mut_slice_unique()
                .expect("private capture slots")
            {
                walk_if_needed(child, runtime, budget).await?;
            }
            let bytes = u64::try_from(Closure::resolved_header_bytes())
                .map_err(|_| allocation(Some(runtime), HostAllocationError::Overflow))?;
            let retained = HostAllocationPlan::retained_exact_output_bytes(bytes, 1)
                .map_err(|source| allocation(Some(runtime), source))?;
            let receipt = HostAllocationReceipt::admit(
                Some(runtime),
                HostAllocationPlan::new(0, retained)
                    .map_err(|source| allocation(Some(runtime), source))?,
            )
            .map_err(|source| allocation(Some(runtime), source))?
            .ok_or_else(|| allocation(Some(runtime), HostAllocationError::MissingManagedOwner))?;
            let mut data = source.clone().into_data();
            data.captures = captures;
            receipt
                .with_output_construction(runtime, |permit| {
                    Closure::try_from_data_with_exact_header(data, permit)
                })
                .map_err(|source| allocation(Some(runtime), source))?
                .map_err(|source| match source {
                    ClosureHeaderError::Allocation(source) => {
                        allocation(Some(runtime), source.into())
                    }
                    ClosureHeaderError::CapturesNotShared => crate::runtime_error::semantic_error(
                        "CallableArguments",
                        "gather capture destination lacks shared storage",
                    ),
                })
        })
        .await
}

#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "managed_gather/tests.rs"]
mod tests;
