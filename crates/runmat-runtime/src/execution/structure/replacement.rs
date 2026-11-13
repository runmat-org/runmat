//! Explicit replacement slots for identity-preserving structure transformations.
//! These transforms can replace resident payloads, but never create new GC
//! identities. Keep the admitted registration of all original descendants
//! through the walk; only map/key/Value-slot backing is copied, not payloads.
use crate::{
    context::RuntimeContext,
    execution::{RootedValue, RootedValueLease},
    RuntimeError,
};
use runmat_value::{CellArray, StructArray, StructValue, Value};

pub(crate) fn record(
    source: &StructValue,
    runtime: Option<&RuntimeContext>,
) -> Result<(StructValue, RootedValueLease), RuntimeError> {
    let owner = crate::context::host_allocation::HostAllocationReceipt::capture_owner(runtime)
        .map_err(|source| allocation(runtime, source))?;
    let _source = root(Value::Struct(source.clone()), runtime)?;
    super::check_owner(runtime, &owner).map_err(|source| allocation(runtime, source))?;
    let output = if let Some(runtime) = runtime {
        crate::execution::construct_struct_from_fields_in_context(runtime, source.fields.iter())?
    } else {
        ordinary()?;
        let mut builder =
            runmat_value::UnboundedStructValueBuilder::with_capacity(source.fields.len());
        for entry in source.fields.entries() {
            builder.insert_text(entry.key().clone(), entry.value().clone());
        }
        RootedValue::new(Value::Struct(builder.finish())).map_err(|e| e.into_runtime_error())?
    };
    let output = output.into_value_and_lease();
    let roots = output.1;
    let Value::Struct(value) = output.0 else {
        unreachable!("structure constructor")
    };
    super::check_owner(runtime, &owner).map_err(|source| allocation(runtime, source))?;
    Ok((value, roots))
}
pub(crate) fn array(
    source: &StructArray,
    runtime: Option<&RuntimeContext>,
) -> Result<(StructArray, RootedValueLease), RuntimeError> {
    let owner = crate::context::host_allocation::HostAllocationReceipt::capture_owner(runtime)
        .map_err(|source| allocation(runtime, source))?;
    let _source = root(Value::StructArray(source.clone()), runtime)?;
    super::check_owner(runtime, &owner).map_err(|source| allocation(runtime, source))?;
    let output = if let Some(runtime) = runtime {
        crate::execution::construct_struct_array_from_values_in_context(
            runtime,
            source
                .column_owners()
                .map(|(name, values)| (name, values.as_slice())),
            source.shape_owner(),
        )?
    } else {
        ordinary()?;
        let mut builder = runmat_value::UnboundedStructArrayBuilder::with_capacity(
            source.shape_owner().clone(),
            source.field_names().len(),
        )
        .map_err(semantic)?;
        for (name, values) in source.column_owners() {
            builder
                .insert(
                    name.to_owned(),
                    runmat_value::OutputListStorage::shared_unbounded(values.to_vec()),
                )
                .map_err(column_error)?;
        }
        RootedValue::new(Value::StructArray(builder.finish()))
            .map_err(|e| e.into_runtime_error())?
    };
    let output = output.into_value_and_lease();
    let roots = output.1;
    let Value::StructArray(value) = output.0 else {
        unreachable!("structure array constructor")
    };
    super::check_owner(runtime, &owner).map_err(|source| allocation(runtime, source))?;
    Ok((value, roots))
}

/// Copy only Cell slots into a private destination while all original GC
/// descendants remain registered. Transformations using these slots must
/// preserve GC identities, or separately root every newly created identity.
pub(crate) fn cell(
    source: &CellArray,
    runtime: Option<&RuntimeContext>,
) -> Result<(CellArray, RootedValueLease), RuntimeError> {
    use crate::context::host_allocation::HostAllocationReceipt;
    let owner = HostAllocationReceipt::capture_owner(runtime)
        .map_err(|source| allocation(runtime, source))?;
    let _source = root(Value::Cell(source.clone()), runtime)?;
    super::check_owner(runtime, &owner).map_err(|source| allocation(runtime, source))?;
    let output = if owner.is_some() {
        let runtime = runtime.expect("managed owner requires its context");
        if let Some(bytes) = source
            .shape
            .unclaimed_shared_bytes()
            .map_err(|source| allocation(Some(runtime), source.into()))?
        {
            super::array::promote(runtime, bytes, |permit| {
                source.shape.try_promote_shared_claim(permit)
            })?;
        }
        super::check_owner(Some(runtime), &owner)
            .map_err(|source| allocation(Some(runtime), source))?;
        let values = crate::execution::RootedValues::try_initialize_in_context(
            source.data().len(),
            runtime,
            |index| Ok::<_, crate::RuntimeError>(source.data()[index].clone()),
        )?;
        values.with_owned_output_storage(|values| {
            let cell =
                CellArray::from_admitted(values, source.shape.clone()).map_err(|source| {
                    crate::runtime_error::semantic_error("CellReplacement", source)
                })?;
            RootedValue::in_context(Value::Cell(cell), runtime)
        })?
    } else {
        super::check_owner(runtime, &owner).map_err(|source| allocation(runtime, source))?;
        let cell = CellArray::new_with_shape(source.data().to_vec(), source.shape().to_vec())
            .map_err(|source| crate::runtime_error::semantic_error("CellReplacement", source))?;
        match runtime {
            Some(runtime) => RootedValue::in_context(Value::Cell(cell), runtime)?,
            None => {
                RootedValue::new(Value::Cell(cell)).map_err(|source| source.into_runtime_error())?
            }
        }
    };
    super::check_owner(runtime, &owner).map_err(|source| allocation(runtime, source))?;
    let output = output.into_value_and_lease();
    let roots = output.1;
    let Value::Cell(value) = output.0 else {
        unreachable!("cell constructor")
    };
    Ok((value, roots))
}
pub(crate) fn lease(
    value: &Value,
    runtime: Option<&RuntimeContext>,
) -> Result<RootedValueLease, RuntimeError> {
    match runtime {
        Some(runtime) => {
            RootedValueLease::for_value_slices_in_context(&[std::slice::from_ref(value)], runtime)
        }
        None => {
            ordinary()?;
            RootedValueLease::for_traced(value).map_err(|e| e.into_runtime_error())
        }
    }
}

pub(crate) fn root(
    value: Value,
    runtime: Option<&RuntimeContext>,
) -> Result<RootedValue, RuntimeError> {
    match runtime {
        Some(runtime) => RootedValue::in_context(value, runtime),
        None => {
            ordinary()?;
            RootedValue::new(value).map_err(|source| source.into_runtime_error())
        }
    }
}
fn ordinary() -> Result<(), RuntimeError> {
    use crate::context::host_allocation::{
        HostAllocationError, HostAllocationPolicy, HostAllocationReceipt,
    };
    match HostAllocationReceipt::policy(None).map_err(|e| allocation(None, e))? {
        HostAllocationPolicy::Unbounded => Ok(()),
        HostAllocationPolicy::Managed => {
            Err(allocation(None, HostAllocationError::MissingManagedOwner))
        }
    }
}
fn allocation(
    runtime: Option<&RuntimeContext>,
    source: crate::context::host_allocation::HostAllocationError,
) -> RuntimeError {
    crate::context::host_allocation::host_allocation_runtime_error(
        runtime,
        "RunMat:gather:SharedStructure",
        "structure replacement admission failed",
        source,
    )
}
fn semantic(source: impl std::error::Error + Send + Sync + 'static) -> RuntimeError {
    crate::runtime_error::semantic_error_with_source(
        "StructReplacement",
        "structure replacement failed",
        source,
    )
}

fn column_error(source: runmat_value::StructArrayFieldError) -> RuntimeError {
    match source {
        runmat_value::StructArrayFieldError::InvalidColumn { .. } => {
            crate::runtime_error::semantic_error(
                "StructReplacement",
                "structure-array column shape or backing is invalid",
            )
        }
        runmat_value::StructArrayFieldError::Map(source) => semantic(source.into_parts().0),
    }
}
