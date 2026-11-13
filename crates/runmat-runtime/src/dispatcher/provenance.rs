//! Publication projects owned value containers, never shared managed referents.
use super::*;

pub(super) fn project(value: &mut Value, explicit: bool) -> Result<(), RuntimeError> {
    match value {
        Value::GpuTensor(handle) => {
            let intent = if explicit || runmat_accelerate_api::handle_is_explicit(handle) {
                runmat_accelerate_api::GpuHandleProvenance::Explicit
            } else {
                runmat_accelerate_api::GpuHandleProvenance::Automatic
            };
            *handle = handle.with_provenance(intent).map_err(|error| {
                build_runtime_error(format!("GPU result provenance: {error}"))
                    .with_identifier("RunMat:Gpu:InvalidResult")
                    .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
                    .with_source(error)
                    .build()
            })?;
        }
        Value::Cell(cell) => {
            if !cell
                .data()
                .iter()
                .any(super::owned_gpu::has_transformable_gpu)
            {
                return Ok(());
            }
            let mut retained = None;
            if cell.try_data_mut_unique().is_none() {
                if super::owned_gather::unbounded_shared_cell_policy()? {
                    cell.make_data_unique_unbounded().map_err(|error| {
                        build_runtime_error(format!("GPU result cell alias: {error}"))
                            .with_identifier("RunMat:Gpu:InvalidResult")
                            .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
                            .with_source(error)
                            .build()
                    })?;
                } else {
                    let context = crate::context::legacy::active().ok_or_else(|| {
                        crate::runtime_error::semantic_error(
                            "ManagedOwnerUnavailable",
                            "shared Cell provenance has no active allocation owner",
                        )
                    })?;
                    let (replacement, receipt) =
                        super::cell_alias::prepare_shared_cell(cell, &context)?;
                    *cell = replacement;
                    retained = Some((context, receipt));
                }
            }
            let values = cell.try_data_mut_unique().ok_or_else(|| {
                crate::sequence::sequence_error_to_runtime(
                    runmat_value::ValueSequenceError::SharedOutputStorage,
                )
            })?;
            for value in values {
                project(value, explicit)?;
            }
            if let Some((context, receipt)) = retained {
                receipt.checkpoint(&context).map_err(|source| {
                    crate::context::host_allocation::host_allocation_runtime_error(
                        Some(&context),
                        "RunMat:Gpu:InvalidResult",
                        "shared Cell provenance authority changed",
                        source,
                    )
                })?;
            }
        }
        Value::Struct(object) => {
            if !object
                .fields
                .values()
                .any(super::owned_gpu::has_transformable_gpu)
            {
                return Ok(());
            }
            let context = crate::context::legacy::active();
            let mut _roots = None;
            if !object.fields.is_unique() {
                let (replacement, roots) =
                    super::structure_alias::record(object, context.as_ref())?;
                *object = replacement;
                _roots = Some(roots);
            }
            for value in object
                .fields
                .try_values_mut_unique()
                .expect("explicit unique map owner")
            {
                project(value, explicit)?;
            }
        }
        Value::StructArray(array) => {
            if !array
                .column_owners()
                .any(|(_, values)| values.iter().any(super::owned_gpu::has_transformable_gpu))
            {
                return Ok(());
            }
            if let Some(result) =
                array.try_for_each_value_mut_unique(|value| project(value, explicit))
            {
                result?;
            } else {
                let context = crate::context::legacy::active();
                let prepared = super::structure_alias::array(array, context.as_ref())?;
                let _roots = prepared.1;
                let mut replacement = prepared.0;
                replacement
                    .try_for_each_value_mut_unique(|value| project(value, explicit))
                    .expect("explicit unique array owner")?;
                *array = replacement;
            }
        }
        Value::Object(object) => {
            if !object
                .properties()
                .values()
                .any(super::owned_gpu::has_transformable_gpu)
            {
                return Ok(());
            }
            let context = crate::context::legacy::active();
            let prepared =
                crate::execution::object::replacement::copy_generation(object, context.as_ref())?;
            let _roots = prepared.1;
            let mut replacement = prepared.0;
            for (name, value) in object.properties() {
                let mut value = value.clone();
                project(&mut value, explicit)?;
                replacement
                    .try_replace_property_unique(name, value)
                    .map_err(|_| {
                        crate::runtime_error::semantic_error(
                            "RunMat:GPU:Provenance",
                            "private object destination lost its unique property slot",
                        )
                    })?;
            }
            *object = replacement;
        }
        Value::ObjectArray(array) => {
            if !array
                .data()
                .iter()
                .any(super::owned_gpu::has_transformable_gpu)
            {
                return Ok(());
            }
            let context = crate::context::legacy::active();
            let mut _retained = None;
            if !array.data_owner().is_unique() {
                let prepared =
                    crate::execution::object_array::copy_source(array, context.as_ref())?
                        .into_value_and_lease();
                _retained = Some(prepared.1);
                let Value::ObjectArray(replacement) = prepared.0 else {
                    unreachable!("canonical object array output")
                };
                *array = replacement;
            }
            for index in 0..array.len() {
                let mut child = array.data()[index].clone();
                project(&mut child, explicit)?;
                array.try_replace_linear_unique(index, child).map_err(|_| {
                    crate::runtime_error::semantic_error(
                        "ObjectArrayConstruction",
                        "provenance projection changed object-array class or storage kind",
                    )
                })?;
            }
        }
        Value::Closure(closure) => {
            if !closure
                .captures
                .iter()
                .any(super::owned_gpu::has_transformable_gpu)
            {
                return Ok(());
            }
            let Value::Closure(closure) = std::mem::replace(value, Value::Num(0.)) else {
                unreachable!("matched closure")
            };
            let mut data = closure.into_data();
            let result = project_values(&mut data.captures, explicit);
            *value = Value::Closure(runmat_value::Closure::from_data(data));
            result?;
        }
        Value::OutputList(values) => {
            project_values(values, explicit)?;
        }
        // These values publish shared identities, not owned payloads. Their
        // producer-stored provenance survives unchanged, including in object
        // arrays. Traversing/mutating target or callback GC values here would
        // rewrite every alias and could recurse through cycles or live borrows.
        Value::HandleObject(_) | Value::Listener(_) => {}
        _ => {}
    }
    Ok(())
}

fn project_values(
    values: &mut runmat_value::OutputListStorage,
    explicit: bool,
) -> Result<(), RuntimeError> {
    if !values.iter().any(super::owned_gpu::has_transformable_gpu) {
        return Ok(());
    }
    let mut retained = None;
    if values.try_as_mut_slice_unique().is_none() {
        if super::owned_gather::unbounded_shared_cell_policy()? {
            values.make_unique_unbounded().map_err(|error| {
                build_runtime_error(format!("GPU result list alias: {error}"))
                    .with_identifier("RunMat:Gpu:InvalidResult")
                    .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
                    .with_source(error)
                    .build()
            })?;
        } else {
            let context = crate::context::legacy::active().ok_or_else(|| {
                crate::runtime_error::semantic_error(
                    "ManagedOwnerUnavailable",
                    "shared result list provenance has no active allocation owner",
                )
            })?;
            let (replacement, receipt) =
                super::cell_alias::prepare_shared_values(values, &context)?;
            *values = replacement;
            retained = Some((context, receipt));
        }
    }
    let elements = values.try_as_mut_slice_unique().ok_or_else(|| {
        crate::sequence::sequence_error_to_runtime(
            runmat_value::ValueSequenceError::SharedOutputStorage,
        )
    })?;
    for value in elements {
        project(value, explicit)?;
    }
    if let Some((context, receipt)) = retained {
        receipt.checkpoint(&context).map_err(|source| {
            crate::context::host_allocation::host_allocation_runtime_error(
                Some(&context),
                "RunMat:Gpu:InvalidResult",
                "shared result list provenance authority changed",
                source,
            )
        })?;
    }
    Ok(())
}
