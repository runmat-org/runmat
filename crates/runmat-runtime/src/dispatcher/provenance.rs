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
                    cell.data.make_unique_unbounded().map_err(|error| {
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
            for value in object.fields.values_mut() {
                project(value, explicit)?;
            }
        }
        Value::StructArray(array) => {
            let mut result = Ok(());
            array.for_each_value_mut(|value| {
                if result.is_ok() {
                    result = project(value, explicit);
                }
            });
            result?;
        }
        Value::Object(object) => {
            for value in object.properties.values_mut() {
                project(value, explicit)?;
            }
        }
        Value::ObjectArray(_) => {
            // Consume owned elements: no host-payload cloning and no unchecked
            // mutable access that could break homogeneous object-array storage.
            let Value::ObjectArray(array) = std::mem::replace(value, Value::Num(0.)) else {
                unreachable!("matched object array")
            };
            let class = array.class_name().clone();
            let shape = array.shape().to_vec();
            let selected_provenance = array.selected_provenance().to_vec();
            let unselected_provenance = array.has_unselected_provenance();
            let mut elements = array.into_data();
            for element in &mut elements {
                project(element, explicit)?;
            }
            let rebuilt =
                runmat_value::ObjectArray::new(class, elements, shape).map_err(|error| {
                    build_runtime_error(format!("GPU result object array: {error}"))
                        .with_identifier("RunMat:Gpu:InvalidResult")
                        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
                        .build()
                })?;
            let rebuilt = if rebuilt.is_empty() {
                let rebuilt = rebuilt
                    .with_selected_provenance(selected_provenance)
                    .map_err(RuntimeError::from)?;
                if unselected_provenance {
                    rebuilt.with_unselected_provenance()
                } else {
                    rebuilt
                }
            } else {
                rebuilt
            };
            *value = Value::ObjectArray(rebuilt);
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
