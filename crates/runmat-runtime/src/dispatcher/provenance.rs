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
            for value in &mut cell.data {
                project(value, explicit)?;
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
            let mut elements = array.into_data();
            for element in &mut elements {
                project(element, explicit)?;
            }
            *value = Value::ObjectArray(
                runmat_value::ObjectArray::new(class, elements, shape).map_err(|error| {
                    build_runtime_error(format!("GPU result object array: {error}"))
                        .with_identifier("RunMat:Gpu:InvalidResult")
                        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
                        .build()
                })?,
            );
        }
        Value::Closure(closure) => {
            for value in &mut closure.captures {
                project(value, explicit)?;
            }
        }
        Value::OutputList(values) => {
            for value in values {
                project(value, explicit)?;
            }
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
