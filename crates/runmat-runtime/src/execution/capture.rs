use std::collections::HashSet;

use runmat_accelerate_api::GpuTensorHandle;
use runmat_value::Value;

use crate::RuntimeError;

/// Validate values captured by a lazy future before they can cross a spawn boundary.
pub fn validate_spawn_capture(value: &Value) -> Result<(), RuntimeError> {
    for_each_gpu_handle(value, &mut |handle| {
        let provider = runmat_accelerate_api::provider_for_handle(handle).ok_or_else(|| {
            crate::runtime_error::semantic_error(
                "SpawnProviderUnavailable",
                format!(
                    "spawn cannot capture GPU handle buffer {} (device {}) without an active provider",
                    handle.buffer_id, handle.device_id
                ),
            )
        })?;
        let policy = provider.spawn_handle_concurrency();
        if matches!(
            policy,
            runmat_accelerate_api::SpawnHandleConcurrency::Reject
        ) {
            return Err(crate::runtime_error::semantic_error(
                "SpawnGpuHandleUnsupported",
                format!(
                    "spawn cannot capture GPU handle buffer {} on provider '{}' (spawn_handle_concurrency={})",
                    handle.buffer_id,
                    provider.device_info(),
                    policy.as_str()
                ),
            ));
        }
        Ok(())
    })
}

fn for_each_gpu_handle(
    value: &Value,
    operation: &mut impl FnMut(&GpuTensorHandle) -> Result<(), RuntimeError>,
) -> Result<(), RuntimeError> {
    visit_gpu_handles(value, operation, &mut HashSet::new())
}

fn visit_gpu_handles(
    value: &Value,
    operation: &mut impl FnMut(&GpuTensorHandle) -> Result<(), RuntimeError>,
    visited_handles: &mut HashSet<usize>,
) -> Result<(), RuntimeError> {
    match value {
        Value::GpuTensor(handle) => operation(handle),
        Value::Cell(cell) => visit_values(&cell.data, operation, visited_handles),
        Value::Struct(value) => {
            for value in value.fields.values() {
                visit_gpu_handles(value, operation, visited_handles)?;
            }
            Ok(())
        }
        Value::Object(value) => {
            for value in value.properties.values() {
                visit_gpu_handles(value, operation, visited_handles)?;
            }
            Ok(())
        }
        Value::ObjectArray(value) => visit_values(value.data(), operation, visited_handles),
        Value::Closure(value) => visit_values(&value.captures, operation, visited_handles),
        Value::OutputList(values) => visit_values(values, operation, visited_handles),
        Value::HandleObject(handle) => {
            let address = runmat_gc::gc_handle_addr(&handle.target);
            if visited_handles.insert(address) {
                runmat_gc::gc_with_value(&handle.target, |value| {
                    visit_gpu_handles(value, operation, visited_handles)
                })
                .map_err(|error| RuntimeError::new(format!("invalid handle target: {error}")))??;
            }
            Ok(())
        }
        Value::Int(_)
        | Value::Num(_)
        | Value::Complex(_, _)
        | Value::Bool(_)
        | Value::LogicalArray(_)
        | Value::String(_)
        | Value::StringArray(_)
        | Value::CharArray(_)
        | Value::Symbolic(_)
        | Value::SymbolicArray(_)
        | Value::Tensor(_)
        | Value::SparseTensor(_)
        | Value::ComplexTensor(_)
        | Value::Listener(_)
        | Value::FunctionHandle(_)
        | Value::ExternalFunctionHandle(_)
        | Value::MethodFunctionHandle(_)
        | Value::BoundFunctionHandle { .. }
        | Value::ClassRef(_)
        | Value::MException(_)
        | Value::Future(_)
        | Value::Task(_)
        | Value::Pool(_)
        | Value::Job(_)
        | Value::Foreign(_) => Ok(()),
    }
}

fn visit_values(
    values: &[Value],
    operation: &mut impl FnMut(&GpuTensorHandle) -> Result<(), RuntimeError>,
    visited_handles: &mut HashSet<usize>,
) -> Result<(), RuntimeError> {
    for value in values {
        visit_gpu_handles(value, operation, visited_handles)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_accelerate_api::{
        AccelDownloadFuture, AccelProvider, HostTensorView, SpawnHandleConcurrency,
        ThreadProviderGuard,
    };
    use runmat_value::{CellArray, HandleRef, StructValue};

    struct RejectProvider;
    static REJECT_PROVIDER: RejectProvider = RejectProvider;

    impl AccelProvider for RejectProvider {
        fn upload(&self, _host: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
            Err(anyhow::anyhow!("unsupported"))
        }

        fn download<'a>(&'a self, _handle: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
            Box::pin(async { Err(anyhow::anyhow!("unsupported")) })
        }

        fn free(&self, _handle: &GpuTensorHandle) -> anyhow::Result<()> {
            Ok(())
        }

        fn device_info(&self) -> String {
            "reject-provider".into()
        }

        fn device_id(&self) -> u32 {
            41
        }
    }

    struct ShareProvider;
    static SHARE_PROVIDER: ShareProvider = ShareProvider;

    impl AccelProvider for ShareProvider {
        fn upload(&self, _host: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
            Err(anyhow::anyhow!("unsupported"))
        }

        fn download<'a>(&'a self, _handle: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
            Box::pin(async { Err(anyhow::anyhow!("unsupported")) })
        }

        fn free(&self, _handle: &GpuTensorHandle) -> anyhow::Result<()> {
            Ok(())
        }

        fn device_info(&self) -> String {
            "share-provider".into()
        }

        fn device_id(&self) -> u32 {
            42
        }

        fn spawn_handle_concurrency(&self) -> SpawnHandleConcurrency {
            SpawnHandleConcurrency::ImmutableShare
        }
    }

    fn gpu(device_id: u32, buffer_id: u64) -> Value {
        Value::GpuTensor(GpuTensorHandle {
            shape: vec![1],
            device_id,
            buffer_id,
            descriptor: Default::default(),
        })
    }

    #[test]
    fn spawn_capture_obeys_provider_concurrency_policy() {
        let _guard = ThreadProviderGuard::set(Some(&REJECT_PROVIDER));
        let error = validate_spawn_capture(&gpu(41, 7)).expect_err("reject capture");
        assert_eq!(error.identifier(), Some("RunMat:SpawnGpuHandleUnsupported"));
        drop(_guard);

        let _guard = ThreadProviderGuard::set(Some(&SHARE_PROVIDER));
        validate_spawn_capture(&gpu(42, 9)).expect("immutable sharing is safe");
    }

    #[test]
    fn spawn_capture_rejects_missing_provider() {
        let _guard = ThreadProviderGuard::set(None);
        let error = validate_spawn_capture(&gpu(99, 13)).expect_err("missing provider");
        assert_eq!(error.identifier(), Some("RunMat:SpawnProviderUnavailable"));
    }

    #[test]
    fn spawn_capture_recurses_through_cells_closures_and_handle_objects() {
        let _guard = ThreadProviderGuard::set(Some(&REJECT_PROVIDER));
        let cell = Value::Cell(
            CellArray::new(vec![Value::Num(1.0), gpu(41, 11)], 1, 2).expect("test cell"),
        );
        assert_eq!(
            validate_spawn_capture(&cell).unwrap_err().identifier(),
            Some("RunMat:SpawnGpuHandleUnsupported")
        );

        let closure = Value::Closure(runmat_value::Closure {
            function_name: "worker".into(),
            bound_function: None,
            captures: vec![gpu(41, 21)],
        });
        assert_eq!(
            validate_spawn_capture(&closure).unwrap_err().identifier(),
            Some("RunMat:SpawnGpuHandleUnsupported")
        );

        let mut payload = StructValue::new();
        payload.fields.insert("nested".into(), gpu(41, 31));
        let target = runmat_gc::gc_allocate(Value::Struct(payload)).expect("gc payload");
        let object = Value::HandleObject(HandleRef {
            class_name: "Payload".into(),
            target,
            valid: true,
        });
        assert_eq!(
            validate_spawn_capture(&object).unwrap_err().identifier(),
            Some("RunMat:SpawnGpuHandleUnsupported")
        );
    }
}
