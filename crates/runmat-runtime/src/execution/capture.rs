use std::collections::HashSet;

use runmat_accelerate_api::GpuTensorHandle;
use runmat_value::Value;

use crate::RuntimeError;

/// Validate values captured by a lazy future before they can cross a spawn boundary.
pub fn validate_spawn_capture(value: &Value) -> Result<(), RuntimeError> {
    super::storable::validate_storable_value(value)?;
    for_each_gpu_handle(value, &mut |handle| {
        let provider = runmat_accelerate_api::provider_for_handle(handle).ok_or_else(|| {
            crate::runtime_error::semantic_error(
                "SpawnProviderUnavailable",
                format!(
                    "spawn cannot capture GPU handle buffer {} (device {}) without an active provider",
                    handle.buffer_id(), handle.device_id()
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
                    handle.buffer_id(),
                    provider.device_info(),
                    policy.as_str()
                ),
            ));
        }
        Ok(())
    })
}

/// Validate an aggregate capture without manufacturing a language value to
/// represent the executor-owned collection.
pub fn validate_spawn_captures(values: &[Value]) -> Result<(), RuntimeError> {
    for value in values {
        validate_spawn_capture(value)?;
    }
    Ok(())
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
        Value::ScopedFunctionHandle(_) => Err(crate::build_runtime_error(
            "invocation-scoped callables cannot cross a spawn boundary",
        )
        .with_identifier("RunMat:ScopedCallable:NotPortable")
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .build()),
        Value::GpuTensor(handle) => operation(handle),
        Value::Cell(cell) => visit_values(cell.data(), operation, visited_handles),
        Value::Struct(value) => {
            for value in value.fields.values() {
                visit_gpu_handles(value, operation, visited_handles)?;
            }
            Ok(())
        }
        Value::StructArray(array) => {
            let mut result = Ok(());
            array.for_each_value(|value| {
                if result.is_ok() {
                    result = visit_gpu_handles(value, operation, visited_handles);
                }
            });
            result
        }
        Value::Object(value) => {
            for value in value.properties().values() {
                visit_gpu_handles(value, operation, visited_handles)?;
            }
            Ok(())
        }
        Value::ObjectArray(value) => visit_values(value.data(), operation, visited_handles),
        Value::Closure(value) => visit_values(&value.captures, operation, visited_handles),
        Value::OutputList(_) => unreachable!("storable validation rejects transient sequences"),
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
        | Value::BoundFunctionHandle(_)
        | Value::ClassRef(_)
        | Value::MException(_)
        | Value::Future(_)
        | Value::Task(_)
        | Value::Pool(_)
        | Value::Job(_)
        | Value::Distributed(_)
        | Value::Composite(_)
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
    use runmat_value::{CellArray, HandleRef, StructArray};

    struct CaptureProvider {
        owner: runmat_accelerate_api::test_support::TestAllocationOwner,
        policy: SpawnHandleConcurrency,
    }

    impl AccelProvider for CaptureProvider {
        fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
            Err(runmat_accelerate_api::unsupported_provider_operation(
                "unused capture upload",
            ))
        }
        fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
            Box::pin(async {
                Err(runmat_accelerate_api::unsupported_provider_operation(
                    "unused capture download",
                ))
            })
        }
        fn device_info(&self) -> String {
            "capture fixture owner".into()
        }
        fn device_id(&self) -> u32 {
            self.owner.device_id()
        }
        fn spawn_handle_concurrency(&self) -> SpawnHandleConcurrency {
            self.policy
        }
    }

    fn reject_provider() -> &'static CaptureProvider {
        static PROVIDER: std::sync::OnceLock<CaptureProvider> = std::sync::OnceLock::new();
        PROVIDER.get_or_init(|| CaptureProvider {
            owner: Default::default(),
            policy: SpawnHandleConcurrency::Reject,
        })
    }
    fn share_provider() -> &'static CaptureProvider {
        static PROVIDER: std::sync::OnceLock<CaptureProvider> = std::sync::OnceLock::new();
        PROVIDER.get_or_init(|| CaptureProvider {
            owner: Default::default(),
            policy: SpawnHandleConcurrency::ImmutableShare,
        })
    }
    fn gpu(owner: &runmat_accelerate_api::test_support::TestAllocationOwner) -> Value {
        Value::GpuTensor(
            owner
                .allocate_zeroed(
                    vec![1],
                    runmat_accelerate_api::NumericElementType::F64,
                    runmat_accelerate_api::GpuTensorStorage::Real,
                )
                .unwrap()
                .handle,
        )
    }

    #[test]
    fn spawn_capture_obeys_provider_concurrency_policy() {
        let _guard = ThreadProviderGuard::set(Some(reject_provider()));
        let error =
            validate_spawn_capture(&gpu(&reject_provider().owner)).expect_err("reject capture");
        assert_eq!(error.identifier(), Some("RunMat:SpawnGpuHandleUnsupported"));
        drop(_guard);

        let _guard = ThreadProviderGuard::set(Some(share_provider()));
        validate_spawn_capture(&gpu(&share_provider().owner)).expect("immutable sharing is safe");
    }

    #[test]
    fn spawn_capture_rejects_missing_provider() {
        let _guard = ThreadProviderGuard::set(None);
        runmat_accelerate_api::clear_provider();
        let owner = runmat_accelerate_api::test_support::TestAllocationOwner::new();
        let error = validate_spawn_capture(&gpu(&owner)).expect_err("missing provider");
        assert_eq!(error.identifier(), Some("RunMat:SpawnProviderUnavailable"));
    }

    #[test]
    fn spawn_capture_rejects_transient_sequences_before_recursive_traversal() {
        let nested = Value::Cell(
            CellArray::new(vec![Value::output_list(vec![Value::Num(1.0)])], 1, 1)
                .expect("nested capture"),
        );
        let error = validate_spawn_capture(&nested).expect_err("transient capture must fail");
        assert_eq!(
            error.identifier(),
            Some("RunMat:TransientSequenceNotStorable")
        );

        let closure = Value::Closure(runmat_value::Closure::from_data(
            runmat_value::ClosureData {
                lexical_environment: None,
                function_name: "capture_target".into(),
                binding: None,
                source_role: runmat_types::FunctionSourceRole::Unknown,
                source_file: None,
                captures: vec![Value::output_list(Vec::new())].into(),
            },
        ));
        let error = validate_spawn_capture(&closure).expect_err("transient closure must fail");
        assert_eq!(
            error.identifier(),
            Some("RunMat:TransientSequenceNotStorable")
        );
    }

    #[test]
    fn spawn_capture_recurses_through_cells_closures_and_handle_objects() {
        let _guard = ThreadProviderGuard::set(Some(reject_provider()));
        let cell = Value::Cell(
            CellArray::new(vec![Value::Num(1.0), gpu(&reject_provider().owner)], 1, 2)
                .expect("test cell"),
        );
        assert_eq!(
            validate_spawn_capture(&cell).unwrap_err().identifier(),
            Some("RunMat:SpawnGpuHandleUnsupported")
        );

        let closure = Value::Closure(runmat_value::Closure::from_data(
            runmat_value::ClosureData {
                lexical_environment: None,
                function_name: "worker".into(),
                binding: None,
                source_role: runmat_types::FunctionSourceRole::Unknown,
                source_file: None,
                captures: vec![gpu(&reject_provider().owner)].into(),
            },
        ));
        assert_eq!(
            validate_spawn_capture(&closure).unwrap_err().identifier(),
            Some("RunMat:SpawnGpuHandleUnsupported")
        );

        let mut payload = runmat_value::UnboundedStructValueBuilder::new();
        payload
            .fields
            .insert("nested".into(), gpu(&reject_provider().owner));
        let target = runmat_gc::gc_allocate(Value::Struct(payload.finish())).expect("gc payload");
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

    #[test]
    fn spawn_capture_recurses_through_nd_structure_arrays() {
        let _guard = ThreadProviderGuard::set(Some(reject_provider()));
        let element = |payload: Value, nested: Value| {
            let mut value = runmat_value::UnboundedStructValueBuilder::new();
            value.insert("payload", payload);
            value.insert("nested", nested);
            value.finish()
        };
        let array = StructArray::with_fields(
            vec!["payload".into(), "nested".into()],
            vec![
                element(Value::Num(1.0), Value::Num(2.0)),
                element(Value::Num(3.0), gpu(&reject_provider().owner)),
            ],
            vec![1, 1, 2],
        )
        .expect("N-D structure array");
        assert_eq!(
            validate_spawn_capture(&Value::StructArray(array))
                .unwrap_err()
                .identifier(),
            Some("RunMat:SpawnGpuHandleUnsupported")
        );

        let empty = StructArray::with_fields(
            vec!["payload".into(), "nested".into()],
            Vec::new(),
            vec![0, 2],
        )
        .expect("empty structure array");
        validate_spawn_capture(&Value::StructArray(empty))
            .expect("empty structure array has no handles");
    }
}
