use super::*;
use crate::builtins::common::test_support;
use futures::executor::block_on;
use runmat_accelerate_api::{AccelProvider as _, HostTensorView};
use runmat_value::{IntValue, IntegerStorage};

#[derive(Clone, Copy)]
enum NativeBehavior {
    Success,
    Error,
    Reject,
}

struct CountingCumtrapzProvider {
    owner: runmat_accelerate_api::test_support::TestAllocationOwner,
    allocations: std::sync::Mutex<Vec<runmat_accelerate_api::test_support::RetirementProbe>>,
    behavior: NativeBehavior,
}

impl CountingCumtrapzProvider {
    fn values(&self, handle: &runmat_accelerate_api::GpuTensorHandle) -> Vec<f64> {
        self.allocations
            .lock()
            .unwrap()
            .iter()
            .find_map(|probe| {
                probe
                    .with_data(handle, |data| {
                        let runmat_accelerate_api::HostNumericDataView::F64(data) = data else {
                            panic!("fixture requires native double payload");
                        };
                        data.to_vec()
                    })
                    .ok()
            })
            .expect("live fixture allocation")
    }
    fn retired(&self) -> usize {
        self.allocations
            .lock()
            .unwrap()
            .iter()
            .map(|probe| probe.retirements())
            .sum()
    }
}

impl runmat_accelerate_api::AccelProvider for CountingCumtrapzProvider {
    fn upload(
        &self,
        host: &HostTensorView,
    ) -> anyhow::Result<runmat_accelerate_api::GpuTensorHandle> {
        let allocation =
            self.owner
                .allocate_numeric(runmat_accelerate_api::HostNumericTensorOwned {
                    data: runmat_accelerate_api::HostNumericDataOwned::F64(host.data.to_vec()),
                    shape: host.shape.to_vec(),
                    storage: runmat_accelerate_api::GpuTensorStorage::Real,
                })?;
        self.allocations.lock().unwrap().push(allocation.probe);
        Ok(allocation.handle)
    }
    fn download<'a>(
        &'a self,
        _: &'a runmat_accelerate_api::GpuTensorHandle,
    ) -> runmat_accelerate_api::AccelDownloadFuture<'a> {
        Box::pin(async { panic!("native completion or terminal failure must not gather") })
    }
    fn device_info(&self) -> String {
        "cumtrapz allocation retirement fixture".into()
    }
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }

    fn cumtrapz_dim(
        &self,
        input: &runmat_accelerate_api::GpuTensorHandle,
        dim: usize,
        spacing: runmat_accelerate_api::ProviderTrapezoidSpacing<'_>,
    ) -> runmat_accelerate_api::ProviderExecutionResult<
        Option<runmat_accelerate_api::GpuTensorHandle>,
    > {
        let result: anyhow::Result<_> = (|| match self.behavior {
            NativeBehavior::Success => {
                use runmat_accelerate_api::ProviderTrapezoidSpacing as S;
                let spacing = match spacing {
                    S::Unit => SpacingSpec::Unit,
                    S::Scalar(value) => SpacingSpec::Scalar(value),
                    S::ScalarHandle(handle) => SpacingSpec::Scalar(self.values(handle)[0]),
                    S::Vector(handle) => SpacingSpec::Vector(self.values(handle)),
                    S::Tensor(handle) => SpacingSpec::Tensor(
                        Tensor::new(self.values(handle), handle.shape().to_vec()).unwrap(),
                    ),
                };
                let tensor = Tensor::new(self.values(input), input.shape().to_vec()).unwrap();
                let result = cumtrapz_tensor(&tensor, &spacing, dim + 1, false)
                    .expect("fixture canonical integration");
                self.upload(&HostTensorView {
                    data: &result.materialize_f64(),
                    shape: result.shape(),
                })
            }
            NativeBehavior::Error => anyhow::bail!("deliberate cumtrapz failure"),
            NativeBehavior::Reject => self.upload(&HostTensorView {
                data: &[0.0],
                shape: &[1, 1],
            }),
        })();
        result.map(Some).map_err(|source| {
            runmat_accelerate_api::ProviderExecutionError::from(source).with_unproven_effects()
        })
    }
}

fn with_counting_provider<R>(
    behavior: NativeBehavior,
    body: impl FnOnce(&'static CountingCumtrapzProvider) -> R,
) -> R {
    let _guard = test_support::accel_test_lock();
    let provider = Box::leak(Box::new(CountingCumtrapzProvider {
        owner: runmat_accelerate_api::test_support::TestAllocationOwner::new(),
        allocations: std::sync::Mutex::new(Vec::new()),
        behavior,
    }));
    unsafe { runmat_accelerate_api::register_provider(provider) };
    let _thread_provider = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    body(provider)
}

fn run_cumtrapz(first: Value, rest: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    block_on(super::cumtrapz_builtin(None, first, rest))
}

#[test]
fn cumtrapz_scalar_is_zero() {
    let value = run_cumtrapz(
        Value::Num(5.0),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    )
    .expect("cumtrapz");
    assert_eq!(value, Value::Num(0.0));
}

#[test]
fn cumtrapz_row_vector_unit_spacing() {
    let y = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
    let value = run_cumtrapz(
        Value::Tensor(y),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    )
    .expect("cumtrapz");
    let Value::Tensor(out) = value else {
        panic!("expected tensor result");
    };
    assert_eq!(out.shape(), &[1, 3]);
    assert_eq!(out.materialize_f64(), vec![0.0, 1.5, 4.0]);
}

#[test]
fn cumtrapz_nonuniform_x_vector() {
    let x = Tensor::new(vec![0.0, 1.0, 3.0], vec![1, 3]).unwrap();
    let y = Tensor::new(vec![0.0, 1.0, 2.0], vec![1, 3]).unwrap();
    let value = run_cumtrapz(
        Value::Tensor(x),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Tensor(y)]),
    )
    .expect("cumtrapz");
    let Value::Tensor(out) = value else {
        panic!("expected tensor result");
    };
    assert_eq!(out.materialize_f64(), vec![0.0, 0.5, 3.5]);
}

#[test]
fn cumtrapz_reads_typed_integer_values_and_spacing_exactly() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let x = Tensor::new_integer(IntegerStorage::U16(vec![0, 1, 3]), vec![1, 3]).expect("x");
    let y = Tensor::new_integer(IntegerStorage::I16(vec![0, 1, 2]), vec![1, 3]).expect("y");

    let value = run_cumtrapz(
        Value::Tensor(x),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Tensor(y)]),
    )
    .expect("cumtrapz");

    let Value::Tensor(out) = value else {
        panic!("expected tensor result");
    };
    assert_eq!(out.materialize_f64(), vec![0.0, 0.5, 3.5]);
}

#[test]
fn cumtrapz_matrix_dimension_two() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let y = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();
    let value = run_cumtrapz(
        Value::Tensor(y),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Int(IntValue::I32(
            2,
        ))]),
    )
    .expect("cumtrapz");
    let Value::Tensor(out) = value else {
        panic!("expected tensor result");
    };
    assert_eq!(out.shape(), &[2, 3]);
    assert_eq!(out.materialize_f64(), vec![0.0, 0.0, 1.5, 4.5, 4.0, 10.0]);
}

#[test]
fn cumtrapz_descriptor_signatures_cover_core_forms() {
    let labels: Vec<&str> = CUMTRAPZ_DESCRIPTOR
        .signatures
        .iter()
        .map(|sig| sig.label)
        .collect();
    assert!(labels.contains(&"Q = cumtrapz(Y)"));
    assert!(labels.contains(&"Q = cumtrapz(Y, dim)"));
    assert!(labels.contains(&"Q = cumtrapz(X, Y)"));
    assert!(labels.contains(&"Q = cumtrapz(X, Y, dim)"));
}

#[test]
fn cumtrapz_descriptor_errors_have_stable_codes() {
    assert!(CUMTRAPZ_DESCRIPTOR
        .errors
        .iter()
        .any(|error| error.code == CUMTRAPZ_ERROR_INVALID_ARGUMENT.code));
    assert!(CUMTRAPZ_DESCRIPTOR
        .errors
        .iter()
        .any(|error| error.code == CUMTRAPZ_ERROR_INVALID_INPUT.code));
    assert!(CUMTRAPZ_DESCRIPTOR
        .errors
        .iter()
        .any(|error| error.code == CUMTRAPZ_ERROR_INTERNAL.code));
}

#[test]
fn cumtrapz_gpu_spec_describes_logical_fallback() {
    assert!(GPU_SPEC
        .notes
        .contains("integer and logical resident inputs"));
    assert!(GPU_SPEC.notes.contains("validated host fallback"));
    assert!(!GPU_SPEC
        .notes
        .contains("Real, logical, and complex-interleaved"));
}

#[test]
fn cumtrapz_invalid_dim_uses_descriptor_identifier() {
    let y = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
    let err = run_cumtrapz(
        Value::Tensor(y),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Int(IntValue::I32(
            0,
        ))]),
    )
    .expect_err("cumtrapz");
    assert_eq!(err.identifier(), CUMTRAPZ_ERROR_INVALID_ARGUMENT.identifier);
}

#[test]
fn cumtrapz_too_many_inputs_uses_descriptor_identifier() {
    let err = run_cumtrapz(
        Value::Num(1.0),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Num(2.0),
            Value::Num(3.0),
            Value::Num(4.0),
        ]),
    )
    .expect_err("cumtrapz");
    assert_eq!(err.identifier(), CUMTRAPZ_ERROR_INVALID_ARGUMENT.identifier);
}

#[test]
fn cumtrapz_gpu_input_preserves_real_result_residency() {
    test_support::with_test_provider(|provider| {
        let y = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
        let handle = provider
            .upload(&HostTensorView {
                data: &y.materialize_f64(),
                shape: y.shape(),
            })
            .expect("upload");
        let result = run_cumtrapz(
            Value::GpuTensor(handle),
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        )
        .expect("cumtrapz gpu");
        let Value::GpuTensor(out) = result else {
            panic!("expected gpu result");
        };
        let gathered = test_support::gather(Value::GpuTensor(out)).expect("gather");
        assert_eq!(gathered.shape(), &[1, 3]);
        assert_eq!(gathered.materialize_f64(), vec![0.0, 1.5, 4.0]);
    });
}

#[test]
fn cumtrapz_complex_gpu_input_preserves_result_residency() {
    test_support::with_test_provider(|provider| {
        let y = ComplexTensor::new(vec![(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)], vec![1, 3]).unwrap();
        let handle = crate::builtins::common::gpu_helpers::upload_complex_tensor(provider, &y)
            .expect("upload complex");
        provider.reset_telemetry();

        let result = run_cumtrapz(
            Value::GpuTensor(handle.clone()),
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        )
        .expect("cumtrapz complex gpu");
        let Value::GpuTensor(out) = result else {
            panic!("expected gpu result");
        };
        assert_eq!(
            runmat_accelerate_api::handle_storage(&out),
            runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
        );
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);

        let gathered = block_on(provider.download(&out)).expect("download");
        assert_eq!(
            gathered.storage,
            runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
        );
        assert_eq!(gathered.shape, vec![1, 3]);
        assert_eq!(gathered.data, vec![0.0, 0.0, 1.5, 1.5, 4.0, 4.0]);
        handle.release().unwrap();
        out.release().unwrap();
    });
}

#[test]
fn cumtrapz_gpu_input_uses_vector_spacing_on_provider() {
    test_support::with_test_provider(|provider| {
        let x = Tensor::new(vec![0.0, 1.0, 3.0], vec![1, 3]).unwrap();
        let y = Tensor::new(vec![0.0, 1.0, 2.0], vec![1, 3]).unwrap();
        let handle = provider
            .upload(&HostTensorView {
                data: &y.materialize_f64(),
                shape: y.shape(),
            })
            .expect("upload y");
        let result = run_cumtrapz(
            Value::Tensor(x),
            crate::builtins::common::test_support::builtin_arguments(vec![Value::GpuTensor(
                handle,
            )]),
        )
        .expect("cumtrapz gpu");
        let Value::GpuTensor(out) = result else {
            panic!("expected gpu result");
        };
        let gathered = test_support::gather(Value::GpuTensor(out)).expect("gather");
        assert_eq!(gathered.shape(), &[1, 3]);
        assert_eq!(gathered.materialize_f64(), vec![0.0, 0.5, 3.5]);
    });
}

#[test]
fn cumtrapz_gpu_input_uses_tensor_spacing_on_provider() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    test_support::with_test_provider(|provider| {
        let x = Tensor::new(vec![0.0, 0.0, 1.0, 1.0, 3.0, 3.0], vec![2, 3]).unwrap();
        let y = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();
        let y_handle = provider
            .upload(&HostTensorView {
                data: &y.materialize_f64(),
                shape: y.shape(),
            })
            .expect("upload y");
        let x_handle = provider
            .upload(&HostTensorView {
                data: &x.materialize_f64(),
                shape: x.shape(),
            })
            .expect("upload x");
        let result = run_cumtrapz(
            Value::GpuTensor(x_handle),
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::GpuTensor(y_handle),
                Value::Int(IntValue::I32(2)),
            ]),
        )
        .expect("cumtrapz gpu");
        let Value::GpuTensor(out) = result else {
            panic!("expected gpu result");
        };
        let gathered = test_support::gather(Value::GpuTensor(out)).expect("gather");
        assert_eq!(gathered.shape(), &[2, 3]);
        assert_eq!(
            gathered.materialize_f64(),
            vec![0.0, 0.0, 1.5, 4.5, 6.5, 15.5]
        );
    });
}

#[test]
fn cumtrapz_extensions_are_independently_mode_gated() {
    let integer =
        || Value::Tensor(Tensor::new_integer(IntegerStorage::U8(vec![1, 2]), vec![1, 2]).unwrap());
    let logical =
        || Value::LogicalArray(runmat_value::LogicalArray::new(vec![1, 0], vec![1, 2]).unwrap());
    let y = || Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap());
    let tensor_x = || Value::Tensor(Tensor::new(vec![0.0, 1.0, 0.0, 1.0], vec![2, 2]).unwrap());
    let matrix_y = || Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap());
    let cases = [
        (
            integer(),
            Vec::new(),
            CUMTRAPZ_INTEGER_Y_EXTENSION.error_identifier,
        ),
        (
            logical(),
            Vec::new(),
            CUMTRAPZ_LOGICAL_Y_EXTENSION.error_identifier,
        ),
        (
            integer(),
            vec![y()],
            CUMTRAPZ_INTEGER_X_EXTENSION.error_identifier,
        ),
        (
            logical(),
            vec![y()],
            CUMTRAPZ_LOGICAL_X_EXTENSION.error_identifier,
        ),
        (
            y(),
            vec![Value::Int(IntValue::U8(2))],
            CUMTRAPZ_INTEGER_DIM_EXTENSION.error_identifier,
        ),
        (
            tensor_x(),
            vec![matrix_y()],
            CUMTRAPZ_TENSOR_SPACING_EXTENSION.error_identifier,
        ),
    ];
    let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
    for (first, rest, identifier) in cases {
        let error = run_cumtrapz(
            first,
            crate::builtins::common::test_support::builtin_arguments(rest),
        )
        .expect_err("strict mode rejects extension");
        assert_eq!(error.identifier(), identifier);
    }
}

#[test]
fn cumtrapz_integer_capabilities_cover_all_classes_and_preserve_single() {
    assert_eq!(CUMTRAPZ_INTEGER_CAPABILITIES.len(), 3);
    assert!(CUMTRAPZ_INTEGER_CAPABILITIES
        .iter()
        .all(|capability| capability.inputs[0].classes.len() == 8));
    let y = Tensor::from_f32(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
    let Value::Tensor(output) = run_cumtrapz(
        Value::Tensor(y),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    )
    .unwrap() else {
        panic!("expected tensor");
    };
    assert_eq!(output.numeric_dtype(), NumericDType::F32);
    let complex = ComplexTensor::from_f32(vec![(1.0, 1.0), (2.0, 2.0)], vec![1, 2]).unwrap();
    let Value::ComplexTensor(output) = run_cumtrapz(
        Value::ComplexTensor(complex),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    )
    .unwrap() else {
        panic!("expected complex tensor");
    };
    assert_eq!(output.numeric_dtype(), NumericDType::F32);
}

#[test]
fn cumtrapz_accepts_all_integer_storage_classes_and_rejects_inexact_boundary() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let storages = [
        IntegerStorage::I8(vec![0, 1, 2]),
        IntegerStorage::I16(vec![0, 1, 2]),
        IntegerStorage::I32(vec![0, 1, 2]),
        IntegerStorage::I64(vec![0, 1, 2]),
        IntegerStorage::U8(vec![0, 1, 2]),
        IntegerStorage::U16(vec![0, 1, 2]),
        IntegerStorage::U32(vec![0, 1, 2]),
        IntegerStorage::U64(vec![0, 1, 2]),
    ];
    for storage in storages {
        let y = Tensor::new_integer(storage, vec![1, 3]).unwrap();
        let Value::Tensor(output) = run_cumtrapz(
            Value::Tensor(y),
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        )
        .unwrap() else {
            panic!("expected tensor");
        };
        assert_eq!(output.materialize_f64(), vec![0.0, 0.5, 2.0]);
    }
    let inexact = Tensor::new_integer(
        IntegerStorage::U64(vec![0, 9_007_199_254_740_993]),
        vec![1, 2],
    )
    .unwrap();
    let error = run_cumtrapz(
        Value::Tensor(inexact),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    )
    .expect_err("inexact boundary");
    assert_eq!(error.identifier(), CUMTRAPZ_ERROR_INVALID_INPUT.identifier);
}

#[test]
fn cumtrapz_resident_extension_roles_reject_before_provider_access() {
    fn resident(shape: Vec<usize>, integer: bool, logical: bool) -> Value {
        use runmat_accelerate_api::{
            test_support::TestAllocationOwner, GpuTensorStorage, NumericElementType,
            ProviderPrecision,
        };
        let owner = TestAllocationOwner::new();
        let allocation = if logical {
            owner.allocate_logical_zeroed(shape, ProviderPrecision::F64)
        } else {
            owner.allocate_zeroed(
                shape,
                if integer {
                    NumericElementType::U64
                } else {
                    NumericElementType::F64
                },
                GpuTensorStorage::Real,
            )
        }
        .unwrap();
        Value::GpuTensor(allocation.handle)
    }
    let y = || Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap());
    let cases = [
        (
            resident(vec![1, 2], true, false),
            Vec::new(),
            CUMTRAPZ_INTEGER_Y_EXTENSION.error_identifier,
        ),
        (
            resident(vec![1, 2], false, true),
            Vec::new(),
            CUMTRAPZ_LOGICAL_Y_EXTENSION.error_identifier,
        ),
        (
            resident(vec![1, 2], true, false),
            vec![y()],
            CUMTRAPZ_INTEGER_X_EXTENSION.error_identifier,
        ),
        (
            resident(vec![1, 2], false, true),
            vec![y()],
            CUMTRAPZ_LOGICAL_X_EXTENSION.error_identifier,
        ),
        (
            Value::Num(1.0),
            vec![y(), resident(vec![1, 1], true, false)],
            CUMTRAPZ_INTEGER_DIM_EXTENSION.error_identifier,
        ),
    ];
    let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
    for (first, rest, identifier) in cases {
        let error = run_cumtrapz(
            first,
            crate::builtins::common::test_support::builtin_arguments(rest),
        )
        .expect_err("resident extension gate");
        assert_eq!(error.identifier(), identifier);
    }
}

#[test]
fn cumtrapz_frees_host_spacing_upload_for_every_native_outcome() {
    for (behavior, expected_frees) in [
        (NativeBehavior::Success, 1),
        (NativeBehavior::Error, 1),
        (NativeBehavior::Reject, 2),
    ] {
        with_counting_provider(behavior, |provider| {
            let y = [0.0, 1.0, 2.0];
            let y_handle = provider
                .upload(&HostTensorView {
                    data: &y,
                    shape: &[1, 3],
                })
                .expect("resident Y");
            let x = Value::Tensor(Tensor::new(vec![0.0, 1.0, 3.0], vec![1, 3]).unwrap());
            let result = run_cumtrapz(
                x,
                crate::builtins::common::test_support::builtin_arguments(vec![Value::GpuTensor(
                    y_handle.clone(),
                )]),
            );
            assert_eq!(provider.retired(), expected_frees);
            match behavior {
                NativeBehavior::Success => {
                    let Value::GpuTensor(result_handle) = result.expect("native cumtrapz") else {
                        panic!("expected resident native result");
                    };
                    assert_eq!(provider.values(&result_handle), vec![0.0, 0.5, 3.5]);
                    result_handle.release().unwrap();
                }
                NativeBehavior::Error | NativeBehavior::Reject => {
                    let error = result.expect_err("operational and malformed outputs are terminal");
                    assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
                }
            }
            y_handle.allocation_lease().validate_live().unwrap();
            y_handle.release().unwrap();
            assert!(provider
                .allocations
                .lock()
                .unwrap()
                .iter()
                .all(|probe| probe.retirements() == 1));
        });
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
#[cfg(feature = "wgpu")]
fn cumtrapz_wgpu_matches_cpu_vector_spacing() {
    let Ok(provider) = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    ) else {
        return;
    };
    runmat_accelerate_api::set_thread_provider(Some(provider));
    let x = Tensor::new(vec![0.0, 1.0, 3.0], vec![1, 3]).unwrap();
    let y = Tensor::new(vec![0.0, 1.0, 2.0], vec![1, 3]).unwrap();
    let cpu = run_cumtrapz(
        Value::Tensor(x.clone()),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Tensor(y.clone())]),
    )
    .unwrap();
    let y_handle = provider
        .upload(&HostTensorView {
            data: &y.materialize_f64(),
            shape: y.shape(),
        })
        .unwrap();
    let gpu = run_cumtrapz(
        Value::Tensor(x),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::GpuTensor(y_handle)]),
    )
    .unwrap();
    let gathered = test_support::gather(gpu).expect("gather gpu");
    let expected = match cpu {
        Value::Tensor(tensor) => tensor,
        other => panic!("unexpected cpu result {other:?}"),
    };
    let tol = match provider.precision() {
        runmat_accelerate_api::ProviderPrecision::F64 => 1e-9,
        runmat_accelerate_api::ProviderPrecision::F32 => 1e-5,
    };
    assert_eq!(gathered.shape(), expected.shape());
    for (actual, expected) in gathered
        .materialize_f64()
        .iter()
        .zip(expected.materialize_f64().iter())
    {
        assert!((actual - expected).abs() < tol);
    }
}

#[test]
fn virtual_integration_axes_do_not_allocate_axis_sized_metadata() {
    let input = Tensor::from_f32(vec![1., 2., 3., 4.], vec![2, 2]).unwrap();
    let result = cumtrapz_tensor(&input, &SpacingSpec::Unit, usize::MAX, true).unwrap();
    assert_eq!(result.shape(), &[2, 2]);
    assert_eq!(result.as_f32_slice().unwrap(), &[0., 0., 0., 0.]);
    let result = super::super::trapz::trapz_tensor(&input, &SpacingSpec::Unit, usize::MAX).unwrap();
    assert_eq!(result.shape(), &[2, 2]);
    assert_eq!(result.as_f32_slice().unwrap(), &[0., 0., 0., 0.]);
}
