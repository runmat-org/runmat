//! MATLAB-compatible `cosh` builtin with GPU-aware semantics for RunMat.

use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::{
    BuiltinErrorDescriptor, COSH_CHARACTER_INPUT_EXTENSION, COSH_ERROR_INTERNAL,
    COSH_ERROR_INVALID_INPUT, COSH_INTEGER_INPUT_EXTENSION, COSH_LOGICAL_INPUT_EXTENSION,
};
#[cfg(test)]
use runmat_builtins::{COSH_DESCRIPTOR, COSH_INTEGER_CAPABILITIES};
use runmat_macros::runtime_builtin;
#[cfg(test)]
use runmat_value::NumericDType;
use runmat_value::{CharArray, ComplexStorage, ComplexTensor, Tensor, Value};

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, FusionError,
    FusionExprContext, FusionKernelTemplate, GpuOpKind, ProviderHook, ReductionNaN,
    ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "cosh";

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::trigonometry::cosh")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "cosh",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Unary { name: "unary_cosh" }],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes:
        "Providers may execute cosh directly on the device; runtimes gather to the host when unary_cosh is unavailable.",
};

fn cosh_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    let mut builder = build_runtime_error(error.message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn cosh_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl std::fmt::Display,
) -> RuntimeError {
    let mut builder =
        build_runtime_error(format!("{}: {}", error.message, detail)).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::trigonometry::cosh")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "cosh",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: |ctx: &FusionExprContext| {
            let input = ctx.inputs.first().ok_or(FusionError::MissingInput(0))?;
            Ok(format!("cosh({input})"))
        },
    }),
    reduction: None,
    emits_nan: false,
    notes: "Fusion planner emits WGSL `cosh` calls; providers may override via fused elementwise kernels.",
};

#[runtime_builtin(
    name = "cosh",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::trigonometry::cosh"
)]
async fn cosh_builtin(value: Value) -> BuiltinResult<Value> {
    super::inverse_helpers::reject_excess_outputs(BUILTIN_NAME)?;
    super::inverse_helpers::ensure_input_extensions(
        &value,
        BUILTIN_NAME,
        &COSH_INTEGER_INPUT_EXTENSION,
        &COSH_LOGICAL_INPUT_EXTENSION,
        &COSH_CHARACTER_INPUT_EXTENSION,
    )?;
    crate::builtins::common::validation::ensure_runmat_integer_f64_boundary(
        &value,
        &COSH_INTEGER_INPUT_EXTENSION,
        BUILTIN_NAME,
        "X",
    )
    .await?;
    crate::builtins::common::validation::reject_typed_complex_integer(&value, "cosh")?;
    match value {
        Value::GpuTensor(handle) => cosh_gpu(handle).await,
        Value::Complex(re, im) => Ok(Value::Complex(
            cosh_complex_re(re, im),
            cosh_complex_im(re, im),
        )),
        Value::ComplexTensor(ct) => cosh_complex_tensor(ct),
        Value::CharArray(ca) => cosh_char_array(ca),
        Value::String(_) | Value::StringArray(_) => Err(cosh_error(&COSH_ERROR_INVALID_INPUT)),
        other => cosh_real(other),
    }
}

async fn cosh_gpu(handle: GpuTensorHandle) -> BuiltinResult<Value> {
    let provider = runmat_accelerate_api::provider_for_handle(&handle);
    let requires_exact_host_path = runmat_accelerate_api::handle_integer_type(&handle).is_some()
        || runmat_accelerate_api::handle_is_logical(&handle)
        || runmat_accelerate_api::handle_storage(&handle)
            != runmat_accelerate_api::GpuTensorStorage::Real;
    if !requires_exact_host_path {
        if let Some(provider) = provider {
            match provider.unary_cosh(&handle).await {
                Ok(output) => {
                    return super::inverse_helpers::validate_real_unary_provider_output(
                        provider,
                        &handle,
                        output,
                        BUILTIN_NAME,
                    )
                }
                Err(error) if gpu_helpers::provider_hook_is_unsupported(&error) => {}
                Err(error) => {
                    return Err(cosh_error_with_detail(
                        &COSH_ERROR_INTERNAL,
                        format!("provider unary_cosh failed: {error}"),
                    ))
                }
            }
        }
    }
    super::inverse_helpers::gather_value_compute_restore(
        handle,
        BUILTIN_NAME,
        |value| match value {
            Value::Complex(re, im) => Ok(Value::Complex(
                cosh_complex_re(re, im),
                cosh_complex_im(re, im),
            )),
            Value::ComplexTensor(tensor) => cosh_complex_tensor(tensor),
            other => cosh_real(other),
        },
    )
    .await
}

fn cosh_real(value: Value) -> BuiltinResult<Value> {
    let tensor = tensor::value_into_tensor_for("cosh", value)
        .map_err(|e| cosh_error_with_detail(&COSH_ERROR_INVALID_INPUT, e))?;
    cosh_tensor(tensor).map(tensor::tensor_into_value)
}

fn cosh_tensor(tensor: Tensor) -> BuiltinResult<Tensor> {
    if tensor.numeric_dtype() == runmat_value::NumericDType::F32 {
        let data = tensor
            .as_f32_slice()
            .expect("single tensor storage")
            .iter()
            .map(|&v| v.cosh())
            .collect();
        return Tensor::from_f32(data, tensor.shape.clone())
            .map_err(|e| cosh_error_with_detail(&COSH_ERROR_INTERNAL, e));
    }
    let data = tensor::tensor_values_f64_cow(&tensor)
        .iter()
        .map(|&v| v.cosh())
        .collect::<Vec<_>>();
    Tensor::new(data, tensor.shape.clone())
        .map_err(|e| cosh_error_with_detail(&COSH_ERROR_INTERNAL, e))
}

fn cosh_complex_tensor(ct: ComplexTensor) -> BuiltinResult<Value> {
    let shape = ct.shape.clone();
    let tensor = match ct.into_complex_storage() {
        ComplexStorage::F32(values) => ComplexTensor::from_f32(
            values
                .into_iter()
                .map(|(re, im)| {
                    (
                        cosh_complex_re(f64::from(re), f64::from(im)) as f32,
                        cosh_complex_im(f64::from(re), f64::from(im)) as f32,
                    )
                })
                .collect(),
            shape,
        ),
        ComplexStorage::F64(values) => ComplexTensor::new(
            values
                .into_iter()
                .map(|(re, im)| (cosh_complex_re(re, im), cosh_complex_im(re, im)))
                .collect(),
            shape,
        ),
        ComplexStorage::Integer(_) => Err("typed complex integer input is unsupported".into()),
    }
    .map_err(|e| cosh_error_with_detail(&COSH_ERROR_INTERNAL, e))?;
    Ok(Value::ComplexTensor(tensor))
}

fn cosh_char_array(ca: CharArray) -> BuiltinResult<Value> {
    let data = ca
        .data
        .iter()
        .map(|&ch| (ch as u32 as f64).cosh())
        .collect::<Vec<_>>();
    let tensor = Tensor::new(data, vec![ca.rows, ca.cols])
        .map_err(|e| cosh_error_with_detail(&COSH_ERROR_INTERNAL, e))?;
    Ok(Value::Tensor(tensor))
}

#[inline]
fn cosh_complex_re(re: f64, im: f64) -> f64 {
    re.cosh() * im.cos()
}

#[inline]
fn cosh_complex_im(re: f64, im: f64) -> f64 {
    re.sinh() * im.sin()
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_accelerate_api::{
        AccelDownloadFuture, AccelProvider, AccelProviderFuture, GpuTensorStorage, HostTensorOwned,
        HostTensorView, ProviderPrecision,
    };
    use runmat_value::{IntValue, LogicalArray, Tensor};
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicU64, AtomicU8, AtomicUsize, Ordering};
    use std::sync::Mutex;

    fn error_message(err: &RuntimeError) -> String {
        err.message().to_string()
    }

    fn cosh_builtin(value: Value) -> BuiltinResult<Value> {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        block_on(super::cosh_builtin(value))
    }

    struct MalformedCoshProvider {
        device_id: u32,
        next_buffer: AtomicU64,
        malformed: AtomicU8,
        allocations: AtomicUsize,
        frees: AtomicUsize,
        buffers: Mutex<HashMap<u64, HostTensorOwned>>,
    }

    impl MalformedCoshProvider {
        fn new() -> Self {
            Self {
                device_id: runmat_accelerate_api::next_device_id(),
                next_buffer: AtomicU64::new(8_710_000_000_000_000_000),
                malformed: AtomicU8::new(0),
                allocations: AtomicUsize::new(0),
                frees: AtomicUsize::new(0),
                buffers: Mutex::new(HashMap::new()),
            }
        }

        fn allocate(
            &self,
            data: Vec<f64>,
            shape: Vec<usize>,
            device_id: u32,
            precision: ProviderPrecision,
            storage: GpuTensorStorage,
        ) -> GpuTensorHandle {
            let buffer_id = self.next_buffer.fetch_add(1, Ordering::Relaxed);
            self.buffers.lock().unwrap().insert(
                buffer_id,
                HostTensorOwned {
                    data,
                    shape: shape.clone(),
                    storage,
                },
            );
            self.allocations.fetch_add(1, Ordering::Relaxed);
            GpuTensorHandle {
                shape,
                device_id,
                buffer_id,
                descriptor: runmat_accelerate_api::GpuTensorDescriptor::numeric(
                    match precision {
                        ProviderPrecision::F32 => runmat_accelerate_api::NumericElementType::F32,
                        ProviderPrecision::F64 => runmat_accelerate_api::NumericElementType::F64,
                    },
                    storage,
                ),
            }
        }
    }

    impl AccelProvider for MalformedCoshProvider {
        fn upload(&self, host: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
            Ok(self.allocate(
                host.data.to_vec(),
                host.shape.to_vec(),
                self.device_id,
                ProviderPrecision::F64,
                GpuTensorStorage::Real,
            ))
        }

        fn download<'a>(&'a self, handle: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
            Box::pin(async move {
                self.buffers
                    .lock()
                    .unwrap()
                    .get(&handle.buffer_id)
                    .cloned()
                    .ok_or_else(|| anyhow::anyhow!("unknown test buffer"))
            })
        }

        fn free(&self, handle: &GpuTensorHandle) -> anyhow::Result<()> {
            if self
                .buffers
                .lock()
                .unwrap()
                .remove(&handle.buffer_id)
                .is_some()
            {
                self.frees.fetch_add(1, Ordering::Relaxed);
            }
            Ok(())
        }

        fn device_info(&self) -> String {
            "malformed-cosh-test-provider".to_string()
        }

        fn device_id(&self) -> u32 {
            self.device_id
        }

        fn unary_cosh<'a>(
            &'a self,
            input: &'a GpuTensorHandle,
        ) -> AccelProviderFuture<'a, GpuTensorHandle> {
            Box::pin(async move {
                let malformed = self.malformed.load(Ordering::Relaxed);
                let device_id = if malformed == 2 {
                    self.device_id.wrapping_add(10_000)
                } else {
                    self.device_id
                };
                let precision = if malformed == 0 {
                    ProviderPrecision::F32
                } else {
                    ProviderPrecision::F64
                };
                let storage = if malformed == 1 {
                    GpuTensorStorage::ComplexInterleaved
                } else {
                    GpuTensorStorage::Real
                };
                Ok(self.allocate(
                    vec![99.0; input.shape.iter().product()],
                    input.shape.clone(),
                    device_id,
                    precision,
                    storage,
                ))
            })
        }
    }

    #[test]
    fn cosh_descriptor_signatures_cover_core_form() {
        let labels: Vec<&str> = COSH_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"Y = cosh(X)"));
        assert_eq!(COSH_INTEGER_CAPABILITIES[0].inputs[0].classes.len(), 8);
    }

    #[test]
    fn cosh_integer_gate_boundary_and_single_precision() {
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        assert!(block_on(super::cosh_builtin(Value::Int(IntValue::I8(0)))).is_err());
        drop(_strict);
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        for value in [
            IntValue::I8(0),
            IntValue::I16(0),
            IntValue::I32(0),
            IntValue::I64(0),
            IntValue::U8(0),
            IntValue::U16(0),
            IntValue::U32(0),
            IntValue::U64(0),
        ] {
            assert!(block_on(super::cosh_builtin(Value::Int(value))).is_ok());
        }
        assert!(block_on(super::cosh_builtin(Value::Int(IntValue::U64(
            (1_u64 << 53) + 1
        ))))
        .is_err());
        assert!(block_on(super::cosh_builtin(Value::Int(IntValue::U64(1_u64 << 54)))).is_ok());
        let Value::Tensor(real) = block_on(super::cosh_builtin(Value::Tensor(
            Tensor::from_f32(vec![0.0, 1.0], vec![2, 1]).unwrap(),
        )))
        .unwrap() else {
            panic!("expected single tensor")
        };
        assert_eq!(real.numeric_dtype(), NumericDType::F32);
        let Value::ComplexTensor(complex) = block_on(super::cosh_builtin(Value::ComplexTensor(
            ComplexTensor::from_f32(vec![(1.0, 2.0)], vec![1, 1]).unwrap(),
        )))
        .unwrap() else {
            panic!("expected complex tensor")
        };
        assert_eq!(complex.numeric_dtype(), NumericDType::F32);
    }

    #[test]
    fn cosh_rejects_excess_outputs() {
        let _outputs = crate::output_count::push_output_count(Some(2));
        let error =
            block_on(super::cosh_builtin(Value::Num(0.0))).expect_err("excess outputs must reject");
        assert_eq!(error.identifier(), Some("RunMat:cosh:TooManyOutputs"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cosh_scalar() {
        let value = Value::Num(2.0);
        let result = cosh_builtin(value).expect("cosh");
        match result {
            Value::Num(v) => assert!((v - 2.0f64.cosh()).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cosh_tensor_elements() {
        let tensor = Tensor::new(vec![-1.0, 0.0, 1.0], vec![3, 1]).unwrap();
        let result = cosh_builtin(Value::Tensor(tensor)).expect("cosh");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![3, 1]);
                let expected = [(-1.0f64).cosh(), 1.0, 1.0f64.cosh()];
                for (got, exp) in t.materialize_f64().iter().zip(expected.iter()) {
                    assert!((got - exp).abs() < 1e-12);
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cosh_reads_typed_integer_tensor_storage_exactly() {
        let tensor = Tensor::new_integer(
            runmat_value::IntegerStorage::I16(vec![-1, 0, 1]),
            vec![3, 1],
        )
        .expect("integer tensor");

        match cosh_builtin(Value::Tensor(tensor)).expect("cosh") {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![3, 1]);
                let expected = [(-1.0f64).cosh(), 1.0, 1.0f64.cosh()];
                for (actual, expected) in out.materialize_f64().iter().zip(expected.iter()) {
                    assert!((actual - expected).abs() < 1e-12);
                }
                assert!(out.integer_storage().is_none());
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cosh_int_value_promotes() {
        let value = Value::Int(IntValue::I32(1));
        let result = cosh_builtin(value).expect("cosh");
        match result {
            Value::Num(v) => assert!((v - 1.0f64.cosh()).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cosh_complex_scalar() {
        let result = cosh_builtin(Value::Complex(1.0, 2.0)).expect("cosh");
        match result {
            Value::Complex(re, im) => {
                assert!((re - cosh_complex_re(1.0, 2.0)).abs() < 1e-12);
                assert!((im - cosh_complex_im(1.0, 2.0)).abs() < 1e-12);
            }
            other => panic!("expected complex result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cosh_char_array_roundtrip() {
        let chars = CharArray::new("AZ".chars().collect(), 1, 2).unwrap();
        let result = cosh_builtin(Value::CharArray(chars)).expect("cosh");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 2]);
                for (idx, ch) in ['A', 'Z'].into_iter().enumerate() {
                    let expected = (ch as u32 as f64).cosh();
                    assert!((t.materialize_f64()[idx] - expected).abs() < 1e-12);
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cosh_logical_array_promotes() {
        let logical = LogicalArray::new(vec![1, 0, 1], vec![1, 3]).unwrap();
        let result = cosh_builtin(Value::LogicalArray(logical)).expect("cosh");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 3]);
                let expected = [1.0f64.cosh(), 0.0f64.cosh(), 1.0f64.cosh()];
                for (got, exp) in t.materialize_f64().iter().zip(expected.iter()) {
                    assert!((got - exp).abs() < 1e-12);
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cosh_string_errors() {
        let err = cosh_builtin(Value::String("runmat".to_string())).expect_err("expected error");
        let message = error_message(&err);
        assert!(message.contains("invalid input"));
        assert_eq!(err.identifier(), COSH_ERROR_INVALID_INPUT.identifier);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cosh_gpu_provider_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![0.0, 0.5, 1.0, 1.5], vec![4, 1]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.materialize_f64(),
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = cosh_builtin(Value::GpuTensor(handle)).expect("cosh");
            let gathered = test_support::gather(result).expect("gather");
            let expected: Vec<f64> = tensor.materialize_f64().iter().map(|&v| v.cosh()).collect();
            assert_eq!(gathered.shape, vec![4, 1]);
            assert_eq!(gathered.materialize_f64(), expected);
        });
    }

    #[test]
    fn cosh_complex_gpu_fallback_restores_through_the_owner() {
        test_support::with_test_provider(|provider| {
            let input = ComplexTensor::new(vec![(0.5, 0.25), (1.0, -0.75)], vec![2, 1])
                .expect("complex tensor");
            let handle = gpu_helpers::upload_complex_tensor(provider, &input).expect("upload");
            let resident = cosh_builtin(Value::GpuTensor(handle)).expect("resident cosh");
            let gathered = block_on(gpu_helpers::gather_value_async(&resident)).expect("gather");
            let Value::ComplexTensor(output) = gathered else {
                panic!("expected complex tensor")
            };
            for (actual, &(real, imag)) in output
                .materialize_f64()
                .iter()
                .zip(input.materialize_f64().iter())
            {
                assert!((actual.0 - cosh_complex_re(real, imag)).abs() < 1e-12);
                assert!((actual.1 - cosh_complex_im(real, imag)).abs() < 1e-12);
            }
        });
    }

    #[test]
    fn cosh_rejects_and_frees_malformed_native_outputs_before_owner_restoration() {
        let _guard = test_support::accel_test_lock();
        let provider = Box::leak(Box::new(MalformedCoshProvider::new()));
        unsafe {
            runmat_accelerate_api::register_provider(provider);
        }

        for malformed in 0..3_u8 {
            provider.malformed.store(malformed, Ordering::Relaxed);
            let input = provider
                .upload(&HostTensorView {
                    data: &[0.0, 1.0],
                    shape: &[2, 1],
                })
                .expect("input upload");
            let error = block_on(super::cosh_gpu(input.clone()))
                .expect_err("malformed native output must reject");
            assert!(error.message().contains("malformed unary output"));

            let completed = usize::from(malformed) + 1;
            assert_eq!(provider.allocations.load(Ordering::Relaxed), completed * 2);
            assert_eq!(provider.frees.load(Ordering::Relaxed), completed * 2 - 1);
            provider.free(&input).expect("free input");
            assert_eq!(provider.frees.load(Ordering::Relaxed), completed * 2);
        }
        assert_eq!(
            provider.allocations.load(Ordering::Relaxed),
            provider.frees.load(Ordering::Relaxed)
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn cosh_wgpu_matches_cpu_elementwise() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let t = Tensor::new(vec![0.0, 0.25, 0.5, 0.75], vec![4, 1]).unwrap();
        let cpu = cosh_real(Value::Tensor(t.clone())).unwrap();
        let view = runmat_accelerate_api::HostTensorView {
            data: &t.materialize_f64(),
            shape: &t.shape,
        };
        let h = runmat_accelerate_api::provider()
            .unwrap()
            .upload(&view)
            .unwrap();
        let gpu = block_on(cosh_gpu(h)).unwrap();
        let gathered = test_support::gather(gpu).expect("gather");
        match (cpu, gathered) {
            (Value::Tensor(ct), gt) => {
                assert_eq!(gt.shape, ct.shape);
                let tol = match runmat_accelerate_api::provider().unwrap().precision() {
                    runmat_accelerate_api::ProviderPrecision::F64 => 1e-12,
                    runmat_accelerate_api::ProviderPrecision::F32 => 1e-5,
                };
                for (a, b) in gt.materialize_f64().iter().zip(ct.materialize_f64().iter()) {
                    assert!((a - b).abs() < tol, "|{} - {}| >= {}", a, b, tol);
                }
            }
            _ => panic!("unexpected shapes"),
        }
    }
}
