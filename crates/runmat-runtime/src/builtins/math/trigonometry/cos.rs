//! MATLAB-compatible `cos` builtin with GPU-aware semantics for RunMat.

use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::{
    BuiltinErrorDescriptor, COS_CHARACTER_INPUT_EXTENSION, COS_ERROR_ARG_COUNT,
    COS_ERROR_GPU_UNAVAILABLE, COS_ERROR_INTERNAL, COS_ERROR_INVALID_INPUT,
    COS_ERROR_INVALID_OPTION, COS_ERROR_LIKE_PROTOTYPE, COS_INTEGER_INPUT_EXTENSION,
    COS_LIKE_OUTPUT_EXTENSION, COS_LOGICAL_INPUT_EXTENSION,
};
#[cfg(test)]
use runmat_builtins::{COS_DESCRIPTOR, COS_INTEGER_CAPABILITIES};
use runmat_macros::runtime_builtin;
#[cfg(test)]
use runmat_value::NumericDType;
use runmat_value::{CharArray, ComplexStorage, ComplexTensor, Tensor, Value};

use crate::builtins::common::random_args::{complex_tensor_into_value, keyword_of};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, FusionKernelTemplate,
    GpuOpKind, ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, map_control_flow_with_builtin, tensor};
use crate::builtins::math::symbolic::symbolic_function;
use crate::context::RuntimeContext;
use crate::execution::{RootedValue, RootedValueLease};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};
use runmat_value::SymbolicFunction;

const BUILTIN_NAME: &str = "cos";

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::trigonometry::cos")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "cos",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Unary { name: "unary_cos" }],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes:
        "Providers may execute cosine directly on device; runtimes gather to host when unary_cos is unavailable.",
};

fn cos_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    let mut builder = build_runtime_error(error.message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn cos_error_with_detail(
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

const FUSION_SEMANTICS: crate::builtins::common::spec::FusionSemanticDeclaration =
    crate::builtins::common::spec::FusionSemanticDeclaration::total_real_unary_math(
        runmat_builtins::BuiltinBindingIdentity {
            builtin: runmat_builtins::COS_CATALOG_ENTRY.identity,
            variant: "default",
        },
        runmat_types::ScalarUnaryMath::Cos,
    );

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::trigonometry::cos",
    semantics = FUSION_SEMANTICS,
    host_resources = crate::builtins::math::host_resources::real_map
)]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "cos",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: crate::builtins::common::spec::FusionExprBuilder::new(|ctx, writer| {
            crate::builtins::common::spec::write_unary_math_expression(
                runmat_types::ScalarUnaryMath::Cos, ctx, writer,
            )
        }),
    }),
    reduction: None,
    emits_nan: false,
    notes: "Fusion planner emits WGSL `cos` calls; providers can override via fused elementwise kernels.",
};

#[runtime_builtin(
    name = "cos",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::trigonometry::cos"
)]
async fn cos_builtin(
    runtime: Option<RuntimeContext>,
    value: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<RootedValue> {
    let _inputs = RootedValueLease::for_values(std::iter::once(&value).chain(rest.iter()))
        .map_err(|cause| cause.into_runtime_error())?;
    crate::output_count::validate_fixed_outputs(
        &runmat_builtins::COS_DESCRIPTOR,
        &runmat_builtins::COS_ERROR_TOO_MANY_OUTPUTS,
        "cos",
    )?;
    let template = parse_output_template(rest.view())?;
    let template = admit_output_template(template)?;
    ensure_cos_extensions(&value, rest.view()).await?;
    if let Some(symbolic) = symbolic_function(&value, SymbolicFunction::Cos) {
        let symbolic = RootedValue::new(symbolic).map_err(|cause| cause.into_runtime_error())?;
        let output = apply_output_template(symbolic.value().clone(), &template, &[]).await?;
        return RootedValue::new(output).map_err(|cause| cause.into_runtime_error());
    }
    let original = match &value {
        Value::GpuTensor(handle) => Some(handle.clone()),
        _ => None,
    };
    let signature = usize::from(!rest.is_empty());
    super::inverse_helpers::preparation::unary_form_real(
        runtime.as_ref(),
        runmat_builtins::COS_CATALOG_ENTRY.identity,
        value,
        signature,
        1 + rest.len(),
        |value: f32| Ok(value.cos()),
        |value: f64| Ok(value.cos()),
        |value| async move {
            super::inverse_helpers::like::evaluate(
                value,
                original.as_ref(),
                &template,
                cos_admitted,
                |result, protected| {
                    let template = &template;
                    async move {
                        let output = apply_output_template(result, template, &protected).await?;
                        RootedValue::new(output).map_err(|cause| cause.into_runtime_error())
                    }
                },
            )
            .await
        },
    )
    .await
}

async fn cos_admitted(value: Value, protected: Vec<GpuTensorHandle>) -> BuiltinResult<RootedValue> {
    let base = match value {
        Value::GpuTensor(handle) => cos_gpu(handle, &protected).await?,
        Value::Complex(re, im) => Value::Complex(cos_complex_re(re, im), cos_complex_im(re, im)),
        Value::ComplexTensor(ct) => cos_complex_tensor(ct)?,
        Value::CharArray(ca) => cos_char_array(ca)?,
        Value::String(_) | Value::StringArray(_) => {
            return Err(cos_error_with_detail(
                &COS_ERROR_INVALID_INPUT,
                "expected numeric input, got string",
            ))
        }
        other => cos_real(other)?,
    };
    RootedValue::new(base).map_err(|cause| cause.into_runtime_error())
}

async fn ensure_cos_extensions(
    value: &Value,
    rest: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<()> {
    crate::builtins::common::validation::ensure_runmat_real_integer_f64_boundary(
        value,
        &COS_INTEGER_INPUT_EXTENSION,
        runmat_builtins::COS_CATALOG_ENTRY.identity,
        &runmat_builtins::COS_ERROR_INVALID_INPUT,
        "X",
    )
    .await?;
    if matches!(value, Value::Bool(_) | Value::LogicalArray(_))
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_is_logical(handle))
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &COS_LOGICAL_INPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    if matches!(value, Value::CharArray(_)) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &COS_CHARACTER_INPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    if !rest.is_empty() {
        crate::compatibility::ensure_builtin_extension_enabled(
            &COS_LIKE_OUTPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    Ok(())
}

async fn cos_gpu(handle: GpuTensorHandle, protected: &[GpuTensorHandle]) -> BuiltinResult<Value> {
    let provider = gpu_helpers::exact_provider_for_handle(&handle);
    let requires_exact_host_path = runmat_accelerate_api::handle_integer_type(&handle).is_some()
        || runmat_accelerate_api::handle_is_logical(&handle);
    if !requires_exact_host_path {
        if let Some(provider) = provider {
            match provider.unary_cos(&handle).await {
                Ok(output)
                    if !protected.iter().any(|input| {
                        gpu_helpers::same_gpu_handle(input, &output)
                            && !gpu_helpers::same_gpu_handle(input, &handle)
                    }) && gpu_helpers::unary_gpu_output_matches(
                        &output,
                        &handle,
                        provider,
                        gpu_helpers::UnaryGpuOutputContract {
                            storage: runmat_accelerate_api::handle_storage(&handle),
                            precision: runmat_accelerate_api::handle_precision(&handle),
                            integer: None,
                            logical: false,
                            alias: gpu_helpers::GpuOutputAliasPolicy::AllowInput,
                        },
                    ) =>
                {
                    return gpu_helpers::resident_gpu_value(output);
                }
                Ok(output) => {
                    let mut error = cos_error_with_detail(
                        &COS_ERROR_INTERNAL,
                        "provider returned an invalid unary cosine result",
                    );
                    error.gpu_gather_retry = crate::GpuGatherRetry::Never;
                    return Err(gpu_helpers::reject_gpu_output(output, error));
                }
                Err(error) if gpu_helpers::provider_hook_allows_host_fallback(&error) => {}
                Err(error) => {
                    let mut failure = cos_error_with_detail(
                        &COS_ERROR_INTERNAL,
                        format!("provider unary cosine failed: {error}"),
                    );
                    failure.gpu_gather_retry = crate::GpuGatherRetry::Never;
                    failure =
                        crate::runtime_error::with_provider_execution_source(failure, error.into());
                    return Err(failure);
                }
            }
        }
    }
    let source = handle.clone();
    let gathered = gpu_helpers::gather_value_async(&Value::GpuTensor(handle))
        .await
        .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
    let host = match gathered {
        Value::Complex(re, im) => Ok(Value::Complex(
            cos_complex_re(re, im),
            cos_complex_im(re, im),
        )),
        Value::ComplexTensor(ct) => cos_complex_tensor(ct),
        Value::Tensor(tensor) => cos_tensor(tensor).map(tensor::tensor_into_value),
        Value::Num(n) => Ok(Value::Num(n.cos())),
        other => cos_real(other),
    }?;
    let host = crate::builtins::math::resident_real_unary::preserve_elementwise_scalar_shape(
        host,
        source.shape(),
        BUILTIN_NAME,
    )?;
    let protected: Vec<_> = protected.iter().collect();
    gpu_helpers::restore_class_preserving_value_protecting(&source, host, BUILTIN_NAME, &protected)
}

fn cos_real(value: Value) -> BuiltinResult<Value> {
    let tensor = tensor::value_into_tensor_for("cos", value)
        .map_err(|e| cos_error_with_detail(&COS_ERROR_INVALID_INPUT, e))?;
    cos_tensor(tensor).map(tensor::tensor_into_value)
}

fn cos_tensor(tensor: Tensor) -> BuiltinResult<Tensor> {
    if tensor.numeric_dtype() == runmat_value::NumericDType::F32 {
        let data = tensor
            .as_f32_slice()
            .expect("single tensor storage")
            .iter()
            .map(|&v| v.cos())
            .collect();
        return Tensor::from_f32(data, tensor.shape().to_vec())
            .map_err(|e| cos_error_with_detail(&COS_ERROR_INTERNAL, e));
    }
    let data = tensor::tensor_values_f64_cow(&tensor)
        .iter()
        .map(|&v| v.cos())
        .collect::<Vec<_>>();
    Tensor::new(data, tensor.shape().to_vec())
        .map_err(|e| cos_error_with_detail(&COS_ERROR_INTERNAL, e))
}

fn cos_complex_tensor(ct: ComplexTensor) -> BuiltinResult<Value> {
    let (storage, shape) = ct.into_complex_storage_and_shape();
    let tensor = match storage {
        ComplexStorage::F32(values) => ComplexTensor::from_f32(
            values
                .into_iter()
                .map(|(re, im)| {
                    (
                        cos_complex_re(f64::from(re), f64::from(im)) as f32,
                        cos_complex_im(f64::from(re), f64::from(im)) as f32,
                    )
                })
                .collect(),
            shape,
        ),
        ComplexStorage::F64(values) => ComplexTensor::new(
            values
                .into_iter()
                .map(|(re, im)| (cos_complex_re(re, im), cos_complex_im(re, im)))
                .collect(),
            shape,
        ),
        ComplexStorage::Integer(_) => Err("typed complex integer input is unsupported".into()),
    }
    .map_err(|e| cos_error_with_detail(&COS_ERROR_INTERNAL, e))?;
    Ok(complex_tensor_into_value(tensor))
}

fn cos_char_array(ca: CharArray) -> BuiltinResult<Value> {
    let data = ca
        .data()
        .iter()
        .map(|&ch| (u32::from(ch.raw()) as f64).cos())
        .collect::<Vec<_>>();
    let tensor = Tensor::new(data, vec![ca.rows, ca.cols])
        .map_err(|e| cos_error_with_detail(&COS_ERROR_INTERNAL, e))?;
    Ok(tensor::tensor_into_value(tensor))
}

#[inline]
fn cos_complex_re(re: f64, im: f64) -> f64 {
    re.cos() * im.cosh()
}

#[inline]
fn cos_complex_im(re: f64, im: f64) -> f64 {
    -re.sin() * im.sinh()
}

mod output_template;
#[cfg(test)]
use output_template::convert_to_host_complex;
use output_template::{admit_output_template, apply_output_template, parse_output_template};

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_accelerate_api::{
        AccelDownloadFuture, AccelProvider, AccelProviderFuture, GpuTensorStorage, HostTensorOwned,
        HostTensorView, ProviderPrecision,
    };
    use runmat_value::{IntValue, StringArray, Tensor};
    use std::sync::atomic::{AtomicU8, Ordering};
    use std::sync::Mutex;

    use crate::builtins::common::{gpu_helpers, test_support};

    fn cos_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        block_on(super::cos_builtin(
            None,
            value,
            crate::builtins::common::test_support::builtin_arguments(rest),
        ))
        .map(|owned| owned.value().clone())
    }

    struct MalformedCosProvider {
        owner: runmat_accelerate_api::test_support::TestAllocationOwner,
        foreign: runmat_accelerate_api::test_support::TestAllocationOwner,
        malformed: AtomicU8,
        probes: Mutex<Vec<runmat_accelerate_api::test_support::RetirementProbe>>,
    }

    impl MalformedCosProvider {
        fn new() -> Self {
            Self {
                owner: runmat_accelerate_api::test_support::TestAllocationOwner::new(),
                foreign: runmat_accelerate_api::test_support::TestAllocationOwner::new(),
                malformed: AtomicU8::new(0),
                probes: Mutex::new(Vec::new()),
            }
        }

        fn allocate(
            &self,
            mut data: Vec<f64>,
            shape: Vec<usize>,
            foreign: bool,
            precision: ProviderPrecision,
            storage: GpuTensorStorage,
        ) -> GpuTensorHandle {
            use runmat_accelerate_api::{HostNumericDataOwned, HostNumericTensorOwned};
            if storage == GpuTensorStorage::ComplexInterleaved {
                data = data.into_iter().flat_map(|real| [real, 0.0]).collect();
            }
            let data = match precision {
                ProviderPrecision::F32 => {
                    HostNumericDataOwned::F32(data.into_iter().map(|v| v as f32).collect())
                }
                ProviderPrecision::F64 => HostNumericDataOwned::F64(data),
            };
            let owner = if foreign { &self.foreign } else { &self.owner };
            let allocation = owner
                .allocate_numeric(HostNumericTensorOwned {
                    data,
                    shape,
                    storage,
                })
                .unwrap();
            self.probes.lock().unwrap().push(allocation.probe);
            allocation.handle
        }

        fn allocations(&self) -> usize {
            self.probes.lock().unwrap().len()
        }
        fn retirements(&self) -> usize {
            self.probes
                .lock()
                .unwrap()
                .iter()
                .map(|probe| probe.retirements())
                .sum()
        }
    }

    impl AccelProvider for MalformedCosProvider {
        fn upload(&self, host: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
            Ok(self.allocate(
                host.data.to_vec(),
                host.shape.to_vec(),
                false,
                ProviderPrecision::F64,
                GpuTensorStorage::Real,
            ))
        }

        fn download<'a>(&'a self, handle: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
            Box::pin(async move {
                self.probes
                    .lock()
                    .unwrap()
                    .iter()
                    .find_map(|probe| {
                        probe
                            .with_data(handle, |data| {
                                use runmat_accelerate_api::HostNumericDataView;
                                let data = match data {
                                    HostNumericDataView::F64(values) => values.to_vec(),
                                    HostNumericDataView::F32(values) => {
                                        values.iter().map(|&v| f64::from(v)).collect()
                                    }
                                    _ => panic!("floating fixture payload"),
                                };
                                HostTensorOwned {
                                    data,
                                    shape: handle.shape().to_vec(),
                                    storage: handle.physical().storage(),
                                }
                            })
                            .ok()
                    })
                    .ok_or_else(|| anyhow::anyhow!("unknown test allocation"))
            })
        }

        fn device_info(&self) -> String {
            "malformed-cos-test-provider".into()
        }
        fn device_id(&self) -> u32 {
            self.owner.device_id()
        }

        fn unary_cos<'a>(
            &'a self,
            input: &'a GpuTensorHandle,
        ) -> AccelProviderFuture<'a, GpuTensorHandle> {
            Box::pin(async move {
                let malformed = self.malformed.load(Ordering::Relaxed);
                Ok(self.allocate(
                    vec![99.0; input.shape().iter().product()],
                    input.shape().to_vec(),
                    malformed == 2,
                    if malformed == 0 {
                        ProviderPrecision::F32
                    } else {
                        ProviderPrecision::F64
                    },
                    if malformed == 1 {
                        GpuTensorStorage::ComplexInterleaved
                    } else {
                        GpuTensorStorage::Real
                    },
                ))
            })
        }
    }

    #[test]
    fn cos_descriptor_signatures_cover_like_overload() {
        let labels: Vec<&str> = COS_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"Y = cos(X)"));
        assert!(labels.contains(&"Y = cos(X, \"like\", P)"));
        assert_eq!(COS_INTEGER_CAPABILITIES[0].inputs[0].classes.len(), 8);
    }

    #[test]
    fn cos_integer_gate_all_classes_boundary_and_single_precision() {
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let err = block_on(super::cos_builtin(
            None,
            Value::Int(IntValue::I8(1)),
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .map(|owned| owned.value().clone())
        .expect_err("strict mode rejects integer extension");
        assert_eq!(
            err.identifier(),
            COS_INTEGER_INPUT_EXTENSION.error_identifier
        );
        drop(_strict);

        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        for value in [
            IntValue::I8(1),
            IntValue::I16(1),
            IntValue::I32(1),
            IntValue::I64(1),
            IntValue::U8(1),
            IntValue::U16(1),
            IntValue::U32(1),
            IntValue::U64(1),
        ] {
            assert!(block_on(super::cos_builtin(
                None,
                Value::Int(value),
                crate::builtins::common::test_support::builtin_arguments(Vec::new())
            ))
            .map(|owned| owned.value().clone())
            .is_ok());
        }
        assert!(block_on(super::cos_builtin(
            None,
            Value::Int(IntValue::U64((1_u64 << 53) + 1)),
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .map(|owned| owned.value().clone())
        .is_err());
        assert!(block_on(super::cos_builtin(
            None,
            Value::Int(IntValue::U64(1_u64 << 54)),
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .map(|owned| owned.value().clone())
        .is_ok());

        let single = Tensor::from_f32(vec![0.0, 1.0], vec![2, 1]).unwrap();
        let Value::Tensor(single_out) = block_on(super::cos_builtin(
            None,
            Value::Tensor(single),
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .map(|owned| owned.value().clone())
        .unwrap() else {
            panic!("expected single tensor")
        };
        assert_eq!(single_out.numeric_dtype(), NumericDType::F32);
        let complex = ComplexTensor::from_f32(vec![(1.0, 2.0)], vec![1, 1]).unwrap();
        let Value::ComplexTensor(complex_out) = block_on(super::cos_builtin(
            None,
            Value::ComplexTensor(complex),
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .map(|owned| owned.value().clone())
        .unwrap() else {
            panic!("expected single complex tensor")
        };
        assert_eq!(complex_out.numeric_dtype(), NumericDType::F32);
    }

    fn error_message(err: RuntimeError) -> String {
        err.message().to_string()
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cos_scalar_zero() {
        let result = cos_builtin(Value::Num(0.0), Vec::new()).expect("cos");
        match result {
            Value::Num(v) => assert!((v - 1.0).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn cos_preserves_rank_zero_scalarization_and_rank_one_shape() {
        let scalar = cos_builtin(
            Value::Tensor(Tensor::new(vec![0.0], Vec::new()).unwrap()),
            Vec::new(),
        )
        .expect("rank-zero cos");
        assert_eq!(scalar, Value::Num(1.0));

        let vector = cos_builtin(
            Value::Tensor(Tensor::new(vec![0.0, 0.0], vec![2]).unwrap()),
            Vec::new(),
        )
        .expect("rank-one cos");
        let Value::Tensor(vector) = vector else {
            panic!("expected rank-one tensor")
        };
        assert_eq!(vector.shape(), &[2]);
        assert_eq!(vector.materialize_f64(), vec![1.0, 1.0]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cos_tensor_elements() {
        let tensor = Tensor::new(vec![0.0, std::f64::consts::PI], vec![2, 1]).unwrap();
        let result = cos_builtin(Value::Tensor(tensor), Vec::new()).expect("cos");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape(), &[2, 1]);
                assert!((t.materialize_f64()[0] - 1.0).abs() < 1e-12);
                assert!((t.materialize_f64()[1] + 1.0).abs() < 1e-12);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cos_reads_typed_integer_tensor_storage_exactly() {
        let tensor =
            Tensor::new_integer(runmat_value::IntegerStorage::I16(vec![0, 1, 2]), vec![3, 1])
                .expect("integer tensor");

        let result = cos_builtin(Value::Tensor(tensor), Vec::new()).expect("cos");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape(), &[3, 1]);
                let expected = [1.0, 1.0f64.cos(), 2.0f64.cos()];
                for (actual, expected) in out.materialize_f64().iter().zip(expected.iter()) {
                    assert!((actual - expected).abs() < 1e-12);
                }
                assert!(out.integer_storage().is_none());
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn cos_host_complex_conversion_reads_typed_integer_storage_exactly() {
        let tensor = Tensor::new_integer(
            runmat_value::IntegerStorage::I64(vec![-3, 0, 5]),
            vec![3, 1],
        )
        .expect("integer tensor");

        let result =
            block_on(convert_to_host_complex(Value::Tensor(tensor))).expect("complex conversion");
        let Value::ComplexTensor(out) = result else {
            panic!("expected complex tensor result");
        };
        assert_eq!(out.shape(), &[3, 1]);
        assert_eq!(
            out.materialize_f64(),
            vec![(-3.0, 0.0), (0.0, 0.0), (5.0, 0.0)]
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cos_int_value_promotes() {
        let value = Value::Int(IntValue::I32(1));
        let result = cos_builtin(value, Vec::new()).expect("cos");
        match result {
            Value::Num(v) => assert!((v - 1.0f64.cos()).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cos_complex_scalar() {
        let result = cos_builtin(Value::Complex(1.0, 2.0), Vec::new()).expect("cos");
        match result {
            Value::Complex(re, im) => {
                assert!((re - (1.0f64.cos() * 2.0f64.cosh())).abs() < 1e-12);
                assert!((im + (1.0f64.sin() * 2.0f64.sinh())).abs() < 1e-12);
            }
            other => panic!("expected complex result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cos_char_array_roundtrip() {
        let chars = CharArray::from_unicode_scalars("abc".chars().collect(), 1, 3).unwrap();
        let result = cos_builtin(Value::CharArray(chars), Vec::new()).expect("cos");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape(), &[1, 3]);
                for (idx, ch) in ['a', 'b', 'c'].into_iter().enumerate() {
                    let expected = (ch as u32 as f64).cos();
                    assert!((t.materialize_f64()[idx] - expected).abs() < 1e-12);
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cos_gpu_provider_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![0.0, 1.0, 2.0, 3.0], vec![4, 1]).unwrap();
            let view = HostTensorView {
                data: &tensor.materialize_f64(),
                shape: tensor.shape(),
            };
            let handle = provider.upload(&view).expect("upload");
            let result = cos_builtin(Value::GpuTensor(handle), Vec::new()).expect("cos");
            let gathered = test_support::gather(result).expect("gather");
            let expected: Vec<f64> = tensor.materialize_f64().iter().map(|&v| v.cos()).collect();
            assert_eq!(gathered.shape(), &[4, 1]);
            assert_eq!(gathered.materialize_f64(), expected);
        });
    }

    #[test]
    fn cos_rejects_and_frees_malformed_native_outputs() {
        let _guard = test_support::accel_test_lock();
        let provider = Box::leak(Box::new(MalformedCosProvider::new()));
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
            let error = block_on(super::cos_gpu(input.clone(), &[]))
                .expect_err("malformed native output must be rejected");
            assert_eq!(error.identifier(), COS_ERROR_INTERNAL.identifier);
            let completed = usize::from(malformed) + 1;
            assert_eq!(provider.allocations(), completed * 2);
            assert_eq!(provider.retirements(), completed * 2 - 1);
            input.release().expect("release input");
            assert_eq!(provider.retirements(), completed * 2);
        }
        assert_eq!(provider.allocations(), provider.retirements());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cos_like_missing_prototype_errors() {
        let err =
            cos_builtin(Value::Num(1.0), vec![Value::from("like")]).expect_err("expected error");
        assert_eq!(err.identifier(), COS_ERROR_INVALID_OPTION.identifier);
        let message = error_message(err);
        assert!(message.contains("prototype"));
    }

    #[test]
    fn cos_validates_like_syntax_before_extension_gate() {
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let error = block_on(super::cos_builtin(
            None,
            Value::Int(IntValue::U8(1)),
            crate::builtins::common::test_support::builtin_arguments(vec![Value::from("like")]),
        ))
        .map(|owned| owned.value().clone())
        .expect_err("malformed like syntax");
        assert_eq!(error.identifier(), COS_ERROR_INVALID_OPTION.identifier);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cos_like_complex_prototype_returns_complex() {
        let result = cos_builtin(
            Value::Num(1.0),
            vec![Value::from("like"), Value::Complex(0.0, 1.0)],
        )
        .expect("cos");
        match result {
            Value::Complex(re, im) => {
                assert!((re - 1.0_f64.cos()).abs() < 1e-12);
                assert!(im.abs() < 1e-12);
            }
            other => panic!("expected complex result, got {other:?}"),
        }
    }

    #[test]
    fn cos_gpu_complex_input_preserves_complex_output() {
        test_support::with_test_provider(|provider| {
            let input = ComplexTensor::new(vec![(0.5, 0.75), (2.0, -0.25)], vec![1, 2]).unwrap();
            let handle = gpu_helpers::upload_complex_tensor(provider, &input).expect("upload");
            let result = cos_builtin(Value::GpuTensor(handle), Vec::new()).expect("cos");
            let out = match result {
                Value::GpuTensor(handle) => {
                    assert_eq!(
                        runmat_accelerate_api::handle_storage(&handle),
                        runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
                    );
                    match block_on(gpu_helpers::gather_value_async(&Value::GpuTensor(handle)))
                        .expect("gather")
                    {
                        Value::ComplexTensor(out) => out,
                        other => panic!("expected gathered complex tensor, got {other:?}"),
                    }
                }
                Value::ComplexTensor(out) => out,
                other => panic!("expected complex output, got {other:?}"),
            };
            assert_eq!(out.shape(), &[1, 2]);
            for (idx, &(re, im)) in input.materialize_f64().iter().enumerate() {
                assert!((out.materialize_f64()[idx].0 - cos_complex_re(re, im)).abs() < 1e-12);
                assert!((out.materialize_f64()[idx].1 - cos_complex_im(re, im)).abs() < 1e-12);
            }
        });
    }

    #[test]
    fn cos_like_complex_gpu_prototype_uploads_complex_result() {
        test_support::with_test_provider(|provider| {
            let input = Tensor::new(vec![0.0, 1.0], vec![2, 1]).unwrap();
            let proto_tensor = ComplexTensor::new(vec![(0.0, 1.0)], vec![1, 1]).unwrap();
            let proto = gpu_helpers::upload_complex_tensor(provider, &proto_tensor)
                .expect("upload complex prototype");
            let result = cos_builtin(
                Value::Tensor(input.clone()),
                vec![Value::from("like"), Value::GpuTensor(proto)],
            )
            .expect("cos");
            let Value::GpuTensor(handle) = result else {
                panic!("expected complex gpu tensor, got {result:?}");
            };
            assert_eq!(
                runmat_accelerate_api::handle_storage(&handle),
                runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
            );
            let gathered =
                block_on(gpu_helpers::gather_value_async(&Value::GpuTensor(handle))).unwrap();
            let Value::ComplexTensor(out) = gathered else {
                panic!("expected gathered complex tensor, got {gathered:?}");
            };
            assert_eq!(out.shape(), &[2, 1]);
            for (idx, &re) in input.materialize_f64().iter().enumerate() {
                assert!((out.materialize_f64()[idx].0 - re.cos()).abs() < 1e-12);
                assert!(out.materialize_f64()[idx].1.abs() < 1e-12);
            }
        });
    }

    #[test]
    fn cos_like_complex_gpu_prototype_converts_resident_real_gpu_result() {
        test_support::with_test_provider(|provider| {
            let input = Tensor::new(vec![0.0, 1.0], vec![2, 1]).unwrap();
            let input_view = HostTensorView {
                data: &input.materialize_f64(),
                shape: input.shape(),
            };
            let input_handle = provider.upload(&input_view).expect("upload input");
            let proto_tensor = ComplexTensor::new(vec![(0.0, 1.0)], vec![1, 1]).unwrap();
            let proto = gpu_helpers::upload_complex_tensor(provider, &proto_tensor)
                .expect("upload complex prototype");
            let result = cos_builtin(
                Value::GpuTensor(input_handle),
                vec![Value::from("like"), Value::GpuTensor(proto)],
            )
            .expect("cos");
            let Value::GpuTensor(handle) = result else {
                panic!("expected complex gpu tensor, got {result:?}");
            };
            assert_eq!(
                runmat_accelerate_api::handle_storage(&handle),
                runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
            );
            let gathered =
                block_on(gpu_helpers::gather_value_async(&Value::GpuTensor(handle))).unwrap();
            let Value::ComplexTensor(out) = gathered else {
                panic!("expected gathered complex tensor, got {gathered:?}");
            };
            assert_eq!(out.shape(), &[2, 1]);
            for (idx, &re) in input.materialize_f64().iter().enumerate() {
                assert!((out.materialize_f64()[idx].0 - re.cos()).abs() < 1e-12);
                assert!(out.materialize_f64()[idx].1.abs() < 1e-12);
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cos_like_gpu_prototype() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![0.0, 1.0, 2.0, 3.0], vec![4, 1]).unwrap();
            let proto_view = HostTensorView {
                data: &[0.0],
                shape: &[1, 1],
            };
            let proto = provider.upload(&proto_view).expect("upload");
            let result = cos_builtin(
                Value::Tensor(tensor.clone()),
                vec![Value::from("like"), Value::GpuTensor(proto.clone())],
            )
            .expect("cos");
            match result {
                Value::GpuTensor(handle) => {
                    let gathered = test_support::gather(Value::GpuTensor(handle)).expect("gather");
                    let expected: Vec<f64> =
                        tensor.materialize_f64().iter().map(|&v| v.cos()).collect();
                    assert_eq!(gathered.shape(), &[4, 1]);
                    assert_eq!(gathered.materialize_f64(), expected);
                }
                other => panic!("expected GPU tensor, got {other:?}"),
            }
        });
    }

    #[test]
    fn cos_like_gpu_prototype_controls_owner_across_providers() {
        let _guard = test_support::accel_test_lock();
        let input_provider = Box::leak(Box::new(
            runmat_accelerate::simple_provider::InProcessProvider::new(),
        ));
        let prototype_provider = Box::leak(Box::new(
            runmat_accelerate::simple_provider::InProcessProvider::new(),
        ));
        unsafe {
            runmat_accelerate_api::register_provider(input_provider);
            runmat_accelerate_api::register_provider(prototype_provider);
        }
        let input = input_provider
            .upload(&HostTensorView {
                data: &[0.0, 1.0],
                shape: &[2, 1],
            })
            .expect("upload input");
        let prototype = prototype_provider
            .upload(&HostTensorView {
                data: &[0.0],
                shape: &[1, 1],
            })
            .expect("upload prototype");

        let result = cos_builtin(
            Value::GpuTensor(input),
            vec![Value::from("like"), Value::GpuTensor(prototype)],
        )
        .expect("mixed-provider like conversion");
        let Value::GpuTensor(output) = result else {
            panic!("expected GPU output")
        };
        assert_eq!(output.device_id(), prototype_provider.device_id());
        let output_owner =
            runmat_accelerate_api::provider_for_handle(&output).expect("registered output owner");
        assert!(runmat_accelerate_api::same_provider_instance(
            output_owner,
            prototype_provider as &dyn runmat_accelerate_api::AccelProvider
        ));
        let gathered = test_support::gather(Value::GpuTensor(output)).expect("gather output");
        assert_eq!(gathered.materialize_f64(), vec![1.0, 1.0_f64.cos()]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cos_like_host_with_gpu_input_gathers() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![0.0, 1.0], vec![2, 1]).unwrap();
            let view = HostTensorView {
                data: &tensor.materialize_f64(),
                shape: tensor.shape(),
            };
            let handle = provider.upload(&view).expect("upload");
            let result = cos_builtin(
                Value::GpuTensor(handle),
                vec![Value::from("like"), Value::Num(0.0)],
            )
            .expect("cos");
            match result {
                Value::Tensor(t) => {
                    let expected: Vec<f64> =
                        tensor.materialize_f64().iter().map(|&v| v.cos()).collect();
                    assert_eq!(t.shape(), &[2, 1]);
                    assert_eq!(t.materialize_f64(), expected);
                }
                Value::GpuTensor(_) => panic!("expected host result"),
                Value::Num(_) => panic!("expected vector output"),
                other => panic!("unexpected result {other:?}"),
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cos_like_rejects_extra_arguments() {
        let err = cos_builtin(
            Value::Num(0.0),
            vec![Value::from("like"), Value::Num(0.0), Value::Num(1.0)],
        )
        .expect_err("expected error");
        let message = error_message(err);
        assert!(message.contains("too many input arguments"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cos_like_keyword_case_insensitive() {
        let tensor = Tensor::new(vec![0.0, 1.0], vec![2, 1]).unwrap();
        let result = cos_builtin(
            Value::Tensor(tensor.clone()),
            vec![Value::from("LIKE"), Value::Num(0.0)],
        )
        .expect("cos");
        match result {
            Value::Tensor(out) => {
                let expected: Vec<f64> =
                    tensor.materialize_f64().iter().map(|&v| v.cos()).collect();
                assert_eq!(out.shape(), &[2, 1]);
                assert_eq!(out.materialize_f64(), expected);
            }
            other => panic!("unexpected result {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cos_like_char_array_keyword() {
        let keyword = CharArray::new_row("like");
        let result = cos_builtin(
            Value::Num(0.0),
            vec![Value::CharArray(keyword), Value::Num(0.0)],
        )
        .expect("cos");
        match result {
            Value::Num(v) => assert!((v - 1.0).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cos_like_string_array_keyword() {
        let keyword = StringArray::new(vec!["LIKE".to_string()], vec![1]).unwrap();
        let result = cos_builtin(
            Value::Num(0.0),
            vec![Value::StringArray(keyword), Value::Num(0.0)],
        )
        .expect("cos");
        match result {
            Value::Num(v) => assert!((v - 1.0).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cos_unrecognised_option_errors() {
        let err =
            cos_builtin(Value::Num(0.0), vec![Value::from("invalid")]).expect_err("expected error");
        let message = error_message(err);
        assert!(message.contains("unrecognised argument"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn cos_wgpu_matches_cpu_elementwise() {
        let _guard = test_support::accel_test_lock();
        let Some(provider) = test_support::wgpu_provider_if_available() else {
            return;
        };
        let t = Tensor::new(vec![0.0, 1.0, 2.0, 3.0], vec![4, 1]).unwrap();
        let cpu = cos_real(Value::Tensor(t.clone())).unwrap();
        let view = HostTensorView {
            data: &t.materialize_f64(),
            shape: t.shape(),
        };
        let h = provider.upload(&view).unwrap();
        let gpu = block_on(cos_gpu(h, &[])).unwrap();
        let gathered = test_support::gather(gpu).expect("gather");
        match (cpu, gathered) {
            (Value::Tensor(ct), gt) => {
                assert_eq!(gt.shape(), ct.shape());
                let tol = match provider.precision() {
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
