//! MATLAB-compatible `atan` builtin with GPU-aware semantics for RunMat.
//!
//! Provides element-wise inverse tangent for scalars, tensors, and complex data while mirroring
//! MATLAB behavior. GPU execution uses provider hooks when available and falls back to the host
//! path if kernels are missing or outputs must become host-resident.

use num_complex::Complex64;
use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::{
    BuiltinErrorDescriptor, ATAN_CHARACTER_INPUT_EXTENSION, ATAN_ERROR_ARG_COUNT,
    ATAN_ERROR_GPU_UNAVAILABLE, ATAN_ERROR_INTERNAL, ATAN_ERROR_INVALID_INPUT,
    ATAN_ERROR_INVALID_OPTION, ATAN_ERROR_LIKE_PROTOTYPE, ATAN_INTEGER_INPUT_EXTENSION,
    ATAN_LIKE_EXTENSION, ATAN_LOGICAL_INPUT_EXTENSION,
};
#[cfg(test)]
use runmat_builtins::{ATAN_DESCRIPTOR, ATAN_ERROR_TOO_MANY_OUTPUTS};
use runmat_macros::runtime_builtin;
use runmat_value::{CharArray, ComplexTensor, Tensor, Value};

use crate::builtins::common::random_args::{complex_tensor_into_value, keyword_of};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, FusionKernelTemplate,
    GpuOpKind, ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
use crate::context::RuntimeContext;
use crate::execution::{RootedValue, RootedValueLease};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "atan";

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::trigonometry::atan")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "atan",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Unary { name: "unary_atan" }],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Providers execute atan on-device via unary_atan; runtimes gather to host when the hook is unavailable.",
};

fn atan_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    let mut builder = build_runtime_error(error.message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn atan_error_with_detail(
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
            builtin: runmat_builtins::ATAN_CATALOG_ENTRY.identity,
            variant: "default",
        },
        runmat_types::ScalarUnaryMath::Atan,
    );

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::trigonometry::atan",
    semantics = FUSION_SEMANTICS,
    host_resources = crate::builtins::math::host_resources::real_map
)]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "atan",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: crate::builtins::common::spec::FusionExprBuilder::new(|ctx, writer| {
            crate::builtins::common::spec::write_unary_math_expression(
                runmat_types::ScalarUnaryMath::Atan, ctx, writer,
            )
        }),
    }),
    reduction: None,
    emits_nan: false,
    notes: "Fusion planner emits WGSL atan calls; providers may override with specialised fused kernels.",
};

#[runtime_builtin(
    name = "atan",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::trigonometry::atan"
)]
async fn atan_builtin(
    runtime: Option<RuntimeContext>,
    value: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<RootedValue> {
    let _inputs = RootedValueLease::for_values(std::iter::once(&value).chain(rest.iter()))
        .map_err(|cause| cause.into_runtime_error())?;
    super::inverse_helpers::reject_excess_outputs(BUILTIN_NAME)?;
    super::inverse_helpers::ensure_input_extensions(
        &value,
        BUILTIN_NAME,
        &ATAN_INTEGER_INPUT_EXTENSION,
        &ATAN_LOGICAL_INPUT_EXTENSION,
        &ATAN_CHARACTER_INPUT_EXTENSION,
    )?;
    if !rest.is_empty() {
        crate::compatibility::ensure_builtin_extension_enabled(&ATAN_LIKE_EXTENSION, BUILTIN_NAME)?;
    }
    let template = parse_output_template(rest.view())?;
    let template = admit_output_template(template)?;
    crate::builtins::common::validation::reject_typed_complex_integer_for(
        &value,
        runmat_builtins::ATAN_CATALOG_ENTRY.identity,
        &runmat_builtins::ATAN_ERROR_INVALID_INPUT,
    )?;
    let original = match &value {
        Value::GpuTensor(handle) => Some(handle.clone()),
        _ => None,
    };
    let signature = usize::from(!rest.is_empty());
    super::inverse_helpers::preparation::unary_form_real(
        runtime.as_ref(),
        runmat_builtins::ATAN_CATALOG_ENTRY.identity,
        value,
        signature,
        1 + rest.len(),
        |value: f32| Ok(value.atan()),
        |value: f64| Ok(value.atan()),
        |value| async move {
            super::inverse_helpers::like::evaluate(
                value,
                original.as_ref(),
                &template,
                atan_admitted,
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

async fn atan_admitted(
    value: Value,
    protected: Vec<GpuTensorHandle>,
) -> BuiltinResult<RootedValue> {
    let base = match value {
        Value::GpuTensor(handle) => atan_gpu(handle, &protected).await?,
        Value::Complex(re, im) => {
            let (out_re, out_im) = atan_complex_components(re, im);
            Value::Complex(out_re, out_im)
        }
        Value::ComplexTensor(ct) => atan_complex_tensor(ct)?,
        Value::CharArray(ca) => atan_char_array(ca)?,
        Value::String(_) | Value::StringArray(_) => {
            return Err(atan_error_with_detail(
                &ATAN_ERROR_INVALID_INPUT,
                "expected numeric input",
            ))
        }
        other => atan_real(other)?,
    };
    RootedValue::new(base).map_err(|cause| cause.into_runtime_error())
}

async fn atan_gpu(handle: GpuTensorHandle, protected: &[GpuTensorHandle]) -> BuiltinResult<Value> {
    if runmat_accelerate_api::handle_storage(&handle)
        == runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
    {
        return super::inverse_helpers::resident_complex::compute_protecting(
            handle,
            BUILTIN_NAME,
            protected,
            |value| match value {
                Value::Complex(re, im) => {
                    let (re, im) = atan_complex_components(re, im);
                    Ok(Value::Complex(re, im))
                }
                Value::ComplexTensor(tensor) => atan_complex_tensor(tensor),
                other => atan_real(other),
            },
        )
        .await;
    }
    if runmat_accelerate_api::handle_integer_type(&handle).is_some()
        || runmat_accelerate_api::handle_is_logical(&handle)
    {
        return crate::builtins::common::provider_restore::gather_compute_restore_protecting(
            handle,
            BUILTIN_NAME,
            protected,
            |tensor| atan_tensor(tensor).map(tensor::tensor_into_value),
        )
        .await;
    }
    if let Some(provider) = gpu_helpers::exact_provider_for_handle(&handle) {
        match provider.unary_atan(&handle).await {
            Ok(output) => {
                return crate::builtins::common::provider_restore::validate_real_unary_provider_output_protecting(
                    provider,
                    &handle,
                    output,
                    BUILTIN_NAME,
                    protected,
                )
            }
            Err(error) if gpu_helpers::provider_hook_allows_host_fallback(&error) => {}
            Err(error) => {
                return Err(atan_error_with_detail(
                    &ATAN_ERROR_INTERNAL,
                    format!("provider unary_atan failed: {error}"),
                ))
            }
        }
    }
    crate::builtins::common::provider_restore::gather_compute_restore_protecting(
        handle,
        BUILTIN_NAME,
        protected,
        |tensor| atan_tensor(tensor).map(tensor::tensor_into_value),
    )
    .await
}

fn atan_real(value: Value) -> BuiltinResult<Value> {
    let tensor = tensor::value_into_tensor_for("atan", value)
        .map_err(|e| atan_error_with_detail(&ATAN_ERROR_INVALID_INPUT, e))?;
    atan_tensor(tensor).map(tensor::tensor_into_value)
}

fn atan_tensor(tensor: Tensor) -> BuiltinResult<Tensor> {
    super::inverse_helpers::map_real_tensor(tensor, BUILTIN_NAME, f64::atan, f32::atan)
}

fn atan_complex_tensor(ct: ComplexTensor) -> BuiltinResult<Value> {
    let tensor = super::inverse_helpers::map_complex_tensor(
        ct,
        BUILTIN_NAME,
        |(real, imag)| atan_complex_components(real, imag),
        |(real, imag)| {
            let result = num_complex::Complex32::new(real, imag).atan();
            (result.re, result.im)
        },
    )?;
    Ok(complex_tensor_into_value(tensor))
}

fn atan_char_array(array: CharArray) -> BuiltinResult<Value> {
    let data = array
        .iter_column_major()
        .map(|ch| f64::from(ch.raw()).atan())
        .collect::<Vec<_>>();
    let tensor = Tensor::new(data, vec![array.rows, array.cols])
        .map_err(|e| atan_error_with_detail(&ATAN_ERROR_INTERNAL, e))?;
    Ok(tensor::tensor_into_value(tensor))
}

#[inline]
fn atan_complex_components(re: f64, im: f64) -> (f64, f64) {
    let value = Complex64::new(re, im).atan();
    (value.re, value.im)
}

mod output_template;
use output_template::{admit_output_template, apply_output_template, parse_output_template};

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_accelerate_api::HostTensorView;
    use runmat_value::{IntValue, Tensor};

    fn atan_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        block_on(super::atan_builtin(
            None,
            value,
            crate::builtins::common::test_support::builtin_arguments(rest),
        ))
        .map(|owned| owned.value().clone())
    }

    #[test]
    fn atan_extensions_like_and_output_arity_are_gated() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let integer = block_on(super::atan_builtin(
            None,
            Value::Int(runmat_value::IntValue::I8(1)),
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .map(|owned| owned.value().clone())
        .expect_err("integer input must be gated");
        assert_eq!(
            integer.identifier(),
            ATAN_INTEGER_INPUT_EXTENSION.error_identifier
        );
        let logical = block_on(super::atan_builtin(
            None,
            Value::Bool(true),
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .map(|owned| owned.value().clone())
        .expect_err("logical input must be gated");
        assert_eq!(
            logical.identifier(),
            ATAN_LOGICAL_INPUT_EXTENSION.error_identifier
        );
        let chars = CharArray::from_unicode_scalars("A".chars().collect(), 1, 1).unwrap();
        let character = block_on(super::atan_builtin(
            None,
            Value::CharArray(chars),
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .map(|owned| owned.value().clone())
        .expect_err("character input must be gated");
        assert_eq!(
            character.identifier(),
            ATAN_CHARACTER_INPUT_EXTENSION.error_identifier
        );
        let like = block_on(super::atan_builtin(
            None,
            Value::Num(1.0),
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::from("like"),
                Value::Num(0.0),
            ]),
        ))
        .map(|owned| owned.value().clone())
        .expect_err("like form must be gated");
        assert_eq!(like.identifier(), ATAN_LIKE_EXTENSION.error_identifier);
        let _outputs = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
        let arity = block_on(super::atan_builtin(
            None,
            Value::Num(0.0),
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .map(|owned| owned.value().clone())
        .expect_err("excess outputs must reject");
        assert_eq!(arity.identifier(), ATAN_ERROR_TOO_MANY_OUTPUTS.identifier);
    }

    #[test]
    fn atan_preserves_native_single_real_and_complex_storage() {
        let real = Tensor::from_f32(vec![0.5, 2.0], vec![2, 1]).unwrap();
        let Value::Tensor(real_output) =
            atan_builtin(Value::Tensor(real), Vec::new()).expect("single atan")
        else {
            panic!("expected single tensor");
        };
        assert_eq!(real_output.numeric_dtype(), runmat_value::NumericDType::F32);
        let complex = ComplexTensor::from_f32(vec![(0.5, 0.25), (2.0, -1.0)], vec![2, 1]).unwrap();
        let Value::ComplexTensor(complex_output) =
            atan_builtin(Value::ComplexTensor(complex), Vec::new()).expect("complex-single atan")
        else {
            panic!("expected complex-single tensor");
        };
        assert_eq!(
            complex_output.numeric_dtype(),
            runmat_value::NumericDType::F32
        );
    }

    fn error_message(err: RuntimeError) -> String {
        err.message().to_string()
    }

    #[test]
    fn atan_descriptor_signatures_cover_like_overload() {
        let labels: Vec<&str> = ATAN_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"Y = atan(X)"));
        assert!(labels.contains(&"Y = atan(X, \"like\", P)"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn atan_scalar() {
        let result = atan_builtin(Value::Num(1.0), Vec::new()).expect("atan");
        match result {
            Value::Num(v) => assert!((v - std::f64::consts::FRAC_PI_4).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn atan_preserves_rank_zero_scalarization_and_rank_one_shape() {
        let scalar = atan_builtin(
            Value::Tensor(Tensor::new(vec![0.0], Vec::new()).unwrap()),
            Vec::new(),
        )
        .expect("rank-zero atan");
        assert_eq!(scalar, Value::Num(0.0));

        let vector = atan_builtin(
            Value::Tensor(Tensor::new(vec![0.0, 0.0], vec![2]).unwrap()),
            Vec::new(),
        )
        .expect("rank-one atan");
        let Value::Tensor(vector) = vector else {
            panic!("expected rank-one tensor")
        };
        assert_eq!(vector.shape(), &[2]);
        assert_eq!(vector.materialize_f64(), vec![0.0, 0.0]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn atan_tensor_elements() {
        let tensor = Tensor::new(vec![-1.0, 0.0, 1.0], vec![3, 1]).unwrap();
        let result = atan_builtin(Value::Tensor(tensor.clone()), Vec::new()).expect("atan");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape(), &[3, 1]);
                for (value, expected) in out
                    .materialize_f64()
                    .iter()
                    .zip(tensor.materialize_f64().iter().map(|v| v.atan()))
                {
                    assert!((value - expected).abs() < 1e-12);
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn atan_reads_typed_integer_tensor_storage_exactly() {
        let tensor =
            Tensor::new_integer(runmat_value::IntegerStorage::I16(vec![0, 1, 2]), vec![3, 1])
                .expect("integer tensor");

        let result = atan_builtin(Value::Tensor(tensor), Vec::new()).expect("atan");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape(), &[3, 1]);
                let expected = [0.0, 1.0f64.atan(), 2.0f64.atan()];
                for (actual, expected) in out.materialize_f64().iter().zip(expected.iter()) {
                    assert!((actual - expected).abs() < 1e-12);
                }
                assert!(out.integer_storage().is_none());
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn atan_like_complex_reads_typed_integer_storage_exactly() {
        let tensor = Tensor::new_integer(
            runmat_value::IntegerStorage::I64(vec![-3, 0, 5]),
            vec![3, 1],
        )
        .expect("integer tensor");

        let result = atan_builtin(
            Value::Tensor(tensor),
            vec![Value::from("like"), Value::Complex(0.0, 1.0)],
        )
        .expect("atan");
        let Value::ComplexTensor(out) = result else {
            panic!("expected complex tensor result");
        };
        assert_eq!(out.shape(), &[3, 1]);
        let expected = [-3.0f64.atan(), 0.0f64.atan(), 5.0f64.atan()];
        for (actual, expected) in out.materialize_f64().iter().zip(expected) {
            assert!((actual.0 - expected).abs() < 1e-12);
            assert_eq!(actual.1, 0.0);
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn atan_all_integer_scalar_classes_cross_the_double_boundary() {
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
            let Value::Num(result) =
                atan_builtin(Value::Int(value), Vec::new()).expect("atan integer")
            else {
                panic!("expected real double scalar")
            };
            assert_eq!(result, std::f64::consts::FRAC_PI_4);
        }
        let Value::Num(result) =
            atan_builtin(Value::Int(IntValue::U64(u64::MAX)), Vec::new()).expect("wide atan")
        else {
            panic!("expected real double scalar")
        };
        assert_eq!(result, (u64::MAX as f64).atan());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn atan_complex_scalar() {
        let result = atan_builtin(Value::Complex(1.0, 2.0), Vec::new()).expect("atan");
        match result {
            Value::Complex(re, im) => {
                let expected = Complex64::new(1.0, 2.0).atan();
                assert!((re - expected.re).abs() < 1e-12);
                assert!((im - expected.im).abs() < 1e-12);
            }
            other => panic!("expected complex result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn atan_complex_tensor_elements() {
        let tensor = ComplexTensor::new(vec![(1.0, 0.5), (-0.5, 1.0)], vec![2, 1]).unwrap();
        let result = atan_builtin(Value::ComplexTensor(tensor.clone()), Vec::new()).expect("atan");
        match result {
            Value::ComplexTensor(out) => {
                assert_eq!(out.shape(), &[2, 1]);
                for (value, expected) in out.materialize_f64().iter().zip(
                    tensor
                        .materialize_f64()
                        .iter()
                        .map(|&(r, i)| atan_complex_components(r, i)),
                ) {
                    assert!((value.0 - expected.0).abs() < 1e-12);
                    assert!((value.1 - expected.1).abs() < 1e-12);
                }
            }
            Value::Complex(re, im) => {
                panic!("expected tensor result, got scalar {re}+{im}i");
            }
            other => panic!("unexpected result {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn atan_char_array_roundtrip() {
        let chars = CharArray::from_code_units(
            [0x0052, 0xD800, 0xDC00, 0xFFFF]
                .map(runmat_value::Utf16CodeUnit::from_raw)
                .to_vec(),
            2,
            2,
        )
        .unwrap();
        let result = atan_builtin(Value::CharArray(chars.clone()), Vec::new()).expect("atan");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape(), &[2, 2]);
                for (value, code) in t
                    .materialize_f64()
                    .iter()
                    .zip([82.0_f64, 56320.0, 55296.0, 65535.0])
                {
                    let expected = code.atan();
                    assert!((value - expected).abs() < 1e-12);
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn atan_string_errors() {
        let err = atan_builtin(Value::from("runmat"), Vec::new()).expect_err("expected error");
        assert_eq!(err.identifier(), ATAN_ERROR_INVALID_INPUT.identifier);
        let message = error_message(err);
        assert!(message.contains("expected numeric input"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn atan_like_missing_prototype_errors() {
        let err =
            atan_builtin(Value::Num(0.0), vec![Value::from("like")]).expect_err("expected error");
        assert_eq!(err.identifier(), ATAN_ERROR_INVALID_OPTION.identifier);
        let message = error_message(err);
        assert!(message.contains("expected prototype"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn atan_like_host_prototype() {
        let tensor = Tensor::new(vec![0.0, 1.0], vec![2, 1]).unwrap();
        let result = atan_builtin(
            Value::Tensor(tensor.clone()),
            vec![Value::from("like"), Value::Num(0.0)],
        )
        .expect("atan");
        match result {
            Value::Tensor(out) => {
                let expected: Vec<f64> =
                    tensor.materialize_f64().iter().map(|&v| v.atan()).collect();
                assert_eq!(out.materialize_f64(), expected);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn atan_like_complex_prototype_promotes() {
        let result = atan_builtin(
            Value::Num(0.5),
            vec![Value::from("like"), Value::Complex(0.0, 1.0)],
        )
        .expect("atan");
        match result {
            Value::Complex(re, im) => {
                assert!((re - 0.5f64.atan()).abs() < 1e-12);
                assert!(im.abs() < 1e-12);
            }
            other => panic!("expected complex result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn atan_gpu_provider_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![-1.0, 0.0, 1.0], vec![3, 1]).unwrap();
            let view = HostTensorView {
                data: &tensor.materialize_f64(),
                shape: tensor.shape(),
            };
            let handle = provider.upload(&view).expect("upload");
            let Value::GpuTensor(result_handle) =
                atan_builtin(Value::GpuTensor(handle), Vec::new()).expect("atan")
            else {
                panic!("expected resident result")
            };
            assert_eq!(result_handle.device_id(), provider.device_id());
            assert!(
                runmat_accelerate_api::provider_for_handle(&result_handle).is_some_and(|owner| {
                    runmat_accelerate_api::same_provider_instance(owner, provider)
                })
            );
            let gathered = test_support::gather(Value::GpuTensor(result_handle)).expect("gather");
            assert_eq!(gathered.shape(), &[3, 1]);
            let expected: Vec<f64> = tensor.materialize_f64().iter().map(|&v| v.atan()).collect();
            assert_eq!(gathered.materialize_f64(), expected);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn atan_like_gpu_prototype_keeps_residency() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![0.0, 1.0], vec![2, 1]).unwrap();
            let view = HostTensorView {
                data: &tensor.materialize_f64(),
                shape: tensor.shape(),
            };
            let input = provider.upload(&view).expect("upload");
            let proto = provider.upload(&view).expect("proto upload");
            let result = atan_builtin(
                Value::GpuTensor(input),
                vec![Value::from("like"), Value::GpuTensor(proto)],
            )
            .expect("atan");
            match result {
                Value::GpuTensor(handle) => {
                    assert_eq!(handle.device_id(), provider.device_id());
                    assert!(
                        runmat_accelerate_api::provider_for_handle(&handle).is_some_and(|owner| {
                            runmat_accelerate_api::same_provider_instance(owner, provider)
                        })
                    );
                    let gathered = test_support::gather(Value::GpuTensor(handle)).expect("gather");
                    let expected: Vec<f64> =
                        tensor.materialize_f64().iter().map(|&v| v.atan()).collect();
                    assert_eq!(gathered.materialize_f64(), expected);
                }
                other => panic!("expected GPU tensor, got {other:?}"),
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn atan_like_real_with_complex_output_errors() {
        let err = atan_builtin(
            Value::Complex(1.0, 1.0),
            vec![Value::from("like"), Value::Num(0.0)],
        )
        .expect_err("expected error");
        let message = error_message(err);
        assert!(message.contains("complex"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn atan_like_gpu_with_complex_output_errors() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![0.0], vec![1, 1]).unwrap();
            let view = HostTensorView {
                data: &tensor.materialize_f64(),
                shape: tensor.shape(),
            };
            let proto = provider.upload(&view).expect("upload");
            let err = atan_builtin(
                Value::Complex(1.0, 1.0),
                vec![Value::from("like"), Value::GpuTensor(proto)],
            )
            .expect_err("expected error");
            let message = error_message(err);
            assert!(message.contains("complex"));
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn atan_like_non_numeric_prototype_errors() {
        let err = atan_builtin(
            Value::Num(0.0),
            vec![Value::from("like"), Value::from("not-a-proto")],
        )
        .expect_err("expected error");
        let message = error_message(err);
        assert!(message.contains("prototype must be numeric"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn atan_like_rejects_extra_arguments() {
        let err = atan_builtin(
            Value::Num(0.0),
            vec![Value::from("like"), Value::Num(0.0), Value::Num(1.0)],
        )
        .expect_err("expected error");
        let message = error_message(err);
        assert!(message.contains("too many input arguments"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn atan_like_keyword_flexible_case() {
        let result = atan_builtin(Value::Num(1.0), vec![Value::from("LIKE"), Value::Num(0.0)])
            .expect("atan");
        match result {
            Value::Num(v) => assert!((v - std::f64::consts::FRAC_PI_4).abs() < 1e-12),
            other => panic!("unexpected result {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn atan_like_char_array_keyword() {
        let keyword = CharArray::new_row("like");
        let result = atan_builtin(
            Value::Num(0.0),
            vec![Value::CharArray(keyword), Value::Num(0.0)],
        )
        .expect("atan");
        match result {
            Value::Num(v) => assert!(v.abs() < 1e-12),
            other => panic!("unexpected result {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn atan_unrecognised_argument_errors() {
        let err = atan_builtin(Value::Num(0.0), vec![Value::from("invalid")])
            .expect_err("expected error");
        let message = error_message(err);
        assert!(message.contains("unrecognised argument"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn atan_wgpu_matches_cpu_elementwise() {
        let _guard = test_support::accel_test_lock();
        let Some(provider) = test_support::wgpu_provider_if_available() else {
            return;
        };
        let tensor = Tensor::new(vec![-2.0, -0.5, 0.0, 0.5, 2.0], vec![5, 1]).unwrap();
        let cpu = atan_real(Value::Tensor(tensor.clone())).unwrap();
        let view = HostTensorView {
            data: &tensor.materialize_f64(),
            shape: tensor.shape(),
        };
        let handle = provider.upload(&view).unwrap();
        let gpu = block_on(atan_gpu(handle, &[])).unwrap();
        let gathered = test_support::gather(gpu).expect("gather");
        match (cpu, gathered) {
            (Value::Tensor(ct), gt) => {
                assert_eq!(gt.shape(), ct.shape());
                let tol = match provider.precision() {
                    runmat_accelerate_api::ProviderPrecision::F64 => 1e-12,
                    runmat_accelerate_api::ProviderPrecision::F32 => 1e-5,
                };
                for (a, b) in gt.materialize_f64().iter().zip(ct.materialize_f64().iter()) {
                    assert!((a - b).abs() < tol, "|{a} - {b}| >= {tol}");
                }
            }
            _ => panic!("unexpected comparison result"),
        }
    }
}
