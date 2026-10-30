//! MATLAB-compatible `fix` builtin with GPU-aware semantics for RunMat.

use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::{
    BuiltinErrorDescriptor, FIX_ERROR_INTERNAL, FIX_ERROR_INVALID_INPUT, FIX_ERROR_TOO_MANY_OUTPUTS,
};
#[cfg(test)]
use runmat_builtins::{FIX_DESCRIPTOR, FIX_INTEGER_CAPABILITIES};
use runmat_macros::runtime_builtin;
use runmat_value::{
    CharArray, ComplexStorage, ComplexTensor, NumericStorage, ObjectInstance, Tensor, Value,
};

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, FusionError,
    FusionKernelTemplate, GpuOpKind, ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType,
    ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
use crate::context::RuntimeContext;
use crate::execution::{RootedValue, RootedValueLease, RootedValuesBuilder};
use crate::{BuiltinResult, RuntimeError};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::rounding::fix")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "fix",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Unary { name: "unary_fix" }],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Providers may implement unary_fix to keep fix on device; otherwise the runtime gathers to host and applies CPU truncation.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::rounding::fix")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "fix",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: crate::builtins::common::spec::FusionExprBuilder::new(|ctx, writer| {
            let input = ctx.inputs.first().ok_or(FusionError::MissingInput(0))?;
            let zero = match ctx.scalar_ty {
                ScalarType::F32 => "0.0", ScalarType::F64 => "f64(0.0)",
                other => return Err(FusionError::UnsupportedPrecision(other)),
            };
            writer.write_formatted(format_args!("select(trunc({input}), {zero}, trunc({input}) == {zero})"))
        }),
    }),
    reduction: None,
    emits_nan: false,
    notes: "Fusion planner emits WGSL truncation; providers can substitute custom kernels when unary_fix is available.",
};

const BUILTIN_NAME: &str = "fix";

fn builtin_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    super::unary::error_with_detail(BUILTIN_NAME, error, detail.as_ref())
}

#[runtime_builtin(
    name = "fix",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::rounding::fix",
    context = true
)]
async fn fix_builtin(runtime: Option<RuntimeContext>, value: Value) -> BuiltinResult<RootedValue> {
    super::unary::reject_excess_outputs(BUILTIN_NAME, &FIX_ERROR_TOO_MANY_OUTPUTS)?;
    crate::builtins::common::validation::reject_typed_complex_integer(&value, BUILTIN_NAME)?;
    super::preparation::unary(
        runtime.as_ref(),
        &runmat_builtins::FIX_CATALOG_ENTRY,
        value,
        fix_value,
    )
    .await
}

async fn fix_value(value: Value) -> BuiltinResult<RootedValue> {
    let _input =
        RootedValueLease::for_traced(&value).map_err(|error| error.into_runtime_error())?;
    match value {
        Value::GpuTensor(handle) => super::unary::root_output(fix_gpu(handle).await?),
        Value::Object(object) if crate::builtins::table::is_tabular_object(&object) => {
            fix_table(object).await
        }
        other => super::unary::root_output(fix_host_value(other)?),
    }
}

async fn fix_table(object: ObjectInstance) -> BuiltinResult<RootedValue> {
    let variables = crate::builtins::table::table_variables(&object)
        .map_err(|err| builtin_error_with_detail(&FIX_ERROR_INVALID_INPUT, err.message))?;
    let mut names = Vec::with_capacity(variables.fields.len());
    let mut rounded = RootedValuesBuilder::with_capacity(variables.fields.len());
    for (name, value) in variables.into_fields_unbounded() {
        crate::builtins::common::validation::reject_typed_complex_integer(&value, BUILTIN_NAME)?;
        let value = match value {
            Value::GpuTensor(handle) => fix_gpu(handle).await?,
            Value::Object(_) => {
                return Err(builtin_error_with_detail(
                    &FIX_ERROR_INVALID_INPUT,
                    format!("table variable {name} does not support fix"),
                ))
            }
            other => fix_host_value(other)?,
        };
        names.push(name);
        rounded.push_owned(super::unary::root_output(value)?);
    }
    let rounded = rounded
        .finish()
        .map_err(|error| error.into_runtime_error())?;
    crate::builtins::table::table_replace_variables_like_owned(&object, names, rounded)
        .map_err(|err| builtin_error_with_detail(&FIX_ERROR_INTERNAL, err.message))
}

fn fix_host_value(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::Complex(re, im) => Ok(Value::Complex(fix_scalar(re), fix_scalar(im))),
        Value::ComplexTensor(ct) => fix_complex_tensor(ct),
        Value::CharArray(ca) => fix_char_array(ca),
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(&logical)
                .map_err(|err| builtin_error_with_detail(&FIX_ERROR_INVALID_INPUT, err))?;
            fix_tensor(tensor).map(tensor::tensor_into_value)
        }
        Value::String(_) | Value::StringArray(_) => Err(builtin_error_with_detail(
            &FIX_ERROR_INVALID_INPUT,
            "expected numeric or logical input",
        )),
        other => fix_numeric(other),
    }
}

async fn fix_gpu(handle: GpuTensorHandle) -> BuiltinResult<Value> {
    let provider = super::unary::input_provider(&handle, BUILTIN_NAME, &FIX_ERROR_INTERNAL)?;
    if runmat_accelerate_api::handle_integer_type(&handle).is_some() {
        return gpu_helpers::resident_gpu_value(handle);
    }
    if !runmat_accelerate_api::handle_is_logical(&handle)
        && runmat_accelerate_api::handle_storage(&handle)
            == runmat_accelerate_api::GpuTensorStorage::Real
    {
        let result = {
            let _metadata = gpu_helpers::ResidencyRestoreGuard::new(&handle);
            provider.unary_fix(&handle).await
        };
        match result {
            Ok(output) => {
                return super::unary::validate_provider_output(
                    provider,
                    &handle,
                    output,
                    BUILTIN_NAME,
                    &FIX_ERROR_INTERNAL,
                )
            }
            Err(error) if gpu_helpers::provider_hook_allows_host_fallback(&error) => {}
            Err(error) => {
                return Err(builtin_error_with_detail(
                    &FIX_ERROR_INTERNAL,
                    format!("provider unary_fix failed: {error}"),
                ))
            }
        }
    }
    let host = {
        let _metadata = gpu_helpers::ResidencyRestoreGuard::new(&handle);
        gpu_helpers::download_value_preserving_residency_async(provider, &handle).await?
    };
    let output = fix_host_value(host)?;
    gpu_helpers::restore_class_preserving_value(&handle, output, BUILTIN_NAME)
}

fn fix_numeric(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::Num(n) => Ok(Value::Num(fix_scalar(n))),
        // MATLAB integer values are already integral. Preserve their exact
        // class and bits instead of routing 64-bit values through f64.
        Value::Int(i) => Ok(Value::Int(i)),
        Value::Bool(b) => Ok(Value::Num(fix_scalar(if b { 1.0 } else { 0.0 }))),
        Value::Tensor(t) => fix_tensor(t).map(tensor::tensor_into_value),
        other => {
            let tensor = tensor::value_into_tensor_for("fix", other)
                .map_err(|err| builtin_error_with_detail(&FIX_ERROR_INVALID_INPUT, err))?;
            Ok(fix_tensor(tensor).map(tensor::tensor_into_value)?)
        }
    }
}

fn fix_tensor(tensor: Tensor) -> BuiltinResult<Tensor> {
    let shape = tensor.shape().to_vec();
    let storage = tensor
        .into_numeric_storage()
        .map_err(|e| builtin_error_with_detail(&FIX_ERROR_INTERNAL, e))?;
    let output = match storage {
        NumericStorage::F64(values) => {
            NumericStorage::F64(values.into_iter().map(fix_scalar).collect())
        }
        NumericStorage::F32(values) => NumericStorage::F32(
            values
                .into_iter()
                .map(|value| {
                    let truncated = value.trunc();
                    if truncated == 0.0 {
                        0.0
                    } else {
                        truncated
                    }
                })
                .collect(),
        ),
        integer => integer,
    };
    Tensor::from_numeric_storage(output, shape)
        .map_err(|e| builtin_error_with_detail(&FIX_ERROR_INTERNAL, e))
}

fn fix_complex_tensor(ct: ComplexTensor) -> BuiltinResult<Value> {
    let (storage, shape) = ct.into_complex_storage_and_shape();
    let storage = match storage {
        ComplexStorage::F64(values) => ComplexStorage::F64(
            values
                .into_iter()
                .map(|(re, im)| (fix_scalar(re), fix_scalar(im)))
                .collect(),
        ),
        ComplexStorage::F32(values) => ComplexStorage::F32(
            values
                .into_iter()
                .map(|(re, im)| (re.trunc(), im.trunc()))
                .collect(),
        ),
        ComplexStorage::Integer(_) => {
            return Err(builtin_error_with_detail(
                &FIX_ERROR_INVALID_INPUT,
                "operations involving complex numbers with integer types are not supported",
            ))
        }
    };
    let tensor = ComplexTensor::from_complex_storage(storage, shape)
        .map_err(|e| builtin_error_with_detail(&FIX_ERROR_INTERNAL, e))?;
    Ok(Value::ComplexTensor(tensor))
}

fn fix_char_array(ca: CharArray) -> BuiltinResult<Value> {
    let data = ca
        .to_column_major()
        .iter()
        .map(|&ch| fix_scalar(u32::from(ch.raw()) as f64))
        .collect::<Vec<_>>();
    let tensor = Tensor::new(data, ca.shape)
        .map_err(|e| builtin_error_with_detail(&FIX_ERROR_INTERNAL, e))?;
    Ok(Value::Tensor(tensor))
}

fn fix_scalar(value: f64) -> f64 {
    if !value.is_finite() {
        return value;
    }
    let truncated = value.trunc();
    if truncated == 0.0 {
        0.0
    } else {
        truncated
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use crate::builtins::math::rounding::unary;
    use crate::RuntimeError;
    use futures::executor::block_on;
    use runmat_builtins::{
        BuiltinIntegerBackendRule, BuiltinIntegerComputationDomain,
        BuiltinIntegerInputAvailability, BuiltinIntegerOutputClassRule,
    };
    use runmat_value::{ComplexStorage, ComplexTensor, IntValue, IntegerStorage, LogicalArray};

    fn fix_builtin(value: Value) -> BuiltinResult<Value> {
        block_on(super::fix_builtin(None, value)).map(|owned| owned.value().clone())
    }

    #[test]
    fn fix_preserves_native_single_storage() {
        let input = Tensor::from_f32(vec![-1.75, -0.0, 2.25], vec![1, 3]).unwrap();
        let output = fix_tensor(input).unwrap();
        assert_eq!(
            output.into_numeric_storage().unwrap(),
            NumericStorage::F32(vec![-1.0, 0.0, 2.0])
        );
    }

    #[test]
    fn fix_rejects_excess_outputs() {
        let _outputs = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
        let error = fix_builtin(Value::Num(1.25)).expect_err("fix must reject excess outputs");
        assert_eq!(error.identifier(), Some("RunMat:fix:TooManyOutputs"));
    }

    fn assert_error_contains(error: &RuntimeError, needle: &str) {
        assert!(
            error.message().contains(needle),
            "unexpected error: {}",
            error.message()
        );
    }

    #[test]
    fn fix_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = FIX_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"Y = fix(X)"));
        assert_eq!(FIX_INTEGER_CAPABILITIES.len(), 1);
        let capability = &FIX_INTEGER_CAPABILITIES[0];
        assert_eq!(capability.inputs[0].classes.len(), 8);
        assert_eq!(
            capability.inputs[0].availability,
            BuiltinIntegerInputAvailability::Documented
        );
        assert_eq!(
            capability.computation_domain,
            BuiltinIntegerComputationDomain::ExactInteger
        );
        assert_eq!(
            capability.output_class,
            BuiltinIntegerOutputClassRule::PreserveInput
        );
        assert_eq!(capability.backend, BuiltinIntegerBackendRule::HostAndGpu);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fix_scalar_positive_and_negative() {
        let input = Value::Tensor(
            Tensor::new(vec![-3.7, -2.4, -0.6, 0.0, 0.6, 2.4, 3.7], vec![7, 1]).unwrap(),
        );
        let result = fix_builtin(input).expect("fix");
        match result {
            Value::Tensor(t) => {
                assert_eq!(
                    t.materialize_f64(),
                    vec![-3.0, -2.0, 0.0, 0.0, 0.0, 2.0, 3.0]
                );
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fix_tensor_matrix() {
        let tensor = Tensor::new(vec![1.9, 4.1, -2.8, 0.5], vec![2, 2]).unwrap();
        let result = fix_builtin(Value::Tensor(tensor)).expect("fix");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape(), [2, 2]);
                assert_eq!(t.materialize_f64(), vec![1.0, 4.0, -2.0, 0.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fix_complex_number() {
        let result = fix_builtin(Value::Complex(1.9, -2.2)).expect("fix");
        match result {
            Value::Complex(re, im) => {
                assert_eq!(re, 1.0);
                assert_eq!(im, -2.0);
            }
            other => panic!("expected complex result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fix_char_array_returns_numeric_tensor() {
        let chars = CharArray::from_unicode_scalars("ABC".chars().collect(), 1, 3).unwrap();
        let result = fix_builtin(Value::CharArray(chars)).expect("fix");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape(), [1, 3]);
                assert_eq!(t.materialize_f64(), vec![65.0, 66.0, 67.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fix_logical_array() {
        let logical = LogicalArray::new(vec![1, 0, 1, 1], vec![2, 2]).unwrap();
        let result = fix_builtin(Value::LogicalArray(logical)).expect("fix");
        match result {
            Value::Tensor(t) => assert_eq!(t.materialize_f64(), vec![1.0, 0.0, 1.0, 1.0]),
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fix_bool_promotes_to_numeric() {
        let result = fix_builtin(Value::Bool(true)).expect("fix");
        match result {
            Value::Num(v) => assert_eq!(v, 1.0),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fix_integer_scalars_preserve_class_and_exact_64_bit_values() {
        let signed = Value::Int(IntValue::I64(i64::MIN));
        assert_eq!(fix_builtin(signed.clone()).expect("fix"), signed);

        let unsigned = Value::Int(IntValue::U64(u64::MAX));
        assert_eq!(fix_builtin(unsigned.clone()).expect("fix"), unsigned);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fix_read_typed_integer_storage_exactly() {
        let scalar =
            Tensor::new_integer(IntegerStorage::I64(vec![i64::MAX]), vec![1, 1]).expect("integer");
        assert_eq!(
            fix_builtin(Value::Tensor(scalar)).expect("fix"),
            Value::Int(IntValue::I64(i64::MAX))
        );

        let tensor = Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX, 3]), vec![1, 2])
            .expect("integer");
        match fix_builtin(Value::Tensor(tensor)).expect("fix") {
            Value::Tensor(out) => {
                assert_eq!(out.shape(), [1, 2]);
                assert_eq!(
                    out.integer_storage(),
                    Some(&IntegerStorage::U64(vec![u64::MAX, 3]))
                );
            }
            other => panic!("expected typed integer tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fix_string_errors() {
        let err = fix_builtin(Value::from("abc")).unwrap_err();
        assert_error_contains(&err, "expected numeric");
        assert_eq!(err.identifier(), FIX_ERROR_INVALID_INPUT.identifier);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fix_preserves_special_values_and_canonicalizes_negative_zero() {
        let tensor = Tensor::new(
            vec![f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -0.0],
            vec![4, 1],
        )
        .unwrap();
        let result = fix_builtin(Value::Tensor(tensor)).expect("fix");
        let Value::Tensor(out) = result else {
            panic!("expected tensor result");
        };
        assert!(out.materialize_f64()[0].is_nan(), "NaN should propagate");
        assert_eq!(out.materialize_f64()[1], f64::INFINITY);
        assert_eq!(out.materialize_f64()[2], f64::NEG_INFINITY);
        assert_eq!(out.materialize_f64()[3], 0.0);
        assert!(
            out.materialize_f64()[3].is_sign_positive(),
            "negative zero should canonicalize to +0"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fix_complex_tensor_rounds_components() {
        let tensor = ComplexTensor::new(vec![(1.9, -2.6), (-3.4, 0.2)], vec![2, 1]).unwrap();
        let result = fix_builtin(Value::ComplexTensor(tensor)).expect("fix");
        let Value::ComplexTensor(out) = result else {
            panic!("expected complex tensor result");
        };
        assert_eq!(out.shape(), [2, 1]);
        assert_eq!(out.materialize_f64(), vec![(1.0, -2.0), (-3.0, 0.0)]);
    }

    #[test]
    fn fix_complex_tensor_preserves_native_single_storage() {
        let input = ComplexTensor::from_complex_storage(
            ComplexStorage::F32(vec![(1.9, -2.6), (-3.4, 0.2)].into()),
            vec![2, 1],
        )
        .unwrap();
        let Value::ComplexTensor(output) = fix_complex_tensor(input).unwrap() else {
            panic!("expected complex tensor");
        };
        assert_eq!(
            output.into_complex_storage(),
            ComplexStorage::F32(vec![(1.0, -2.0), (-3.0, 0.0)].into())
        );
    }

    #[test]
    fn fix_table_preserves_every_integer_variable_class() {
        let columns = vec![
            Tensor::new_integer(IntegerStorage::I8(vec![-3, 4]), vec![2, 1]).unwrap(),
            Tensor::new_integer(IntegerStorage::I16(vec![-3, 4]), vec![2, 1]).unwrap(),
            Tensor::new_integer(IntegerStorage::I32(vec![-3, 4]), vec![2, 1]).unwrap(),
            Tensor::new_integer(IntegerStorage::I64(vec![i64::MIN, i64::MAX]), vec![2, 1]).unwrap(),
            Tensor::new_integer(IntegerStorage::U8(vec![3, 4]), vec![2, 1]).unwrap(),
            Tensor::new_integer(IntegerStorage::U16(vec![3, 4]), vec![2, 1]).unwrap(),
            Tensor::new_integer(IntegerStorage::U32(vec![3, 4]), vec![2, 1]).unwrap(),
            Tensor::new_integer(IntegerStorage::U64(vec![0, u64::MAX]), vec![2, 1]).unwrap(),
        ];
        let names = (0..columns.len())
            .map(|index| format!("V{index}"))
            .collect::<Vec<_>>();
        let expected = columns.clone();
        let input = crate::builtins::table::table_from_columns(
            names,
            columns.into_iter().map(Value::Tensor).collect(),
        )
        .unwrap();
        let output = block_on(super::fix_builtin(None, input)).unwrap();
        let Value::Object(output) = output.value() else {
            panic!("expected table output");
        };
        let variables = crate::builtins::table::table_variables(output).unwrap();
        for (value, expected) in variables.fields.values().zip(expected) {
            assert_eq!(value, &Value::Tensor(expected));
        }
    }

    #[test]
    fn fix_resident_integer_is_an_exact_identity() {
        test_support::with_test_provider(|provider| {
            let input =
                Tensor::new_integer(IntegerStorage::U64(vec![0, u64::MAX]), vec![1, 2]).unwrap();
            let handle = gpu_helpers::upload_tensor(provider, &input).expect("integer upload");
            let buffer_id = handle.buffer_id();
            let Value::GpuTensor(output) = block_on(super::fix_gpu(handle)).unwrap() else {
                panic!("expected resident integer output");
            };
            assert_eq!(output.buffer_id(), buffer_id);
            assert_eq!(
                test_support::gather(Value::GpuTensor(output)).unwrap(),
                input
            );
        });
    }

    #[test]
    fn fix_rejects_aliased_or_mistyped_native_outputs_without_freeing_input() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.25], vec![1, 1]).unwrap();
            let input = gpu_helpers::upload_tensor(provider, &tensor).expect("input upload");
            assert!(unary::validate_provider_output(
                provider,
                &input,
                input.clone(),
                BUILTIN_NAME,
                &FIX_ERROR_INTERNAL,
            )
            .is_err());
            assert_eq!(
                test_support::gather(Value::GpuTensor(input.clone()))
                    .expect("input remains live")
                    .materialize_f64(),
                vec![1.25]
            );

            let logical = provider
                .upload_logical(&runmat_accelerate_api::HostLogicalTensorView {
                    data: &[1],
                    shape: &[1, 1],
                })
                .expect("logical upload");
            let logical_physical = logical.allocation_lease().downgrade();
            assert!(unary::validate_provider_output(
                provider,
                &input,
                logical,
                BUILTIN_NAME,
                &FIX_ERROR_INTERNAL,
            )
            .is_err());

            assert!(logical_physical.acquire().is_none());

            let integer_tensor =
                Tensor::new_integer(IntegerStorage::U8(vec![1]), vec![1, 1]).unwrap();
            let integer =
                gpu_helpers::upload_tensor(provider, &integer_tensor).expect("integer upload");
            let integer_physical = integer.allocation_lease().downgrade();
            assert!(unary::validate_provider_output(
                provider,
                &input,
                integer,
                BUILTIN_NAME,
                &FIX_ERROR_INTERNAL,
            )
            .is_err());
            assert!(integer_physical.acquire().is_none());
            input.allocation_lease().validate_live().unwrap();
            let physical = input.allocation_lease().downgrade();
            input.release().expect("release input");
            assert!(physical.acquire().is_none());
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fix_gpu_provider_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![-1.9, -0.1, 0.1, 2.6], vec![4, 1]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.materialize_f64(),
                shape: tensor.shape(),
            };
            let handle = provider.upload(&view).expect("upload");
            let result = fix_builtin(Value::GpuTensor(handle)).expect("fix");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.materialize_f64(), vec![-1.0, 0.0, 0.0, 2.0]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn fix_wgpu_matches_cpu() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let tensor = Tensor::new(vec![-3.7, -0.4, 0.4, 3.7], vec![4, 1]).unwrap();
        let cpu = fix_tensor(tensor.clone()).unwrap();
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.materialize_f64(),
            shape: tensor.shape(),
        };
        let handle = runmat_accelerate_api::provider()
            .unwrap()
            .upload(&view)
            .unwrap();
        let gpu = block_on(fix_gpu(handle)).unwrap();
        let gathered = test_support::gather(gpu).expect("gather");
        assert_eq!(gathered.shape(), cpu.shape());
        assert_eq!(gathered.materialize_f64(), cpu.materialize_f64());
    }
}
