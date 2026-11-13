//! MATLAB-compatible `trace` builtin with GPU-aware semantics for RunMat.

#[cfg(test)]
use runmat_accelerate_api::HostTensorView;
use runmat_builtins::BuiltinErrorDescriptor;
#[cfg(test)]
use runmat_builtins::TRACE_DESCRIPTOR;
use runmat_builtins::{
    TRACE_CHAR_INPUT_EXTENSION, TRACE_ERROR_INTERNAL, TRACE_ERROR_INVALID_INPUT,
    TRACE_INTEGER_INPUT_EXTENSION, TRACE_LOGICAL_INPUT_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::{CharArray, ComplexTensor, IntValue, Tensor, Value};
use runmat_value::{NumericDType, SparseTensor};

use crate::builtins::common::gpu_helpers;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::tensor;
#[cfg(test)]
use crate::builtins::math::linalg::type_resolvers::numeric_scalar_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

mod resident;
use resident::trace_gpu;

#[cfg(test)]
mod resident_tests;

const NAME: &str = "trace";

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::linalg::ops::trace")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: NAME,
    op_kind: GpuOpKind::Reduction,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[
        ProviderHook::Custom("diag_extract"),
        ProviderHook::Reduction {
            name: "reduce_sum",
        },
    ],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::InheritInputs,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: Some(256),
    workgroup_size: Some(256),
    accepts_nan_mode: false,
    notes:
        "Resident inputs retain their exact owner. A provider with double accumulation extracts the diagonal and sums it in double precision, then restores single precision when required; scalar identity results may share the input allocation. A provider that certifies its double-sum hook unavailable takes the class-preserving host route before resident staging; other unsupported hooks follow effect-aware fallback rules.",
};

fn trace_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    trace_error_with_message(error.message, error)
}

fn trace_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn trace_invalid_input(message: impl Into<String>) -> RuntimeError {
    trace_error_with_message(message, &TRACE_ERROR_INVALID_INPUT)
}

fn map_control_flow(err: RuntimeError) -> RuntimeError {
    if err.message() == "interaction pending..." {
        return build_runtime_error("interaction pending...")
            .with_builtin(NAME)
            .build();
    }
    let mut builder = build_runtime_error(err.message()).with_builtin(NAME);
    if let Some(identifier) = err.identifier() {
        builder = builder.with_identifier(identifier.to_string());
    }
    if let Some(task_id) = err.context.task_id.clone() {
        builder = builder.with_task_id(task_id);
    }
    if !err.context.call_stack.is_empty() {
        builder = builder.with_call_stack(err.context.call_stack.clone());
    }
    if let Some(phase) = err.context.phase.clone() {
        builder = builder.with_phase(phase);
    }
    builder.with_source(err).build()
}

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::linalg::ops::trace")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: NAME,
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: true,
    notes: "Trace is treated as a scalar reduction boundary; fusion wrappers stop at trace so producers/consumers can fuse independently.",
};

#[runtime_builtin(
    name = "trace",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::linalg::ops::trace"
)]
async fn trace_builtin(value: Value) -> BuiltinResult<Value> {
    crate::builtins::common::validation::reject_typed_complex_integer(&value, NAME)?;
    ensure_trace_extensions(&value)?;
    match value {
        Value::GpuTensor(handle) => trace_gpu(handle).await,
        Value::ComplexTensor(ct) => trace_complex_tensor(ct),
        Value::Complex(re, im) => Ok(Value::Complex(re, im)),
        Value::SparseTensor(sparse) => trace_sparse(sparse),
        Value::CharArray(ca) => trace_char_array(ca),
        other => trace_numeric(other),
    }
}

fn trace_numeric(value: Value) -> BuiltinResult<Value> {
    let tensor = tensor::value_into_tensor_for(NAME, value).map_err(trace_invalid_input)?;
    let (rows, cols) = ensure_matrix_shape(NAME, tensor.shape())?;
    if tensor.integer_storage().is_some() {
        return Ok(Value::Num(trace_integer_tensor_sum(&tensor, rows)?));
    }
    let sum = trace_tensor_sum(&tensor, rows, cols);
    if tensor.numeric_dtype() == NumericDType::F32 {
        let scalar = Tensor::from_f32(vec![sum as f32], vec![1, 1]).map_err(|error| {
            trace_error_with_message(format!("trace: {error}"), &TRACE_ERROR_INTERNAL)
        })?;
        Ok(Value::Tensor(scalar))
    } else {
        Ok(Value::Num(sum))
    }
}

fn trace_sparse(sparse: SparseTensor) -> BuiltinResult<Value> {
    ensure_square(sparse.rows, sparse.cols)?;
    if sparse.integer_storage().is_some() {
        let mut sum = 0_i128;
        for index in 0..sparse.rows {
            if let Some(value) = sparse.integer_at(index, index) {
                sum = sum
                    .checked_add(int_value_i128(&value))
                    .ok_or_else(|| trace_invalid_input("trace: integer diagonal sum overflowed"))?;
            }
        }
        return exact_integer_sum_to_double(sum);
    }
    if sparse.is_complex() {
        let (real, imaginary) = (0..sparse.rows).fold((0.0, 0.0), |sum, index| {
            let value = sparse.complex_at(index, index).unwrap_or((0.0, 0.0));
            (sum.0 + value.0, sum.1 + value.1)
        });
        if sparse.numeric_dtype() == Some(NumericDType::F32) {
            return Ok(Value::ComplexTensor(
                ComplexTensor::from_f32(vec![(real as f32, imaginary as f32)], vec![1, 1])
                    .map_err(trace_invalid_input)?,
            ));
        }
        return Ok(Value::Complex(real, imaginary));
    }
    let sum = (0..sparse.rows)
        .map(|index| sparse.get(index, index).unwrap_or(0.0))
        .sum::<f64>();
    if sparse.numeric_dtype() == Some(NumericDType::F32) {
        Ok(Value::Tensor(
            Tensor::from_f32(vec![sum as f32], vec![1, 1]).map_err(trace_invalid_input)?,
        ))
    } else {
        Ok(Value::Num(sum))
    }
}

fn trace_complex_tensor(ct: ComplexTensor) -> BuiltinResult<Value> {
    let (rows, cols) = ensure_matrix_shape(NAME, ct.shape())?;
    let dtype = ct.numeric_dtype();
    let diag_len = rows.min(cols);
    let mut sum_re = 0.0;
    let mut sum_im = 0.0;
    let values = ct.materialize_f64();
    for idx in 0..diag_len {
        let linear = idx + idx * rows;
        let (re, im) = values[linear];
        sum_re += re;
        sum_im += im;
    }
    if dtype == NumericDType::F32 {
        Ok(Value::ComplexTensor(
            ComplexTensor::from_f32(vec![(sum_re as f32, sum_im as f32)], vec![1, 1])
                .map_err(trace_invalid_input)?,
        ))
    } else {
        Ok(Value::Complex(sum_re, sum_im))
    }
}

fn trace_char_array(ca: CharArray) -> BuiltinResult<Value> {
    let (rows, cols) = ensure_matrix_shape(NAME, &[ca.rows, ca.cols])?;
    let diag_len = rows.min(cols);
    let mut sum = 0.0;
    for idx in 0..diag_len {
        let linear = idx * ca.cols + idx;
        if let Some(ch) = ca.data().get(linear) {
            sum += f64::from(ch.raw());
        }
    }
    Ok(Value::Num(sum))
}

fn trace_tensor_sum(tensor: &Tensor, rows: usize, cols: usize) -> f64 {
    let diag_len = rows.min(cols);
    let mut sum = 0.0;
    for idx in 0..diag_len {
        let linear = idx + idx * rows;
        sum += tensor::tensor_value_f64(tensor, linear);
    }
    sum
}

fn ensure_matrix_shape(name: &str, shape: &[usize]) -> BuiltinResult<(usize, usize)> {
    if shape.len() > 2 && shape.iter().skip(2).any(|&d| d != 1) {
        let _ = name;
        Err(trace_error(&TRACE_ERROR_INVALID_INPUT))
    } else {
        let (rows, cols) = matrix_extents_from_shape(shape);
        ensure_square(rows, cols)?;
        Ok((rows, cols))
    }
}

fn ensure_square(rows: usize, cols: usize) -> BuiltinResult<()> {
    if rows == cols {
        Ok(())
    } else {
        Err(trace_invalid_input("trace: input must be a square matrix"))
    }
}

fn trace_integer_tensor_sum(tensor: &Tensor, rows: usize) -> BuiltinResult<f64> {
    let mut sum = 0_i128;
    for index in 0..rows {
        let value = tensor
            .integer_storage()
            .and_then(|storage| storage.value_at(index + index * rows))
            .ok_or_else(|| trace_invalid_input("trace: invalid integer diagonal storage"))?;
        sum = sum
            .checked_add(int_value_i128(&value))
            .ok_or_else(|| trace_invalid_input("trace: integer diagonal sum overflowed"))?;
    }
    match exact_integer_sum_to_double(sum)? {
        Value::Num(value) => Ok(value),
        _ => unreachable!(),
    }
}

fn exact_integer_sum_to_double(sum: i128) -> BuiltinResult<Value> {
    const MAX_EXACT: i128 = 1_i128 << 53;
    if !(-MAX_EXACT..=MAX_EXACT).contains(&sum) {
        return Err(trace_invalid_input(
            "trace: integer diagonal sum must be exactly representable as double",
        ));
    }
    Ok(Value::Num(sum as f64))
}

fn int_value_i128(value: &IntValue) -> i128 {
    match value {
        IntValue::I8(value) => i128::from(*value),
        IntValue::I16(value) => i128::from(*value),
        IntValue::I32(value) => i128::from(*value),
        IntValue::I64(value) => i128::from(*value),
        IntValue::U8(value) => i128::from(*value),
        IntValue::U16(value) => i128::from(*value),
        IntValue::U32(value) => i128::from(*value),
        IntValue::U64(value) => i128::from(*value),
    }
}

fn ensure_trace_extensions(value: &Value) -> BuiltinResult<()> {
    if matches!(value, Value::Int(_))
        || matches!(value, Value::Tensor(tensor) if tensor.integer_storage().is_some())
        || matches!(value, Value::SparseTensor(sparse) if sparse.integer_storage().is_some())
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_integer_type(handle).is_some())
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &TRACE_INTEGER_INPUT_EXTENSION,
            NAME,
        )?;
    }
    if matches!(value, Value::Bool(_) | Value::LogicalArray(_))
        || matches!(value, Value::SparseTensor(sparse) if sparse.is_logical())
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_is_logical(handle))
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &TRACE_LOGICAL_INPUT_EXTENSION,
            NAME,
        )?;
    }
    if matches!(value, Value::CharArray(_)) {
        crate::compatibility::ensure_builtin_extension_enabled(&TRACE_CHAR_INPUT_EXTENSION, NAME)?;
    }
    Ok(())
}

fn matrix_extents_from_shape(shape: &[usize]) -> (usize, usize) {
    match shape.len() {
        0 => (1, 1),
        1 => (shape[0], 1),
        _ => (shape[0], shape[1]),
    }
}

#[path = "trace/geometry.rs"]
mod geometry;
pub(in crate::builtins::math) use geometry::{trace_geometry, TraceGeometry};

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_builtins::{ResolveContext, Type};
    use runmat_value::{IntValue, IntegerStorage, LogicalArray};
    fn unwrap_error(err: crate::RuntimeError) -> crate::RuntimeError {
        err
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn trace_scalar_num() {
        let result = trace_builtin(Value::Num(7.0)).expect("trace");
        assert_eq!(result, Value::Num(7.0));
    }

    #[test]
    fn trace_type_returns_scalar() {
        let out = numeric_scalar_type(
            &[Type::Tensor {
                shape: Some(vec![Some(2), Some(2)]),
            }],
            &ResolveContext::new(Vec::new()),
        );
        assert_eq!(out, Type::Num);
    }

    #[test]
    fn trace_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = TRACE_DESCRIPTOR
            .signatures
            .iter()
            .map(|signature| signature.label)
            .collect();
        assert!(labels.contains(&"t = trace(A)"));
    }

    #[test]
    fn trace_descriptor_errors_have_stable_codes() {
        let codes: Vec<&str> = TRACE_DESCRIPTOR.errors.iter().map(|err| err.code).collect();
        assert!(codes.contains(&"RM.TRACE.INVALID_INPUT"));
        assert!(codes.contains(&"RM.TRACE.INTERNAL"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn trace_rejects_rectangular_matrix() {
        let tensor = Tensor::new(vec![4.0, 1.0, 5.0, 2.0, 6.0, 3.0], vec![3, 2]).unwrap();
        let error = trace_builtin(Value::Tensor(tensor)).unwrap_err();
        assert_eq!(error.identifier(), TRACE_ERROR_INVALID_INPUT.identifier);
    }

    #[test]
    fn trace_reads_typed_integer_diagonal_storage_exactly() {
        let tensor =
            Tensor::new_integer(IntegerStorage::I16(vec![4, 1, 2, 6]), vec![2, 2]).expect("tensor");
        let result = trace_builtin(Value::Tensor(tensor)).expect("trace");
        assert_eq!(result, Value::Num(10.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn trace_rejects_nonscalar_vector() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        assert!(trace_builtin(Value::Tensor(tensor)).is_err());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn trace_empty_matrix_returns_zero() {
        let tensor = Tensor::new(Vec::new(), vec![0, 0]).unwrap();
        let result = trace_builtin(Value::Tensor(tensor)).expect("trace");
        assert_eq!(result, Value::Num(0.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn trace_rank_zero_and_singleton_vector_use_validated_matrix_extents() {
        let scalar = Tensor::new(vec![7.0], Vec::new()).expect("rank-zero real scalar");
        assert_eq!(
            trace_builtin(Value::Tensor(scalar)).expect("real scalar trace"),
            Value::Num(7.0)
        );

        let single = Tensor::from_f32(vec![3.5], vec![1]).expect("single vector");
        let Value::Tensor(single_trace) =
            trace_builtin(Value::Tensor(single)).expect("single vector trace")
        else {
            panic!("single trace must retain its typed carrier")
        };
        assert_eq!(single_trace.numeric_dtype(), NumericDType::F32);
        assert_eq!(single_trace.materialize_f64(), vec![3.5]);

        let exact = Tensor::new_integer(IntegerStorage::I16(vec![-4]), Vec::new())
            .expect("rank-zero exact scalar");
        assert_eq!(
            trace_builtin(Value::Tensor(exact)).expect("exact scalar trace"),
            Value::Num(-4.0)
        );

        let complex =
            ComplexTensor::new(vec![(2.0, -5.0)], Vec::new()).expect("rank-zero complex scalar");
        assert_eq!(
            trace_builtin(Value::ComplexTensor(complex)).expect("complex scalar trace"),
            Value::Complex(2.0, -5.0)
        );

        let complex_single =
            ComplexTensor::from_f32(vec![(1.25, -0.5)], vec![1]).expect("single complex vector");
        let Value::ComplexTensor(complex_trace) =
            trace_builtin(Value::ComplexTensor(complex_single)).expect("complex vector trace")
        else {
            panic!("single complex trace must retain its typed carrier")
        };
        assert_eq!(complex_trace.numeric_dtype(), NumericDType::F32);
        assert_eq!(complex_trace.materialize_f64(), vec![(1.25, -0.5)]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn trace_empty_typed_arrays_use_zero_matrix_extents() {
        let single = Tensor::from_f32(Vec::new(), vec![0, 0]).expect("empty single matrix");
        let Value::Tensor(single_trace) =
            trace_builtin(Value::Tensor(single)).expect("empty single trace")
        else {
            panic!("empty single trace must retain its typed carrier")
        };
        assert_eq!(single_trace.numeric_dtype(), NumericDType::F32);
        assert_eq!(single_trace.materialize_f64(), vec![0.0]);

        let exact = Tensor::new_integer(IntegerStorage::U16(Vec::new()), vec![0, 0])
            .expect("empty exact matrix");
        assert_eq!(
            trace_builtin(Value::Tensor(exact)).expect("empty exact trace"),
            Value::Num(0.0)
        );

        let complex =
            ComplexTensor::from_f32(Vec::new(), vec![0, 0]).expect("empty complex single matrix");
        let Value::ComplexTensor(complex_trace) =
            trace_builtin(Value::ComplexTensor(complex)).expect("empty complex single trace")
        else {
            panic!("empty complex single trace must retain its typed carrier")
        };
        assert_eq!(complex_trace.numeric_dtype(), NumericDType::F32);
        assert_eq!(complex_trace.materialize_f64(), vec![(0.0, 0.0)]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn trace_complex_matrix() {
        let data = vec![(1.0, 2.0), (3.0, -4.0), (5.0, 6.0), (7.0, 8.0)];
        let ct = ComplexTensor::new(data, vec![2, 2]).unwrap();
        let result = trace_builtin(Value::ComplexTensor(ct)).expect("trace");
        match result {
            Value::Complex(re, im) => {
                assert!((re - 8.0).abs() < 1e-12);
                assert!((im - 10.0).abs() < 1e-12);
            }
            other => panic!("expected complex result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn trace_char_array_promotes_to_double() {
        let chars = CharArray::from_unicode_scalars("a".chars().collect(), 1, 1).unwrap();
        let result = trace_builtin(Value::CharArray(chars)).expect("trace");
        match result {
            Value::Num(value) => assert!((value - ('a' as u32 as f64)).abs() < 1e-12),
            other => panic!("expected numeric result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn trace_char_array_square_matrix_uses_diagonal() {
        let chars = CharArray::from_code_units(
            [0xD800, 0x0042, 0x0043, 0xDC00]
                .map(runmat_value::Utf16CodeUnit::from_raw)
                .to_vec(),
            2,
            2,
        )
        .unwrap();
        let result = trace_builtin(Value::CharArray(chars)).expect("trace");
        match result {
            Value::Num(value) => {
                let expected = 111616.0;
                assert!((value - expected).abs() < 1e-12);
            }
            other => panic!("expected numeric result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn trace_gpu_provider_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0], vec![2, 2]).expect("tensor");
            let view = HostTensorView {
                data: &tensor.materialize_f64(),
                shape: tensor.shape(),
            };
            let handle = provider.upload(&view).expect("upload");
            let result = trace_builtin(Value::GpuTensor(handle)).expect("trace");
            match result {
                Value::GpuTensor(out) => {
                    let host = test_support::gather(Value::GpuTensor(out.clone())).expect("gather");
                    assert_eq!(host.shape(), [1, 1]);
                    assert_eq!(host.len(), 1);
                    assert!((host.materialize_f64()[0] - 6.0).abs() < 1e-12);
                    drop(out);
                }
                other => panic!("expected gpu result, got {other:?}"),
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn trace_gpu_fallback_uploads_scalar() {
        // Force gather path by using a zero-length diagonal
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(Vec::new(), vec![0, 0]).unwrap();
            let view = HostTensorView {
                data: &tensor.materialize_f64(),
                shape: tensor.shape(),
            };
            let handle = provider.upload(&view).expect("upload");
            let result = trace_builtin(Value::GpuTensor(handle)).expect("trace");
            match result {
                Value::GpuTensor(out) => {
                    let host = test_support::gather(Value::GpuTensor(out.clone())).expect("gather");
                    assert_eq!(host.materialize_f64(), vec![0.0]);
                    drop(out);
                }
                other => panic!("expected gpu result, got {other:?}"),
            }
        });
    }

    #[test]
    fn trace_single_resident_sum_restores_single_without_download() {
        test_support::with_test_provider(|provider| {
            let matrix = [16_777_216.0_f32, 2.0, 3.0, 1.0];
            let input = provider
                .upload_numeric(&runmat_accelerate_api::HostNumericTensorView {
                    data: runmat_accelerate_api::HostNumericDataView::F32(&matrix),
                    shape: &[2, 2],
                    storage: runmat_accelerate_api::GpuTensorStorage::Real,
                })
                .unwrap();
            provider.reset_telemetry();
            let result = trace_builtin(Value::GpuTensor(input.clone())).unwrap();
            let Value::GpuTensor(output) = result else {
                panic!("single trace should stay resident")
            };
            assert_eq!(
                runmat_accelerate_api::handle_precision(&output),
                Some(runmat_accelerate_api::ProviderPrecision::F32)
            );
            assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
            assert_eq!(
                test_support::gather(Value::GpuTensor(output))
                    .unwrap()
                    .materialize_f64(),
                vec![16_777_216.0],
            );
            input.allocation_lease().validate_live().unwrap();
        });
    }

    #[test]
    fn complex_resident_trace_uses_class_preserving_fallback() {
        test_support::with_test_provider(|provider| {
            let matrix = ComplexTensor::new(
                vec![(1.0, 2.0), (3.0, 4.0), (5.0, 6.0), (7.0, 8.0)],
                vec![2, 2],
            )
            .unwrap();
            let input = gpu_helpers::upload_complex_tensor(provider, &matrix).unwrap();
            let result = trace_builtin(Value::GpuTensor(input.clone())).unwrap();
            let Value::GpuTensor(output) = result else {
                panic!("complex trace must retain resident class")
            };
            let Value::ComplexTensor(host) = futures::executor::block_on(
                gpu_helpers::gather_value_async(&Value::GpuTensor(output)),
            )
            .unwrap() else {
                panic!("complex trace must preserve complex storage")
            };
            assert_eq!(host.materialize_f64(), vec![(8.0, 10.0)]);
            input.allocation_lease().validate_live().unwrap();
        });
    }

    #[test]
    fn asymmetric_single_complex_trace_matches_host_without_repeated_materialization() {
        let matrix = ComplexTensor::from_f32(
            vec![(1.0, 2.0), (100.0, 200.0), (-50.0, -60.0), (7.0, 8.0)],
            vec![2, 2],
        )
        .unwrap();
        let Value::ComplexTensor(host) =
            trace_builtin(Value::ComplexTensor(matrix.clone())).unwrap()
        else {
            panic!("single complex trace must preserve its typed carrier")
        };
        assert_eq!(host.numeric_dtype(), NumericDType::F32);
        assert_eq!(host.materialize_f64(), vec![(8.0, 10.0)]);

        test_support::with_test_provider(|provider| {
            let input = gpu_helpers::upload_complex_tensor(provider, &matrix).unwrap();
            let result = trace_builtin(Value::GpuTensor(input.clone())).unwrap();
            let Value::GpuTensor(output) = result else {
                panic!("single complex trace must stay typed on its owner")
            };
            let Value::ComplexTensor(gathered) = futures::executor::block_on(
                gpu_helpers::gather_value_async(&Value::GpuTensor(output)),
            )
            .unwrap() else {
                panic!("single complex trace must retain complex storage")
            };
            assert_eq!(gathered.numeric_dtype(), NumericDType::F32);
            assert_eq!(gathered.materialize_f64(), host.materialize_f64());
            input.allocation_lease().validate_live().unwrap();
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn trace_gpu_integer_uses_exact_host_sum_and_returns_double() {
        test_support::with_test_provider(|provider| {
            let handle = provider
                .upload_integer(&runmat_accelerate_api::HostIntegerTensorView {
                    data: runmat_accelerate_api::HostIntegerDataView::U64(&[3, 0, 0, 7]),
                    shape: &[2, 2],
                })
                .expect("upload integer matrix");
            let result = trace_builtin(Value::GpuTensor(handle)).expect("trace");
            let Value::GpuTensor(out) = result else {
                panic!("expected resident double result");
            };
            assert_eq!(runmat_accelerate_api::handle_integer_type(&out), None);
            let host = test_support::gather(Value::GpuTensor(out.clone())).expect("gather");
            assert_eq!(host.shape(), [1, 1]);
            assert_eq!(host.materialize_f64(), vec![10.0]);
            drop(out);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn trace_integer_promotes_to_double() {
        let value = Value::Int(IntValue::I32(5));
        let result = trace_builtin(value).expect("trace");
        assert_eq!(result, Value::Num(5.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn trace_bool_promotes_to_double() {
        let result = trace_builtin(Value::Bool(true)).expect("trace");
        assert_eq!(result, Value::Num(1.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn trace_logical_array_matches_numeric() {
        let data = vec![1, 0, 0, 0, 1, 0, 0, 0, 1];
        let logical = LogicalArray::new(data, vec![3, 3]).expect("logical");
        let result = trace_builtin(Value::LogicalArray(logical)).expect("trace");
        assert_eq!(result, Value::Num(3.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn trace_complex_empty_matrix_returns_zero() {
        let complex = ComplexTensor::new(Vec::new(), vec![0, 0]).expect("complex");
        let result = trace_builtin(Value::ComplexTensor(complex)).expect("trace");
        match result {
            Value::Complex(re, im) => {
                assert_eq!(re, 0.0);
                assert_eq!(im, 0.0);
            }
            other => panic!("expected complex zero, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn trace_rejects_higher_dimensional_inputs() {
        let tensor = Tensor::new(vec![1.0; 8], vec![2, 2, 2]).unwrap();
        let err = unwrap_error(trace_builtin(Value::Tensor(tensor)).unwrap_err());
        assert_eq!(err.identifier(), TRACE_ERROR_INVALID_INPUT.identifier);
        assert_eq!(err.message(), TRACE_ERROR_INVALID_INPUT.message);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn trace_wgpu_matches_cpu() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 8.0], vec![2, 2]).unwrap();
        let cpu = trace_numeric(Value::Tensor(tensor.clone())).unwrap();
        let view = HostTensorView {
            data: &tensor.materialize_f64(),
            shape: tensor.shape(),
        };
        let handle = runmat_accelerate_api::provider()
            .unwrap()
            .upload(&view)
            .unwrap();
        let gpu = trace_builtin(Value::GpuTensor(handle)).unwrap();
        let gathered = test_support::gather(gpu).expect("gather");
        let expected = match cpu {
            Value::Num(n) => n,
            Value::Tensor(t) if !t.materialize_f64().is_empty() => t.materialize_f64()[0],
            Value::Tensor(_) => 0.0,
            other => panic!("unexpected cpu comparison value {other:?}"),
        };
        assert_eq!(gathered.shape(), [1, 1]);
        let actual = gathered
            .materialize_f64()
            .first()
            .copied()
            .expect("gathered tensor should contain one element");
        assert!((expected - actual).abs() < 1e-9);
    }

    fn trace_builtin(value: Value) -> BuiltinResult<Value> {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        block_on(super::trace_builtin(value))
    }

    #[test]
    fn trace_integer_extension_is_rejected_in_matlab_mode() {
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let error = block_on(super::trace_builtin(Value::Int(IntValue::I32(5)))).unwrap_err();
        assert_eq!(
            error.identifier(),
            TRACE_INTEGER_INPUT_EXTENSION.error_identifier
        );
    }

    #[test]
    fn trace_rejects_inexact_wide_integer_sum() {
        let tensor = Tensor::new_integer(
            IntegerStorage::U64(vec![9_007_199_254_740_993, 0, 0, 0]),
            vec![2, 2],
        )
        .unwrap();
        let error = trace_builtin(Value::Tensor(tensor)).unwrap_err();
        assert!(error.message().contains("exactly representable"));
    }

    #[test]
    fn trace_supports_sparse_floating_and_exact_integer_matrices() {
        let sparse = SparseTensor::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![2.5, 3.5]).unwrap();
        assert_eq!(
            trace_builtin(Value::SparseTensor(sparse)).unwrap(),
            Value::Num(6.0)
        );

        let sparse = SparseTensor::new_integer(
            2,
            2,
            vec![0, 1, 2],
            vec![0, 1],
            IntegerStorage::U64(vec![9_007_199_254_740_992, 1]),
        )
        .unwrap();
        let error = trace_builtin(Value::SparseTensor(sparse)).unwrap_err();
        assert!(error.message().contains("exactly representable"));
    }

    #[test]
    fn trace_preserves_complex_sparse_diagonal() {
        let sparse = SparseTensor::new_complex(
            2,
            2,
            vec![0, 1, 2],
            vec![0, 1],
            vec![(2.5, -1.0), (3.5, 4.0)],
        )
        .expect("complex sparse");
        assert_eq!(
            trace_builtin(Value::SparseTensor(sparse)).expect("trace"),
            Value::Complex(6.0, 3.0)
        );

        let sparse_single = SparseTensor::new_complex_f32(
            2,
            2,
            vec![0, 1, 2],
            vec![0, 1],
            vec![(2.5, -1.0), (3.5, 4.0)],
        )
        .expect("complex single sparse");
        let Value::ComplexTensor(single) =
            trace_builtin(Value::SparseTensor(sparse_single)).expect("single trace")
        else {
            panic!("single complex sparse trace must retain its typed carrier")
        };
        assert_eq!(single.numeric_dtype(), NumericDType::F32);
        assert_eq!(single.materialize_f64(), vec![(6.0, 3.0)]);
    }
}
