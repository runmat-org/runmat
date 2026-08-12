//! MATLAB-compatible `pagetranspose` builtin.

use crate::builtins::array::shape::permute::{
    permute_complex_tensor, permute_gpu, permute_logical_array, permute_string_array,
    permute_tensor,
};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::tensor;
use crate::builtins::math::linalg::ops::transpose_real_sparse_tensor;
use crate::builtins::math::linalg::type_resolvers::page_transpose_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    CellArray, Value,
};
use runmat_macros::runtime_builtin;

const NAME: &str = "pagetranspose";

const PAGETRANSPOSE_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input value with the first two dimensions swapped on every page.",
}];

const PAGETRANSPOSE_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input scalar or array value.",
}];

const PAGETRANSPOSE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "Y = pagetranspose(X)",
    inputs: &PAGETRANSPOSE_INPUTS,
    outputs: &PAGETRANSPOSE_OUTPUT,
}];

const PAGETRANSPOSE_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PAGETRANSPOSE.INVALID_ARGUMENT",
    identifier: Some("RunMat:pagetranspose:InvalidArgument"),
    when: "Call does not provide exactly one input argument.",
    message: "pagetranspose: expected exactly one input argument",
};

const PAGETRANSPOSE_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PAGETRANSPOSE.INVALID_INPUT",
    identifier: Some("RunMat:pagetranspose:InvalidInput"),
    when: "Input type is unsupported for page-wise transpose.",
    message: "pagetranspose: unsupported input type",
};

const PAGETRANSPOSE_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PAGETRANSPOSE.INTERNAL",
    identifier: Some("RunMat:pagetranspose:Internal"),
    when: "Runtime cannot materialize the page-wise transpose output.",
    message: "pagetranspose: internal runtime failure",
};

const PAGETRANSPOSE_ERRORS: [BuiltinErrorDescriptor; 3] = [
    PAGETRANSPOSE_ERROR_INVALID_ARGUMENT,
    PAGETRANSPOSE_ERROR_INVALID_INPUT,
    PAGETRANSPOSE_ERROR_INTERNAL,
];

pub const PAGETRANSPOSE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &PAGETRANSPOSE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &PAGETRANSPOSE_ERRORS,
};

#[runmat_macros::register_gpu_spec(
    builtin_path = "crate::builtins::math::linalg::ops::pagetranspose"
)]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: NAME,
    op_kind: GpuOpKind::Custom("permute"),
    supported_precisions: &[
        ScalarType::F32,
        ScalarType::F64,
        ScalarType::I32,
        ScalarType::Bool,
    ],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("permute")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes:
        "Equivalent to permute(X,[2 1 3:ndims(X)]); uses the provider permute hook when available and otherwise falls back to gather->host transpose->upload.",
};

#[runmat_macros::register_fusion_spec(
    builtin_path = "crate::builtins::math::linalg::ops::pagetranspose"
)]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: NAME,
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Page-wise transpose acts as a fusion boundary because it changes array layout.",
};

fn builtin_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    builtin_error_with_message(error.message, error)
}

fn builtin_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn invalid_argument(message: impl Into<String>) -> RuntimeError {
    builtin_error_with_message(message, &PAGETRANSPOSE_ERROR_INVALID_ARGUMENT)
}

fn invalid_input(message: impl Into<String>) -> RuntimeError {
    builtin_error_with_message(message, &PAGETRANSPOSE_ERROR_INVALID_INPUT)
}

fn internal_error(message: impl Into<String>) -> RuntimeError {
    builtin_error_with_message(message, &PAGETRANSPOSE_ERROR_INTERNAL)
}

#[runtime_builtin(
    name = "pagetranspose",
    category = "math/linalg/ops",
    summary = "Apply the nonconjugate transpose to each matrix page of an array.",
    keywords = "pagetranspose,page-wise transpose,non-conjugate,permute,gpu",
    accel = "custom",
    type_resolver(page_transpose_type),
    descriptor(crate::builtins::math::linalg::ops::pagetranspose::PAGETRANSPOSE_DESCRIPTOR),
    builtin_path = "crate::builtins::math::linalg::ops::pagetranspose"
)]
async fn pagetranspose_builtin(mut args: Vec<Value>) -> BuiltinResult<Value> {
    let value = match args.len() {
        0 => return Err(builtin_error(&PAGETRANSPOSE_ERROR_INVALID_ARGUMENT)),
        1 => args.remove(0),
        _ => return Err(invalid_argument("pagetranspose: too many input arguments")),
    };
    pagetranspose_value(value).await
}

async fn pagetranspose_value(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::GpuTensor(handle) => {
            let order = page_transpose_order(handle.shape.len());
            permute_gpu(NAME, handle, &order).await
        }
        Value::Tensor(t) => {
            let order = page_transpose_order_for_shape_len(t.shape.len());
            Ok(tensor::tensor_into_value(permute_tensor(NAME, t, &order)?))
        }
        Value::SparseTensor(s) => Ok(Value::SparseTensor(
            transpose_real_sparse_tensor(s).map_err(|e| internal_error(format!("{NAME}: {e}")))?,
        )),
        Value::ComplexTensor(ct) => {
            let order = page_transpose_order_for_shape_len(ct.shape.len());
            Ok(Value::ComplexTensor(permute_complex_tensor(
                NAME, ct, &order,
            )?))
        }
        Value::LogicalArray(la) => {
            let order = page_transpose_order_for_shape_len(la.shape.len());
            Ok(Value::LogicalArray(permute_logical_array(
                NAME, la, &order,
            )?))
        }
        Value::CharArray(ca) => Ok(Value::CharArray(pagetranspose_char_array(ca)?)),
        Value::StringArray(sa) => {
            let order = page_transpose_order_for_shape_len(sa.shape.len());
            Ok(Value::StringArray(permute_string_array(NAME, sa, &order)?))
        }
        Value::Cell(ca) => Ok(Value::Cell(pagetranspose_cell_array(ca)?)),
        Value::Struct(s) => Ok(Value::Struct(s)),
        Value::Complex(re, im) => Ok(Value::Complex(re, im)),
        Value::Num(n) => Ok(Value::Num(n)),
        Value::Int(i) => Ok(Value::Int(i)),
        Value::Bool(b) => Ok(Value::Bool(b)),
        Value::String(s) => Ok(Value::String(s)),
        other => Err(invalid_input(format!(
            "pagetranspose: unsupported input type {other:?}"
        ))),
    }
}

fn page_transpose_order(rank: usize) -> Vec<usize> {
    page_transpose_order_for_shape_len(rank)
}

fn page_transpose_order_for_shape_len(rank: usize) -> Vec<usize> {
    let rank = rank.max(2);
    let mut order: Vec<usize> = (1..=rank).collect();
    order.swap(0, 1);
    order
}

fn pagetranspose_cell_array(ca: CellArray) -> BuiltinResult<CellArray> {
    if ca.shape.len() > 2 {
        let order = page_transpose_order_for_shape_len(ca.shape.len());
        let CellArray { data, shape, .. } = ca;
        let (data, shape) = permute_cell_data(&data, &shape, &order)?;
        return CellArray::new_with_shape(data, shape)
            .map_err(|e| internal_error(format!("{NAME}: {e}")));
    }

    let rows = ca.rows;
    let cols = ca.cols;
    let mut out = Vec::with_capacity(ca.data.len());
    for r_out in 0..cols {
        for c_out in 0..rows {
            let src = c_out * cols + r_out;
            out.push(ca.data[src].clone());
        }
    }
    CellArray::new(out, cols, rows).map_err(|e| internal_error(format!("{NAME}: {e}")))
}

fn pagetranspose_char_array(
    ca: runmat_builtins::CharArray,
) -> BuiltinResult<runmat_builtins::CharArray> {
    let rows = ca.rows;
    let cols = ca.cols;
    if ca.data.is_empty() {
        return runmat_builtins::CharArray::new(Vec::new(), cols, rows)
            .map_err(|e| internal_error(format!("{NAME}: {e}")));
    }
    let mut out = vec!['\0'; ca.data.len()];
    for r in 0..rows {
        for c in 0..cols {
            let src = r * cols + c;
            let dst = c * rows + r;
            if src < ca.data.len() && dst < out.len() {
                out[dst] = ca.data[src];
            }
        }
    }
    runmat_builtins::CharArray::new(out, cols, rows)
        .map_err(|e| internal_error(format!("{NAME}: {e}")))
}

fn permute_cell_data(
    data: &[Value],
    shape: &[usize],
    order: &[usize],
) -> BuiltinResult<(Vec<Value>, Vec<usize>)> {
    let mut src_shape = shape.to_vec();
    if src_shape.len() < order.len() {
        src_shape.extend(std::iter::repeat_n(1, order.len() - src_shape.len()));
    }
    let total = src_shape
        .iter()
        .try_fold(1usize, |acc, dim| acc.checked_mul(*dim));
    if total != Some(data.len()) {
        return Err(internal_error(format!(
            "{NAME}: cell data length does not match shape product"
        )));
    }

    let zero_based: Vec<usize> = order.iter().map(|idx| idx - 1).collect();
    let mut dst_shape = vec![0usize; order.len()];
    for (dst_dim, &src_dim) in zero_based.iter().enumerate() {
        dst_shape[dst_dim] = src_shape[src_dim];
    }

    let src_strides = compute_strides(&src_shape);
    let dst_total: usize = dst_shape.iter().product();
    let mut dst_coords = vec![0usize; order.len()];
    let mut src_coords = vec![0usize; order.len()];
    let mut out = Vec::with_capacity(dst_total);

    for dst_index in 0..dst_total {
        let mut rem = dst_index;
        for (dim, &size) in dst_shape.iter().enumerate() {
            dst_coords[dim] = if size == 0 { 0 } else { rem % size };
            if size != 0 {
                rem /= size;
            }
        }
        for (dst_dim, &src_dim) in zero_based.iter().enumerate() {
            src_coords[src_dim] = dst_coords[dst_dim];
        }
        let src_index = src_coords
            .iter()
            .enumerate()
            .map(|(dim, coord)| coord * src_strides[dim])
            .sum::<usize>();
        out.push(data[src_index].clone());
    }

    Ok((out, dst_shape))
}

fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = Vec::with_capacity(shape.len());
    let mut stride = 1usize;
    for &dim in shape {
        strides.push(stride);
        stride = stride.saturating_mul(dim);
    }
    strides
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_builtins::{
        CellArray, CharArray, ComplexTensor, IntegerComplexStorage, IntegerStorage, LogicalArray,
        ResolveContext, SparseTensor, StringArray, StructValue, Tensor, Type,
    };

    fn call(args: Vec<Value>) -> BuiltinResult<Value> {
        block_on(super::pagetranspose_builtin(args))
    }

    fn call_one(value: Value) -> BuiltinResult<Value> {
        call(vec![value])
    }

    #[test]
    fn pagetranspose_preserves_typed_complex_integer_components_exactly() {
        let input = ComplexTensor::new_integer(
            IntegerComplexStorage::new(
                IntegerStorage::U64(vec![u64::MAX, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 1_u64 << 63]),
                IntegerStorage::U64((1..=12).collect()),
            )
            .expect("storage"),
            vec![2, 3, 2],
        )
        .expect("tensor");

        let Value::ComplexTensor(result) =
            call_one(Value::ComplexTensor(input)).expect("pagetranspose")
        else {
            panic!("expected complex tensor");
        };
        let storage = result.integer_storage().expect("exact integer storage");
        assert_eq!(result.shape, vec![3, 2, 2]);
        assert_eq!(
            storage.real,
            IntegerStorage::U64(vec![u64::MAX, 3, 5, 2, 4, 6, 7, 9, 11, 8, 10, 1_u64 << 63,])
        );
        assert_eq!(
            storage.imag,
            IntegerStorage::U64(vec![1, 3, 5, 2, 4, 6, 7, 9, 11, 8, 10, 12])
        );
    }

    fn tensor(data: &[f64], shape: &[usize]) -> Tensor {
        Tensor::new(data.to_vec(), shape.to_vec()).unwrap()
    }

    #[test]
    fn descriptor_covers_core_form_and_errors() {
        assert_eq!(
            PAGETRANSPOSE_DESCRIPTOR.signatures[0].label,
            "Y = pagetranspose(X)"
        );
        let codes: Vec<&str> = PAGETRANSPOSE_DESCRIPTOR
            .errors
            .iter()
            .map(|error| error.code)
            .collect();
        assert!(codes.contains(&"RM.PAGETRANSPOSE.INVALID_ARGUMENT"));
        assert!(codes.contains(&"RM.PAGETRANSPOSE.INVALID_INPUT"));
        assert!(codes.contains(&"RM.PAGETRANSPOSE.INTERNAL"));
    }

    #[test]
    fn rejects_wrong_arity_with_stable_identifier() {
        let err = call(Vec::new()).expect_err("missing input should fail");
        assert_eq!(
            err.identifier(),
            PAGETRANSPOSE_ERROR_INVALID_ARGUMENT.identifier
        );

        let err =
            call(vec![Value::Num(1.0), Value::Num(2.0)]).expect_err("extra input should fail");
        assert_eq!(
            err.identifier(),
            PAGETRANSPOSE_ERROR_INVALID_ARGUMENT.identifier
        );
    }

    #[test]
    fn type_resolver_swaps_first_two_dims() {
        let out = page_transpose_type(
            &[Type::Tensor {
                shape: Some(vec![Some(2), Some(3), Some(4)]),
            }],
            &ResolveContext::new(Vec::new()),
        );
        assert_eq!(
            out,
            Type::Tensor {
                shape: Some(vec![Some(3), Some(2), Some(4)])
            }
        );
    }

    #[test]
    fn type_resolver_preserves_scalar_struct_type() {
        let out = page_transpose_type(
            &[Type::Struct {
                known_fields: Some(vec!["name".to_string()]),
            }],
            &ResolveContext::new(Vec::new()),
        );
        assert_eq!(
            out,
            Type::Struct {
                known_fields: Some(vec!["name".to_string()])
            }
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn transposes_each_numeric_page() {
        let data: Vec<f64> = (1..=12).map(|n| n as f64).collect();
        let value = call_one(Value::Tensor(tensor(&data, &[2, 3, 2]))).expect("pagetranspose");
        match value {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![3, 2, 2]);
                assert_eq!(
                    out.materialize_f64(),
                    vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0, 7.0, 9.0, 11.0, 8.0, 10.0, 12.0]
                );
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn transposes_matrix_like_transpose() {
        let value = call_one(Value::Tensor(tensor(
            &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0],
            &[2, 3],
        )))
        .expect("pagetranspose");
        match value {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![3, 2]);
                assert_eq!(out.materialize_f64(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn preserves_complex_values_without_conjugation() {
        let ct = ComplexTensor::new(vec![(1.0, 2.0), (3.0, -4.0)], vec![1, 2]).unwrap();
        let value = call_one(Value::ComplexTensor(ct)).expect("pagetranspose");
        match value {
            Value::ComplexTensor(out) => {
                assert_eq!(out.shape, vec![2, 1]);
                assert_eq!(out.materialize_f64(), vec![(1.0, 2.0), (3.0, -4.0)]);
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn handles_logical_char_and_cell_arrays() {
        let mask = LogicalArray::new(vec![1, 0, 0, 1, 1, 0], vec![2, 3]).unwrap();
        match call_one(Value::LogicalArray(mask)).expect("pagetranspose") {
            Value::LogicalArray(out) => {
                assert_eq!(out.shape, vec![3, 2]);
                assert_eq!(out.data, vec![1, 0, 1, 0, 1, 0]);
            }
            other => panic!("expected logical array, got {other:?}"),
        }

        let chars = CharArray::new("runmat".chars().collect(), 2, 3).unwrap();
        match call_one(Value::CharArray(chars)).expect("pagetranspose") {
            Value::CharArray(out) => {
                assert_eq!(out.rows, 3);
                assert_eq!(out.cols, 2);
                assert_eq!(out.data, vec!['r', 'm', 'u', 'a', 'n', 't']);
            }
            other => panic!("expected char array, got {other:?}"),
        }

        let cells = CellArray::new(
            vec![
                Value::from(1),
                Value::from(2),
                Value::from(3),
                Value::from(4),
                Value::from(5),
                Value::from(6),
            ],
            2,
            3,
        )
        .unwrap();
        match call_one(Value::Cell(cells)).expect("pagetranspose") {
            Value::Cell(out) => {
                assert_eq!(out.shape, vec![3, 2]);
                assert_eq!(
                    out.data,
                    vec![
                        Value::from(1),
                        Value::from(4),
                        Value::from(2),
                        Value::from(5),
                        Value::from(3),
                        Value::from(6),
                    ]
                );
            }
            other => panic!("expected cell array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn handles_string_sparse_struct_and_empty_arrays() {
        let strings = StringArray::new(
            vec!["a".into(), "b".into(), "c".into(), "d".into()],
            vec![2, 2],
        )
        .unwrap();
        match call_one(Value::StringArray(strings)).expect("pagetranspose") {
            Value::StringArray(out) => {
                assert_eq!(out.shape, vec![2, 2]);
                assert_eq!(out.data, vec!["a", "c", "b", "d"]);
            }
            other => panic!("expected string array, got {other:?}"),
        }

        let sparse =
            SparseTensor::new(2, 3, vec![0, 1, 2, 3], vec![1, 0, 1], vec![4.0, 5.0, 6.0]).unwrap();
        match call_one(Value::SparseTensor(sparse)).expect("pagetranspose") {
            Value::SparseTensor(out) => {
                assert_eq!(out.rows, 3);
                assert_eq!(out.cols, 2);
                assert_eq!(out.col_ptrs, vec![0, 1, 3]);
                assert_eq!(out.row_indices, vec![1, 0, 2]);
                assert_eq!(out.materialize_f64(), vec![5.0, 4.0, 6.0]);
            }
            other => panic!("expected sparse tensor, got {other:?}"),
        }

        let mut structure = StructValue::new();
        structure.insert("answer", Value::from(42));
        assert_eq!(
            call_one(Value::Struct(structure.clone())).unwrap(),
            Value::Struct(structure)
        );

        match call_one(Value::Tensor(tensor(&[], &[0, 3]))).expect("pagetranspose") {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![3, 0]);
                assert!(out.materialize_f64().is_empty());
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn scalar_values_are_identity() {
        assert_eq!(call_one(Value::Num(4.0)).unwrap(), Value::Num(4.0));
        assert_eq!(call_one(Value::Bool(true)).unwrap(), Value::Bool(true));
        assert_eq!(
            call_one(Value::Complex(1.0, -2.0)).unwrap(),
            Value::Complex(1.0, -2.0)
        );
    }
}
