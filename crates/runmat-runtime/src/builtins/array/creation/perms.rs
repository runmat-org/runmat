//! MATLAB-compatible `perms` builtin.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    CellArray, CharArray, ComplexTensor, LogicalArray, ResolveContext, StringArray, Tensor, Type,
    Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::{
    gpu_helpers,
    spec::{
        BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
        ReductionNaN, ResidencyPolicy, ShapeRequirements,
    },
};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "perms";
const MAX_INPUT_LEN: usize = 10;
const MAX_OUTPUT_ELEMENTS: usize = 50_000_000;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::array::creation::perms")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: BUILTIN_NAME,
    op_kind: GpuOpKind::Custom("array_construct"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Deterministic permutation materialisation runs on the host; gpuArray inputs are gathered before constructing the output.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::array::creation::perms")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: BUILTIN_NAME,
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "perms materialises factorial-size outputs and is not eligible for elementwise fusion.",
};

fn perms_type(args: &[Type], _context: &ResolveContext) -> Type {
    match args.first() {
        Some(Type::Logical { .. }) | Some(Type::Bool) => Type::logical(),
        Some(Type::String) => Type::String,
        Some(Type::Cell { element_type, .. }) => Type::Cell {
            element_type: element_type.clone(),
            length: None,
        },
        Some(Type::Unknown) | None => Type::Unknown,
        _ => Type::tensor(),
    }
}

const PERMS_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "P",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Matrix whose rows contain every permutation of the input vector.",
}];

const PERMS_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "v",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input vector to permute.",
}];

const PERMS_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "P = perms(v)",
    inputs: &PERMS_INPUTS,
    outputs: &PERMS_OUTPUT,
}];

const ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PERMS.INVALID_INPUT",
    identifier: Some("RunMat:perms:InvalidInput"),
    when: "The input is missing, not a vector, or has an unsupported type.",
    message: "perms: input must be a vector",
};

const ERROR_TOO_LARGE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PERMS.TOO_LARGE",
    identifier: Some("RunMat:perms:TooLarge"),
    when: "The factorial-size output would exceed RunMat's supported materialisation limit.",
    message: "perms: output is too large",
};

const ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PERMS.INTERNAL",
    identifier: Some("RunMat:perms:Internal"),
    when: "Output allocation, GPU gather, or container construction failed.",
    message: "perms: internal error",
};

const PERMS_ERRORS: [BuiltinErrorDescriptor; 3] =
    [ERROR_INVALID_INPUT, ERROR_TOO_LARGE, ERROR_INTERNAL];

pub const PERMS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &PERMS_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &PERMS_ERRORS,
};

#[runtime_builtin(
    name = "perms",
    category = "array/creation",
    summary = "Return all permutations of a vector as rows.",
    keywords = "perms,permutation,combinatorics,vector",
    accel = "array_construct",
    type_resolver(perms_type),
    descriptor(crate::builtins::array::creation::perms::PERMS_DESCRIPTOR),
    builtin_path = "crate::builtins::array::creation::perms"
)]
async fn perms_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    if !rest.is_empty() {
        return Err(perms_error_with(
            &ERROR_INVALID_INPUT,
            "perms: too many input arguments",
        ));
    }
    evaluate(value).await
}

async fn evaluate(value: Value) -> BuiltinResult<Value> {
    if let Value::GpuTensor(handle) = value {
        let gathered = gpu_helpers::gather_value_async(&Value::GpuTensor(handle))
            .await
            .map_err(|e| perms_error_with(&ERROR_INTERNAL, format!("perms: {e}")))?;
        return evaluate_host(gathered);
    }
    evaluate_host(value)
}

fn evaluate_host(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::Num(n) => Ok(Value::Num(n)),
        Value::Int(i) => Ok(Value::Int(i)),
        Value::Complex(re, im) => Ok(Value::Complex(re, im)),
        Value::Bool(flag) => Ok(Value::Bool(flag)),
        Value::Tensor(tensor) => perms_tensor(tensor),
        Value::ComplexTensor(tensor) => perms_complex_tensor(tensor),
        Value::LogicalArray(array) => perms_logical(array),
        Value::CharArray(chars) => perms_chars(chars),
        Value::String(s) => Ok(Value::String(s)),
        Value::StringArray(array) => perms_strings(array),
        Value::Cell(cell) => perms_cells(cell),
        Value::SparseTensor(_)
        | Value::Symbolic(_)
        | Value::SymbolicArray(_)
        | Value::Struct(_)
        | Value::Object(_)
        | Value::HandleObject(_)
        | Value::Listener(_)
        | Value::FunctionHandle(_)
        | Value::ExternalFunctionHandle(_)
        | Value::MethodFunctionHandle(_)
        | Value::BoundFunctionHandle { .. }
        | Value::Closure(_)
        | Value::GpuTensor(_)
        | Value::ClassRef(_)
        | Value::MException(_)
        | Value::OutputList(_) => Err(perms_error(&ERROR_INVALID_INPUT)),
    }
}

fn perms_tensor(tensor: Tensor) -> BuiltinResult<Value> {
    let elements = vector_len(&tensor.shape)?;
    let rows = checked_output_rows(elements)?;
    let data = permuted_columns(&tensor.data, rows, elements)?;
    Tensor::new_with_dtype(data, vec![rows, elements], tensor.dtype)
        .map(Value::Tensor)
        .map_err(|e| perms_error_with(&ERROR_INTERNAL, format!("perms: {e}")))
}

fn perms_complex_tensor(tensor: ComplexTensor) -> BuiltinResult<Value> {
    let elements = vector_len(&tensor.shape)?;
    let rows = checked_output_rows(elements)?;
    let data = permuted_columns(&tensor.data, rows, elements)?;
    ComplexTensor::new(data, vec![rows, elements])
        .map(Value::ComplexTensor)
        .map_err(|e| perms_error_with(&ERROR_INTERNAL, format!("perms: {e}")))
}

fn perms_logical(array: LogicalArray) -> BuiltinResult<Value> {
    let elements = vector_len(&array.shape)?;
    let rows = checked_output_rows(elements)?;
    let data = permuted_columns(&array.data, rows, elements)?;
    LogicalArray::new(data, vec![rows, elements])
        .map(Value::LogicalArray)
        .map_err(|e| perms_error_with(&ERROR_INTERNAL, format!("perms: {e}")))
}

fn perms_chars(chars: CharArray) -> BuiltinResult<Value> {
    let elements = vector_len(&[chars.rows, chars.cols])?;
    let rows = checked_output_rows(elements)?;
    let data = permuted_rows(&chars.data, rows, elements)?;
    CharArray::new(data, rows, elements)
        .map(Value::CharArray)
        .map_err(|e| perms_error_with(&ERROR_INTERNAL, format!("perms: {e}")))
}

fn perms_strings(array: StringArray) -> BuiltinResult<Value> {
    let elements = vector_len(&array.shape)?;
    let rows = checked_output_rows(elements)?;
    let data = permuted_columns(&array.data, rows, elements)?;
    StringArray::new(data, vec![rows, elements])
        .map(Value::StringArray)
        .map_err(|e| perms_error_with(&ERROR_INTERNAL, format!("perms: {e}")))
}

fn perms_cells(cell: CellArray) -> BuiltinResult<Value> {
    let elements = vector_len(&cell.shape)?;
    let rows = checked_output_rows(elements)?;
    let data = permuted_rows(&cell.data, rows, elements)?;
    CellArray::new(data, rows, elements)
        .map(Value::Cell)
        .map_err(|e| perms_error_with(&ERROR_INTERNAL, format!("perms: {e}")))
}

fn vector_len(shape: &[usize]) -> BuiltinResult<usize> {
    if shape.len() > 2 {
        return Err(perms_error(&ERROR_INVALID_INPUT));
    }
    match shape {
        [] => Err(perms_error(&ERROR_INVALID_INPUT)),
        [0, 0] => Ok(0),
        [n] => Ok(*n),
        [rows, cols] if *rows == 1 || *cols == 1 => Ok(rows.saturating_mul(*cols)),
        _ => Err(perms_error(&ERROR_INVALID_INPUT)),
    }
}

fn checked_output_rows(elements: usize) -> BuiltinResult<usize> {
    if elements > MAX_INPUT_LEN {
        return Err(perms_error_with(
            &ERROR_TOO_LARGE,
            format!("perms: input length {elements} exceeds supported limit {MAX_INPUT_LEN}"),
        ));
    }
    let rows = factorial(elements)?;
    let total = rows.checked_mul(elements).ok_or_else(|| {
        perms_error_with(&ERROR_TOO_LARGE, "perms: output element count overflows")
    })?;
    if total > MAX_OUTPUT_ELEMENTS {
        return Err(perms_error_with(
            &ERROR_TOO_LARGE,
            "perms: output exceeds supported materialisation limit",
        ));
    }
    Ok(rows)
}

fn factorial(n: usize) -> BuiltinResult<usize> {
    let mut acc = 1usize;
    for value in 2..=n {
        acc = acc.checked_mul(value).ok_or_else(|| {
            perms_error_with(&ERROR_TOO_LARGE, "perms: factorial output size overflows")
        })?;
    }
    Ok(acc)
}

fn permuted_columns<T: Clone>(input: &[T], rows: usize, cols: usize) -> BuiltinResult<Vec<T>> {
    if cols == 0 {
        return Ok(Vec::new());
    }

    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| perms_error_with(&ERROR_TOO_LARGE, "perms: output size overflows"))?;
    let mut output = vec![input[0].clone(); total];
    let mut indices: Vec<usize> = (0..cols).rev().collect();

    for row in 0..rows {
        for (col, &source_index) in indices.iter().enumerate() {
            output[col * rows + row] = input[source_index].clone();
        }
        if row + 1 < rows && !previous_permutation(&mut indices) {
            return Err(perms_error_with(
                &ERROR_INTERNAL,
                "perms: permutation sequence ended early",
            ));
        }
    }

    Ok(output)
}

fn permuted_rows<T: Clone>(input: &[T], rows: usize, cols: usize) -> BuiltinResult<Vec<T>> {
    if cols == 0 {
        return Ok(Vec::new());
    }

    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| perms_error_with(&ERROR_TOO_LARGE, "perms: output size overflows"))?;
    let mut output = vec![input[0].clone(); total];
    let mut indices: Vec<usize> = (0..cols).rev().collect();

    for row in 0..rows {
        for (col, &source_index) in indices.iter().enumerate() {
            output[row * cols + col] = input[source_index].clone();
        }
        if row + 1 < rows && !previous_permutation(&mut indices) {
            return Err(perms_error_with(
                &ERROR_INTERNAL,
                "perms: permutation sequence ended early",
            ));
        }
    }

    Ok(output)
}

fn previous_permutation(values: &mut [usize]) -> bool {
    if values.len() < 2 {
        return false;
    }

    let Some(pivot) = (0..values.len() - 1).rfind(|&i| values[i] > values[i + 1]) else {
        return false;
    };
    let swap_with = (pivot + 1..values.len())
        .rfind(|&i| values[i] < values[pivot])
        .expect("pivot guarantees a smaller suffix value");
    values.swap(pivot, swap_with);
    values[pivot + 1..].reverse();
    true
}

fn perms_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    perms_error_with(error, error.message)
}

fn perms_error_with(
    error: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::{gpu_helpers, test_support};
    use futures::executor::block_on;
    use runmat_builtins::{IntValue, NumericDType};

    fn call(value: Value) -> BuiltinResult<Value> {
        block_on(super::perms_builtin(value, Vec::new()))
    }

    fn tensor_rows(tensor: &Tensor) -> Vec<Vec<f64>> {
        (0..tensor.rows)
            .map(|row| {
                (0..tensor.cols)
                    .map(|col| tensor.data[col * tensor.rows + row])
                    .collect()
            })
            .collect()
    }

    fn complex_rows(tensor: &ComplexTensor) -> Vec<Vec<(f64, f64)>> {
        (0..tensor.rows)
            .map(|row| {
                (0..tensor.cols)
                    .map(|col| tensor.data[col * tensor.rows + row])
                    .collect()
            })
            .collect()
    }

    #[test]
    fn descriptor_exposes_public_signature_and_errors() {
        assert_eq!(PERMS_DESCRIPTOR.signatures[0].label, "P = perms(v)");
        assert_eq!(PERMS_DESCRIPTOR.output_mode, BuiltinOutputMode::Fixed);
        assert!(PERMS_DESCRIPTOR
            .errors
            .iter()
            .any(|err| err.identifier == Some("RunMat:perms:TooLarge")));
    }

    #[test]
    fn type_resolver_returns_container_compatible_types() {
        assert_eq!(
            super::perms_type(
                &[Type::Logical { shape: None }],
                &ResolveContext::new(Vec::new())
            ),
            Type::logical()
        );
        assert_eq!(
            super::perms_type(&[Type::Num], &ResolveContext::new(Vec::new())),
            Type::tensor()
        );
    }

    #[test]
    fn numeric_vector_uses_matlab_reverse_lexicographic_order() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
        let Value::Tensor(out) = call(Value::Tensor(tensor)).expect("perms") else {
            panic!("expected tensor");
        };
        assert_eq!(out.shape, vec![6, 3]);
        assert_eq!(out.dtype, NumericDType::F64);
        assert_eq!(
            tensor_rows(&out),
            vec![
                vec![3.0, 2.0, 1.0],
                vec![3.0, 1.0, 2.0],
                vec![2.0, 3.0, 1.0],
                vec![2.0, 1.0, 3.0],
                vec![1.0, 3.0, 2.0],
                vec![1.0, 2.0, 3.0],
            ]
        );
    }

    #[test]
    fn column_vector_preserves_element_order_before_permuting() {
        let tensor = Tensor::new(vec![10.0, 20.0, 30.0], vec![3, 1]).unwrap();
        let Value::Tensor(out) = call(Value::Tensor(tensor)).expect("perms") else {
            panic!("expected tensor");
        };
        assert_eq!(tensor_rows(&out)[0], vec![30.0, 20.0, 10.0]);
        assert_eq!(tensor_rows(&out)[5], vec![10.0, 20.0, 30.0]);
    }

    #[test]
    fn duplicate_values_are_not_deduplicated() {
        let tensor = Tensor::new(vec![1.0, 1.0, 2.0], vec![1, 3]).unwrap();
        let Value::Tensor(out) = call(Value::Tensor(tensor)).expect("perms") else {
            panic!("expected tensor");
        };
        assert_eq!(out.rows, 6);
        assert_eq!(tensor_rows(&out)[0], vec![2.0, 1.0, 1.0]);
        assert_eq!(tensor_rows(&out)[1], vec![2.0, 1.0, 1.0]);
    }

    #[test]
    fn scalar_inputs_return_scalar_values() {
        assert_eq!(call(Value::Num(7.0)).expect("perms"), Value::Num(7.0));
        assert_eq!(
            call(Value::Int(IntValue::U32(7))).expect("perms"),
            Value::Int(IntValue::U32(7))
        );
        assert_eq!(
            call(Value::Complex(1.0, -2.0)).expect("perms"),
            Value::Complex(1.0, -2.0)
        );
        assert_eq!(call(Value::Bool(true)).expect("perms"), Value::Bool(true));
        assert_eq!(
            call(Value::String("token".to_string())).expect("perms"),
            Value::String("token".to_string())
        );
    }

    #[test]
    fn numeric_dtype_is_preserved_for_tensors() {
        let tensor = Tensor::new_with_dtype(vec![1.0, 2.0], vec![1, 2], NumericDType::U32).unwrap();
        let Value::Tensor(out) = call(Value::Tensor(tensor)).expect("perms") else {
            panic!("expected tensor");
        };
        assert_eq!(out.dtype, NumericDType::U32);
        assert_eq!(tensor_rows(&out), vec![vec![2.0, 1.0], vec![1.0, 2.0]]);
    }

    #[test]
    fn complex_tensor_vectors_are_permuted_by_position() {
        let tensor =
            ComplexTensor::new(vec![(1.0, 1.0), (2.0, -2.0), (3.0, 0.5)], vec![1, 3]).unwrap();
        let Value::ComplexTensor(out) = call(Value::ComplexTensor(tensor)).expect("perms") else {
            panic!("expected complex tensor");
        };
        assert_eq!(out.shape, vec![6, 3]);
        assert_eq!(
            complex_rows(&out)[0],
            vec![(3.0, 0.5), (2.0, -2.0), (1.0, 1.0)]
        );
        assert_eq!(
            complex_rows(&out)[5],
            vec![(1.0, 1.0), (2.0, -2.0), (3.0, 0.5)]
        );
    }

    #[test]
    fn logical_char_string_and_cell_vectors_are_supported() {
        let logical = LogicalArray::new(vec![0, 1, 1], vec![1, 3]).unwrap();
        let Value::LogicalArray(logical_out) = call(Value::LogicalArray(logical)).expect("logical")
        else {
            panic!("expected logical");
        };
        assert_eq!(logical_out.shape, vec![6, 3]);
        let logical_rows = logical_out.shape[0];
        let logical_cols = logical_out.shape[1];
        assert_eq!(
            (0..logical_cols)
                .map(|col| logical_out.data[col * logical_rows])
                .collect::<Vec<_>>(),
            vec![1, 1, 0]
        );

        let chars = CharArray::new_row("abc");
        let Value::CharArray(char_out) = call(Value::CharArray(chars)).expect("char") else {
            panic!("expected char");
        };
        assert_eq!(char_out.rows, 6);
        assert_eq!(char_out.cols, 3);
        let first_row: String = char_out.data[0..char_out.cols].iter().collect();
        assert_eq!(first_row, "cba");
        assert!(format!("{char_out}").contains("cba"));

        let strings = StringArray::new(
            vec!["red".to_string(), "green".to_string(), "blue".to_string()],
            vec![1, 3],
        )
        .unwrap();
        let Value::StringArray(string_out) = call(Value::StringArray(strings)).expect("strings")
        else {
            panic!("expected string array");
        };
        assert_eq!(string_out.shape, vec![6, 3]);
        assert_eq!(
            (0..string_out.cols)
                .map(|col| string_out.data[col * string_out.rows].as_str())
                .collect::<Vec<_>>(),
            vec!["blue", "green", "red"]
        );

        let cell = CellArray::new(
            vec![Value::Num(1.0), Value::Num(2.0), Value::Num(3.0)],
            1,
            3,
        )
        .unwrap();
        let Value::Cell(cell_out) = call(Value::Cell(cell)).expect("cell") else {
            panic!("expected cell");
        };
        assert_eq!(cell_out.shape, vec![6, 3]);
        assert_eq!(
            (0..cell_out.cols)
                .map(|col| cell_out.get(0, col).expect("cell get"))
                .collect::<Vec<_>>(),
            vec![Value::Num(3.0), Value::Num(2.0), Value::Num(1.0)]
        );
    }

    #[test]
    fn empty_vector_returns_one_empty_permutation() {
        let tensor = Tensor::new(Vec::new(), vec![1, 0]).unwrap();
        let Value::Tensor(out) = call(Value::Tensor(tensor)).expect("perms") else {
            panic!("expected tensor");
        };
        assert_eq!(out.shape, vec![1, 0]);
        assert!(out.data.is_empty());

        let empty_literal = Tensor::new(Vec::new(), vec![0, 0]).unwrap();
        let Value::Tensor(out) = call(Value::Tensor(empty_literal)).expect("perms []") else {
            panic!("expected tensor");
        };
        assert_eq!(out.shape, vec![1, 0]);
        assert!(out.data.is_empty());
    }

    #[test]
    fn rejects_non_vector_and_unsupported_inputs() {
        let matrix = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let err = call(Value::Tensor(matrix)).unwrap_err();
        assert_eq!(err.identifier.as_deref(), Some("RunMat:perms:InvalidInput"));

        let sparse = Value::SparseTensor(runmat_builtins::SparseTensor {
            rows: 1,
            cols: 1,
            col_ptrs: vec![0, 0],
            row_indices: Vec::new(),
            values: Vec::new(),
            integer_data: None,
        });
        let err = call(sparse).unwrap_err();
        assert_eq!(err.identifier.as_deref(), Some("RunMat:perms:InvalidInput"));
    }

    #[test]
    fn rejects_factorial_outputs_that_are_too_large() {
        let tensor = Tensor::new((1..=11).map(|v| v as f64).collect(), vec![1, 11]).unwrap();
        let err = call(Value::Tensor(tensor)).unwrap_err();
        assert_eq!(err.identifier.as_deref(), Some("RunMat:perms:TooLarge"));
    }

    #[test]
    fn gpu_inputs_gather_to_host_before_permuting() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
            let handle = provider
                .upload(&runmat_accelerate_api::HostTensorView {
                    data: &tensor.data,
                    shape: &tensor.shape,
                })
                .expect("upload");
            let Value::Tensor(out) = call(Value::GpuTensor(handle)).expect("gpu perms") else {
                panic!("expected tensor");
            };
            assert_eq!(tensor_rows(&out)[0], vec![3.0, 2.0, 1.0]);
        });
    }

    #[test]
    fn gpu_logical_and_complex_inputs_preserve_host_output_class() {
        test_support::with_test_provider(|provider| {
            let logical_tensor = Tensor::new(vec![0.0, 1.0, 1.0], vec![1, 3]).unwrap();
            let handle = provider
                .upload(&runmat_accelerate_api::HostTensorView {
                    data: &logical_tensor.data,
                    shape: &logical_tensor.shape,
                })
                .expect("upload logical");
            let Value::LogicalArray(out) =
                call(gpu_helpers::logical_gpu_value(handle)).expect("logical gpu perms")
            else {
                panic!("expected logical array");
            };
            assert_eq!(out.shape, vec![6, 3]);
            assert_eq!(
                (0..3)
                    .map(|col| out.data[col * out.shape[0]])
                    .collect::<Vec<_>>(),
                vec![1, 1, 0]
            );

            let complex =
                ComplexTensor::new(vec![(1.0, 1.0), (2.0, -2.0), (3.0, 0.5)], vec![1, 3]).unwrap();
            let handle = gpu_helpers::upload_complex_tensor(provider, &complex).expect("upload");
            let Value::ComplexTensor(out) =
                call(gpu_helpers::complex_gpu_value(handle)).expect("complex gpu perms")
            else {
                panic!("expected complex tensor");
            };
            assert_eq!(
                complex_rows(&out)[0],
                vec![(3.0, 0.5), (2.0, -2.0), (1.0, 1.0)]
            );
        });
    }
}
