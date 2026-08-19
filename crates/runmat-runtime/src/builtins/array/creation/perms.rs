//! MATLAB-compatible `perms` builtin.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    ResolveContext, Type,
};
use runmat_builtins::{
    BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor, BuiltinIntegerComputationDomain,
    BuiltinIntegerInputAvailability, BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule,
    BuiltinIntegerOverflowRule, BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule,
};
use runmat_macros::runtime_builtin;
use runmat_value::{
    CellArray, CharArray, ComplexTensor, LogicalArray, NumericStorage, StringArray, Tensor, Value,
};

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

const PERMS_INTEGER_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "v",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The compatibility target explicitly documents every integer class and shows an int16 example. Values are rearranged without arithmetic or conversion.",
    }];
pub const PERMS_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "P = perms(integer_v)",
        inputs: &PERMS_INTEGER_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
        notes: "Every output element is copied from authoritative native storage in reverse lexicographic index order. Documented gpuArray inputs gather through their owner and the same-class result is restored to that owner.",
    }];

#[runtime_builtin(
    name = "perms",
    category = "array/creation",
    summary = "Return all permutations of a vector as rows.",
    keywords = "perms,permutation,combinatorics,vector",
    accel = "array_construct",
    type_resolver(perms_type),
    descriptor(crate::builtins::array::creation::perms::PERMS_DESCRIPTOR),
    integer_capabilities(crate::builtins::array::creation::perms::PERMS_INTEGER_CAPABILITIES),
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
        let gathered = gpu_helpers::gather_value_async(&Value::GpuTensor(handle.clone()))
            .await
            .map_err(|e| perms_error_with(&ERROR_INTERNAL, format!("perms: {e}")))?;
        let output = evaluate_host(gathered)?;
        let restored = gpu_helpers::restore_class_preserving_value(&handle, output, BUILTIN_NAME)
            .map_err(|e| perms_error_with(&ERROR_INTERNAL, format!("perms: {e}")))?;
        if runmat_accelerate_api::handle_is_explicit(&handle)
            && !matches!(restored, Value::GpuTensor(_))
        {
            return Err(perms_error_with(
                &ERROR_INTERNAL,
                "perms: provider cannot preserve explicit gpuArray output residency",
            ));
        }
        return Ok(restored);
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
        | Value::ObjectArray(_)
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
        | Value::Future(_)
        | Value::Task(_)
        | Value::Pool(_)
        | Value::Job(_)
        | Value::Foreign(_)
        | Value::OutputList(_) => Err(perms_error(&ERROR_INVALID_INPUT)),
    }
}

fn perms_tensor(tensor: Tensor) -> BuiltinResult<Value> {
    let elements = vector_len(&tensor.shape)?;
    let rows = checked_output_rows(elements)?;
    let storage = tensor
        .into_numeric_storage()
        .map_err(|error| perms_error_with(&ERROR_INTERNAL, format!("perms: {error}")))?;
    let storage = permute_numeric_storage(storage, rows, elements)?;
    Tensor::from_numeric_storage(storage, vec![rows, elements])
        .map(Value::Tensor)
        .map_err(|e| perms_error_with(&ERROR_INTERNAL, format!("perms: {e}")))
}

fn permute_numeric_storage(
    storage: NumericStorage,
    rows: usize,
    elements: usize,
) -> BuiltinResult<NumericStorage> {
    macro_rules! permute {
        ($values:expr, $variant:ident) => {
            NumericStorage::$variant(permuted_columns(&$values, rows, elements)?)
        };
    }
    Ok(match storage {
        NumericStorage::F64(values) => permute!(values, F64),
        NumericStorage::F32(values) => permute!(values, F32),
        NumericStorage::I8(values) => permute!(values, I8),
        NumericStorage::I16(values) => permute!(values, I16),
        NumericStorage::I32(values) => permute!(values, I32),
        NumericStorage::I64(values) => permute!(values, I64),
        NumericStorage::U8(values) => permute!(values, U8),
        NumericStorage::U16(values) => permute!(values, U16),
        NumericStorage::U32(values) => permute!(values, U32),
        NumericStorage::U64(values) => permute!(values, U64),
    })
}

fn perms_complex_tensor(tensor: ComplexTensor) -> BuiltinResult<Value> {
    let elements = vector_len(&tensor.shape)?;
    let rows = checked_output_rows(elements)?;
    if let Some(storage) = tensor.integer_storage() {
        let storage = storage
            .reorder(|values| {
                permuted_columns(values, rows, elements).map_err(|error| error.to_string())
            })
            .map_err(|error| perms_error_with(&ERROR_INTERNAL, format!("perms: {error}")))?;
        return ComplexTensor::new_integer(storage, vec![rows, elements])
            .map(Value::ComplexTensor)
            .map_err(|error| perms_error_with(&ERROR_INTERNAL, format!("perms: {error}")));
    }
    let data = permuted_columns(&tensor.materialize_f64(), rows, elements)?;
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
    use runmat_value::{IntValue, IntegerStorage, NumericDType};

    fn call(value: Value) -> BuiltinResult<Value> {
        block_on(super::perms_builtin(value, Vec::new()))
    }

    fn tensor_rows(tensor: &Tensor) -> Vec<Vec<f64>> {
        (0..tensor.rows)
            .map(|row| {
                (0..tensor.cols)
                    .map(|col| tensor.materialize_f64()[col * tensor.rows + row])
                    .collect()
            })
            .collect()
    }

    fn complex_rows(tensor: &ComplexTensor) -> Vec<Vec<(f64, f64)>> {
        (0..tensor.rows)
            .map(|row| {
                (0..tensor.cols)
                    .map(|col| tensor.materialize_f64()[col * tensor.rows + row])
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
        assert_eq!(out.numeric_dtype(), NumericDType::F64);
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
    fn native_single_storage_is_preserved_for_tensors() {
        let tensor = Tensor::from_f32(vec![1.0, 2.0], vec![1, 2]).unwrap();
        let Value::Tensor(out) = call(Value::Tensor(tensor)).expect("perms") else {
            panic!("expected tensor");
        };
        assert_eq!(
            out.into_numeric_storage().expect("single storage"),
            NumericStorage::F32(vec![2.0, 1.0, 1.0, 2.0])
        );
    }

    #[test]
    fn exact_integer_tensor_classes_and_values_are_preserved() {
        let storages = [
            IntegerStorage::I8(vec![-2, 7]),
            IntegerStorage::I16(vec![-300, 400]),
            IntegerStorage::I32(vec![i32::MIN, i32::MAX]),
            IntegerStorage::I64(vec![i64::MIN, i64::MAX]),
            IntegerStorage::U8(vec![0, u8::MAX]),
            IntegerStorage::U16(vec![0, u16::MAX]),
            IntegerStorage::U32(vec![0, u32::MAX]),
            IntegerStorage::U64(vec![9_007_199_254_740_993, u64::MAX]),
        ];

        for storage in storages {
            let values = storage.exact_values();
            let expected = storage
                .from_exact_values_like(vec![
                    values[1].clone(),
                    values[0].clone(),
                    values[0].clone(),
                    values[1].clone(),
                ])
                .expect("expected storage");
            let input = Tensor::new_integer(storage, vec![1, 2]).expect("integer tensor");
            let Value::Tensor(output) = call(Value::Tensor(input)).expect("perms") else {
                panic!("expected exact integer tensor");
            };
            assert_eq!(output.shape, vec![2, 2]);
            assert_eq!(output.integer_storage(), Some(&expected));
        }
    }

    #[test]
    fn resident_integer_perms_restores_exact_class_and_explicit_residency() {
        test_support::with_test_provider(|provider| {
            let input = Tensor::new_integer(
                IntegerStorage::U64(vec![9_007_199_254_740_993, 9_007_199_254_740_994]),
                vec![1, 2],
            )
            .expect("integer input");
            let handle = gpu_helpers::upload_tensor(provider, &input).expect("integer upload");
            let handle =
                handle.with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit);
            let output = call(Value::GpuTensor(handle)).expect("resident perms");
            let Value::GpuTensor(output_handle) = &output else {
                panic!("expected resident output");
            };
            assert_eq!(
                runmat_accelerate_api::handle_integer_type(output_handle),
                Some(runmat_accelerate_api::IntegerElementType::U64)
            );
            let gathered = test_support::gather(output).expect("gather output");
            assert_eq!(
                gathered.integer_storage(),
                Some(&IntegerStorage::U64(vec![
                    9_007_199_254_740_994,
                    9_007_199_254_740_993,
                    9_007_199_254_740_993,
                    9_007_199_254_740_994,
                ]))
            );
        });
    }

    #[cfg(feature = "wgpu")]
    #[test]
    fn integer_perms_wgpu_fallback_preserves_every_class_and_explicit_residency() {
        let _accel_guard = test_support::accel_test_lock();
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let Some(provider) = runmat_accelerate_api::provider() else {
            return;
        };
        let cases = [
            (
                IntegerStorage::I8(vec![1, 2]),
                IntegerStorage::I8(vec![2, 1, 1, 2]),
            ),
            (
                IntegerStorage::I16(vec![1, 2]),
                IntegerStorage::I16(vec![2, 1, 1, 2]),
            ),
            (
                IntegerStorage::I32(vec![1, 2]),
                IntegerStorage::I32(vec![2, 1, 1, 2]),
            ),
            (
                IntegerStorage::I64(vec![1, 2]),
                IntegerStorage::I64(vec![2, 1, 1, 2]),
            ),
            (
                IntegerStorage::U8(vec![1, 2]),
                IntegerStorage::U8(vec![2, 1, 1, 2]),
            ),
            (
                IntegerStorage::U16(vec![1, 2]),
                IntegerStorage::U16(vec![2, 1, 1, 2]),
            ),
            (
                IntegerStorage::U32(vec![1, 2]),
                IntegerStorage::U32(vec![2, 1, 1, 2]),
            ),
            (
                IntegerStorage::U64(vec![1, 2]),
                IntegerStorage::U64(vec![2, 1, 1, 2]),
            ),
        ];
        for (input, expected) in cases {
            let tensor = Tensor::new_integer(input, vec![1, 2]).expect("integer input");
            let handle = gpu_helpers::upload_tensor(provider, &tensor).expect("integer upload");
            let handle =
                handle.with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit);
            let output = call(Value::GpuTensor(handle)).expect("resident integer perms");
            let Value::GpuTensor(output_handle) = &output else {
                panic!("expected resident output");
            };
            assert!(runmat_accelerate_api::handle_is_explicit(output_handle));
            let gathered = test_support::gather(output).expect("gather output");
            assert_eq!(gathered.integer_storage(), Some(&expected));
        }
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
        assert!(out.materialize_f64().is_empty());

        let empty_literal = Tensor::new(Vec::new(), vec![0, 0]).unwrap();
        let Value::Tensor(out) = call(Value::Tensor(empty_literal)).expect("perms []") else {
            panic!("expected tensor");
        };
        assert_eq!(out.shape, vec![1, 0]);
        assert!(out.materialize_f64().is_empty());
    }

    #[test]
    fn rejects_non_vector_and_unsupported_inputs() {
        let matrix = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let err = call(Value::Tensor(matrix)).unwrap_err();
        assert_eq!(err.identifier.as_deref(), Some("RunMat:perms:InvalidInput"));

        let sparse = Value::SparseTensor(runmat_value::SparseTensor::zeros(1, 1));
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
    fn gpu_inputs_restore_permutations_to_the_owning_provider() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
            let handle = provider
                .upload(&runmat_accelerate_api::HostTensorView {
                    data: &tensor.materialize_f64(),
                    shape: &tensor.shape,
                })
                .expect("upload");
            let output = call(Value::GpuTensor(handle)).expect("gpu perms");
            assert!(matches!(output, Value::GpuTensor(_)));
            let out = test_support::gather(output).expect("gather output");
            assert_eq!(tensor_rows(&out)[0], vec![3.0, 2.0, 1.0]);
        });
    }

    #[test]
    fn gpu_logical_and_complex_inputs_restore_output_class_and_residency() {
        test_support::with_test_provider(|provider| {
            let logical_tensor = Tensor::new(vec![0.0, 1.0, 1.0], vec![1, 3]).unwrap();
            let handle = provider
                .upload(&runmat_accelerate_api::HostTensorView {
                    data: &logical_tensor.materialize_f64(),
                    shape: &logical_tensor.shape,
                })
                .expect("upload logical");
            let output = call(gpu_helpers::logical_gpu_value(handle)).expect("logical gpu perms");
            let Value::GpuTensor(output_handle) = &output else {
                panic!("expected resident logical output");
            };
            assert!(runmat_accelerate_api::handle_is_logical(output_handle));
            let Value::LogicalArray(out) =
                block_on(gpu_helpers::gather_value_async(&output)).expect("gather logical output")
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
            let output = call(gpu_helpers::complex_gpu_value(handle)).expect("complex gpu perms");
            let Value::GpuTensor(output_handle) = &output else {
                panic!("expected resident complex output");
            };
            assert_eq!(
                runmat_accelerate_api::handle_storage(output_handle),
                runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
            );
            let Value::ComplexTensor(out) =
                block_on(gpu_helpers::gather_value_async(&output)).expect("gather complex output")
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
