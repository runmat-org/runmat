//! MATLAB-compatible `char` builtin with GPU-aware conversion semantics for RunMat.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    CellArray, CharArray, LogicalArray, StringArray, Tensor, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::map_control_flow_with_builtin;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::strings::type_resolvers::string_array_type;
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::strings::core::char")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "char",
    op_kind: GpuOpKind::Custom("conversion"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes:
        "Conversion always runs on the CPU; GPU tensors are gathered before building the result.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::strings::core::char")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "char",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Character materialisation runs outside of fusion; results always live on the host.",
};

const BUILTIN_NAME: &str = "char";

const CHAR_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "C",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Character array result.",
}];

const CHAR_INPUT_SINGLE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input value to convert into character data.",
}];

const CHAR_INPUT_VARIADIC: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X...",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Multiple inputs converted row-wise and padded.",
}];

const CHAR_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "C = char()",
        inputs: &[],
        outputs: &CHAR_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "C = char(X)",
        inputs: &CHAR_INPUT_SINGLE,
        outputs: &CHAR_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "C = char(X...)",
        inputs: &CHAR_INPUT_VARIADIC,
        outputs: &CHAR_OUTPUT,
    },
];

const CHAR_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CHAR.INVALID_INPUT",
    identifier: Some("RunMat:char:InvalidInput"),
    when: "Input type cannot be converted to character data.",
    message: "char: invalid input",
};

const CHAR_ERROR_INVALID_CODEPOINT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CHAR.INVALID_CODEPOINT",
    identifier: Some("RunMat:char:InvalidCodePoint"),
    when: "Numeric input is not a finite integer Unicode code point.",
    message: "char: numeric inputs must be finite Unicode code points",
};

const CHAR_ERROR_DIMENSION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CHAR.INVALID_DIMENSION",
    identifier: Some("RunMat:char:InvalidDimension"),
    when: "Array inputs are not 2-D (or trailing singleton dimensions).",
    message: "char: inputs must be 2-D",
};

const CHAR_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CHAR.INTERNAL",
    identifier: Some("RunMat:char:InternalError"),
    when: "Internal character array construction failed.",
    message: "char: internal error",
};

const CHAR_ERRORS: [BuiltinErrorDescriptor; 4] = [
    CHAR_ERROR_INVALID_INPUT,
    CHAR_ERROR_INVALID_CODEPOINT,
    CHAR_ERROR_DIMENSION,
    CHAR_ERROR_INTERNAL,
];

pub const CHAR_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &CHAR_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &CHAR_ERRORS,
};

fn char_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    char_error_with_message(error.message, error)
}

fn char_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn char_flow(message: impl Into<String>) -> RuntimeError {
    char_error_with_message(message, &CHAR_ERROR_INTERNAL)
}

fn remap_char_flow(err: RuntimeError) -> RuntimeError {
    map_control_flow_with_builtin(err, BUILTIN_NAME)
}

#[runtime_builtin(
    name = "char",
    category = "strings/core",
    summary = "Convert numeric codes and text values into character arrays.",
    keywords = "char,character,string,gpu",
    accel = "conversion",
    type_resolver(string_array_type),
    descriptor(crate::builtins::strings::core::char::CHAR_DESCRIPTOR),
    builtin_path = "crate::builtins::strings::core::char"
)]
async fn char_builtin(rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    if rest.is_empty() {
        let empty =
            CharArray::new(Vec::new(), 0, 0).map_err(|_| char_error(&CHAR_ERROR_INTERNAL))?;
        return Ok(Value::CharArray(empty));
    }

    let mut rows: Vec<Vec<char>> = Vec::new();
    let mut max_width = 0usize;

    for arg in rest {
        let gathered = gather_if_needed_async(&arg)
            .await
            .map_err(remap_char_flow)?;
        let mut produced = value_to_char_rows(&gathered)?;
        for row in &produced {
            if row.len() > max_width {
                max_width = row.len();
            }
        }
        rows.append(&mut produced);
    }

    if rows.is_empty() {
        let empty =
            CharArray::new(Vec::new(), 0, 0).map_err(|_| char_error(&CHAR_ERROR_INTERNAL))?;
        return Ok(Value::CharArray(empty));
    }

    let cols = max_width;
    let total_rows = rows.len();
    let mut data = vec![' '; total_rows * cols];
    for (row_idx, row) in rows.into_iter().enumerate() {
        for (col_idx, ch) in row.into_iter().enumerate() {
            if col_idx < cols {
                data[row_idx * cols + col_idx] = ch;
            }
        }
    }

    let array =
        CharArray::new(data, total_rows, cols).map_err(|_| char_error(&CHAR_ERROR_INTERNAL))?;
    Ok(Value::CharArray(array))
}

fn value_to_char_rows(value: &Value) -> BuiltinResult<Vec<Vec<char>>> {
    if let Some(array) = crate::builtins::datetime::datetime_char_array(value)
        .map_err(|err| char_flow(err.message().to_string()))?
    {
        return Ok(char_array_rows(&array));
    }
    if let Some(array) = crate::builtins::duration::duration_char_array(value)
        .map_err(|err| char_flow(err.message().to_string()))?
    {
        return Ok(char_array_rows(&array));
    }
    match value {
        Value::CharArray(ca) => Ok(char_array_rows(ca)),
        Value::String(s) => Ok(vec![s.chars().collect()]),
        Value::StringArray(sa) => string_array_rows(sa),
        Value::Num(n) => Ok(vec![vec![number_to_char(*n)?]]),
        Value::Int(i) => {
            let as_double = i.to_f64();
            Ok(vec![vec![number_to_char(as_double)?]])
        }
        Value::Bool(b) => {
            let code = if *b { 1.0 } else { 0.0 };
            Ok(vec![vec![number_to_char(code)?]])
        }
        Value::Tensor(t) => tensor_rows(t),
        Value::SparseTensor(s) => {
            let dense = s.to_dense().map_err(char_flow)?;
            tensor_rows(&dense)
        }
        Value::LogicalArray(la) => logical_rows(la),
        Value::Cell(ca) => cell_rows(ca),
        Value::GpuTensor(_) => Err(char_error(&CHAR_ERROR_INVALID_INPUT)),
        Value::Complex(_, _) | Value::ComplexTensor(_) => Err(char_error_with_message(
            "char: complex inputs are not supported",
            &CHAR_ERROR_INVALID_INPUT,
        )),
        Value::Struct(_)
        | Value::Object(_)
        | Value::HandleObject(_)
        | Value::Listener(_)
        | Value::FunctionHandle(_)
        | Value::ExternalFunctionHandle(_)
        | Value::MethodFunctionHandle(_)
        | Value::BoundFunctionHandle { .. }
        | Value::Closure(_)
        | Value::ClassRef(_)
        | Value::MException(_)
        | Value::OutputList(_) => Err(char_error_with_message(
            format!("char: unsupported input type {:?}", value),
            &CHAR_ERROR_INVALID_INPUT,
        )),
    }
}

fn char_array_rows(ca: &CharArray) -> Vec<Vec<char>> {
    let mut rows = Vec::with_capacity(ca.rows);
    for r in 0..ca.rows {
        let mut row = Vec::with_capacity(ca.cols);
        for c in 0..ca.cols {
            row.push(ca.data[r * ca.cols + c]);
        }
        rows.push(row);
    }
    rows
}

fn string_array_rows(sa: &StringArray) -> BuiltinResult<Vec<Vec<char>>> {
    ensure_two_dimensional(&sa.shape, "char")?;
    if sa.data.is_empty() {
        return Ok(Vec::new());
    }
    let mut rows = Vec::with_capacity(sa.data.len());
    let rows_count = sa.rows();
    let cols_count = sa.cols();
    if rows_count == 0 || cols_count == 0 {
        return Ok(Vec::new());
    }
    for c in 0..cols_count {
        for r in 0..rows_count {
            let idx = r + c * rows_count;
            rows.push(sa.data[idx].chars().collect());
        }
    }
    Ok(rows)
}

fn tensor_rows(t: &Tensor) -> BuiltinResult<Vec<Vec<char>>> {
    ensure_two_dimensional(&t.shape, "char")?;
    let (rows, cols) = infer_rows_cols(&t.shape, t.data.len());
    if rows == 0 {
        return Ok(Vec::new());
    }
    let mut out = Vec::with_capacity(rows);
    for r in 0..rows {
        let mut row = Vec::with_capacity(cols);
        for c in 0..cols {
            if cols == 0 {
                continue;
            }
            let idx = r + c * rows;
            let value = t.data[idx];
            row.push(number_to_char(value)?);
        }
        out.push(row);
    }
    Ok(out)
}

fn logical_rows(la: &LogicalArray) -> BuiltinResult<Vec<Vec<char>>> {
    ensure_two_dimensional(&la.shape, "char")?;
    let (rows, cols) = infer_rows_cols(&la.shape, la.data.len());
    if rows == 0 {
        return Ok(Vec::new());
    }
    let mut out = Vec::with_capacity(rows);
    for r in 0..rows {
        let mut row = Vec::with_capacity(cols);
        for c in 0..cols {
            if cols == 0 {
                continue;
            }
            let idx = r + c * rows;
            let code = if la.data[idx] != 0 { 1.0 } else { 0.0 };
            row.push(number_to_char(code)?);
        }
        out.push(row);
    }
    Ok(out)
}

fn cell_rows(ca: &CellArray) -> BuiltinResult<Vec<Vec<char>>> {
    let mut rows = Vec::with_capacity(ca.data.len());
    for ptr in &ca.data {
        let element = (**ptr).clone();
        let mut converted = value_to_char_rows(&element)?;
        match converted.len() {
            0 => rows.push(Vec::new()),
            1 => rows.push(converted.remove(0)),
            _ => {
                return Err(char_error_with_message(
                    "char: cell elements must be character vectors or string scalars",
                    &CHAR_ERROR_INVALID_INPUT,
                ))
            }
        }
    }
    Ok(rows)
}

fn number_to_char(value: f64) -> BuiltinResult<char> {
    if !value.is_finite() {
        return Err(char_error_with_message(
            "char: numeric inputs must be finite",
            &CHAR_ERROR_INVALID_CODEPOINT,
        ));
    }
    let rounded = value.round();
    if (value - rounded).abs() > 1e-9 {
        return Err(char_error_with_message(
            format!("char: numeric inputs must be integers in the Unicode range (got {value})"),
            &CHAR_ERROR_INVALID_CODEPOINT,
        ));
    }
    if rounded < 0.0 {
        return Err(char_error_with_message(
            format!("char: negative code points are invalid (got {rounded})"),
            &CHAR_ERROR_INVALID_CODEPOINT,
        ));
    }
    if rounded > 0x10FFFF as f64 {
        return Err(char_error_with_message(
            format!("char: code point {} exceeds Unicode range", rounded as u64),
            &CHAR_ERROR_INVALID_CODEPOINT,
        ));
    }
    let code = rounded as u32;
    char::from_u32(code).ok_or_else(|| {
        char_error_with_message(
            format!("char: invalid code point {code}"),
            &CHAR_ERROR_INVALID_CODEPOINT,
        )
    })
}

fn ensure_two_dimensional(shape: &[usize], context: &str) -> BuiltinResult<()> {
    if shape.len() <= 2 {
        return Ok(());
    }
    if shape.iter().skip(2).all(|&d| d == 1) {
        return Ok(());
    }
    Err(char_error_with_message(
        format!("{context}: inputs must be 2-D"),
        &CHAR_ERROR_DIMENSION,
    ))
}

fn infer_rows_cols(shape: &[usize], len: usize) -> (usize, usize) {
    match shape.len() {
        0 => {
            if len == 0 {
                (0, 0)
            } else {
                (1, 1)
            }
        }
        1 => (1, shape[0]),
        2 => (shape[0], shape[1]),
        _ => {
            let rows = shape[0];
            let cols = if shape.len() > 1 { shape[1] } else { 1 };
            (rows, cols)
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use runmat_builtins::{ResolveContext, Type};

    fn char_builtin(rest: Vec<Value>) -> BuiltinResult<Value> {
        futures::executor::block_on(super::char_builtin(rest))
    }
    use runmat_builtins::StringArray;

    fn error_message(err: crate::RuntimeError) -> String {
        err.message().to_string()
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn char_no_arguments_returns_empty() {
        let result = char_builtin(Vec::new()).expect("char");
        match result {
            Value::CharArray(ca) => {
                assert_eq!(ca.rows, 0);
                assert_eq!(ca.cols, 0);
                assert!(ca.data.is_empty());
            }
            other => panic!("expected char array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn char_from_string_scalar() {
        let value = Value::String("RunMat".to_string());
        let result = char_builtin(vec![value]).expect("char");
        match result {
            Value::CharArray(ca) => {
                assert_eq!(ca.rows, 1);
                assert_eq!(ca.cols, 6);
                assert_eq!(ca.data, "RunMat".chars().collect::<Vec<_>>());
            }
            other => panic!("expected char array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn char_from_numeric_tensor() {
        let tensor =
            Tensor::new(vec![82.0, 85.0, 78.0, 77.0, 65.0, 84.0], vec![1, 6]).expect("tensor");
        let result = char_builtin(vec![Value::Tensor(tensor)]).expect("char");
        match result {
            Value::CharArray(ca) => {
                assert_eq!(ca.rows, 1);
                assert_eq!(ca.cols, 6);
                assert_eq!(ca.data, "RUNMAT".chars().collect::<Vec<_>>());
            }
            other => panic!("expected char array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn char_from_string_array_with_padding() {
        let data = vec!["cat".to_string(), "giraffe".to_string()];
        let sa = StringArray::new(data, vec![2, 1]).expect("string array");
        let result = char_builtin(vec![Value::StringArray(sa)]).expect("char from string array");
        match result {
            Value::CharArray(ca) => {
                assert_eq!(ca.rows, 2);
                assert_eq!(ca.cols, 7);
                assert_eq!(
                    ca.data,
                    vec!['c', 'a', 't', ' ', ' ', ' ', ' ', 'g', 'i', 'r', 'a', 'f', 'f', 'e']
                );
            }
            other => panic!("expected char array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn char_from_cell_array_of_strings() {
        let cell = CellArray::new(
            vec![
                Value::from("north"),
                Value::from("east"),
                Value::from("west"),
            ],
            3,
            1,
        )
        .expect("cell array");
        let result = char_builtin(vec![Value::Cell(cell)]).expect("char");
        match result {
            Value::CharArray(ca) => {
                assert_eq!(ca.rows, 3);
                assert_eq!(ca.cols, 5);
                assert_eq!(
                    ca.data,
                    vec!['n', 'o', 'r', 't', 'h', 'e', 'a', 's', 't', ' ', 'w', 'e', 's', 't', ' ']
                );
            }
            other => panic!("expected char array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn char_numeric_and_text_arguments_concatenate() {
        let text = Value::String("hi".to_string());
        let codes = Tensor::new(vec![65.0, 66.0], vec![1, 2]).expect("tensor");
        let result = char_builtin(vec![text, Value::Tensor(codes)]).expect("char");
        match result {
            Value::CharArray(ca) => {
                assert_eq!(ca.rows, 2);
                assert_eq!(ca.cols, 2);
                assert_eq!(ca.data, vec!['h', 'i', 'A', 'B']);
            }
            other => panic!("expected char array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn char_gpu_tensor_round_trip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![82.0, 85.0, 78.0], vec![1, 3]).expect("tensor");
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = char_builtin(vec![Value::GpuTensor(handle)]).expect("char");
            match result {
                Value::CharArray(ca) => {
                    assert_eq!(ca.rows, 1);
                    assert_eq!(ca.cols, 3);
                    assert_eq!(ca.data, vec!['R', 'U', 'N']);
                }
                other => panic!("expected char array, got {other:?}"),
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn char_rejects_non_integer_numeric() {
        let err =
            error_message(char_builtin(vec![Value::Num(65.5)]).expect_err("non-integer numeric"));
        assert!(err.contains("integers"), "unexpected error message: {err}");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn char_rejects_high_dimension_tensor() {
        let tensor =
            Tensor::new(vec![65.0, 66.0], vec![1, 1, 2]).expect("tensor construction failed");
        let err = error_message(
            char_builtin(vec![Value::Tensor(tensor)]).expect_err("should reject >2D tensor"),
        );
        assert!(err.contains("2-D"), "expected dimension error, got {err}");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn char_string_array_column_major_order() {
        let data = vec![
            "c0r0".to_string(),
            "c0r1".to_string(),
            "c1r0".to_string(),
            "c1r1".to_string(),
        ];
        let sa = StringArray::new(data, vec![2, 2]).expect("string array");
        let result = char_builtin(vec![Value::StringArray(sa)]).expect("char");
        match result {
            Value::CharArray(ca) => {
                assert_eq!(ca.rows, 4);
                assert_eq!(ca.cols, 4);
                assert_eq!(ca.data, "c0r0c0r1c1r0c1r1".chars().collect::<Vec<char>>());
            }
            other => panic!("expected char array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn char_rejects_high_dimension_string_array() {
        let sa = StringArray::new(vec!["a".to_string(), "b".to_string()], vec![1, 1, 2])
            .expect("string array");
        let err = error_message(
            char_builtin(vec![Value::StringArray(sa)]).expect_err("should reject >2D string array"),
        );
        assert!(err.contains("2-D"), "expected dimension error, got {err}");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn char_rejects_complex_input() {
        let err =
            error_message(char_builtin(vec![Value::Complex(1.0, 2.0)]).expect_err("complex input"));
        assert!(
            err.contains("complex"),
            "expected complex error message, got {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn char_wgpu_numeric_codes_matches_cpu() {
        use runmat_accelerate::backend::wgpu::provider::{
            register_wgpu_provider, WgpuProviderOptions,
        };

        let _ = register_wgpu_provider(WgpuProviderOptions::default());

        let tensor = Tensor::new(vec![82.0, 85.0, 78.0], vec![1, 3]).unwrap();
        let cpu = char_builtin(vec![Value::Tensor(tensor.clone())]).expect("char cpu");

        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = runmat_accelerate_api::provider()
            .expect("wgpu provider")
            .upload(&view)
            .expect("upload");
        let gpu = char_builtin(vec![Value::GpuTensor(handle)]).expect("char gpu");

        match (cpu, gpu) {
            (Value::CharArray(expected), Value::CharArray(actual)) => {
                assert_eq!(actual, expected);
            }
            other => panic!("unexpected results {other:?}"),
        }
    }

    #[test]
    fn char_type_is_string_array() {
        assert_eq!(
            string_array_type(&[Type::Num], &ResolveContext::new(Vec::new())),
            Type::cell_of(Type::String)
        );
    }
}
