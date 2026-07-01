//! MATLAB-compatible `int2str` builtin with GPU-aware host formatting.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    CharArray, Tensor, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::gpu_helpers;
use crate::builtins::common::map_control_flow_with_builtin;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::common::tensor;
use crate::builtins::strings::type_resolvers::string_scalar_type;
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "int2str";

const INT2STR_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "chr",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Character array containing the rounded integer text.",
}];

const INT2STR_INPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "N",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Numeric or logical scalar, vector, or matrix input.",
}];

const INT2STR_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "chr = int2str(N)",
    inputs: &INT2STR_INPUT,
    outputs: &INT2STR_OUTPUT,
}];

const INT2STR_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.INT2STR.INVALID_INPUT",
    identifier: Some("RunMat:int2str:InvalidInput"),
    when: "Input value is not a supported numeric/logical scalar, vector, or matrix.",
    message: "int2str: unsupported input type",
};

const INT2STR_ERROR_INVALID_OPTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.INT2STR.INVALID_OPTION",
    identifier: Some("RunMat:int2str:InvalidOption"),
    when: "Arguments are malformed or too many were supplied.",
    message: "int2str: invalid option arguments",
};

const INT2STR_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.INT2STR.INTERNAL",
    identifier: Some("RunMat:int2str:InternalError"),
    when: "Internal char-array assembly failed.",
    message: "int2str: internal error",
};

const INT2STR_ERRORS: [BuiltinErrorDescriptor; 3] = [
    INT2STR_ERROR_INVALID_INPUT,
    INT2STR_ERROR_INVALID_OPTION,
    INT2STR_ERROR_INTERNAL,
];

pub const INT2STR_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &INT2STR_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &INT2STR_ERRORS,
};

fn int2str_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    int2str_error_with_message(error.message, error)
}

fn int2str_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn remap_int2str_flow(err: RuntimeError) -> RuntimeError {
    map_control_flow_with_builtin(err, BUILTIN_NAME)
}

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::strings::core::int2str")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "int2str",
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
    notes: "Accepts GPU inputs but gathers them to host memory before rounding and formatting.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::strings::core::int2str")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "int2str",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes:
        "Conversion builtin; not eligible for fusion and always materialises host character arrays.",
};

#[runtime_builtin(
    name = "int2str",
    category = "strings/core",
    summary = "Convert rounded integer values to a character array.",
    keywords = "int2str,integer to string,number to string,round",
    examples = "chr = int2str([5 10 20; 100 200 400]);",
    type_resolver(string_scalar_type),
    descriptor(crate::builtins::strings::core::int2str::INT2STR_DESCRIPTOR),
    builtin_path = "crate::builtins::strings::core::int2str"
)]
async fn int2str_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    if !rest.is_empty() {
        return Err(int2str_error_with_message(
            "int2str: too many input arguments",
            &INT2STR_ERROR_INVALID_OPTION,
        ));
    }

    let gathered = gather_if_needed_async(&value)
        .await
        .map_err(remap_int2str_flow)?;
    let data = extract_numeric_data(gathered).await?;
    Ok(Value::CharArray(format_numeric_data(data)?))
}

enum NumericData {
    Real {
        data: Vec<f64>,
        rows: usize,
        cols: usize,
    },
}

async fn extract_numeric_data(value: Value) -> BuiltinResult<NumericData> {
    match value {
        Value::Num(n) => Ok(NumericData::Real {
            data: vec![n],
            rows: 1,
            cols: 1,
        }),
        Value::Int(i) => Ok(NumericData::Real {
            data: vec![i.to_f64()],
            rows: 1,
            cols: 1,
        }),
        Value::Bool(b) => Ok(NumericData::Real {
            data: vec![if b { 1.0 } else { 0.0 }],
            rows: 1,
            cols: 1,
        }),
        Value::Tensor(tensor) => tensor_to_numeric_data(tensor),
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(&logical)
                .map_err(|_| int2str_error(&INT2STR_ERROR_INVALID_INPUT))?;
            tensor_to_numeric_data(tensor)
        }
        Value::Complex(_, _) | Value::ComplexTensor(_) => Err(int2str_error_with_message(
            "int2str: complex input is not supported",
            &INT2STR_ERROR_INVALID_INPUT,
        )),
        Value::GpuTensor(handle) => {
            let gathered = gpu_helpers::gather_tensor_async(&handle)
                .await
                .map_err(remap_int2str_flow)?;
            tensor_to_numeric_data(gathered)
        }
        other => Err(int2str_error_with_message(
            format!(
                "{} {:?}; expected numeric or logical values",
                INT2STR_ERROR_INVALID_INPUT.message, other
            ),
            &INT2STR_ERROR_INVALID_INPUT,
        )),
    }
}

fn tensor_to_numeric_data(tensor: Tensor) -> BuiltinResult<NumericData> {
    if tensor.shape.len() > 2 {
        return Err(int2str_error_with_message(
            "int2str: input must be scalar, vector, or 2-D matrix",
            &INT2STR_ERROR_INVALID_INPUT,
        ));
    }
    let rows = tensor.rows();
    let cols = tensor.cols();
    Ok(NumericData::Real {
        data: tensor.data,
        rows,
        cols,
    })
}

#[derive(Clone)]
struct CellEntry {
    text: String,
    width: usize,
}

fn format_numeric_data(data: NumericData) -> BuiltinResult<CharArray> {
    match data {
        NumericData::Real { data, rows, cols } => format_real_matrix(&data, rows, cols),
    }
}

fn format_real_matrix(data: &[f64], rows: usize, cols: usize) -> BuiltinResult<CharArray> {
    if rows == 0 || cols == 0 {
        return char_array(Vec::new(), 0, 0);
    }

    format_entries(rows, cols, |row, col| {
        let idx = row + col * rows;
        format_rounded_real(data.get(idx).copied().unwrap_or(0.0))
    })
}

fn format_entries<F>(rows: usize, cols: usize, mut value_at: F) -> BuiltinResult<CharArray>
where
    F: FnMut(usize, usize) -> String,
{
    let total_cells = rows.checked_mul(cols).ok_or_else(|| {
        int2str_error_with_message(
            "int2str: output dimensions are too large",
            &INT2STR_ERROR_INTERNAL,
        )
    })?;
    if total_cells > isize::MAX as usize {
        return Err(int2str_error_with_message(
            "int2str: output dimensions are too large",
            &INT2STR_ERROR_INTERNAL,
        ));
    }

    let mut entries = vec![
        vec![
            CellEntry {
                text: String::new(),
                width: 0,
            };
            cols
        ];
        rows
    ];
    let mut col_widths = vec![0usize; cols];

    for (col, width) in col_widths.iter_mut().enumerate() {
        for (row, row_entries) in entries.iter_mut().enumerate() {
            let text = value_at(row, col);
            let entry_width = text.chars().count();
            row_entries[col] = CellEntry {
                text,
                width: entry_width,
            };
            *width = (*width).max(entry_width);
        }
    }

    if cols > 1 {
        for (idx, width) in col_widths.iter_mut().enumerate() {
            if idx > 0 {
                *width += 1;
            }
        }
    }

    rows_to_char_array(assemble_rows(entries, col_widths))
}

fn format_rounded_real(value: f64) -> String {
    format_integer_like(matlab_round(value))
}

fn matlab_round(value: f64) -> f64 {
    if value.is_finite() {
        value.round()
    } else {
        value
    }
}

fn format_integer_like(value: f64) -> String {
    if value.is_nan() {
        return "NaN".to_string();
    }
    if value.is_infinite() {
        return if value.is_sign_negative() {
            "-Inf".to_string()
        } else {
            "Inf".to_string()
        };
    }
    if value == 0.0 {
        return "0".to_string();
    }
    if value >= i64::MIN as f64 && value <= i64::MAX as f64 {
        return format!("{}", value as i64);
    }
    format!("{value:.0}")
}

fn assemble_rows(entries: Vec<Vec<CellEntry>>, col_widths: Vec<usize>) -> Vec<String> {
    entries
        .into_iter()
        .map(|row_entries| {
            row_entries
                .into_iter()
                .enumerate()
                .fold(String::new(), |mut acc, (col, entry)| {
                    if col > 0 {
                        acc.push(' ');
                    }
                    let target = col_widths[col];
                    let pad = target.saturating_sub(entry.width);
                    acc.extend(std::iter::repeat_n(' ', pad));
                    acc.push_str(&entry.text);
                    acc
                })
        })
        .collect()
}

fn rows_to_char_array(rows: Vec<String>) -> BuiltinResult<CharArray> {
    if rows.is_empty() {
        return char_array(Vec::new(), 0, 0);
    }

    let row_count = rows.len();
    let col_count = rows
        .iter()
        .map(|row| row.chars().count())
        .max()
        .unwrap_or(0);
    let capacity = row_count.checked_mul(col_count).ok_or_else(|| {
        int2str_error_with_message(
            "int2str: output dimensions are too large",
            &INT2STR_ERROR_INTERNAL,
        )
    })?;

    let mut data = Vec::with_capacity(capacity);
    for row in rows {
        let mut chars: Vec<char> = row.chars().collect();
        if chars.len() < col_count {
            chars.extend(std::iter::repeat_n(' ', col_count - chars.len()));
        }
        data.extend(chars);
    }
    char_array(data, row_count, col_count)
}

fn char_array(data: Vec<char>, rows: usize, cols: usize) -> BuiltinResult<CharArray> {
    let expected = rows.checked_mul(cols).ok_or_else(|| {
        int2str_error_with_message(
            "int2str: output dimensions are too large",
            &INT2STR_ERROR_INTERNAL,
        )
    })?;
    if expected != data.len() {
        return Err(int2str_error(&INT2STR_ERROR_INTERNAL));
    }
    CharArray::new(data, rows, cols).map_err(|_| int2str_error(&INT2STR_ERROR_INTERNAL))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use runmat_builtins::{IntValue, LogicalArray, ResolveContext, Type};

    fn int2str_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        futures::executor::block_on(super::int2str_builtin(value, rest))
    }

    fn char_rows(value: Value) -> Vec<String> {
        match value {
            Value::CharArray(ca) => ca
                .data
                .chunks(ca.cols)
                .map(|chunk| chunk.iter().collect())
                .collect(),
            other => panic!("expected char array, got {other:?}"),
        }
    }

    fn error_message(err: RuntimeError) -> String {
        err.message().to_string()
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn int2str_rounds_scalar_float() {
        let out = int2str_builtin(Value::Num(3.14159), Vec::new()).expect("int2str");
        assert_eq!(char_rows(out), vec!["3"]);

        let out = int2str_builtin(Value::Num(4.5), Vec::new()).expect("int2str");
        assert_eq!(char_rows(out), vec!["5"]);

        let out = int2str_builtin(Value::Num(-4.5), Vec::new()).expect("int2str");
        assert_eq!(char_rows(out), vec!["-5"]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn int2str_formats_integer_matrix_with_column_alignment() {
        let tensor =
            Tensor::new(vec![5.0, 100.0, 10.0, 200.0, 20.0, 400.0], vec![2, 3]).expect("tensor");
        let out = int2str_builtin(Value::Tensor(tensor), Vec::new()).expect("int2str");
        assert_eq!(char_rows(out), vec!["  5   10   20", "100  200  400"]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn int2str_accepts_int_and_logical_inputs() {
        let out = int2str_builtin(Value::Int(IntValue::I32(-12)), Vec::new()).expect("int2str int");
        assert_eq!(char_rows(out), vec!["-12"]);

        let logical = LogicalArray::new(vec![1, 0, 1], vec![1, 3]).expect("logical");
        let out =
            int2str_builtin(Value::LogicalArray(logical), Vec::new()).expect("int2str logical");
        assert_eq!(char_rows(out), vec!["1  0  1"]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn int2str_preserves_nonfinite_text() {
        let tensor = Tensor::new(vec![f64::NAN, f64::INFINITY, f64::NEG_INFINITY], vec![1, 3])
            .expect("tensor");
        let out = int2str_builtin(Value::Tensor(tensor), Vec::new()).expect("int2str");
        assert_eq!(char_rows(out), vec!["NaN  Inf  -Inf"]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn int2str_empty_matrix_shapes() {
        let empty = Tensor::new(Vec::new(), vec![0, 3]).expect("empty rows");
        let out = int2str_builtin(Value::Tensor(empty), Vec::new()).expect("int2str");
        match out {
            Value::CharArray(ca) => {
                assert_eq!(ca.rows, 0);
                assert_eq!(ca.cols, 0);
                assert!(ca.data.is_empty());
            }
            other => panic!("expected char array, got {other:?}"),
        }

        let empty_cols = Tensor::new(Vec::new(), vec![2, 0]).expect("empty cols");
        let out = int2str_builtin(Value::Tensor(empty_cols), Vec::new()).expect("int2str");
        match out {
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
    fn int2str_gpu_tensor_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![10.2, 20.8], vec![1, 2]).expect("tensor");
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let out = int2str_builtin(Value::GpuTensor(handle), Vec::new()).expect("int2str");
            assert_eq!(char_rows(out), vec!["10  21"]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn int2str_rejects_complex_input() {
        let err = error_message(int2str_builtin(Value::Complex(3.0, 4.0), Vec::new()).unwrap_err());
        assert!(err.contains("complex input is not supported"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn int2str_rejects_extra_arguments() {
        let err =
            error_message(int2str_builtin(Value::Num(1.0), vec![Value::Num(2.0)]).unwrap_err());
        assert!(err.contains("too many input arguments"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn int2str_rejects_non_numeric_input() {
        let err =
            error_message(int2str_builtin(Value::String("hello".into()), Vec::new()).unwrap_err());
        assert!(err.contains("unsupported input type"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn int2str_rejects_nd_input() {
        let tensor = Tensor::new(vec![1.0; 8], vec![2, 2, 2]).expect("tensor");
        let err = error_message(int2str_builtin(Value::Tensor(tensor), Vec::new()).unwrap_err());
        assert!(err.contains("scalar, vector, or 2-D matrix"));
    }

    #[test]
    fn int2str_type_is_string_scalar() {
        assert_eq!(
            string_scalar_type(&[Type::Num], &ResolveContext::new(Vec::new())),
            Type::String
        );
    }
}
