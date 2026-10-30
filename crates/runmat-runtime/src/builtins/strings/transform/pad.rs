//! MATLAB-compatible `pad` builtin with GPU-aware semantics for RunMat.

#[cfg(test)]
use runmat_accelerate_api::{HostIntegerDataView, HostIntegerTensorView};
use runmat_builtins::{
    BuiltinErrorDescriptor, PAD_ERROR_ARGUMENT_CONFIG, PAD_ERROR_ARG_COUNT, PAD_ERROR_CELL_ELEMENT,
    PAD_ERROR_DIRECTION, PAD_ERROR_INTERNAL, PAD_ERROR_INVALID_INPUT, PAD_ERROR_LENGTH,
    PAD_ERROR_PAD_CHAR,
};

use runmat_macros::runtime_builtin;
use runmat_value::{CellArray, CharArray, IntValue, StringArray, StringElement, Value};

use crate::builtins::common::map_control_flow_with_builtin;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::common::tensor;
use crate::builtins::strings::common::{char_row_to_string_slice, is_missing_string};
use crate::{
    build_runtime_error, gather_if_needed_async, make_cell_with_shape, BuiltinResult, RuntimeError,
};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::strings::transform::pad")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "pad",
    op_kind: GpuOpKind::Custom("string-transform"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Executes on the CPU; GPU-resident inputs are gathered before padding to preserve MATLAB semantics.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::strings::transform::pad")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "pad",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "String transformation builtin; always gathers inputs and is not eligible for fusion.",
};

const BUILTIN_NAME: &str = "pad";

const MAX_PAD_TARGET_LENGTH: usize = isize::MAX as usize;

fn map_flow(err: RuntimeError) -> RuntimeError {
    map_control_flow_with_builtin(err, BUILTIN_NAME)
}

fn pad_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn pad_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    pad_error_with_message(error.message, error)
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum PadDirection {
    Left,
    Right,
    Both,
}

#[derive(Clone, Copy)]
enum PadTarget {
    Auto,
    Length(usize),
}

#[derive(Clone, Copy)]
struct PadOptions {
    target: PadTarget,
    direction: PadDirection,
    pad_char: char,
}

impl Default for PadOptions {
    fn default() -> Self {
        Self {
            target: PadTarget::Auto,
            direction: PadDirection::Right,
            pad_char: ' ',
        }
    }
}

impl PadOptions {
    fn base_target(&self, auto_target: usize) -> usize {
        match self.target {
            PadTarget::Auto => auto_target,
            PadTarget::Length(len) => len,
        }
    }
}

#[runtime_builtin(
    name = "pad",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::strings::transform::pad"
)]
async fn pad_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    value: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &runmat_builtins::PAD_DESCRIPTOR,
        &runmat_builtins::PAD_ERROR_TOO_MANY_OUTPUTS,
        "pad",
    )?;
    if crate::dispatcher::value_contains_gpu(&value) {
        return Err(pad_error(&PAD_ERROR_INVALID_INPUT));
    }
    if crate::builtins::strings::common::has_unsupported_character_pages(&value)
        || rest
            .iter()
            .any(crate::builtins::strings::common::has_unsupported_character_pages)
    {
        return Err(pad_error(&PAD_ERROR_INVALID_INPUT));
    }
    if rest.iter().any(
        |arg| matches!(arg, Value::GpuTensor(handle) if runmat_accelerate_api::handle_is_explicit(handle)),
    ) {
        return Err(pad_error(&PAD_ERROR_LENGTH));
    }
    let gathered_rest = rest
        .try_map_preserving_roots_async(rest.len(), runtime.as_ref(), async |index| {
            gather_if_needed_async(&rest[index]).await.map_err(map_flow)
        })
        .await?;
    let options = parse_arguments(gathered_rest.view())?;
    let gathered = gather_if_needed_async(&value).await.map_err(map_flow)?;
    match gathered {
        Value::String(text) => pad_string(text, options),
        Value::StringArray(array) => pad_string_array(array, options),
        Value::CharArray(array) => pad_char_array(array, options),
        Value::Cell(cell) => pad_cell_array(cell, options).await,
        _ => Err(pad_error(&PAD_ERROR_INVALID_INPUT)),
    }
}

fn pad_string(text: StringElement, options: PadOptions) -> BuiltinResult<Value> {
    if is_missing_string(&text) {
        return Ok(Value::String(text));
    }
    let char_count = string_length(text.as_str().expect("nonmissing subject"));
    let base_target = options.base_target(char_count);
    let target_len = element_target_length(&options, base_target, char_count);
    if char_count >= target_len {
        return Ok(Value::String(text));
    }
    Ok(Value::string(apply_padding_owned(
        text.as_str().expect("nonmissing subject"),
        char_count,
        target_len,
        &options,
    )?))
}

fn pad_string_array(array: StringArray, options: PadOptions) -> BuiltinResult<Value> {
    let (data, shape, _, _) = array.into_parts();
    let mut auto_len: usize = 0;
    if matches!(options.target, PadTarget::Auto) {
        for text in &data {
            if !is_missing_string(text) {
                auto_len = auto_len.max(string_length(text.as_str().expect("nonmissing subject")));
            }
        }
    }
    let base_target = options.base_target(auto_len);
    let mut padded: Vec<StringElement> = Vec::with_capacity(data.len());
    let values: Vec<StringElement> = match data.try_into_owned() {
        Ok(values) => values.into_iter().collect(),
        Err(data) => data.iter().cloned().collect(),
    };
    for text in values {
        if is_missing_string(&text) {
            padded.push(text);
            continue;
        }
        let char_count = string_length(text.as_str().expect("nonmissing subject"));
        let target_len = element_target_length(&options, base_target, char_count);
        if char_count >= target_len {
            padded.push(text);
        } else {
            padded.push(StringElement::text(apply_padding_owned(
                text.as_str().expect("nonmissing subject"),
                char_count,
                target_len,
                &options,
            )?));
        }
    }
    let result = StringArray::from_elements_unbounded(padded, shape.to_vec())
        .map_err(|e| pad_error_with_message(format!("{BUILTIN_NAME}: {e}"), &PAD_ERROR_INTERNAL))?;
    Ok(Value::StringArray(result))
}

fn pad_char_array(array: CharArray, options: PadOptions) -> BuiltinResult<Value> {
    if array.rows == 0 {
        return Ok(Value::CharArray(array));
    }
    let (data, _, rows, cols) = array.into_parts();

    let mut rows_text: Vec<String> = Vec::with_capacity(rows);
    let mut auto_len = 0usize;
    for row in 0..rows {
        let text = char_row_to_string_slice(&data, cols, row).map_err(|e| {
            pad_error_with_message(format!("{BUILTIN_NAME}: {e}"), &PAD_ERROR_INTERNAL)
        })?;
        auto_len = auto_len.max(text.encode_utf16().count());
        rows_text.push(text);
    }

    let base_target = options.base_target(auto_len);
    let mut padded_rows: Vec<String> = Vec::with_capacity(rows);
    let mut final_cols: usize = 0;
    for row_text in rows_text.into_iter() {
        let char_count = row_text.encode_utf16().count();
        let target_len = element_target_length(&options, base_target, char_count);
        let padded = apply_padding_owned(&row_text, char_count, target_len, &options)?;
        final_cols = final_cols.max(padded.encode_utf16().count());
        padded_rows.push(padded);
    }

    let total_chars = rows
        .checked_mul(final_cols)
        .ok_or_else(|| pad_error(&PAD_ERROR_LENGTH))?;
    let mut new_data: Vec<runmat_value::Utf16CodeUnit> = Vec::new();
    new_data
        .try_reserve_exact(total_chars)
        .map_err(|_| pad_error(&PAD_ERROR_LENGTH))?;
    for row_text in padded_rows.into_iter() {
        let mut units = row_text
            .encode_utf16()
            .map(runmat_value::Utf16CodeUnit::from_raw)
            .collect::<Vec<_>>();
        if units.len() < final_cols {
            units.resize(
                final_cols,
                runmat_value::Utf16CodeUnit::from_raw(b' ' as u16),
            );
        }
        new_data.extend(units);
    }

    CharArray::from_code_units(new_data, rows, final_cols)
        .map(Value::CharArray)
        .map_err(|e| pad_error_with_message(format!("{BUILTIN_NAME}: {e}"), &PAD_ERROR_INTERNAL))
}

async fn pad_cell_array(cell: CellArray, options: PadOptions) -> BuiltinResult<Value> {
    let shape = cell.shape().to_vec();
    let total = cell.data().len();
    let mut items: Vec<CellItem> = Vec::with_capacity(total);
    let mut auto_len = 0usize;

    for idx in 0..total {
        let value = &cell.data()[idx];
        let gathered = gather_if_needed_async(value).await.map_err(map_flow)?;
        let item = match gathered {
            Value::String(text) => {
                let is_missing = is_missing_string(&text);
                let len = if is_missing {
                    0
                } else {
                    string_length(text.as_str().expect("nonmissing subject"))
                };
                if !is_missing {
                    auto_len = auto_len.max(len);
                }
                CellItem {
                    kind: CellKind::String,
                    text,
                    char_count: len,
                }
            }
            Value::StringArray(sa) if sa.data().len() == 1 => {
                let text = sa.data().first().cloned().unwrap_or_default();
                let is_missing = is_missing_string(&text);
                let len = if is_missing {
                    0
                } else {
                    string_length(text.as_str().expect("nonmissing subject"))
                };
                if !is_missing {
                    auto_len = auto_len.max(len);
                }
                CellItem {
                    kind: CellKind::String,
                    text,
                    char_count: len,
                }
            }
            Value::CharArray(ca) if ca.rows == 0 => CellItem {
                kind: CellKind::EmptyChar(ca),
                text: StringElement::default(),
                char_count: 0,
            },
            Value::CharArray(ca) if ca.rows == 1 => {
                let text = char_row_to_string_slice(ca.data(), ca.cols, 0).map_err(|e| {
                    pad_error_with_message(format!("{BUILTIN_NAME}: {e}"), &PAD_ERROR_INTERNAL)
                })?;
                let len = text.encode_utf16().count();
                auto_len = auto_len.max(len);
                CellItem {
                    kind: CellKind::Char { rows: ca.rows },
                    text: StringElement::text(text),
                    char_count: len,
                }
            }
            Value::CharArray(_) => return Err(pad_error(&PAD_ERROR_CELL_ELEMENT)),
            _ => return Err(pad_error(&PAD_ERROR_CELL_ELEMENT)),
        };
        items.push(item);
    }

    let base_target = options.base_target(auto_len);
    let mut results: Vec<Value> = Vec::with_capacity(total);
    for item in items.into_iter() {
        match item.kind {
            // Preserve both the original empty payload owner and full shape.
            // A positive target must not create nonempty data with zero rows.
            CellKind::EmptyChar(array) => results.push(Value::CharArray(array)),
            CellKind::String if item.text.is_missing() => results.push(Value::String(item.text)),
            CellKind::String => {
                let target_len = element_target_length(&options, base_target, item.char_count);
                let padded = apply_padding_owned(
                    item.text.as_str().expect("nonmissing subject"),
                    item.char_count,
                    target_len,
                    &options,
                )?;
                results.push(Value::string(padded));
            }
            CellKind::Char { rows } => {
                let target_len = element_target_length(&options, base_target, item.char_count);
                let padded = apply_padding_owned(
                    item.text.as_str().expect("nonmissing subject"),
                    item.char_count,
                    target_len,
                    &options,
                )?;
                let chars: Vec<char> = padded.chars().collect();
                let cols = padded.encode_utf16().count();
                let array = CharArray::from_unicode_scalars(chars, rows, cols).map_err(|e| {
                    pad_error_with_message(format!("{BUILTIN_NAME}: {e}"), &PAD_ERROR_INTERNAL)
                })?;
                results.push(Value::CharArray(array));
            }
        }
    }

    make_cell_with_shape(results, shape)
        .map_err(|e| pad_error_with_message(format!("{BUILTIN_NAME}: {e}"), &PAD_ERROR_INTERNAL))
}

#[derive(Clone)]
struct CellItem {
    kind: CellKind,
    text: StringElement,
    char_count: usize,
}

#[derive(Clone)]
enum CellKind {
    String,
    Char { rows: usize },
    EmptyChar(CharArray),
}

fn parse_arguments(args: crate::call::FunctionArgumentView<'_>) -> BuiltinResult<PadOptions> {
    let mut options = PadOptions::default();
    match args.len() {
        0 => Ok(options),
        1 => {
            if let Some(length) = parse_length(&args[0])? {
                options.target = PadTarget::Length(length);
                return Ok(options);
            }
            if let Some(direction) = try_parse_direction(&args[0], false)? {
                options.direction = direction;
                return Ok(options);
            }
            let pad_char = parse_pad_char(&args[0])?;
            options.pad_char = pad_char;
            Ok(options)
        }
        2 => {
            if let Some(length) = parse_length(&args[0])? {
                options.target = PadTarget::Length(length);
                if let Some(direction) = try_parse_direction(&args[1], false)? {
                    options.direction = direction;
                } else {
                    match parse_pad_char(&args[1]) {
                        Ok(pad_char) => options.pad_char = pad_char,
                        Err(_) => return Err(pad_error(&PAD_ERROR_DIRECTION)),
                    }
                }
                Ok(options)
            } else if let Some(direction) = try_parse_direction(&args[0], false)? {
                options.direction = direction;
                let pad_char = parse_pad_char(&args[1])?;
                options.pad_char = pad_char;
                Ok(options)
            } else {
                Err(pad_error(&PAD_ERROR_ARGUMENT_CONFIG))
            }
        }
        3 => {
            let length = parse_length(&args[0])?.ok_or_else(|| pad_error(&PAD_ERROR_LENGTH))?;
            let direction = try_parse_direction(&args[1], true)?
                .ok_or_else(|| pad_error(&PAD_ERROR_DIRECTION))?;
            let pad_char = parse_pad_char(&args[2])?;
            options.target = PadTarget::Length(length);
            options.direction = direction;
            options.pad_char = pad_char;
            Ok(options)
        }
        _ => Err(pad_error(&PAD_ERROR_ARG_COUNT)),
    }
}

fn parse_length(value: &Value) -> BuiltinResult<Option<usize>> {
    match value {
        Value::Num(n) => parse_numeric_length(*n).map(Some),
        Value::Int(i) => i
            .try_to_usize()
            .filter(|length| *length <= MAX_PAD_TARGET_LENGTH)
            .map(Some)
            .ok_or_else(|| pad_error(&PAD_ERROR_LENGTH)),
        Value::Tensor(t) if tensor::is_scalar_tensor(t) => {
            if let Some(value) = t.integer_storage().and_then(|storage| storage.value_at(0)) {
                return parse_integer_length(&value).map(Some);
            }
            parse_numeric_length(tensor::tensor_value_f64(t, 0)).map(Some)
        }
        Value::Tensor(_) => Err(pad_error(&PAD_ERROR_LENGTH)),
        _ => Ok(None),
    }
}

fn parse_integer_length(value: &IntValue) -> BuiltinResult<usize> {
    value
        .try_to_usize()
        .filter(|length| *length <= MAX_PAD_TARGET_LENGTH)
        .ok_or_else(|| pad_error(&PAD_ERROR_LENGTH))
}

fn parse_numeric_length(value: f64) -> BuiltinResult<usize> {
    if !value.is_finite() || value < 0.0 {
        return Err(pad_error(&PAD_ERROR_LENGTH));
    }
    if (value.fract()).abs() > f64::EPSILON {
        return Err(pad_error(&PAD_ERROR_LENGTH));
    }
    if value >= MAX_PAD_TARGET_LENGTH as f64 {
        return Err(pad_error(&PAD_ERROR_LENGTH));
    }
    Ok(value as usize)
}

fn try_parse_direction(value: &Value, strict: bool) -> BuiltinResult<Option<PadDirection>> {
    let Some(text) = value_to_single_string(value)? else {
        return if strict {
            Err(pad_error(&PAD_ERROR_DIRECTION))
        } else {
            Ok(None)
        };
    };
    let lowered = text.trim().to_ascii_lowercase();
    if lowered.is_empty() {
        return if strict {
            Err(pad_error(&PAD_ERROR_DIRECTION))
        } else {
            Ok(None)
        };
    }
    let direction = match lowered.as_str() {
        "left" => PadDirection::Left,
        "right" => PadDirection::Right,
        "both" => PadDirection::Both,
        _ => {
            return if strict {
                Err(pad_error(&PAD_ERROR_DIRECTION))
            } else {
                Ok(None)
            };
        }
    };
    Ok(Some(direction))
}

fn parse_pad_char(value: &Value) -> BuiltinResult<char> {
    let text = value_to_single_string(value)?.ok_or_else(|| pad_error(&PAD_ERROR_PAD_CHAR))?;
    let mut chars = text.chars();
    let Some(first) = chars.next() else {
        return Err(pad_error(&PAD_ERROR_PAD_CHAR));
    };
    if chars.next().is_some() {
        return Err(pad_error(&PAD_ERROR_PAD_CHAR));
    }
    Ok(first)
}

fn value_to_single_string(value: &Value) -> BuiltinResult<Option<String>> {
    Ok(match value {
        Value::String(text) => text.as_str().map(str::to_owned),
        Value::StringArray(sa) => {
            if sa.data().len() == 1 {
                sa.data()[0].as_str().map(str::to_owned)
            } else {
                None
            }
        }
        Value::CharArray(ca) if ca.rows <= 1 => {
            if ca.rows == 0 {
                Some(String::new())
            } else {
                Some(
                    char_row_to_string_slice(ca.data(), ca.cols, 0).map_err(|e| {
                        pad_error_with_message(format!("{BUILTIN_NAME}: {e}"), &PAD_ERROR_INTERNAL)
                    })?,
                )
            }
        }
        _ => None,
    })
}

fn string_length(text: &str) -> usize {
    text.chars().count()
}

fn element_target_length(options: &PadOptions, base_target: usize, current_len: usize) -> usize {
    match options.target {
        PadTarget::Auto => base_target.max(current_len),
        PadTarget::Length(_) => base_target.max(current_len),
    }
}

fn apply_padding_owned(
    text: &str,
    current_len: usize,
    target_len: usize,
    options: &PadOptions,
) -> BuiltinResult<String> {
    if current_len >= target_len {
        return Ok(text.to_owned());
    }
    let delta = target_len - current_len;
    let (left_pad, right_pad) = match options.direction {
        PadDirection::Left => (delta, 0),
        PadDirection::Right => (0, delta),
        PadDirection::Both => {
            let left = delta / 2;
            (left, delta - left)
        }
    };
    let pad_bytes = delta
        .checked_mul(options.pad_char.len_utf8())
        .ok_or_else(|| pad_error(&PAD_ERROR_LENGTH))?;
    let capacity = text
        .len()
        .checked_add(pad_bytes)
        .ok_or_else(|| pad_error(&PAD_ERROR_LENGTH))?;
    let mut result = String::new();
    result
        .try_reserve_exact(capacity)
        .map_err(|_| pad_error(&PAD_ERROR_LENGTH))?;
    for _ in 0..left_pad {
        result.push(options.pad_char);
    }
    result.push_str(text);
    for _ in 0..right_pad {
        result.push(options.pad_char);
    }
    Ok(result)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use runmat_value::{IntValue, IntegerStorage, Tensor};

    fn pad_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        futures::executor::block_on(super::pad_builtin(
            None,
            value,
            crate::builtins::common::test_support::builtin_arguments(rest),
        ))
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn pad_string_length_right() {
        let result = pad_builtin(Value::String("GPU".into()), vec![Value::Num(5.0)]).expect("pad");
        assert_eq!(result, Value::String("GPU  ".into()));
    }

    #[test]
    fn pad_preserves_every_nd_cell_element_and_shape() {
        let empty = CharArray::from_code_units_with_shape(Vec::new(), vec![0, 5, 1])
            .expect("empty full shape");
        let empty_shape = empty.shape.clone();
        let input = make_cell_with_shape(
            vec![
                Value::CharArray(CharArray::new_row("x")),
                Value::string("yz"),
                Value::CharArray(empty),
            ],
            vec![1, 1, 3],
        )
        .expect("nd text cells");
        let Value::Cell(output) = pad_builtin(input, vec![Value::Num(3.0)]).expect("pad nd cells")
        else {
            panic!("cell output");
        };
        assert_eq!(output.shape(), &[1, 1, 3]);
        assert_eq!(output.data().len(), 3);
        assert_eq!(
            output.data()[0],
            Value::CharArray(CharArray::new_row("x  "))
        );
        assert_eq!(output.data()[1], Value::string("yz "));
        let Value::CharArray(empty_output) = &output.data()[2] else {
            panic!("empty character cell");
        };
        assert_eq!(empty_output.shape(), &[0, 5, 1]);
        assert!(empty_output.data().is_empty());
        empty_output
            .validate_layout()
            .expect("empty layout remains valid");
        assert!(
            std::ptr::eq(empty_shape.as_ptr(), empty_output.shape().as_ptr()),
            "retain the original full-shape owner"
        );
    }

    #[test]
    fn pad_length_reads_typed_integer_tensor_exactly() {
        let length = Tensor::new_integer(IntegerStorage::U64(vec![5]), vec![1, 1]).expect("length");

        let result =
            pad_builtin(Value::String("GPU".into()), vec![Value::Tensor(length)]).expect("pad");
        assert_eq!(result, Value::String("GPU  ".into()));
    }

    #[test]
    fn pad_gathers_automatic_integer_length_but_rejects_explicit_length() {
        test_support::with_test_provider(|provider| {
            let values = [5_u64];
            let handle = provider
                .upload_integer(&HostIntegerTensorView {
                    data: HostIntegerDataView::U64(&values),
                    shape: &[1, 1],
                })
                .expect("automatic integer length");
            let handle = handle
                .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Automatic)
                .expect("admit provenance view");
            let result = pad_builtin(
                Value::String("GPU".into()),
                vec![Value::GpuTensor(handle.clone())],
            )
            .expect("automatic residency is transparent");
            assert_eq!(result, Value::String("GPU  ".into()));

            let handle = handle
                .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
                .expect("admit provenance view");
            let error = pad_builtin(
                Value::String("GPU".into()),
                vec![Value::GpuTensor(handle.clone())],
            )
            .expect_err("explicit gpuArray length is unsupported");
            assert_eq!(error.identifier(), PAD_ERROR_LENGTH.identifier);
            handle.release().expect("release fixture handle");
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn pad_string_left_with_custom_char() {
        let result = pad_builtin(
            Value::String("42".into()),
            vec![
                Value::Num(4.0),
                Value::String("left".into()),
                Value::String("0".into()),
            ],
        )
        .expect("pad");
        assert_eq!(result, Value::String("0042".into()));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn pad_string_both_with_odd_count() {
        let result = pad_builtin(
            Value::String("core".into()),
            vec![
                Value::Num(9.0),
                Value::String("both".into()),
                Value::String("*".into()),
            ],
        )
        .expect("pad");
        assert_eq!(result, Value::String("**core***".into()));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn pad_string_array_auto_uses_longest_element() {
        let strings =
            StringArray::new(vec!["GPU".into(), "Accelerate".into()], vec![2, 1]).unwrap();
        let result = pad_builtin(Value::StringArray(strings), Vec::new()).expect("pad");
        match result {
            Value::StringArray(sa) => {
                assert_eq!(sa.data()[0], "GPU       ");
                assert_eq!(sa.data()[1], "Accelerate");
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn pad_string_array_pad_character_only() {
        let strings = StringArray::new(vec!["A".into(), "Run".into()], vec![2, 1]).unwrap();
        let result =
            pad_builtin(Value::StringArray(strings), vec![Value::String("*".into())]).expect("pad");
        match result {
            Value::StringArray(sa) => {
                assert_eq!(sa.data()[0], "A**");
                assert_eq!(sa.data()[1], "Run");
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn pad_string_array_length_with_pad_character() {
        let strings = StringArray::new(vec!["7".into(), "512".into()], vec![2, 1]).unwrap();
        let result = pad_builtin(
            Value::StringArray(strings),
            vec![Value::Num(4.0), Value::String("0".into())],
        )
        .expect("pad");
        match result {
            Value::StringArray(sa) => {
                assert_eq!(sa.data()[0], "7000");
                assert_eq!(sa.data()[1], "5120");
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn pad_string_array_direction_only() {
        let strings =
            StringArray::new(vec!["Mary".into(), "Elizabeth".into()], vec![2, 1]).unwrap();
        let result = pad_builtin(
            Value::StringArray(strings),
            vec![Value::String("left".into())],
        )
        .expect("pad");
        match result {
            Value::StringArray(sa) => {
                assert_eq!(sa.data()[0], "     Mary");
                assert_eq!(sa.data()[1], "Elizabeth");
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn pad_single_string_pad_character_only_leaves_length() {
        let result =
            pad_builtin(Value::String("GPU".into()), vec![Value::String("-".into())]).expect("pad");
        assert_eq!(result, Value::String("GPU".into()));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn pad_char_array_resizes_columns() {
        let chars: Vec<char> = "GPUrun".chars().collect();
        let array = CharArray::from_unicode_scalars(chars, 2, 3).unwrap();
        let result = pad_builtin(Value::CharArray(array), vec![Value::Num(5.0)]).expect("pad");
        match result {
            Value::CharArray(ca) => {
                assert_eq!(ca.rows, 2);
                assert_eq!(ca.cols, 5);
                assert_eq!(
                    runmat_value::decode_utf16_code_units(ca.data()).unwrap(),
                    "GPU  run  "
                );
            }
            other => panic!("expected char array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn pad_cell_array_mixed_content() {
        let cell = CellArray::new(
            vec![
                Value::String("solver".into()),
                Value::CharArray(CharArray::new_row("jit")),
                Value::String("planner".into()),
            ],
            1,
            3,
        )
        .unwrap();
        let result = pad_builtin(
            Value::Cell(cell),
            vec![Value::String("right".into()), Value::String(".".into())],
        )
        .expect("pad");
        match result {
            Value::Cell(out) => {
                assert_eq!(out.rows, 1);
                assert_eq!(out.cols, 3);
                assert_eq!(out.get(0, 0).unwrap(), Value::String("solver.".into()));
                assert_eq!(
                    out.get(0, 1).unwrap(),
                    Value::CharArray(CharArray::new_row("jit...."))
                );
                assert_eq!(out.get(0, 2).unwrap(), Value::String("planner".into()));
            }
            other => panic!("expected cell array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn pad_preserves_missing_string() {
        let result = pad_builtin(
            Value::String(runmat_value::StringElement::Missing),
            vec![Value::Num(8.0)],
        )
        .expect("pad");
        assert_eq!(result, Value::String(runmat_value::StringElement::Missing));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn pad_errors_on_invalid_input_type() {
        let err = pad_builtin(Value::Num(1.0), Vec::new()).unwrap_err();
        assert_eq!(err.to_string(), PAD_ERROR_INVALID_INPUT.message);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn pad_errors_on_negative_length() {
        let err = pad_builtin(Value::String("data".into()), vec![Value::Num(-1.0)]).unwrap_err();
        assert_eq!(err.to_string(), PAD_ERROR_LENGTH.message);
    }

    #[test]
    fn pad_length_rejects_negative_typed_integer_tensor() {
        let length =
            Tensor::new_integer(IntegerStorage::I64(vec![-1]), vec![1, 1]).expect("length");

        let err =
            pad_builtin(Value::String("data".into()), vec![Value::Tensor(length)]).unwrap_err();
        assert_eq!(err.to_string(), PAD_ERROR_LENGTH.message);
    }

    #[test]
    fn pad_length_rejects_oversized_exact_integer_and_double_lengths() {
        let length =
            Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX]), vec![1, 1]).expect("length");
        let err =
            pad_builtin(Value::String("data".into()), vec![Value::Tensor(length)]).unwrap_err();
        assert_eq!(err.to_string(), PAD_ERROR_LENGTH.message);

        let err = pad_builtin(
            Value::String("data".into()),
            vec![Value::Int(IntValue::U64(u64::MAX))],
        )
        .unwrap_err();
        assert_eq!(err.to_string(), PAD_ERROR_LENGTH.message);

        let err = pad_builtin(
            Value::String("data".into()),
            vec![Value::Num(MAX_PAD_TARGET_LENGTH as f64)],
        )
        .unwrap_err();
        assert_eq!(err.to_string(), PAD_ERROR_LENGTH.message);
    }

    #[test]
    fn pad_length_rejects_nonscalar_typed_integer_tensor() {
        let length =
            Tensor::new_integer(IntegerStorage::I32(vec![2, 3]), vec![1, 2]).expect("length");

        for args in [
            vec![Value::Tensor(length.clone())],
            vec![Value::Tensor(length.clone()), Value::String("left".into())],
            vec![
                Value::Tensor(length.clone()),
                Value::String("left".into()),
                Value::String("*".into()),
            ],
        ] {
            let err = pad_builtin(Value::String("data".into()), args).unwrap_err();
            assert_eq!(err.to_string(), PAD_ERROR_LENGTH.message);
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn pad_errors_on_invalid_direction() {
        let err = pad_builtin(
            Value::String("data".into()),
            vec![Value::Num(6.0), Value::String("around".into())],
        )
        .unwrap_err();
        assert_eq!(err.to_string(), PAD_ERROR_DIRECTION.message);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn pad_errors_on_invalid_pad_character() {
        let err = pad_builtin(
            Value::String("data".into()),
            vec![Value::String("left".into()), Value::String("##".into())],
        )
        .unwrap_err();
        assert_eq!(err.to_string(), PAD_ERROR_PAD_CHAR.message);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn pad_works_with_wgpu_provider_active() {
        test_support::with_test_provider(|_| {
            let result =
                pad_builtin(Value::String("GPU".into()), vec![Value::Num(6.0)]).expect("pad");
            assert_eq!(result, Value::String("GPU   ".into()));
        });
    }
}
