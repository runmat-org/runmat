//! MATLAB-compatible `sprintf` builtin that mirrors printf-style formatting semantics.

use std::char;

use runmat_builtins::{CharArray, LogicalArray, StringArray, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::format::{format_variadic_with_cursor, ArgCursor};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
#[cfg(feature = "doc_export")]
use crate::register_builtin_doc_text;
use crate::{gather_if_needed, register_builtin_fusion_spec, register_builtin_gpu_spec};

#[cfg(feature = "doc_export")]
pub const DOC_MD: &str = r#"---
title: "sprintf"
category: "strings/core"
keywords: ["sprintf", "format", "printf", "text", "gpu"]
summary: "Format data into a MATLAB-compatible character vector using printf-style placeholders."
references:
  - https://www.mathworks.com/help/matlab/ref/sprintf.html
gpu_support:
  elementwise: false
  reduction: false
  precisions: []
  broadcasting: "none"
  notes: "Formatting executes on the CPU. GPU tensors are gathered via the active provider before substitution."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 1
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::strings::core::sprintf::tests"
  integration: "builtins::strings::core::sprintf::tests::sprintf_gpu_numeric"
---

# What does the `sprintf` function do in MATLAB / RunMat?
`sprintf(formatSpec, A1, ..., An)` formats data into a character row vector. It honours
MATLAB's printf-style placeholders, including flags (`-`, `+`, space, `0`, `#`), field width,
precision, and star (`*`) width/precision arguments. Arrays are processed column-major and the
format string repeats automatically until every element has been consumed.

## How does the `sprintf` function behave in MATLAB / RunMat?
- `formatSpec` must be text: a character row vector or string scalar. Cell arrays and multi-row
  character arrays are rejected.
- Arguments can be numeric, logical, strings, character arrays, or cell arrays containing supported
  scalar types. Numeric inputs accept scalar, vector, or matrix shapes; elements are flattened in
  column-major order.
- Escape sequences such as `\n`, `\t`, `\r`, `\f`, `\b`, `\a`, `\v`, octal (`\123`), and hexadecimal
  (`\x7F`) are interpreted before formatting so that `sprintf` can build multi-line text easily.
- Width and precision may be literal digits or drawn from the input list using `*` or `.*`. Star
  arguments must be scalar numerics and are consumed in order.
- Flags match MATLAB behaviour: `-` left-aligns, `+` forces a sign, space reserves a leading space
  for positive numbers, `0` enables zero padding, and `#` produces alternate forms for octal,
  hexadecimal, and binary outputs or preserves the decimal point in fixed-point conversions.
- `%%` emits a literal percent character without consuming arguments.
- Complex inputs are formatted as scalar text (`a+bi`) when used with `%s`; numeric conversions
  expect real scalars.
- Additional arguments beyond those required by the repeating format string raise an error to match
  MATLAB diagnostics.

## `sprintf` Function GPU Execution Behaviour
`sprintf` is a residency sink. When inputs include GPU tensors, RunMat gathers them back to host
memory via the active acceleration provider before executing the formatter. Formatting itself runs
entirely on the CPU, ensuring consistent behaviour regardless of device availability.

## Examples of using the `sprintf` function in MATLAB / RunMat

### Formatting A Scalar Value
```matlab
txt = sprintf('Value: %d', 42);
```
Expected output:
```matlab
txt =
    'Value: 42'
```

### Formatting With Precision And Width
```matlab
txt = sprintf('pi ~= %8.4f', pi);
```
Expected output:
```matlab
txt =
    'pi ~=   3.1416'
```

### Combining Strings And Numbers
```matlab
label = sprintf('Trial %02d: %.1f%% complete', 7, 84.95);
```
Expected output:
```matlab
label =
    'Trial 07: 85.0% complete'
```

### Formatting Vector Inputs
```matlab
data = [10 20 30];
txt = sprintf('%d ', data);
```
Expected output:
```matlab
txt =
    '10 20 30 '
```

### Using Star Width/Precision Arguments
```matlab
txt = sprintf('%*.*f %*.*f', 4, 2, 15.2, 6, 4, 0.001);
```
Expected output:
```matlab
txt =
    '15.20  0.0010'
```

### Quoting Text With `%s`
```matlab
names = ["Ada", "Grace"];
txt = sprintf('Hello, %s! ', names);
```
Expected output:
```matlab
txt =
    'Hello, Ada! Hello, Grace! '
```

### Formatting GPU-Resident Data
```matlab
G = gpuArray([1.5 2.5 3.5]);
txt = sprintf('%.1f;', G);
```
Expected output:
```matlab
txt =
    '1.5;2.5;3.5;'
```
RunMat gathers `G` to host memory automatically before formatting so the behaviour matches CPU inputs.

### Creating Multi-line Text
```matlab
message = sprintf('First line\\nSecond line\\t(indented)');
```
Expected output (showing control characters):
```matlab
message =
    'First line
Second line	(indented)'
```

## FAQ

### What happens if there are not enough input values for the format specifiers?
RunMat raises `sprintf: not enough input arguments for format specifier` as soon as a placeholder
cannot be satisfied. This matches MATLAB's error message.

### How are additional values treated when the format string contains fewer specifiers?
The format string repeats until all array elements are consumed. If the format string consumes no
arguments (for example, it contains only literal text) and extra values remain, RunMat raises an
error because MATLAB would also treat this as a mismatch.

### Can I use star (`*`) width or precision arguments with arrays?
Yes. Each `*` consumes the next scalar value from the input stream. When you provide arrays for
width or precision, elements are read in column-major order the same way data arguments are.

### Does `sprintf` support complex numbers?
Complex values can be formatted with `%s`, which renders MATLAB's canonical `a+bi` text. Numeric
conversions require real-valued inputs and raise an error for complex scalars.

### Are logical values supported?
Yes. Logical inputs are promoted to numeric (`1` or `0`) before formatting, so you can combine them
with integer or floating-point conversions.

### Does `sprintf` allocate string scalars?
No. `sprintf` always returns a character row vector for MATLAB compatibility. Use `string` or
`compose` if you need string scalars or string arrays.

### Does GPU hardware change formatting behaviour?
No. Formatting occurs on the CPU. When GPU tensors appear in the input list, RunMat gathers them to
host memory before substitution so the results match MATLAB exactly.

### How do I emit a literal percent sign?
Use `%%` inside the format specification. The formatter converts `%%` into a single `%` without
consuming an argument.

### How are empty inputs handled?
If all argument arrays are empty, `sprintf` returns an empty character array (`1×0`). If the format
string itself is empty, the result is also empty.

### What happens with multi-row character arrays?
MATLAB requires `formatSpec` to be a row vector. RunMat follows the same rule: multi-row character
arrays raise `sprintf: formatSpec must be a character row vector or string scalar`.

## See Also
[compose](./compose), [string](./string), [num2str](./num2str), [strlength](./strlength)
"#;

pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "sprintf",
    op_kind: GpuOpKind::Custom("format"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Formatting runs on the CPU; GPU tensors are gathered before substitution.",
};

register_builtin_gpu_spec!(GPU_SPEC);

pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "sprintf",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Formatting is a residency sink and is not fused; callers should treat sprintf as a CPU-only builtin.",
};

register_builtin_fusion_spec!(FUSION_SPEC);

#[cfg(feature = "doc_export")]
register_builtin_doc_text!("sprintf", DOC_MD);

#[runtime_builtin(
    name = "sprintf",
    category = "strings/core",
    summary = "Format data into a character vector using printf-style placeholders.",
    keywords = "sprintf,format,printf,text",
    accel = "format",
    sink = true
)]
fn sprintf_builtin(format_spec: Value, rest: Vec<Value>) -> Result<Value, String> {
    let gathered_spec = gather_if_needed(&format_spec).map_err(|e| format!("sprintf: {e}"))?;
    let raw_format = extract_format_string(&gathered_spec)?;
    let format_string = decode_escape_sequences(&raw_format)?;
    let flattened_args = flatten_arguments(&rest)?;
    let mut cursor = ArgCursor::new(&flattened_args);
    let mut output = String::new();

    loop {
        let step = format_variadic_with_cursor(&format_string, &mut cursor)?;
        output.push_str(&step.output);

        if step.consumed == 0 {
            if cursor.remaining() > 0 {
                return Err(
                    "sprintf: formatSpec contains no conversion specifiers but additional arguments were supplied"
                        .to_string(),
                );
            }
            break;
        }

        if cursor.remaining() == 0 {
            break;
        }
    }

    char_row_value(&output)
}

fn extract_format_string(value: &Value) -> Result<String, String> {
    match value {
        Value::String(s) => Ok(s.clone()),
        Value::CharArray(ca) => {
            if ca.rows != 1 {
                return Err(
                    "sprintf: formatSpec must be a character row vector or string scalar"
                        .to_string(),
                );
            }
            Ok(ca.data.iter().collect())
        }
        Value::StringArray(sa) if sa.data.len() == 1 => Ok(sa.data[0].clone()),
        _ => Err("sprintf: formatSpec must be a character row vector or string scalar".to_string()),
    }
}

fn decode_escape_sequences(input: &str) -> Result<String, String> {
    let mut result = String::with_capacity(input.len());
    let mut chars = input.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch != '\\' {
            result.push(ch);
            continue;
        }
        let Some(next) = chars.next() else {
            result.push('\\');
            break;
        };
        match next {
            '\\' => result.push('\\'),
            'a' => result.push('\u{0007}'),
            'b' => result.push('\u{0008}'),
            'f' => result.push('\u{000C}'),
            'n' => result.push('\n'),
            'r' => result.push('\r'),
            't' => result.push('\t'),
            'v' => result.push('\u{000B}'),
            'x' => {
                let mut hex = String::new();
                for _ in 0..2 {
                    match chars.peek().copied() {
                        Some(c) if c.is_ascii_hexdigit() => {
                            hex.push(chars.next().unwrap());
                        }
                        _ => break,
                    }
                }
                if hex.is_empty() {
                    result.push('\\');
                    result.push('x');
                } else {
                    let value = u32::from_str_radix(&hex, 16)
                        .map_err(|_| format!("sprintf: invalid hexadecimal escape \\x{hex}"))?;
                    if let Some(chr) = char::from_u32(value) {
                        result.push(chr);
                    } else {
                        return Err(format!(
                            "sprintf: \\x{hex} escape outside valid Unicode range"
                        ));
                    }
                }
            }
            '0'..='7' => {
                let mut oct = String::new();
                oct.push(next);
                for _ in 0..2 {
                    match chars.peek().copied() {
                        Some(c) if ('0'..='7').contains(&c) => {
                            oct.push(chars.next().unwrap());
                        }
                        _ => break,
                    }
                }
                let value = u32::from_str_radix(&oct, 8)
                    .map_err(|_| format!("sprintf: invalid octal escape \\{oct}"))?;
                if let Some(chr) = char::from_u32(value) {
                    result.push(chr);
                } else {
                    return Err(format!(
                        "sprintf: \\{oct} escape outside valid Unicode range"
                    ));
                }
            }
            other => {
                result.push('\\');
                result.push(other);
            }
        }
    }
    Ok(result)
}

fn flatten_arguments(args: &[Value]) -> Result<Vec<Value>, String> {
    let mut flattened = Vec::new();
    for value in args {
        let gathered = gather_if_needed(value).map_err(|e| format!("sprintf: {e}"))?;
        flatten_value(gathered, &mut flattened)?;
    }
    Ok(flattened)
}

fn flatten_value(value: Value, output: &mut Vec<Value>) -> Result<(), String> {
    match value {
        Value::Num(_)
        | Value::Int(_)
        | Value::Bool(_)
        | Value::String(_)
        | Value::Complex(_, _) => {
            output.push(value);
        }
        Value::Tensor(tensor) => {
            for &n in &tensor.data {
                output.push(Value::Num(n));
            }
        }
        Value::ComplexTensor(tensor) => {
            for &(re, im) in &tensor.data {
                output.push(Value::Complex(re, im));
            }
        }
        Value::LogicalArray(LogicalArray { data, .. }) => {
            for byte in data {
                output.push(Value::Bool(byte != 0));
            }
        }
        Value::StringArray(StringArray { data, .. }) => {
            for s in data {
                output.push(Value::String(s));
            }
        }
        Value::CharArray(ca) => {
            if ca.rows == 1 {
                output.push(Value::String(ca.data.iter().collect()));
            } else {
                for row in 0..ca.rows {
                    let mut line = String::with_capacity(ca.cols);
                    for col in 0..ca.cols {
                        line.push(ca.data[row * ca.cols + col]);
                    }
                    output.push(Value::String(line));
                }
            }
        }
        Value::Cell(cell) => {
            for col in 0..cell.cols {
                for row in 0..cell.rows {
                    let idx = row * cell.cols + col;
                    let inner = (*cell.data[idx]).clone();
                    let gathered = gather_if_needed(&inner).map_err(|e| format!("sprintf: {e}"))?;
                    flatten_value(gathered, output)?;
                }
            }
        }
        Value::GpuTensor(handle) => {
            let gathered =
                gather_if_needed(&Value::GpuTensor(handle)).map_err(|e| format!("sprintf: {e}"))?;
            flatten_value(gathered, output)?;
        }
        Value::MException(_)
        | Value::HandleObject(_)
        | Value::Listener(_)
        | Value::Object(_)
        | Value::Struct(_)
        | Value::FunctionHandle(_)
        | Value::Closure(_)
        | Value::ClassRef(_) => {
            return Err("sprintf: unsupported argument type".to_string());
        }
    }
    Ok(())
}

fn char_row_value(text: &str) -> Result<Value, String> {
    let chars: Vec<char> = text.chars().collect();
    let len = chars.len();
    let array = CharArray::new(chars, 1, len).map_err(|e| format!("sprintf: {e}"))?;
    Ok(Value::CharArray(array))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{builtins::common::test_support, make_cell};
    use runmat_builtins::{CharArray, IntValue, StringArray, Tensor};

    fn char_value_to_string(value: Value) -> String {
        match value {
            Value::CharArray(ca) => ca.data.into_iter().collect(),
            other => panic!("expected char output, got {other:?}"),
        }
    }

    #[test]
    fn sprintf_basic_integer() {
        let result = sprintf_builtin(
            Value::String("Value: %d".to_string()),
            vec![Value::Int(IntValue::I32(42))],
        )
        .expect("sprintf");
        assert_eq!(char_value_to_string(result), "Value: 42");
    }

    #[test]
    fn sprintf_float_precision() {
        let result = sprintf_builtin(
            Value::String("pi ~= %.3f".to_string()),
            vec![Value::Num(std::f64::consts::PI)],
        )
        .expect("sprintf");
        assert_eq!(char_value_to_string(result), "pi ~= 3.142");
    }

    #[test]
    fn sprintf_array_repeat() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let result = sprintf_builtin(
            Value::String("%d ".to_string()),
            vec![Value::Tensor(tensor)],
        )
        .expect("sprintf");
        assert_eq!(char_value_to_string(result), "1 2 3 ");
    }

    #[test]
    fn sprintf_star_width() {
        let args = vec![
            Value::Int(IntValue::I32(6)),
            Value::Int(IntValue::I32(2)),
            Value::Num(12.345),
        ];
        let result = sprintf_builtin(Value::String("%*.*f".to_string()), args).expect("sprintf");
        assert_eq!(char_value_to_string(result), " 12.35");
    }

    #[test]
    fn sprintf_literal_percent() {
        let result =
            sprintf_builtin(Value::String("%% complete".to_string()), Vec::new()).expect("sprintf");
        assert_eq!(char_value_to_string(result), "% complete");
    }

    #[test]
    fn sprintf_gpu_numeric() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let value = Value::GpuTensor(handle);
            let result =
                sprintf_builtin(Value::String("%0.1f,".to_string()), vec![value]).expect("sprintf");
            assert_eq!(char_value_to_string(result), "1.0,2.0,");
        });
    }

    #[test]
    fn sprintf_matrix_column_major() {
        let tensor = Tensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).unwrap();
        let result = sprintf_builtin(
            Value::String("%0.0f ".to_string()),
            vec![Value::Tensor(tensor)],
        )
        .expect("sprintf");
        assert_eq!(char_value_to_string(result), "1 3 2 4 ");
    }

    #[test]
    fn sprintf_not_enough_arguments_error() {
        let err = sprintf_builtin(
            Value::String("%d %d".to_string()),
            vec![Value::Int(IntValue::I32(1))],
        )
        .expect_err("sprintf should error");
        assert!(
            err.contains("not enough input arguments"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn sprintf_extra_arguments_error() {
        let err = sprintf_builtin(
            Value::String("literal text".to_string()),
            vec![Value::Int(IntValue::I32(1))],
        )
        .expect_err("sprintf should error");
        assert!(
            err.contains("contains no conversion specifiers"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn sprintf_format_spec_multirow_error() {
        let chars = CharArray::new("hi!".chars().collect(), 3, 1).unwrap();
        let err = sprintf_builtin(Value::CharArray(chars), Vec::new()).expect_err("sprintf");
        assert!(
            err.contains("formatSpec must be a character row vector"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn sprintf_percent_c_from_numeric() {
        let result = sprintf_builtin(
            Value::String("%c".to_string()),
            vec![Value::Int(IntValue::I32(65))],
        )
        .expect("sprintf");
        assert_eq!(char_value_to_string(result), "A");
    }

    #[test]
    fn sprintf_cell_arguments() {
        let cell = make_cell(
            vec![
                Value::Num(1.0),
                Value::String("two".to_string()),
                Value::Num(3.0),
            ],
            3,
            1,
        )
        .expect("cell");
        let result = sprintf_builtin(Value::String("%0.0f %s %0.0f".to_string()), vec![cell])
            .expect("sprintf");
        assert_eq!(char_value_to_string(result), "1 two 3");
    }

    #[test]
    fn sprintf_string_array_column_major() {
        let data = vec!["alpha".to_string(), "beta".to_string(), "gamma".to_string()];
        let array =
            StringArray::new(data, vec![3, 1]).expect("string array construction must succeed");
        let result = sprintf_builtin(
            Value::String("%s ".to_string()),
            vec![Value::StringArray(array)],
        )
        .expect("sprintf");
        assert_eq!(char_value_to_string(result), "alpha beta gamma ");
    }

    #[test]
    fn sprintf_complex_s_conversion() {
        let result = sprintf_builtin(
            Value::String("%s".to_string()),
            vec![Value::Complex(1.5, -2.0)],
        )
        .expect("sprintf");
        assert_eq!(char_value_to_string(result), "1.5-2i");
    }

    #[test]
    fn sprintf_escape_sequences() {
        let result = sprintf_builtin(
            Value::String("Line 1\\nLine 2\\t(tab)".to_string()),
            Vec::new(),
        )
        .expect("sprintf");
        assert_eq!(char_value_to_string(result), "Line 1\nLine 2\t(tab)");
    }

    #[test]
    fn sprintf_hex_and_octal_escapes() {
        let result =
            sprintf_builtin(Value::String("\\x41\\101".to_string()), Vec::new()).expect("sprintf");
        assert_eq!(char_value_to_string(result), "AA");
    }

    #[test]
    fn sprintf_unknown_escape_preserved() {
        let result =
            sprintf_builtin(Value::String("Value\\q".to_string()), Vec::new()).expect("sprintf");
        assert_eq!(char_value_to_string(result), "Value\\q");
    }

    #[test]
    #[cfg(feature = "doc_export")]
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }
}
