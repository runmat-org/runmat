//! MATLAB-compatible `num2str` builtin with GPU-aware semantics for RunMat.

use regex::Regex;
use runmat_builtins::{CharArray, ComplexTensor, Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::gpu_helpers;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::common::tensor;
use crate::gather_if_needed;

const DEFAULT_PRECISION: usize = 15;
const MAX_PRECISION: usize = 52;

#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(
        name = "num2str",
        builtin_path = "crate::builtins::strings::core::num2str"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "num2str"
category: "strings/core"
keywords: ["num2str", "number to string", "format", "precision", "gpu"]
summary: "Convert numeric scalars, vectors, and matrices into MATLAB-style character arrays using general or custom formats."
references:
  - https://www.mathworks.com/help/matlab/ref/num2str.html
gpu_support:
  elementwise: false
  reduction: false
  precisions: []
  broadcasting: "none"
  notes: "Always formats on the CPU. GPU tensors are gathered to host memory before conversion."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 1
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::strings::core::num2str::tests"
  integration: "builtins::strings::core::num2str::tests::num2str_gpu_tensor_roundtrip"
---

# What does the `num2str` function do in MATLAB / RunMat?
`num2str(x)` converts numeric scalars, vectors, and matrices into a character array where each
row of `x` becomes a row of text. Values use MATLAB's short-`g` formatting by default, and you can
provide a precision or an explicit format specifier to control the output. Complex inputs produce
`a ± bi` strings, and logical data is converted to `0` or `1`.

## How does the `num2str` function behave in MATLAB / RunMat?
- Default formatting uses up to 15 significant digits with MATLAB-style `g` behaviour (switching to
  scientific notation when needed).
- `num2str(x, p)` formats using `p` significant digits (`0 ≤ p ≤ 52`).
- `num2str(x, fmt)` accepts a single-number `printf`-style format such as `'%0.3f'`, `'%10.4e'`, or
  `'%.5g'`. Width, `+`, `-`, and `0` flags are supported.
- A trailing `'local'` argument switches the decimal separator to the one inferred from the active
  locale (or the `RUNMAT_DECIMAL_SEPARATOR` environment variable).
- Vector inputs return single-row character arrays; matrices return one textual row per numeric row.
- Empty matrices return empty character arrays that match MATLAB's dimension rules.
- Non-numeric types raise MATLAB-compatible errors.

## `num2str` Function GPU Execution Behaviour
When the input resides on the GPU, RunMat gathers the data back to host memory using the active
RunMat Accelerate provider before applying the formatting logic. The formatted character array
always lives on the CPU, so providers do not need to implement specialised kernels.

## Examples of using the `num2str` function in MATLAB / RunMat

### Converting A Scalar With Default Precision
```matlab
label = num2str(pi);
```
Expected output:
```matlab
label =
    '3.14159265358979'
```

### Formatting With A Specific Number Of Significant Digits
```matlab
digits = num2str(pi, 4);
```
Expected output:
```matlab
digits =
    '3.142'
```

### Using A Custom Format String
```matlab
row = num2str([1.234 5.678], '%.2f');
```
Expected output:
```matlab
row =
    '1.23  5.68'
```

### Displaying A Matrix With Column Alignment
```matlab
block = num2str([1 23 456; 78 9 10]);
```
Expected output:
```matlab
block =
    ' 1  23  456'
    '78   9   10'
```

### Formatting Complex Numbers
```matlab
z = num2str([3+4i 5-6i]);
```
Expected output:
```matlab
z =
    '3 + 4i  5 - 6i'
```

### Respecting Locale-Specific Decimal Separators
```matlab
text = num2str(0.125, 'local');
```
On locales that use a comma for decimals:
```matlab
text =
    '0,125'
```

### Converting GPU-Resident Data
```matlab
G = gpuArray([10.5 20.5]);
txt = num2str(G, '%.1f');
```
Expected output:
```matlab
txt =
    '10.5  20.5'
```
RunMat gathers the tensor to host memory before formatting.

## FAQ

### Can I request more than 15 digits?
Yes. Pass a precision between 0 and 52 to control the number of significant digits, e.g.
`num2str(x, 20)`.

### What format strings are supported?
RunMat supports single-value `printf` conversions using `%f`, `%e`, `%E`, `%g`, and `%G`, including
optional width, `+`, `-`, and `0` flags. Unsupported flags raise descriptive errors.

### Does `num2str` alter the size of my array?
No. The textual result has the same number of rows as the input and aligns each column with spaces.

### How are complex numbers rendered?
Real and imaginary components are formatted separately using the selected precision. The result is
`a + bi` or `a - bi`, with zero real parts simplifying to `bi`.

### How does the `'local'` flag work?
`num2str(..., 'local')` replaces the decimal point with the separator inferred from the active
locale. You can override the detected character with `RUNMAT_DECIMAL_SEPARATOR`, e.g.
`RUNMAT_DECIMAL_SEPARATOR=,`.

### What happens with non-numeric inputs?
Passing structs, objects, handles, or text raises a MATLAB-compatible error. Convert the data to
numeric form first or use `string` for rich text conversions.

## See Also
`sprintf`, `string`, `mat2str`, `str2double`
"#;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::strings::core::num2str")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "num2str",
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
    notes: "Always gathers GPU data to host memory before formatting numeric text.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::strings::core::num2str")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "num2str",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes:
        "Conversion builtin; not eligible for fusion and always materialises host character arrays.",
};

#[cfg_attr(
    feature = "doc_export",
    runtime_builtin(
        name = "num2str",
        category = "strings/core",
        summary = "Format numeric scalars, vectors, and matrices as character arrays.",
        keywords = "num2str,number,string,format,precision,gpu",
        accel = "sink",
        builtin_path = "crate::builtins::strings::core::num2str"
    )
)]
#[cfg_attr(
    not(feature = "doc_export"),
    runtime_builtin(
        name = "num2str",
        category = "strings/core",
        summary = "Format numeric scalars, vectors, and matrices as character arrays.",
        keywords = "num2str,number,string,format,precision,gpu",
        accel = "sink",
        builtin_path = "crate::builtins::strings::core::num2str"
    )
)]
fn num2str_builtin(value: Value, rest: Vec<Value>) -> Result<Value, String> {
    let gathered = gather_if_needed(&value).map_err(|e| format!("num2str: {e}"))?;
    let data = extract_numeric_data(gathered)?;

    let options = parse_options(rest)?;
    let char_array = format_numeric_data(data, &options)?;
    Ok(Value::CharArray(char_array))
}

struct FormatOptions {
    spec: FormatSpec,
    decimal: char,
}

#[derive(Clone)]
enum FormatSpec {
    General { digits: usize },
    Custom(CustomFormat),
}

#[derive(Clone)]
struct CustomFormat {
    kind: CustomKind,
    width: Option<usize>,
    precision: Option<usize>,
    sign_always: bool,
    left_align: bool,
    zero_pad: bool,
    uppercase: bool,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum CustomKind {
    Fixed,
    Exponent,
    General,
}

enum NumericData {
    Real {
        data: Vec<f64>,
        rows: usize,
        cols: usize,
    },
    Complex {
        data: Vec<(f64, f64)>,
        rows: usize,
        cols: usize,
    },
}

fn parse_options(args: Vec<Value>) -> Result<FormatOptions, String> {
    if args.is_empty() {
        return Ok(FormatOptions {
            spec: FormatSpec::General {
                digits: DEFAULT_PRECISION,
            },
            decimal: '.',
        });
    }

    let mut gathered = Vec::with_capacity(args.len());
    for arg in args {
        gathered.push(gather_if_needed(&arg).map_err(|e| format!("num2str: {e}"))?);
    }

    let mut iter = gathered.into_iter();
    let mut spec = FormatSpec::General {
        digits: DEFAULT_PRECISION,
    };
    let mut decimal = '.';

    if let Some(first) = iter.next() {
        if is_local_token(&first)? {
            decimal = detect_decimal_separator(true);
            if iter.next().is_some() {
                return Err("num2str: too many input arguments".to_string());
            }
            return Ok(FormatOptions { spec, decimal });
        }

        spec = if let Some(digits) = try_extract_precision(&first)? {
            FormatSpec::General { digits }
        } else if let Some(text) = value_to_text(&first) {
            FormatSpec::Custom(parse_custom_format(&text)?)
        } else {
            return Err(
                "num2str: second argument must be a precision or format string".to_string(),
            );
        };
    }

    if let Some(second) = iter.next() {
        if !is_local_token(&second)? {
            return Err("num2str: expected 'local' as the third argument".to_string());
        }
        decimal = detect_decimal_separator(true);
    }

    if iter.next().is_some() {
        return Err("num2str: too many input arguments".to_string());
    }

    Ok(FormatOptions { spec, decimal })
}

fn is_local_token(value: &Value) -> Result<bool, String> {
    let Some(text) = value_to_text(value) else {
        return Ok(false);
    };
    Ok(text.trim().eq_ignore_ascii_case("local"))
}

fn try_extract_precision(value: &Value) -> Result<Option<usize>, String> {
    match value {
        Value::Int(i) => {
            let digits = i.to_i64();
            validate_precision(digits)?;
            Ok(Some(digits as usize))
        }
        Value::Num(n) => {
            if !n.is_finite() {
                return Err("num2str: precision must be finite".to_string());
            }
            let rounded = n.round();
            if (rounded - n).abs() > f64::EPSILON {
                return Err("num2str: precision must be an integer".to_string());
            }
            validate_precision(rounded as i64)?;
            Ok(Some(rounded as usize))
        }
        Value::Tensor(t) if t.data.len() == 1 => {
            let value = t.data[0];
            if !value.is_finite() {
                return Err("num2str: precision must be finite".to_string());
            }
            let rounded = value.round();
            if (rounded - value).abs() > f64::EPSILON {
                return Err("num2str: precision must be an integer".to_string());
            }
            validate_precision(rounded as i64)?;
            Ok(Some(rounded as usize))
        }
        Value::LogicalArray(la) if la.data.len() == 1 => {
            let digits = if la.data[0] != 0 { 1 } else { 0 };
            validate_precision(digits)?;
            Ok(Some(digits as usize))
        }
        Value::Bool(b) => {
            let digits = if *b { 1 } else { 0 };
            Ok(Some(digits))
        }
        _ => Ok(None),
    }
}

fn validate_precision(value: i64) -> Result<(), String> {
    if value < 0 || value > MAX_PRECISION as i64 {
        return Err(format!(
            "num2str: precision must satisfy 0 <= p <= {MAX_PRECISION}"
        ));
    }
    Ok(())
}

fn value_to_text(value: &Value) -> Option<String> {
    match value {
        Value::String(s) => Some(s.clone()),
        Value::StringArray(sa) if sa.data.len() == 1 => Some(sa.data[0].clone()),
        Value::CharArray(ca) if ca.rows == 1 => Some(ca.data.iter().collect()),
        _ => None,
    }
}

fn detect_decimal_separator(local: bool) -> char {
    if !local {
        return '.';
    }

    if let Ok(custom) = std::env::var("RUNMAT_DECIMAL_SEPARATOR") {
        let trimmed = custom.trim();
        if let Some(ch) = trimmed.chars().next() {
            return ch;
        }
    }

    let locale = std::env::var("LC_NUMERIC")
        .or_else(|_| std::env::var("RUNMAT_LOCALE"))
        .or_else(|_| std::env::var("LANG"))
        .unwrap_or_default()
        .to_lowercase();

    if locale.is_empty() {
        return '.';
    }

    let comma_locales = [
        "af", "bs", "ca", "cs", "da", "de", "el", "es", "eu", "fi", "fr", "gl", "hr", "hu", "id",
        "is", "it", "lt", "lv", "nb", "nl", "pl", "pt", "ro", "ru", "sk", "sl", "sr", "sv", "tr",
        "uk", "vi",
    ];
    let locale_prefix = locale.split(['.', '_', '@']).next().unwrap_or(&locale);
    for prefix in &comma_locales {
        if locale_prefix.starts_with(prefix) {
            return ',';
        }
    }
    '.'
}

fn parse_custom_format(text: &str) -> Result<CustomFormat, String> {
    if !text.starts_with('%') {
        return Err("num2str: format must start with '%'".to_string());
    }
    if text == "%%" {
        return Err("num2str: '%' escape is not supported for numeric conversion".to_string());
    }

    static FORMAT_RE: once_cell::sync::Lazy<Regex> = once_cell::sync::Lazy::new(|| {
        Regex::new(r"^%([+\-0]*)(\d+)?(?:\.(\d*))?([fFeEgG])$").expect("format regex")
    });

    let captures = FORMAT_RE.captures(text).ok_or_else(|| {
        "num2str: unsupported format string; expected variants like '%0.3f' or '%.5g'".to_string()
    })?;

    let flags = captures.get(1).map(|m| m.as_str()).unwrap_or("");
    let width = captures
        .get(2)
        .map(|m| m.as_str().parse::<usize>().expect("width parse"));
    let precision = captures.get(3).map(|m| {
        if m.as_str().is_empty() {
            0usize
        } else {
            m.as_str().parse::<usize>().expect("precision parse")
        }
    });
    let conversion = captures
        .get(4)
        .map(|m| m.as_str().chars().next().unwrap())
        .unwrap();

    let mut sign_always = false;
    let mut left_align = false;
    let mut zero_pad = false;

    for ch in flags.chars() {
        match ch {
            '+' => sign_always = true,
            '-' => left_align = true,
            '0' => zero_pad = true,
            _ => {
                return Err(format!(
                    "num2str: unsupported format flag '{}'; only '+', '-', and '0' are supported",
                    ch
                ))
            }
        }
    }

    if let Some(p) = precision {
        if p > MAX_PRECISION {
            return Err(format!(
                "num2str: precision must satisfy 0 <= p <= {MAX_PRECISION}"
            ));
        }
    }

    let (kind, uppercase) = match conversion {
        'f' => (CustomKind::Fixed, false),
        'F' => (CustomKind::Fixed, true),
        'e' => (CustomKind::Exponent, false),
        'E' => (CustomKind::Exponent, true),
        'g' => (CustomKind::General, false),
        'G' => (CustomKind::General, true),
        _ => unreachable!(),
    };

    Ok(CustomFormat {
        kind,
        width,
        precision,
        sign_always,
        left_align,
        zero_pad,
        uppercase,
    })
}

fn extract_numeric_data(value: Value) -> Result<NumericData, String> {
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
        Value::Tensor(t) => tensor_to_numeric_data(t),
        Value::LogicalArray(la) => {
            let tensor = tensor::logical_to_tensor(&la)?;
            tensor_to_numeric_data(tensor)
        }
        Value::Complex(re, im) => Ok(NumericData::Complex {
            data: vec![(re, im)],
            rows: 1,
            cols: 1,
        }),
        Value::ComplexTensor(t) => complex_tensor_to_data(t),
        Value::GpuTensor(handle) => {
            let gathered = gpu_helpers::gather_tensor(&handle)?;
            tensor_to_numeric_data(gathered)
        }
        other => Err(format!(
            "num2str: unsupported input type {:?}; expected numeric or logical values",
            other
        )),
    }
}

fn tensor_to_numeric_data(tensor: Tensor) -> Result<NumericData, String> {
    if tensor.shape.len() > 2 {
        return Err("num2str: input must be scalar, vector, or 2-D matrix".to_string());
    }
    let rows = tensor.rows();
    let cols = tensor.cols();
    if rows == 0 || cols == 0 {
        return Ok(NumericData::Real {
            data: tensor.data,
            rows,
            cols,
        });
    }
    Ok(NumericData::Real {
        data: tensor.data,
        rows,
        cols,
    })
}

fn complex_tensor_to_data(tensor: ComplexTensor) -> Result<NumericData, String> {
    if tensor.shape.len() > 2 {
        return Err("num2str: complex input must be scalar, vector, or 2-D matrix".to_string());
    }
    let rows = tensor.rows;
    let cols = tensor.cols;
    Ok(NumericData::Complex {
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

fn format_numeric_data(data: NumericData, options: &FormatOptions) -> Result<CharArray, String> {
    match data {
        NumericData::Real { data, rows, cols } => format_real_matrix(&data, rows, cols, options),
        NumericData::Complex { data, rows, cols } => {
            format_complex_matrix(&data, rows, cols, options)
        }
    }
}

fn format_real_matrix(
    data: &[f64],
    rows: usize,
    cols: usize,
    options: &FormatOptions,
) -> Result<CharArray, String> {
    if rows == 0 {
        return CharArray::new(Vec::new(), 0, 0).map_err(|e| format!("num2str: {e}"));
    }
    if cols == 0 {
        return CharArray::new(Vec::new(), rows, 0).map_err(|e| format!("num2str: {e}"));
    }

    let mut entries = vec![
        vec![
            CellEntry {
                text: String::new(),
                width: 0
            };
            cols
        ];
        rows
    ];
    let mut col_widths = vec![0usize; cols];

    for (col, width) in col_widths.iter_mut().enumerate() {
        for (row, row_entries) in entries.iter_mut().enumerate() {
            let idx = row + col * rows;
            let value = data.get(idx).copied().unwrap_or(0.0);
            let text = format_real(value, &options.spec, options.decimal);
            let entry_width = text.chars().count();
            row_entries[col] = CellEntry {
                text,
                width: entry_width,
            };
            if entry_width > *width {
                *width = entry_width;
            }
        }
    }

    if cols > 1 {
        for (idx, width) in col_widths.iter_mut().enumerate() {
            if idx > 0 {
                *width += 1;
            }
        }
    }

    let rows_str = assemble_rows(entries, col_widths);
    rows_to_char_array(rows_str)
}

fn format_complex_matrix(
    data: &[(f64, f64)],
    rows: usize,
    cols: usize,
    options: &FormatOptions,
) -> Result<CharArray, String> {
    if rows == 0 {
        return CharArray::new(Vec::new(), 0, 0).map_err(|e| format!("num2str: {e}"));
    }
    if cols == 0 {
        return CharArray::new(Vec::new(), rows, 0).map_err(|e| format!("num2str: {e}"));
    }

    let mut entries = vec![
        vec![
            CellEntry {
                text: String::new(),
                width: 0
            };
            cols
        ];
        rows
    ];
    let mut col_widths = vec![0usize; cols];

    for (col, width) in col_widths.iter_mut().enumerate() {
        for (row, row_entries) in entries.iter_mut().enumerate() {
            let idx = row + col * rows;
            let (re, im) = data.get(idx).copied().unwrap_or((0.0, 0.0));
            let text = format_complex(re, im, &options.spec, options.decimal);
            let entry_width = text.chars().count();
            row_entries[col] = CellEntry {
                text,
                width: entry_width,
            };
            if entry_width > *width {
                *width = entry_width;
            }
        }
    }

    if cols > 1 {
        for (idx, width) in col_widths.iter_mut().enumerate() {
            if idx > 0 {
                *width += 1;
            }
        }
    }

    let rows_str = assemble_rows(entries, col_widths);
    rows_to_char_array(rows_str)
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

fn rows_to_char_array(rows: Vec<String>) -> Result<CharArray, String> {
    if rows.is_empty() {
        return CharArray::new(Vec::new(), 0, 0).map_err(|e| format!("num2str: {e}"));
    }
    let row_count = rows.len();
    let col_count = rows
        .iter()
        .map(|row| row.chars().count())
        .max()
        .unwrap_or(0);

    let mut data = Vec::with_capacity(row_count * col_count);
    for row in rows {
        let mut chars: Vec<char> = row.chars().collect();
        if chars.len() < col_count {
            chars.extend(std::iter::repeat_n(' ', col_count - chars.len()));
        }
        data.extend(chars);
    }

    CharArray::new(data, row_count, col_count).map_err(|e| format!("num2str: {e}"))
}

fn format_real(value: f64, spec: &FormatSpec, decimal: char) -> String {
    let text = match spec {
        FormatSpec::General { digits } => format_general(value, *digits, false),
        FormatSpec::Custom(custom) => format_custom(value, custom),
    };
    apply_decimal_locale(text, decimal)
}

fn format_complex(re: f64, im: f64, spec: &FormatSpec, decimal: char) -> String {
    let real_str = format_real(re, spec, decimal);
    let imag_sign = if im.is_sign_negative() { '-' } else { '+' };
    let abs_im = if im == 0.0 { 0.0 } else { im.abs() };
    let imag_str = format_real(abs_im, spec, decimal);

    if abs_im == 0.0 && !im.is_nan() {
        return real_str;
    }

    if re == 0.0 && !re.is_sign_negative() && !re.is_nan() {
        if im.is_sign_negative() && !im.is_nan() {
            return format!(
                "{}i",
                if imag_str.starts_with('-') {
                    imag_str.clone()
                } else {
                    format!("-{imag_str}")
                }
            );
        }
        return format!("{imag_str}i");
    }

    format!("{real_str} {imag_sign} {imag_str}i")
}

fn format_general(value: f64, digits: usize, uppercase: bool) -> String {
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

    let sig_digits = digits.max(1);
    let abs_val = value.abs();
    let exp10 = abs_val.log10().floor() as i32;
    let use_scientific = exp10 < -4 || exp10 >= sig_digits as i32;

    if use_scientific {
        let precision = sig_digits.saturating_sub(1);
        let s = if uppercase {
            format!("{:.*E}", precision, value)
        } else {
            format!("{:.*e}", precision, value)
        };
        let marker = if uppercase { 'E' } else { 'e' };
        if let Some(idx) = s.find(marker) {
            let (mantissa, exponent) = s.split_at(idx);
            let mut mant = mantissa.to_string();
            trim_trailing_zeros(&mut mant);
            normalize_negative_zero(&mut mant);
            let mut result = mant;
            result.push_str(exponent);
            return result;
        }
        s
    } else {
        let decimals = if sig_digits as i32 - 1 - exp10 < 0 {
            0
        } else {
            (sig_digits as i32 - 1 - exp10) as usize
        };
        let mut s = format!("{:.*}", decimals, value);
        trim_trailing_zeros(&mut s);
        normalize_negative_zero(&mut s);
        s
    }
}

fn trim_trailing_zeros(text: &mut String) {
    if let Some(dot_pos) = text.find('.') {
        let mut end = text.len();
        while end > dot_pos + 1 && text.as_bytes()[end - 1] == b'0' {
            end -= 1;
        }
        if end > dot_pos && text.as_bytes()[end - 1] == b'.' {
            end -= 1;
        }
        text.truncate(end);
    }
}

fn normalize_negative_zero(text: &mut String) {
    if text.starts_with('-') && text.chars().skip(1).all(|ch| ch == '0') {
        *text = "0".to_string();
    }
}

fn format_custom(value: f64, fmt: &CustomFormat) -> String {
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

    let precision = fmt.precision.unwrap_or(match fmt.kind {
        CustomKind::Fixed | CustomKind::Exponent => 6,
        CustomKind::General => DEFAULT_PRECISION,
    });

    let mut text = match fmt.kind {
        CustomKind::Fixed => format!("{:.*}", precision, value),
        CustomKind::Exponent => {
            let mut s = format!("{:.*e}", precision, value);
            if fmt.uppercase {
                s = s.to_uppercase();
            }
            s
        }
        CustomKind::General => format_general(value, precision.max(1), fmt.uppercase),
    };

    if fmt.kind != CustomKind::Fixed {
        trim_trailing_zeros(&mut text);
        normalize_negative_zero(&mut text);
    }

    apply_format_flags(text, fmt)
}

fn apply_decimal_locale(text: String, decimal: char) -> String {
    if decimal == '.' {
        return text;
    }
    let mut replaced = false;
    text.chars()
        .map(|ch| {
            if ch == '.' && !replaced {
                replaced = true;
                decimal
            } else {
                ch
            }
        })
        .collect()
}

fn apply_format_flags(mut text: String, fmt: &CustomFormat) -> String {
    if fmt.sign_always && !text.starts_with('-') && !text.starts_with('+') && text != "NaN" {
        text.insert(0, '+');
    }

    let width = fmt.width.unwrap_or(0);
    if width == 0 {
        return text;
    }

    let len = text.chars().count();
    if len >= width {
        return text;
    }

    let pad_count = width - len;
    let pad_char = if fmt.zero_pad && !fmt.left_align {
        '0'
    } else {
        ' '
    };

    if fmt.left_align {
        let mut result = text.clone();
        result.extend(std::iter::repeat_n(' ', pad_count));
        return result;
    }

    if pad_char == '0' && (text.starts_with('+') || text.starts_with('-')) {
        let mut chars = text.chars();
        let sign = chars.next().unwrap();
        let remainder: String = chars.collect();
        let mut result = String::with_capacity(width);
        result.push(sign);
        result.extend(std::iter::repeat_n('0', pad_count));
        result.push_str(&remainder);
        return result;
    }

    let mut result = String::with_capacity(width);
    result.extend(std::iter::repeat_n(' ', pad_count));
    result.push_str(&text);
    result
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use runmat_builtins::{IntValue, LogicalArray, Tensor};

    #[test]
    fn num2str_scalar_default_precision() {
        let value = Value::Num(std::f64::consts::PI);
        let out = num2str_builtin(value, Vec::new()).expect("num2str");
        match out {
            Value::CharArray(ca) => {
                let text: String = ca.data.iter().collect();
                assert_eq!(ca.rows, 1);
                assert!(text.starts_with("3.1415926535897"));
            }
            other => panic!("expected char array, got {other:?}"),
        }
    }

    #[test]
    fn num2str_precision_argument() {
        let value = Value::Num(std::f64::consts::PI);
        let out = num2str_builtin(value, vec![Value::Int(IntValue::I32(4))]).expect("num2str");
        match out {
            Value::CharArray(ca) => {
                let text: String = ca.data.iter().collect();
                assert_eq!(text.trim(), "3.142");
            }
            other => panic!("expected char array, got {other:?}"),
        }
    }

    #[test]
    fn num2str_matrix_alignment() {
        let tensor =
            Tensor::new(vec![1.0, 78.0, 23.0, 9.0, 456.0, 10.0], vec![2, 3]).expect("tensor");
        let out = num2str_builtin(Value::Tensor(tensor), Vec::new()).expect("num2str");
        match out {
            Value::CharArray(ca) => {
                assert_eq!(ca.rows, 2);
                assert_eq!(ca.cols, 11);
                let rows: Vec<String> = ca
                    .data
                    .chunks(ca.cols)
                    .map(|chunk| chunk.iter().collect())
                    .collect();
                assert_eq!(rows[0], " 1  23  456");
                assert_eq!(rows[1], "78   9   10");
            }
            other => panic!("expected char array, got {other:?}"),
        }
    }

    #[test]
    fn num2str_custom_format() {
        let tensor = Tensor::new(vec![1.234, 5.678], vec![1, 2]).expect("tensor");
        let fmt = Value::String("%.2f".to_string());
        let out = num2str_builtin(Value::Tensor(tensor), vec![fmt]).expect("num2str");
        match out {
            Value::CharArray(ca) => {
                let text: String = ca.data.iter().collect();
                assert_eq!(text, "1.23  5.68");
            }
            other => panic!("expected char array, got {other:?}"),
        }
    }

    #[test]
    fn num2str_complex_values() {
        let complex = ComplexTensor::new(vec![(3.0, 4.0), (5.0, -6.0)], vec![1, 2]).expect("cplx");
        let out = num2str_builtin(Value::ComplexTensor(complex), Vec::new()).expect("num2str");
        match out {
            Value::CharArray(ca) => {
                let text: String = ca.data.iter().collect();
                assert_eq!(text, "3 + 4i  5 - 6i");
            }
            other => panic!("expected char array, got {other:?}"),
        }
    }

    #[test]
    fn num2str_local_decimal() {
        std::env::set_var("RUNMAT_DECIMAL_SEPARATOR", ",");
        let out =
            num2str_builtin(Value::Num(0.5), vec![Value::String("local".into())]).expect("num2str");
        std::env::remove_var("RUNMAT_DECIMAL_SEPARATOR");
        match out {
            Value::CharArray(ca) => {
                let text: String = ca.data.iter().collect();
                assert_eq!(text, "0,5");
            }
            other => panic!("expected char array, got {other:?}"),
        }
    }

    #[test]
    fn num2str_logical_array() {
        let logical = LogicalArray::new(vec![1, 0, 1], vec![1, 3]).expect("logical");
        let out = num2str_builtin(Value::LogicalArray(logical), Vec::new()).expect("num2str");
        match out {
            Value::CharArray(ca) => {
                let text: String = ca.data.iter().collect();
                assert_eq!(text, "1  0  1");
            }
            other => panic!("expected char array, got {other:?}"),
        }
    }

    #[test]
    fn num2str_gpu_tensor_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![10.5, 20.5], vec![1, 2]).expect("tensor");
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let out = num2str_builtin(Value::GpuTensor(handle), vec![Value::String("%.1f".into())])
                .expect("num2str");
            match out {
                Value::CharArray(ca) => {
                    let text: String = ca.data.iter().collect();
                    assert_eq!(text, "10.5  20.5");
                }
                other => panic!("expected char array, got {other:?}"),
            }
        });
    }

    #[test]
    fn num2str_invalid_input_type() {
        let err = num2str_builtin(Value::String("hello".into()), Vec::new()).unwrap_err();
        assert!(err.contains("unsupported input type"));
    }

    #[test]
    fn num2str_invalid_format_string() {
        let err = num2str_builtin(Value::Num(1.0), vec![Value::String("%q".into())]).unwrap_err();
        assert!(err.contains("unsupported format string"));
    }

    #[test]
    fn doc_examples_present() {
        let blocks = crate::builtins::common::test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }
}
