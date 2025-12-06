//! MATLAB-compatible `csvwrite` builtin for RunMat.
//!
//! `csvwrite` is an older convenience wrapper that persists numeric matrices to
//! comma-separated text files. Modern MATLAB code typically prefers
//! `writematrix`, but many legacy scripts still depend on `csvwrite`'s terse
//! API and zero-based offset arguments. This implementation mirrors those
//! semantics while integrating with RunMat's builtin framework.

use std::io::Write;
use std::path::{Path, PathBuf};

use runmat_builtins::{Tensor, Value};
use runmat_filesystem::OpenOptions;
use runmat_macros::runtime_builtin;

use crate::builtins::common::fs::expand_user_path;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::common::tensor;
#[cfg(feature = "doc_export")]
use crate::register_builtin_doc_text;
use crate::{gather_if_needed, register_builtin_fusion_spec, register_builtin_gpu_spec};

#[cfg(feature = "doc_export")]
pub const DOC_MD: &str = r#"---
title: "csvwrite"
category: "io/tabular"
keywords: ["csvwrite", "csv", "write", "comma-separated values", "numeric export", "row offset", "column offset"]
summary: "Write numeric matrices to comma-separated text files using MATLAB-compatible offsets."
references:
  - https://www.mathworks.com/help/matlab/ref/csvwrite.html
gpu_support:
  elementwise: false
  reduction: false
  precisions: []
  broadcasting: "none"
  notes: "Runs entirely on the CPU. gpuArray inputs are gathered before serialisation."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 2
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::io::tabular::csvwrite::tests"
  integration:
    - "builtins::io::tabular::csvwrite::tests::csvwrite_writes_basic_matrix"
    - "builtins::io::tabular::csvwrite::tests::csvwrite_honours_offsets"
    - "builtins::io::tabular::csvwrite::tests::csvwrite_handles_gpu_tensors"
    - "builtins::io::tabular::csvwrite::tests::csvwrite_expands_home_directory"
    - "builtins::io::tabular::csvwrite::tests::csvwrite_formats_with_short_g_precision"
    - "builtins::io::tabular::csvwrite::tests::csvwrite_handles_wgpu_provider_gather"
    - "builtins::io::tabular::csvwrite::tests::csvwrite_rejects_negative_offsets"
---

# What does the `csvwrite` function do in MATLAB / RunMat?
`csvwrite(filename, M)` writes a numeric matrix to a comma-separated text file.
The builtin honours MATLAB's historical zero-based row/column offset arguments so
that existing scripts continue to behave identically in RunMat.

## How does the `csvwrite` function behave in MATLAB / RunMat?
- Only real numeric or logical inputs are accepted. Logical values are converted
  to `0` and `1` before writing. Complex and textual inputs raise descriptive
  errors.
- `csvwrite(filename, M, row, col)` starts writing at zero-based row `row` and
  column `col`, leaving earlier rows blank and earlier columns empty within each
  row. Offsets must be non-negative integers.
- Matrices must be 2-D (trailing singleton dimensions are ignored). Column-major
  ordering is respected when serialising to text.
- Numbers are emitted using MATLAB-compatible short `g` formatting (`%.5g`). `NaN`, `Inf`,
  and `-Inf` tokens are written verbatim.
- Existing files are overwritten. `csvwrite` does not support appending; switch
  to `writematrix` with `'WriteMode','append'` when the behaviour is required.
- Paths that begin with `~` expand to the user's home directory before writing.

## `csvwrite` Function GPU Execution Behaviour
`csvwrite` always executes on the host CPU. When the matrix resides on the GPU,
RunMat gathers the data through the active acceleration provider before
serialisation. No provider hooks are required, and the return value reports the
number of bytes written after the gather completes.

## Examples of using the `csvwrite` function in MATLAB / RunMat

### Writing a numeric matrix to CSV
```matlab
A = [1 2 3; 4 5 6];
csvwrite("scores.csv", A);
```
Expected contents of `scores.csv`:
```matlab
1,2,3
4,5,6
```

### Starting output after a header row
```matlab
fid = fopen("with_header.csv", "w");
fprintf(fid, "Name,Jan,Feb\nalpha,1,2\nbeta,3,4\n");
fclose(fid);

csvwrite("with_header.csv", [10 20; 30 40], 1, 0);
```
Expected contents of `with_header.csv`:
```matlab
Name,Jan,Feb

10,20
30,40
```

### Skipping leading columns before data
```matlab
B = magic(3);
csvwrite("offset_columns.csv", B, 0, 2);
```
Expected contents of `offset_columns.csv`:
```matlab
,,8,1,6
,,3,5,7
,,4,9,2
```

### Exporting logical masks as numeric zeros and ones
```matlab
mask = [true false true; false true false];
csvwrite("mask.csv", mask);
```
Expected contents of `mask.csv`:
```matlab
1,0,1
0,1,0
```

### Writing GPU-resident data without manual gather
```matlab
G = gpuArray(single([0.1 0.2 0.3]));
csvwrite("gpu_values.csv", G);
```
Expected behaviour:
```matlab
% Data is gathered automatically from the GPU and written to disk.
```

### Persisting a scalar value for downstream tools
```matlab
total = sum(rand(5));
csvwrite("scalar.csv", total);
```
Expected contents of `scalar.csv`:
```matlab
2.5731
```

## GPU residency in RunMat (Do I need `gpuArray`?)
No additional steps are necessary. `csvwrite` treats GPU arrays as residency
sinks: data is gathered back to host memory prior to writing. This matches
MATLAB's behaviour, where file I/O always operates on host-resident values.

## FAQ

### Why must the input be numeric or logical?
`csvwrite` predates MATLAB's table and string support and only serialises numeric
values. Provide numeric matrices or logical masks, or switch to `writematrix`
when you need to mix text and numbers.

### Are row and column offsets zero-based?
Yes. `row = 1` skips one full line before writing, and `col = 2` inserts two
empty comma-separated fields at the start of each written row.

### Can I append to an existing CSV with `csvwrite`?
No. `csvwrite` always overwrites the destination file. Use `writematrix` with
`'WriteMode','append'` or manipulate the file with lower-level I/O functions.

### How are `NaN` and `Inf` values written?
They are emitted verbatim as `NaN`, `Inf`, or `-Inf`, matching MATLAB's text
representation so that downstream tools can parse them consistently.

### What line ending does `csvwrite` use?
The builtin uses the platform default (`\r\n` on Windows, `\n` elsewhere). Most
CSV consumers handle either convention transparently.

## See Also
[csvread](./csvread), [readmatrix](./readmatrix), [writematrix](./writematrix), [fprintf](../filetext/fprintf), [gpuArray](../../acceleration/gpu/gpuArray), [gather](../../acceleration/gpu/gather)

## Source & Feedback
- The full source code for `csvwrite` lives at: [`crates/runmat-runtime/src/builtins/io/tabular/csvwrite.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/tabular/csvwrite.rs)
- Found a behavioural difference? [Open an issue](https://github.com/runmat-org/runmat/issues/new/choose) with details and a minimal reproduction.
"#;

pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "csvwrite",
    op_kind: GpuOpKind::Custom("io-csvwrite"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Runs entirely on the host; gpuArray inputs are gathered before serialisation.",
};

register_builtin_gpu_spec!(GPU_SPEC);

pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "csvwrite",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Not eligible for fusion; performs host-side file I/O.",
};

register_builtin_fusion_spec!(FUSION_SPEC);

#[cfg(feature = "doc_export")]
register_builtin_doc_text!("csvwrite", DOC_MD);

#[runtime_builtin(
    name = "csvwrite",
    category = "io/tabular",
    summary = "Write numeric matrices to comma-separated text files using MATLAB-compatible offsets.",
    keywords = "csvwrite,csv,write,row offset,column offset",
    accel = "cpu"
)]
fn csvwrite_builtin(filename: Value, data: Value, rest: Vec<Value>) -> Result<Value, String> {
    let filename_value = gather_if_needed(&filename).map_err(|e| format!("csvwrite: {e}"))?;
    let path = resolve_path(&filename_value)?;

    let (row_offset, col_offset) = parse_offsets(&rest)?;

    let gathered_data = gather_if_needed(&data).map_err(|e| format!("csvwrite: {e}"))?;
    let tensor = tensor::value_into_tensor_for("csvwrite", gathered_data)?;
    ensure_matrix_shape(&tensor)?;

    let bytes = write_csv(&path, &tensor, row_offset, col_offset)?;
    Ok(Value::Num(bytes as f64))
}

fn resolve_path(value: &Value) -> Result<PathBuf, String> {
    let raw = match value {
        Value::String(s) => s.clone(),
        Value::CharArray(ca) if ca.rows == 1 => ca.data.iter().collect(),
        Value::StringArray(sa) if sa.data.len() == 1 => sa.data[0].clone(),
        _ => {
            return Err(
                "csvwrite: filename must be a string scalar or character vector".to_string(),
            )
        }
    };

    if raw.trim().is_empty() {
        return Err("csvwrite: filename must not be empty".to_string());
    }

    let expanded = expand_user_path(&raw, "csvwrite").map_err(|e| format!("csvwrite: {e}"))?;
    Ok(Path::new(&expanded).to_path_buf())
}

fn parse_offsets(args: &[Value]) -> Result<(usize, usize), String> {
    match args.len() {
        0 => Ok((0, 0)),
        2 => {
            let row = parse_offset(&args[0], "row offset")?;
            let col = parse_offset(&args[1], "column offset")?;
            Ok((row, col))
        }
        _ => Err(
            "csvwrite: offsets must be provided as two numeric arguments (row, column)".to_string(),
        ),
    }
}

fn parse_offset(value: &Value, context: &str) -> Result<usize, String> {
    match value {
        Value::Int(i) => {
            let raw = i.to_i64();
            if raw < 0 {
                return Err(format!("csvwrite: {context} must be >= 0"));
            }
            Ok(raw as usize)
        }
        Value::Num(n) => coerce_offset_from_float(*n, context),
        Value::Bool(b) => Ok(if *b { 1 } else { 0 }),
        Value::Tensor(t) => {
            if t.data.len() != 1 {
                return Err(format!(
                    "csvwrite: {context} must be a scalar, got {} elements",
                    t.data.len()
                ));
            }
            coerce_offset_from_float(t.data[0], context)
        }
        Value::LogicalArray(logical) => {
            if logical.data.len() != 1 {
                return Err(format!(
                    "csvwrite: {context} must be a scalar, got {} elements",
                    logical.data.len()
                ));
            }
            Ok(if logical.data[0] != 0 { 1 } else { 0 })
        }
        other => Err(format!(
            "csvwrite: {context} must be numeric, got {:?}",
            other
        )),
    }
}

fn coerce_offset_from_float(value: f64, context: &str) -> Result<usize, String> {
    if !value.is_finite() {
        return Err(format!("csvwrite: {context} must be finite"));
    }
    let rounded = value.round();
    if (rounded - value).abs() > 1e-9 {
        return Err(format!("csvwrite: {context} must be an integer"));
    }
    if rounded < 0.0 {
        return Err(format!("csvwrite: {context} must be >= 0"));
    }
    Ok(rounded as usize)
}

fn ensure_matrix_shape(tensor: &Tensor) -> Result<(), String> {
    if tensor.shape.len() <= 2 {
        return Ok(());
    }
    if tensor.shape[2..].iter().all(|&dim| dim == 1) {
        return Ok(());
    }
    Err("csvwrite: input must be 2-D; reshape before writing".to_string())
}

fn write_csv(
    path: &Path,
    tensor: &Tensor,
    row_offset: usize,
    col_offset: usize,
) -> Result<usize, String> {
    let mut options = OpenOptions::new();
    options.create(true).write(true).truncate(true);
    let mut file = options.open(path).map_err(|err| {
        format!(
            "csvwrite: unable to open \"{}\" for writing ({err})",
            path.display()
        )
    })?;

    let line_ending = default_line_ending();
    let rows = tensor.rows();
    let cols = tensor.cols();

    let mut bytes_written = 0usize;

    for _ in 0..row_offset {
        file.write_all(line_ending.as_bytes())
            .map_err(|err| format!("csvwrite: failed to write line ending ({err})"))?;
        bytes_written += line_ending.len();
    }

    if rows == 0 || cols == 0 {
        file.flush()
            .map_err(|err| format!("csvwrite: failed to flush output ({err})"))?;
        return Ok(bytes_written);
    }

    for row in 0..rows {
        let mut fields = Vec::with_capacity(col_offset + cols);
        for _ in 0..col_offset {
            fields.push(String::new());
        }
        for col in 0..cols {
            let idx = row + col * rows;
            let value = tensor.data[idx];
            fields.push(format_numeric(value));
        }
        let line = fields.join(",");
        if !line.is_empty() {
            file.write_all(line.as_bytes())
                .map_err(|err| format!("csvwrite: failed to write value ({err})"))?;
            bytes_written += line.len();
        }
        file.write_all(line_ending.as_bytes())
            .map_err(|err| format!("csvwrite: failed to write line ending ({err})"))?;
        bytes_written += line_ending.len();
    }

    file.flush()
        .map_err(|err| format!("csvwrite: failed to flush output ({err})"))?;

    Ok(bytes_written)
}

fn default_line_ending() -> &'static str {
    if cfg!(windows) {
        "\r\n"
    } else {
        "\n"
    }
}

fn format_numeric(value: f64) -> String {
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

    let precision: i32 = 5;
    let abs = value.abs();
    let exp10 = abs.log10().floor() as i32;
    let use_scientific = exp10 < -4 || exp10 >= precision;

    let raw = if use_scientific {
        let digits_after = (precision - 1).max(0) as usize;
        format!("{:.*e}", digits_after, value)
    } else {
        let decimals = (precision - 1 - exp10).max(0) as usize;
        format!("{:.*}", decimals, value)
    };

    let mut trimmed = trim_trailing_zeros(raw);
    if trimmed == "-0" {
        trimmed = "0".to_string();
    }
    trimmed
}

fn trim_trailing_zeros(mut value: String) -> String {
    if let Some(exp_pos) = value.find(['e', 'E']) {
        let exponent = value.split_off(exp_pos);
        while value.ends_with('0') {
            value.pop();
        }
        if value.ends_with('.') {
            value.pop();
        }
        value.push_str(&normalize_exponent(&exponent));
        value
    } else {
        if value.contains('.') {
            while value.ends_with('0') {
                value.pop();
            }
            if value.ends_with('.') {
                value.pop();
            }
        }
        if value.is_empty() {
            "0".to_string()
        } else {
            value
        }
    }
}

fn normalize_exponent(exponent: &str) -> String {
    if exponent.len() <= 1 {
        return exponent.to_string();
    }
    let mut chars = exponent.chars();
    let marker = chars.next().unwrap();
    let rest: String = chars.collect();
    match rest.parse::<i32>() {
        Ok(parsed) => format!("{}{:+03}", marker, parsed),
        Err(_) => exponent.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    use runmat_accelerate_api::HostTensorView;
    use runmat_builtins::{IntValue, LogicalArray};

    use crate::builtins::common::fs as fs_helpers;
    use crate::builtins::common::test_support;

    static NEXT_ID: AtomicU64 = AtomicU64::new(0);

    fn temp_path(ext: &str) -> PathBuf {
        let millis = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis();
        let unique = NEXT_ID.fetch_add(1, Ordering::Relaxed);
        let mut path = std::env::temp_dir();
        path.push(format!(
            "runmat_csvwrite_{}_{}_{}.{}",
            std::process::id(),
            millis,
            unique,
            ext
        ));
        path
    }

    fn line_ending() -> &'static str {
        default_line_ending()
    }

    #[test]
    fn csvwrite_writes_basic_matrix() {
        let path = temp_path("csv");
        let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();
        let filename = path.to_string_lossy().into_owned();

        csvwrite_builtin(Value::from(filename), Value::Tensor(tensor), Vec::new())
            .expect("csvwrite");

        let contents = fs::read_to_string(&path).expect("read contents");
        assert_eq!(contents, format!("1,2,3{le}4,5,6{le}", le = line_ending()));
        let _ = fs::remove_file(path);
    }

    #[test]
    fn csvwrite_honours_offsets() {
        let path = temp_path("csv");
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let filename = path.to_string_lossy().into_owned();

        csvwrite_builtin(
            Value::from(filename),
            Value::Tensor(tensor),
            vec![Value::Int(IntValue::I32(1)), Value::Int(IntValue::I32(2))],
        )
        .expect("csvwrite");

        let contents = fs::read_to_string(&path).expect("read contents");
        assert_eq!(
            contents,
            format!("{le},,1,3{le},,2,4{le}", le = line_ending())
        );
        let _ = fs::remove_file(path);
    }

    #[test]
    fn csvwrite_handles_gpu_tensors() {
        test_support::with_test_provider(|provider| {
            let path = temp_path("csv");
            let tensor = Tensor::new(vec![0.5, 1.5], vec![1, 2]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let filename = path.to_string_lossy().into_owned();

            csvwrite_builtin(Value::from(filename), Value::GpuTensor(handle), Vec::new())
                .expect("csvwrite");

            let contents = fs::read_to_string(&path).expect("read contents");
            assert_eq!(contents, format!("0.5,1.5{le}", le = line_ending()));
            let _ = fs::remove_file(path);
        });
    }

    #[test]
    fn csvwrite_formats_with_short_g_precision() {
        let path = temp_path("csv");
        let values =
            Tensor::new(vec![12.3456, 1_234_567.0, 0.000123456, -0.0], vec![1, 4]).unwrap();
        let filename = path.to_string_lossy().into_owned();

        csvwrite_builtin(Value::from(filename), Value::Tensor(values), Vec::new())
            .expect("csvwrite");

        let contents = fs::read_to_string(&path).expect("read contents");
        assert_eq!(
            contents,
            format!("12.346,1.2346e+06,0.00012346,0{le}", le = line_ending())
        );
        let _ = fs::remove_file(path);
    }

    #[test]
    fn csvwrite_rejects_negative_offsets() {
        let path = temp_path("csv");
        let tensor = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
        let filename = path.to_string_lossy().into_owned();
        let err = csvwrite_builtin(
            Value::from(filename),
            Value::Tensor(tensor),
            vec![Value::Num(-1.0), Value::Num(0.0)],
        )
        .expect_err("negative offsets should be rejected");
        assert!(
            err.contains("row offset"),
            "unexpected error message: {err}"
        );
    }

    #[cfg(feature = "wgpu")]
    #[test]
    fn csvwrite_handles_wgpu_provider_gather() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let Some(provider) = runmat_accelerate_api::provider() else {
            panic!("wgpu provider not registered");
        };

        let path = temp_path("csv");
        let tensor = Tensor::new(vec![2.0, 4.0], vec![1, 2]).unwrap();
        let view = HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = provider.upload(&view).expect("upload");
        let filename = path.to_string_lossy().into_owned();

        csvwrite_builtin(Value::from(filename), Value::GpuTensor(handle), Vec::new())
            .expect("csvwrite");

        let contents = fs::read_to_string(&path).expect("read contents");
        assert_eq!(contents, format!("2,4{le}", le = line_ending()));
        let _ = fs::remove_file(path);
    }

    #[test]
    fn csvwrite_expands_home_directory() {
        let Some(mut home) = fs_helpers::home_directory() else {
            // Skip when home directory cannot be determined.
            return;
        };
        let filename = format!(
            "runmat_csvwrite_home_{}_{}.csv",
            std::process::id(),
            NEXT_ID.fetch_add(1, Ordering::Relaxed)
        );
        home.push(&filename);

        let tilde_path = format!("~/{}", filename);
        let tensor = Tensor::new(vec![42.0], vec![1, 1]).unwrap();

        csvwrite_builtin(Value::from(tilde_path), Value::Tensor(tensor), Vec::new())
            .expect("csvwrite");

        let contents = fs::read_to_string(&home).expect("read contents");
        assert_eq!(contents, format!("42{le}", le = line_ending()));
        let _ = fs::remove_file(home);
    }

    #[test]
    fn csvwrite_rejects_non_numeric_inputs() {
        let path = temp_path("csv");
        let filename = path.to_string_lossy().into_owned();
        let err = csvwrite_builtin(
            Value::from(filename),
            Value::String("abc".into()),
            Vec::new(),
        )
        .expect_err("csvwrite should fail");
        assert!(err.contains("csvwrite"), "unexpected error message: {err}");
    }

    #[test]
    fn csvwrite_accepts_logical_arrays() {
        let path = temp_path("csv");
        let logical = LogicalArray::new(vec![1, 0, 1, 0], vec![2, 2]).unwrap();
        let filename = path.to_string_lossy().into_owned();

        csvwrite_builtin(
            Value::from(filename),
            Value::LogicalArray(logical),
            Vec::new(),
        )
        .expect("csvwrite");

        let contents = fs::read_to_string(&path).expect("read contents");
        assert_eq!(contents, format!("1,1{le}0,0{le}", le = line_ending()));
        let _ = fs::remove_file(path);
    }

    #[test]
    #[cfg(feature = "doc_export")]
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }
}
