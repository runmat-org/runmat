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
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "csvwrite";

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::io::tabular::csvwrite")]
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

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::io::tabular::csvwrite")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "csvwrite",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Not eligible for fusion; performs host-side file I/O.",
};

fn csvwrite_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .build()
}

fn csvwrite_error_with_source<E>(message: impl Into<String>, source: E) -> RuntimeError
where
    E: std::error::Error + Send + Sync + 'static,
{
    build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .with_source(source)
        .build()
}

fn map_control_flow(err: RuntimeError) -> RuntimeError {
    let identifier = err.identifier().map(|value| value.to_string());
    let message = err.message().to_string();
    let mut builder = build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .with_source(err);
    if let Some(identifier) = identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runtime_builtin(
    name = "csvwrite",
    category = "io/tabular",
    summary = "Write numeric matrices to comma-separated text files using MATLAB-compatible offsets.",
    keywords = "csvwrite,csv,write,row offset,column offset",
    accel = "cpu",
    type_resolver(crate::builtins::io::type_resolvers::num_type),
    builtin_path = "crate::builtins::io::tabular::csvwrite"
)]
async fn csvwrite_builtin(
    filename: Value,
    data: Value,
    rest: Vec<Value>,
) -> crate::BuiltinResult<Value> {
    let filename_value = gather_if_needed_async(&filename)
        .await
        .map_err(map_control_flow)?;
    let path = resolve_path(&filename_value)?;

    let mut gathered_offsets = Vec::with_capacity(rest.len());
    for value in &rest {
        gathered_offsets.push(
            gather_if_needed_async(value)
                .await
                .map_err(map_control_flow)?,
        );
    }
    let (row_offset, col_offset) = parse_offsets(&gathered_offsets)?;

    let gathered_data = gather_if_needed_async(&data)
        .await
        .map_err(map_control_flow)?;
    let tensor =
        tensor::value_into_tensor_for("csvwrite", gathered_data).map_err(csvwrite_error)?;
    ensure_matrix_shape(&tensor)?;

    let bytes = write_csv(&path, &tensor, row_offset, col_offset)?;
    Ok(Value::Num(bytes as f64))
}

fn resolve_path(value: &Value) -> BuiltinResult<PathBuf> {
    let raw = match value {
        Value::String(s) => s.clone(),
        Value::CharArray(ca) if ca.rows == 1 => ca.data.iter().collect(),
        Value::StringArray(sa) if sa.data.len() == 1 => sa.data[0].clone(),
        _ => {
            return Err(csvwrite_error(
                "csvwrite: filename must be a string scalar or character vector",
            ))
        }
    };

    if raw.trim().is_empty() {
        return Err(csvwrite_error("csvwrite: filename must not be empty"));
    }

    let expanded = expand_user_path(&raw, BUILTIN_NAME).map_err(csvwrite_error)?;
    Ok(Path::new(&expanded).to_path_buf())
}

fn parse_offsets(args: &[Value]) -> BuiltinResult<(usize, usize)> {
    match args.len() {
        0 => Ok((0, 0)),
        2 => {
            let row = parse_offset(&args[0], "row offset")?;
            let col = parse_offset(&args[1], "column offset")?;
            Ok((row, col))
        }
        _ => Err(csvwrite_error(
            "csvwrite: offsets must be provided as two numeric arguments (row, column)",
        )),
    }
}

fn parse_offset(value: &Value, context: &str) -> BuiltinResult<usize> {
    match value {
        Value::Int(i) => {
            let raw = i.to_i64();
            if raw < 0 {
                return Err(csvwrite_error(format!("csvwrite: {context} must be >= 0")));
            }
            Ok(raw as usize)
        }
        Value::Num(n) => coerce_offset_from_float(*n, context),
        Value::Bool(b) => Ok(if *b { 1 } else { 0 }),
        Value::Tensor(t) => {
            if t.data.len() != 1 {
                return Err(csvwrite_error(format!(
                    "csvwrite: {context} must be a scalar, got {} elements",
                    t.data.len()
                )));
            }
            coerce_offset_from_float(t.data[0], context)
        }
        Value::LogicalArray(logical) => {
            if logical.data.len() != 1 {
                return Err(csvwrite_error(format!(
                    "csvwrite: {context} must be a scalar, got {} elements",
                    logical.data.len()
                )));
            }
            Ok(if logical.data[0] != 0 { 1 } else { 0 })
        }
        other => Err(csvwrite_error(format!(
            "csvwrite: {context} must be numeric, got {:?}",
            other
        ))),
    }
}

fn coerce_offset_from_float(value: f64, context: &str) -> BuiltinResult<usize> {
    if !value.is_finite() {
        return Err(csvwrite_error(format!(
            "csvwrite: {context} must be finite"
        )));
    }
    let rounded = value.round();
    if (rounded - value).abs() > 1e-9 {
        return Err(csvwrite_error(format!(
            "csvwrite: {context} must be an integer"
        )));
    }
    if rounded < 0.0 {
        return Err(csvwrite_error(format!("csvwrite: {context} must be >= 0")));
    }
    Ok(rounded as usize)
}

fn ensure_matrix_shape(tensor: &Tensor) -> BuiltinResult<()> {
    if tensor.shape.len() <= 2 {
        return Ok(());
    }
    if tensor.shape[2..].iter().all(|&dim| dim == 1) {
        return Ok(());
    }
    Err(csvwrite_error(
        "csvwrite: input must be 2-D; reshape before writing",
    ))
}

fn write_csv(
    path: &Path,
    tensor: &Tensor,
    row_offset: usize,
    col_offset: usize,
) -> BuiltinResult<usize> {
    let mut options = OpenOptions::new();
    options.create(true).write(true).truncate(true);
    let mut file = options.open(path).map_err(|err| {
        csvwrite_error_with_source(
            format!(
                "csvwrite: unable to open \"{}\" for writing ({err})",
                path.display()
            ),
            err,
        )
    })?;

    let line_ending = default_line_ending();
    let rows = tensor.rows();
    let cols = tensor.cols();

    let mut bytes_written = 0usize;

    for _ in 0..row_offset {
        file.write_all(line_ending.as_bytes()).map_err(|err| {
            csvwrite_error_with_source(
                format!("csvwrite: failed to write line ending ({err})"),
                err,
            )
        })?;
        bytes_written += line_ending.len();
    }

    if rows == 0 || cols == 0 {
        file.flush().map_err(|err| {
            csvwrite_error_with_source(format!("csvwrite: failed to flush output ({err})"), err)
        })?;
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
            file.write_all(line.as_bytes()).map_err(|err| {
                csvwrite_error_with_source(format!("csvwrite: failed to write value ({err})"), err)
            })?;
            bytes_written += line.len();
        }
        file.write_all(line_ending.as_bytes()).map_err(|err| {
            csvwrite_error_with_source(
                format!("csvwrite: failed to write line ending ({err})"),
                err,
            )
        })?;
        bytes_written += line_ending.len();
    }

    file.flush().map_err(|err| {
        csvwrite_error_with_source(format!("csvwrite: failed to flush output ({err})"), err)
    })?;

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
pub(crate) mod tests {
    use super::*;
    use runmat_time::unix_timestamp_ms;
    use std::fs;
    use std::sync::atomic::{AtomicU64, Ordering};

    use runmat_accelerate_api::HostTensorView;
    use runmat_builtins::{IntValue, LogicalArray};

    use crate::builtins::common::fs as fs_helpers;
    use crate::builtins::common::test_support;

    fn csvwrite_builtin(filename: Value, data: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        futures::executor::block_on(super::csvwrite_builtin(filename, data, rest))
    }

    static NEXT_ID: AtomicU64 = AtomicU64::new(0);

    fn temp_path(ext: &str) -> PathBuf {
        let millis = unix_timestamp_ms();
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

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
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

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
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

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
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

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
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

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
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
        let message = err.message().to_string();
        assert!(
            message.contains("row offset"),
            "unexpected error message: {message}"
        );
    }

    #[cfg(feature = "wgpu")]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
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

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
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

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
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
        let message = err.message().to_string();
        assert!(
            message.contains("csvwrite"),
            "unexpected error message: {message}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
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
}
