//! MATLAB-compatible `fullfile` builtin for RunMat.

use std::path::{Path, PathBuf};

use runmat_builtins::{CharArray, StringArray, Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const ERR_ARG_TYPE: &str = "fullfile: arguments must be character vectors or string scalars";

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::io::repl_fs::fullfile")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "fullfile",
    op_kind: GpuOpKind::Custom("io"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Host-only path assembly. GPU-resident text inputs are gathered before joining.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::io::repl_fs::fullfile")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "fullfile",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Filesystem path assembly is host-side and not eligible for fusion.",
};

const BUILTIN_NAME: &str = "fullfile";

fn fullfile_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .build()
}

fn map_control_flow(err: RuntimeError) -> RuntimeError {
    let identifier = err.identifier().map(str::to_string);
    let mut builder = build_runtime_error(format!("{BUILTIN_NAME}: {}", err.message()))
        .with_builtin(BUILTIN_NAME)
        .with_source(err);
    if let Some(identifier) = identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runtime_builtin(
    name = "fullfile",
    category = "io/repl_fs",
    summary = "Build a platform-correct file path from multiple path segments.",
    keywords = "fullfile,join paths,filesystem,filesep,path assembly",
    accel = "cpu",
    type_resolver(crate::builtins::io::type_resolvers::fullfile_type),
    builtin_path = "crate::builtins::io::repl_fs::fullfile"
)]
async fn fullfile_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    let gathered = gather_arguments(&args).await?;
    if gathered.is_empty() {
        return Err(fullfile_error("fullfile: not enough input arguments"));
    }

    let mut parts = Vec::with_capacity(gathered.len());
    for value in &gathered {
        parts.push(extract_text(value)?);
    }

    Ok(char_array_value(&join_parts(&parts)))
}

async fn gather_arguments(args: &[Value]) -> BuiltinResult<Vec<Value>> {
    let mut out = Vec::with_capacity(args.len());
    for value in args {
        out.push(
            gather_if_needed_async(value)
                .await
                .map_err(map_control_flow)?,
        );
    }
    Ok(out)
}

fn extract_text(value: &Value) -> BuiltinResult<String> {
    match value {
        Value::String(text) => Ok(text.clone()),
        Value::StringArray(StringArray { data, .. }) => {
            if data.len() != 1 {
                Err(fullfile_error(ERR_ARG_TYPE))
            } else {
                Ok(data[0].clone())
            }
        }
        Value::CharArray(chars) => {
            if chars.rows != 1 {
                return Err(fullfile_error(ERR_ARG_TYPE));
            }
            Ok(chars.data.iter().collect())
        }
        Value::Tensor(tensor) => tensor_to_string(tensor),
        Value::GpuTensor(_) => Err(fullfile_error(ERR_ARG_TYPE)),
        _ => Err(fullfile_error(ERR_ARG_TYPE)),
    }
}

fn tensor_to_string(tensor: &Tensor) -> BuiltinResult<String> {
    if tensor.shape.len() > 2 {
        return Err(fullfile_error(ERR_ARG_TYPE));
    }

    if tensor.rows() != 1 {
        return Err(fullfile_error(ERR_ARG_TYPE));
    }

    let mut text = String::with_capacity(tensor.data.len());
    for &code in &tensor.data {
        if !code.is_finite() {
            return Err(fullfile_error(ERR_ARG_TYPE));
        }
        let rounded = code.round();
        if (code - rounded).abs() > 1e-6 {
            return Err(fullfile_error(ERR_ARG_TYPE));
        }
        let int_code = rounded as i64;
        if !(0..=0x10FFFF).contains(&int_code) {
            return Err(fullfile_error(ERR_ARG_TYPE));
        }
        let ch = char::from_u32(int_code as u32).ok_or_else(|| fullfile_error(ERR_ARG_TYPE))?;
        text.push(ch);
    }

    Ok(text)
}

fn join_parts(parts: &[String]) -> String {
    let mut combined: Option<PathBuf> = None;
    for part in parts {
        if part.is_empty() {
            continue;
        }
        let normalized = normalize_component(part);
        if normalized.is_empty() {
            continue;
        }
        let path = Path::new(&normalized);
        match combined.as_mut() {
            None => combined = Some(PathBuf::from(path)),
            Some(buf) => {
                if path.is_absolute() {
                    *buf = PathBuf::from(path);
                } else {
                    buf.push(path);
                }
            }
        }
    }

    let mut text = combined
        .map(|path| path.to_string_lossy().into_owned())
        .unwrap_or_default();
    if cfg!(windows) {
        text = normalize_windows_separators(&text);
    }
    text
}

fn normalize_component(text: &str) -> String {
    if cfg!(windows) {
        text.replace('/', "\\")
    } else {
        text.to_string()
    }
}

fn normalize_windows_separators(text: &str) -> String {
    text.replace('/', "\\")
}

fn char_array_value(text: &str) -> Value {
    Value::CharArray(CharArray::new_row(text))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::super::REPL_FS_TEST_LOCK;
    use super::*;
    use std::convert::TryFrom;
    use std::path::PathBuf;

    fn fullfile_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
        futures::executor::block_on(super::fullfile_builtin(args))
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fullfile_joins_segments() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());

        let value = fullfile_builtin(vec![
            Value::from("data"),
            Value::from("raw"),
            Value::from("sample.dat"),
        ])
        .expect("fullfile");
        let text = String::try_from(&value).expect("string conversion");
        let sep = std::path::MAIN_SEPARATOR;
        let expected = format!("data{sep}raw{sep}sample.dat");
        assert_eq!(text, expected);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fullfile_skips_empty_segments() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());

        let value = fullfile_builtin(vec![
            Value::from(""),
            Value::from("data"),
            Value::from(""),
            Value::from("raw"),
        ])
        .expect("fullfile");
        let text = String::try_from(&value).expect("string conversion");
        let sep = std::path::MAIN_SEPARATOR;
        let expected = format!("data{sep}raw");
        assert_eq!(text, expected);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fullfile_resets_on_absolute_segment() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());

        let absolute = format!("{}root", std::path::MAIN_SEPARATOR);
        let value = fullfile_builtin(vec![
            Value::from("data"),
            Value::from(absolute.clone()),
            Value::from("file.txt"),
        ])
        .expect("fullfile");
        let text = String::try_from(&value).expect("string conversion");
        let mut expected = PathBuf::from(&absolute)
            .join("file.txt")
            .to_string_lossy()
            .into_owned();
        if cfg!(windows) {
            expected = expected.replace('/', "\\");
        }
        assert_eq!(text, expected);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fullfile_rejects_multi_row_char_array() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());

        let chars = CharArray::new(vec!['a', 'b', 'c', 'd'], 2, 2).expect("char array");
        let err = fullfile_builtin(vec![Value::CharArray(chars)]).expect_err("expected error");
        assert_eq!(err.message(), ERR_ARG_TYPE);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fullfile_rejects_multi_element_string_array() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());

        let array = StringArray::new(vec!["a".into(), "b".into()], vec![1, 2]).expect("array");
        let err = fullfile_builtin(vec![Value::StringArray(array)]).expect_err("expected error");
        assert_eq!(err.message(), ERR_ARG_TYPE);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fullfile_rejects_invalid_argument_types() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());

        let err = fullfile_builtin(vec![Value::Num(1.0)]).expect_err("expected error");
        assert!(err.message().contains("fullfile: arguments"));
    }
}
