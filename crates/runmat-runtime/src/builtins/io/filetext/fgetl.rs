//! MATLAB-compatible `fgetl` builtin for RunMat.

use runmat_builtins::Value;
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::io::filetext::{
    helpers::{bytes_to_char_array, read_text_line},
    registry,
};
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const INVALID_IDENTIFIER_MESSAGE: &str =
    "Invalid file identifier. Use fopen to generate a valid file ID.";
const BUILTIN_NAME: &str = "fgetl";

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::io::filetext::fgetl")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "fgetl",
    op_kind: GpuOpKind::Custom("file-io"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Host-only file I/O; arguments gathered from the GPU when necessary.",
};

fn fgetl_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .build()
}

fn map_control_flow(err: RuntimeError) -> RuntimeError {
    let message = err.message().to_string();
    let identifier = err.identifier().map(|value| value.to_string());
    let mut builder = build_runtime_error(format!("{BUILTIN_NAME}: {message}"))
        .with_builtin(BUILTIN_NAME)
        .with_source(err);
    if let Some(identifier) = identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::io::filetext::fgetl")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "fgetl",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "File I/O calls are not eligible for fusion.",
};

#[runtime_builtin(
    name = "fgetl",
    category = "io/filetext",
    summary = "Read the next line from a file, excluding newline characters.",
    keywords = "fgetl,file,io,line,newline",
    accel = "cpu",
    type_resolver(crate::builtins::io::type_resolvers::fgetl_type),
    builtin_path = "crate::builtins::io::filetext::fgetl"
)]
async fn fgetl_builtin(fid: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    let eval = evaluate(&fid, &rest).await?;
    if let Some(out_count) = crate::output_count::current_output_count() {
        if out_count == 0 {
            return Ok(Value::OutputList(Vec::new()));
        }
        return Ok(crate::output_count::output_list_with_padding(
            out_count,
            vec![eval.first_output()],
        ));
    }
    Ok(eval.first_output())
}

#[derive(Clone, Debug)]
pub struct FgetlEval {
    line: Value,
}

impl FgetlEval {
    fn new(line: Value) -> Self {
        Self { line }
    }

    fn end_of_file() -> Self {
        Self {
            line: Value::Num(-1.0),
        }
    }

    pub fn first_output(&self) -> Value {
        self.line.clone()
    }
}

pub async fn evaluate(fid_value: &Value, rest: &[Value]) -> BuiltinResult<FgetlEval> {
    if !rest.is_empty() {
        return Err(fgetl_error("fgetl: too many input arguments"));
    }

    let fid_host = gather_value(fid_value).await?;
    let fid = parse_fid(&fid_host)?;
    if fid < 0 {
        return Err(fgetl_error("fgetl: file identifier must be non-negative"));
    }
    if fid < 3 {
        return Err(fgetl_error(
            "fgetl: standard input/output identifiers are not supported yet",
        ));
    }

    let info = registry::info_for(fid)
        .ok_or_else(|| fgetl_error(format!("fgetl: {INVALID_IDENTIFIER_MESSAGE}")))?;
    if !permission_allows_read(&info.permission) {
        return Err(fgetl_error(
            "fgetl: file identifier is not open for reading",
        ));
    }
    let handle = registry::take_handle(fid)
        .ok_or_else(|| fgetl_error(format!("fgetl: {INVALID_IDENTIFIER_MESSAGE}")))?;

    let mut guard = handle
        .lock()
        .map_err(|_| fgetl_error("fgetl: failed to lock file handle (poisoned mutex)"))?;
    let file = guard
        .as_mut()
        .ok_or_else(|| fgetl_error(format!("fgetl: {INVALID_IDENTIFIER_MESSAGE}")))?;
    let read = read_text_line(file, None, BUILTIN_NAME)?;
    if read.eof_before_any {
        return Ok(FgetlEval::end_of_file());
    }

    let encoding = if info.encoding.trim().is_empty() {
        "UTF-8".to_string()
    } else {
        info.encoding.clone()
    };

    let line_len = read.data.len().saturating_sub(read.terminators.len());
    let line_value = bytes_to_char_array(&read.data[..line_len], &encoding, BUILTIN_NAME)?;
    Ok(FgetlEval::new(line_value))
}

async fn gather_value(value: &Value) -> BuiltinResult<Value> {
    gather_if_needed_async(value)
        .await
        .map_err(map_control_flow)
}

fn parse_fid(value: &Value) -> BuiltinResult<i32> {
    match value {
        Value::Num(n) => {
            if !n.is_finite() {
                return Err(fgetl_error("fgetl: file identifier must be finite"));
            }
            if (n.fract()).abs() > f64::EPSILON {
                return Err(fgetl_error(
                    "fgetl: file identifier must be an integer scalar",
                ));
            }
            Ok(*n as i32)
        }
        Value::Int(i) => Ok(i.to_i64() as i32),
        Value::Tensor(t) if t.data.len() == 1 => {
            let n = t.data[0];
            if !n.is_finite() {
                return Err(fgetl_error("fgetl: file identifier must be finite"));
            }
            if (n.fract()).abs() > f64::EPSILON {
                return Err(fgetl_error(
                    "fgetl: file identifier must be an integer scalar",
                ));
            }
            Ok(n as i32)
        }
        _ => Err(fgetl_error(
            "fgetl: file identifier must be a numeric scalar",
        )),
    }
}

fn permission_allows_read(permission: &str) -> bool {
    let lower = permission.to_ascii_lowercase();
    lower.starts_with('r') || lower.contains('+')
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use crate::builtins::io::filetext::{fopen, registry};
    use crate::RuntimeError;
    use runmat_accelerate_api::HostTensorView;
    use runmat_time::system_time_now;
    use std::path::{Path, PathBuf};
    use std::time::UNIX_EPOCH;

    fn unwrap_error_message(err: RuntimeError) -> String {
        err.message().to_string()
    }

    fn run_evaluate(fid_value: &Value, rest: &[Value]) -> BuiltinResult<FgetlEval> {
        futures::executor::block_on(evaluate(fid_value, rest))
    }

    fn run_fopen(args: &[Value]) -> BuiltinResult<fopen::FopenEval> {
        futures::executor::block_on(fopen::evaluate(args))
    }

    fn registry_guard() -> std::sync::MutexGuard<'static, ()> {
        registry::test_guard()
    }

    fn unique_path(prefix: &str) -> PathBuf {
        let now = system_time_now()
            .duration_since(UNIX_EPOCH)
            .expect("time went backwards");
        let filename = format!("{}_{}_{}.tmp", prefix, now.as_secs(), now.subsec_nanos());
        std::env::temp_dir().join(filename)
    }

    fn fopen_path(path: &Path) -> FopenHandle {
        let eval = run_fopen(&[Value::from(path.to_string_lossy().to_string())]).expect("fopen");
        let open = eval.as_open().expect("open outputs");
        assert!(open.fid >= 3.0);
        FopenHandle {
            fid: open.fid as i32,
        }
    }

    struct FopenHandle {
        fid: i32,
    }

    impl Drop for FopenHandle {
        fn drop(&mut self) {
            let _ = registry::close(self.fid);
        }
    }

    fn char_text(value: Value) -> String {
        match value {
            Value::CharArray(ca) => ca.data.iter().collect(),
            other => panic!("expected char array, got {other:?}"),
        }
    }

    fn assert_empty_char(value: Value) {
        match value {
            Value::CharArray(ca) => {
                assert_eq!(ca.rows, 1);
                assert_eq!(ca.cols, 0);
                assert!(ca.data.is_empty());
            }
            other => panic!("expected empty char array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fgetl_reads_line_without_newline() {
        let _guard = registry_guard();
        registry::reset_for_tests();
        let path = unique_path("fgetl_line");
        test_support::fs::write(&path, "Hello world\nSecond line\n").unwrap();

        let handle = fopen_path(&path);
        let eval = run_evaluate(&Value::Num(handle.fid as f64), &[]).expect("fgetl");
        assert_eq!(char_text(eval.first_output()), "Hello world");

        test_support::fs::remove_file(&path).unwrap();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fgetl_returns_minus_one_at_eof() {
        let _guard = registry_guard();
        registry::reset_for_tests();
        let path = unique_path("fgetl_eof");
        test_support::fs::write(&path, "line\n").unwrap();
        let handle = fopen_path(&path);

        let _ = run_evaluate(&Value::Num(handle.fid as f64), &[]).expect("first read");
        let eval = run_evaluate(&Value::Num(handle.fid as f64), &[]).expect("second read");
        assert_eq!(eval.first_output(), Value::Num(-1.0));

        test_support::fs::remove_file(&path).unwrap();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fgetl_distinguishes_empty_line_from_eof() {
        let _guard = registry_guard();
        registry::reset_for_tests();
        let path = unique_path("fgetl_empty");
        test_support::fs::write(&path, "\nlast").unwrap();
        let handle = fopen_path(&path);

        let first = run_evaluate(&Value::Num(handle.fid as f64), &[]).expect("first");
        assert_empty_char(first.first_output());
        let second = run_evaluate(&Value::Num(handle.fid as f64), &[]).expect("second");
        assert_eq!(char_text(second.first_output()), "last");
        let third = run_evaluate(&Value::Num(handle.fid as f64), &[]).expect("third");
        assert_eq!(third.first_output(), Value::Num(-1.0));

        test_support::fs::remove_file(&path).unwrap();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fgetl_handles_crlf_lf_and_cr_newlines() {
        let _guard = registry_guard();
        registry::reset_for_tests();
        let path = unique_path("fgetl_newlines");
        test_support::fs::write(&path, b"first\r\nsecond\nthird\rfourth").unwrap();
        let handle = fopen_path(&path);

        let first = run_evaluate(&Value::Num(handle.fid as f64), &[]).expect("first");
        assert_eq!(char_text(first.first_output()), "first");
        let second = run_evaluate(&Value::Num(handle.fid as f64), &[]).expect("second");
        assert_eq!(char_text(second.first_output()), "second");
        let third = run_evaluate(&Value::Num(handle.fid as f64), &[]).expect("third");
        assert_eq!(char_text(third.first_output()), "third");
        let fourth = run_evaluate(&Value::Num(handle.fid as f64), &[]).expect("fourth");
        assert_eq!(char_text(fourth.first_output()), "fourth");

        test_support::fs::remove_file(&path).unwrap();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fgetl_preserves_position_across_repeated_reads() {
        let _guard = registry_guard();
        registry::reset_for_tests();
        let path = unique_path("fgetl_repeated");
        test_support::fs::write(&path, "one\ntwo\nthree\n").unwrap();
        let handle = fopen_path(&path);

        let one = run_evaluate(&Value::Num(handle.fid as f64), &[]).expect("one");
        let two = run_evaluate(&Value::Num(handle.fid as f64), &[]).expect("two");
        let three = run_evaluate(&Value::Num(handle.fid as f64), &[]).expect("three");
        assert_eq!(char_text(one.first_output()), "one");
        assert_eq!(char_text(two.first_output()), "two");
        assert_eq!(char_text(three.first_output()), "three");

        test_support::fs::remove_file(&path).unwrap();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fgetl_errors_for_write_only_identifier() {
        let _guard = registry_guard();
        registry::reset_for_tests();
        let path = unique_path("fgetl_write_only");
        test_support::fs::write(&path, "payload").unwrap();
        let eval = run_fopen(&[
            Value::from(path.to_string_lossy().to_string()),
            Value::from("w"),
        ])
        .expect("fopen");
        let open = eval.as_open().expect("open outputs");
        assert!(open.fid >= 3.0);

        let err = unwrap_error_message(run_evaluate(&Value::Num(open.fid), &[]).unwrap_err());
        assert_eq!(err, "fgetl: file identifier is not open for reading");
        let _ = registry::close(open.fid as i32);
        test_support::fs::remove_file(&path).unwrap();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fgetl_errors_for_closed_identifier() {
        let _guard = registry_guard();
        registry::reset_for_tests();
        let path = unique_path("fgetl_closed");
        test_support::fs::write(&path, "payload").unwrap();
        let handle = fopen_path(&path);
        let fid = handle.fid;
        let _ = registry::close(fid);
        std::mem::forget(handle);

        let err = unwrap_error_message(run_evaluate(&Value::Num(fid as f64), &[]).unwrap_err());
        assert_eq!(
            err,
            "fgetl: Invalid file identifier. Use fopen to generate a valid file ID."
        );
        test_support::fs::remove_file(&path).unwrap();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fgetl_gathers_gpu_scalar_argument() {
        let _guard = registry_guard();
        registry::reset_for_tests();
        let path = unique_path("fgetl_gpu_args");
        test_support::fs::write(&path, b"abcdef\nextra").unwrap();
        let handle = fopen_path(&path);

        test_support::with_test_provider(|provider| {
            let fid_host = [handle.fid as f64];
            let fid_view = HostTensorView {
                data: &fid_host,
                shape: &[1, 1],
            };
            let fid_gpu = Value::GpuTensor(provider.upload(&fid_view).expect("upload fid"));

            let eval = run_evaluate(&fid_gpu, &[]).expect("fgetl");
            assert_eq!(char_text(eval.first_output()), "abcdef");
        });

        test_support::fs::remove_file(&path).unwrap();
    }
}
