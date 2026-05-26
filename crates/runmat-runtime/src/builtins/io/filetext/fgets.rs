//! MATLAB-compatible `fgets` builtin for RunMat.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::io::filetext::{
    helpers::{bytes_to_char_array, empty_numeric_row, numeric_row, read_text_line},
    registry,
};
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "fgets";
const NCHAR_NONNEGATIVE_INTEGER_DETAIL: &str = "nchar must be a non-negative integer scalar";

const FGETS_OUTPUT_LINE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "tline",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Next line including terminators, or -1 at end-of-file.",
}];
const FGETS_OUTPUT_LINE_TERMINATORS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "tline",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Next line including terminators, or -1 at end-of-file.",
    },
    BuiltinParamDescriptor {
        name: "terminators",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Optional,
        default: Some("[]"),
        description: "Numeric row vector of terminator byte values.",
    },
];
const FGETS_INPUTS_FID: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "fid",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "File identifier opened by fopen.",
}];
const FGETS_INPUTS_FID_NCHAR: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "fid",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "File identifier opened by fopen.",
    },
    BuiltinParamDescriptor {
        name: "nchar",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Maximum number of characters to read.",
    },
];
const FGETS_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "tline = fgets(fid)",
        inputs: &FGETS_INPUTS_FID,
        outputs: &FGETS_OUTPUT_LINE,
    },
    BuiltinSignatureDescriptor {
        label: "tline = fgets(fid, nchar)",
        inputs: &FGETS_INPUTS_FID_NCHAR,
        outputs: &FGETS_OUTPUT_LINE,
    },
    BuiltinSignatureDescriptor {
        label: "[tline, terminators] = fgets(fid, ...)",
        inputs: &FGETS_INPUTS_FID_NCHAR,
        outputs: &FGETS_OUTPUT_LINE_TERMINATORS,
    },
];

const FGETS_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FGETS.INVALID_INPUT",
    identifier: Some("RunMat:fgets:InvalidInput"),
    when: "Input argument count or scalar constraints are invalid.",
    message: "fgets: invalid input arguments",
};
const FGETS_ERROR_INVALID_IDENTIFIER: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FGETS.INVALID_IDENTIFIER",
    identifier: Some("RunMat:fgets:InvalidIdentifier"),
    when: "Identifier does not refer to an open readable file.",
    message: "fgets: invalid file identifier. Use fopen to generate a valid file ID.",
};
const FGETS_ERROR_IO: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FGETS.IO",
    identifier: Some("RunMat:fgets:IoFailure"),
    when: "File read or decode operation failed.",
    message: "fgets: file I/O failed",
};
const FGETS_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FGETS.INTERNAL",
    identifier: None,
    when: "Internal control-flow conversion failed.",
    message: "fgets: internal error",
};
const FGETS_ERRORS: [BuiltinErrorDescriptor; 4] = [
    FGETS_ERROR_INVALID_INPUT,
    FGETS_ERROR_INVALID_IDENTIFIER,
    FGETS_ERROR_IO,
    FGETS_ERROR_INTERNAL,
];
pub const FGETS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &FGETS_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &FGETS_ERRORS,
};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::io::filetext::fgets")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "fgets",
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

fn fgets_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn fgets_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    fgets_error_with_message(error.message, error)
}

fn fgets_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    fgets_error_with_message(format!("{}: {}", error.message, detail.as_ref()), error)
}

fn map_control_flow(err: RuntimeError) -> RuntimeError {
    let mut builder = build_runtime_error(format!("{BUILTIN_NAME}: {}", err.message()))
        .with_builtin(BUILTIN_NAME)
        .with_source(err);
    if let Some(identifier) = FGETS_ERROR_INTERNAL.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::io::filetext::fgets")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "fgets",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "File I/O calls are not eligible for fusion.",
};

#[runtime_builtin(
    name = "fgets",
    category = "io/filetext",
    summary = "Read the next line from a file, including newline characters.",
    keywords = "fgets,file,io,line,newline",
    accel = "cpu",
    type_resolver(crate::builtins::io::type_resolvers::fgets_type),
    descriptor(crate::builtins::io::filetext::fgets::FGETS_DESCRIPTOR),
    builtin_path = "crate::builtins::io::filetext::fgets"
)]
async fn fgets_builtin(fid: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    let eval = evaluate(&fid, &rest).await?;
    if let Some(out_count) = crate::output_count::current_output_count() {
        if out_count == 0 {
            return Ok(Value::OutputList(Vec::new()));
        }
        return Ok(crate::output_count::output_list_with_padding(
            out_count,
            eval.outputs(),
        ));
    }
    Ok(eval.first_output())
}

#[derive(Clone, Debug)]
pub struct FgetsEval {
    line: Value,
    terminators: Value,
}

impl FgetsEval {
    fn new(line: Value, terminators: Value) -> Self {
        Self { line, terminators }
    }

    fn end_of_file() -> Self {
        Self {
            line: Value::Num(-1.0),
            terminators: Value::Num(-1.0),
        }
    }

    pub fn first_output(&self) -> Value {
        self.line.clone()
    }

    pub fn outputs(&self) -> Vec<Value> {
        vec![self.line.clone(), self.terminators.clone()]
    }
}

pub async fn evaluate(fid_value: &Value, rest: &[Value]) -> BuiltinResult<FgetsEval> {
    if rest.len() > 1 {
        return Err(fgets_error_with_detail(
            &FGETS_ERROR_INVALID_INPUT,
            "too many input arguments",
        ));
    }

    let fid_host = gather_value(fid_value).await?;
    let fid = parse_fid(&fid_host)?;
    if fid < 0 {
        return Err(fgets_error_with_detail(
            &FGETS_ERROR_INVALID_INPUT,
            "file identifier must be non-negative",
        ));
    }
    if fid < 3 {
        return Err(fgets_error_with_detail(
            &FGETS_ERROR_INVALID_INPUT,
            "standard input/output identifiers are not supported yet",
        ));
    }

    let info =
        registry::info_for(fid).ok_or_else(|| fgets_error(&FGETS_ERROR_INVALID_IDENTIFIER))?;
    if !permission_allows_read(&info.permission) {
        return Err(fgets_error_with_detail(
            &FGETS_ERROR_INVALID_IDENTIFIER,
            "file identifier is not open for reading",
        ));
    }
    let handle =
        registry::take_handle(fid).ok_or_else(|| fgets_error(&FGETS_ERROR_INVALID_IDENTIFIER))?;

    let nchar_limit = parse_nchar(rest).await?;
    let max_bytes = apply_matlab_nchar_limit(nchar_limit);
    let mut guard = handle.lock().map_err(|_| {
        fgets_error_with_detail(
            &FGETS_ERROR_INTERNAL,
            "failed to lock file handle (poisoned mutex)",
        )
    })?;
    let file = guard
        .as_mut()
        .ok_or_else(|| fgets_error(&FGETS_ERROR_INVALID_IDENTIFIER))?;
    let read = read_text_line(file, max_bytes, BUILTIN_NAME)
        .map_err(|e| fgets_error_with_detail(&FGETS_ERROR_IO, e.message()))?;
    if read.eof_before_any {
        return Ok(FgetsEval::end_of_file());
    }

    let encoding = if info.encoding.trim().is_empty() {
        "UTF-8".to_string()
    } else {
        info.encoding.clone()
    };

    let line_value = bytes_to_char_array(&read.data, &encoding, BUILTIN_NAME)
        .map_err(|e| fgets_error_with_detail(&FGETS_ERROR_IO, e.message()))?;
    let terminators_value = if read.terminators.is_empty() {
        empty_numeric_row()
    } else {
        numeric_row(&read.terminators, BUILTIN_NAME)
            .map_err(|e| fgets_error_with_detail(&FGETS_ERROR_IO, e.message()))?
    };

    Ok(FgetsEval::new(line_value, terminators_value))
}

async fn gather_value(value: &Value) -> BuiltinResult<Value> {
    gather_if_needed_async(value)
        .await
        .map_err(map_control_flow)
}

fn parse_fid(value: &Value) -> BuiltinResult<i32> {
    fn checked_f64_to_i32(n: f64) -> BuiltinResult<i32> {
        if n < i32::MIN as f64 || n > i32::MAX as f64 {
            return Err(fgets_error_with_detail(
                &FGETS_ERROR_INVALID_INPUT,
                "file identifier is out of range",
            ));
        }
        Ok(n as i32)
    }

    fn checked_i64_to_i32(n: i64) -> BuiltinResult<i32> {
        i32::try_from(n).map_err(|_| {
            fgets_error_with_detail(
                &FGETS_ERROR_INVALID_INPUT,
                "file identifier is out of range",
            )
        })
    }

    match value {
        Value::Num(n) => {
            if !n.is_finite() {
                return Err(fgets_error_with_detail(
                    &FGETS_ERROR_INVALID_INPUT,
                    "file identifier must be finite",
                ));
            }
            if (n.fract()).abs() > f64::EPSILON {
                return Err(fgets_error_with_detail(
                    &FGETS_ERROR_INVALID_INPUT,
                    "file identifier must be an integer scalar",
                ));
            }
            checked_f64_to_i32(*n)
        }
        Value::Int(i) => checked_i64_to_i32(i.to_i64()),
        Value::Tensor(t) if t.data.len() == 1 => {
            let n = t.data[0];
            if !n.is_finite() {
                return Err(fgets_error_with_detail(
                    &FGETS_ERROR_INVALID_INPUT,
                    "file identifier must be finite",
                ));
            }
            if (n.fract()).abs() > f64::EPSILON {
                return Err(fgets_error_with_detail(
                    &FGETS_ERROR_INVALID_INPUT,
                    "file identifier must be an integer scalar",
                ));
            }
            checked_f64_to_i32(n)
        }
        _ => Err(fgets_error_with_detail(
            &FGETS_ERROR_INVALID_INPUT,
            "file identifier must be a numeric scalar",
        )),
    }
}

async fn parse_nchar(args: &[Value]) -> BuiltinResult<Option<usize>> {
    if args.is_empty() {
        return Ok(None);
    }
    let value = gather_value(&args[0]).await?;
    match value {
        Value::Num(n) => {
            if !n.is_finite() {
                if n.is_sign_positive() {
                    return Ok(None);
                }
                return Err(fgets_error_with_detail(
                    &FGETS_ERROR_INVALID_INPUT,
                    NCHAR_NONNEGATIVE_INTEGER_DETAIL,
                ));
            }
            if n < 0.0 {
                return Err(fgets_error_with_detail(
                    &FGETS_ERROR_INVALID_INPUT,
                    NCHAR_NONNEGATIVE_INTEGER_DETAIL,
                ));
            }
            if (n.fract()).abs() > f64::EPSILON {
                return Err(fgets_error_with_detail(
                    &FGETS_ERROR_INVALID_INPUT,
                    NCHAR_NONNEGATIVE_INTEGER_DETAIL,
                ));
            }
            Ok(Some(n as usize))
        }
        Value::Int(i) => {
            let raw = i.to_i64();
            if raw < 0 {
                return Err(fgets_error_with_detail(
                    &FGETS_ERROR_INVALID_INPUT,
                    NCHAR_NONNEGATIVE_INTEGER_DETAIL,
                ));
            }
            Ok(Some(raw as usize))
        }
        Value::Tensor(t) if t.data.len() == 1 => {
            let n = t.data[0];
            if !n.is_finite() {
                if n.is_sign_positive() {
                    return Ok(None);
                }
                return Err(fgets_error_with_detail(
                    &FGETS_ERROR_INVALID_INPUT,
                    NCHAR_NONNEGATIVE_INTEGER_DETAIL,
                ));
            }
            if n < 0.0 {
                return Err(fgets_error_with_detail(
                    &FGETS_ERROR_INVALID_INPUT,
                    NCHAR_NONNEGATIVE_INTEGER_DETAIL,
                ));
            }
            if (n.fract()).abs() > f64::EPSILON {
                return Err(fgets_error_with_detail(
                    &FGETS_ERROR_INVALID_INPUT,
                    NCHAR_NONNEGATIVE_INTEGER_DETAIL,
                ));
            }
            Ok(Some(n as usize))
        }
        _ => Err(fgets_error_with_detail(
            &FGETS_ERROR_INVALID_INPUT,
            NCHAR_NONNEGATIVE_INTEGER_DETAIL,
        )),
    }
}

fn permission_allows_read(permission: &str) -> bool {
    let lower = permission.to_ascii_lowercase();
    lower.starts_with('r') || lower.contains('+')
}

fn apply_matlab_nchar_limit(nchar_limit: Option<usize>) -> Option<usize> {
    nchar_limit.map(|nchar| nchar.saturating_sub(1))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use crate::builtins::io::filetext::{fopen, registry};
    use crate::RuntimeError;
    use runmat_accelerate_api::HostTensorView;
    use runmat_builtins::IntValue;
    use runmat_time::system_time_now;
    use std::path::{Path, PathBuf};
    use std::time::UNIX_EPOCH;

    fn unwrap_error_message(err: RuntimeError) -> String {
        err.message().to_string()
    }

    fn run_evaluate(fid_value: &Value, rest: &[Value]) -> BuiltinResult<FgetsEval> {
        futures::executor::block_on(evaluate(fid_value, rest))
    }

    fn run_fopen(args: &[Value]) -> BuiltinResult<fopen::FopenEval> {
        futures::executor::block_on(fopen::evaluate(args))
    }

    fn registry_guard() -> std::sync::MutexGuard<'static, ()> {
        registry::test_guard()
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fgets_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = FGETS_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"tline = fgets(fid)"));
        assert!(labels.contains(&"tline = fgets(fid, nchar)"));
        assert!(labels.contains(&"[tline, terminators] = fgets(fid, ...)"));
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

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fgets_reads_line_with_newline() {
        let _guard = registry_guard();
        registry::reset_for_tests();
        let path = unique_path("fgets_line");
        test_support::fs::write(&path, "Hello world\nSecond line\n").unwrap();

        let handle = fopen_path(&path);
        let eval = run_evaluate(&Value::Num(handle.fid as f64), &[]).expect("fgets");
        let line = eval.first_output();
        match line {
            Value::CharArray(ca) => {
                let text: String = ca.data.iter().collect();
                assert_eq!(text, "Hello world\n");
            }
            other => panic!("expected char array, got {other:?}"),
        }
        let ltout = eval.outputs()[1].clone();
        match ltout {
            Value::Tensor(t) => {
                assert_eq!(t.data, vec![10.0]);
                assert_eq!(t.shape, vec![1, 1]);
            }
            other => panic!("expected numeric tensor, got {other:?}"),
        }

        test_support::fs::remove_file(&path).unwrap();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fgets_returns_minus_one_at_eof() {
        let _guard = registry_guard();
        registry::reset_for_tests();
        let path = unique_path("fgets_eof");
        test_support::fs::write(&path, "line\n").unwrap();
        let handle = fopen_path(&path);

        let _ = run_evaluate(&Value::Num(handle.fid as f64), &[]).expect("first read");
        let eval = run_evaluate(&Value::Num(handle.fid as f64), &[]).expect("second read");
        assert_eq!(eval.first_output(), Value::Num(-1.0));
        assert_eq!(eval.outputs()[1], Value::Num(-1.0));

        test_support::fs::remove_file(&path).unwrap();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fgets_honours_nchar_limit() {
        let _guard = registry_guard();
        registry::reset_for_tests();
        let path = unique_path("fgets_limit");
        test_support::fs::write(&path, "abcdefghij\nrest\n").unwrap();
        let handle = fopen_path(&path);

        let eval =
            run_evaluate(&Value::Num(handle.fid as f64), &[Value::Num(5.0)]).expect("limited read");
        match eval.first_output() {
            Value::CharArray(ca) => {
                let text: String = ca.data.iter().collect();
                assert_eq!(text, "abcd");
            }
            other => panic!("expected char array, got {other:?}"),
        }
        match &eval.outputs()[1] {
            Value::Tensor(t) => {
                assert!(t.data.is_empty());
            }
            other => panic!("expected empty numeric tensor, got {other:?}"),
        }

        test_support::fs::remove_file(&path).unwrap();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fgets_errors_for_write_only_identifier() {
        let _guard = registry_guard();
        registry::reset_for_tests();
        let path = unique_path("fgets_write_only");
        test_support::fs::write(&path, "payload").unwrap();
        let eval = run_fopen(&[
            Value::from(path.to_string_lossy().to_string()),
            Value::from("w"),
        ])
        .expect("fopen");
        let open = eval.as_open().expect("open outputs");
        assert!(open.fid >= 3.0);
        let err = unwrap_error_message(run_evaluate(&Value::Num(open.fid), &[]).unwrap_err());
        assert_eq!(
            err,
            format!(
                "{}: file identifier is not open for reading",
                FGETS_ERROR_INVALID_IDENTIFIER.message
            )
        );
        test_support::fs::remove_file(&path).unwrap();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fgets_respects_limit_before_crlf_sequence() {
        let _guard = registry_guard();
        registry::reset_for_tests();
        let path = unique_path("fgets_limit_crlf");
        test_support::fs::write(&path, b"ABCDE\r\nnext\n").unwrap();
        let handle = fopen_path(&path);

        let first =
            run_evaluate(&Value::Num(handle.fid as f64), &[Value::Num(3.0)]).expect("first");
        match first.first_output() {
            Value::CharArray(ca) => {
                let text: String = ca.data.iter().collect();
                assert_eq!(text, "AB");
            }
            other => panic!("expected char array, got {other:?}"),
        }
        match &first.outputs()[1] {
            Value::Tensor(t) => assert!(t.data.is_empty()),
            other => panic!("expected empty numeric tensor, got {other:?}"),
        }

        let second = run_evaluate(&Value::Num(handle.fid as f64), &[]).expect("second");
        match second.first_output() {
            Value::CharArray(ca) => {
                let text: String = ca.data.iter().collect();
                assert_eq!(text, "CDE\r\n");
            }
            other => panic!("expected char array, got {other:?}"),
        }
        match &second.outputs()[1] {
            Value::Tensor(t) => assert_eq!(t.data, vec![13.0, 10.0]),
            other => panic!("expected CRLF terminators, got {other:?}"),
        }

        let third = run_evaluate(&Value::Num(handle.fid as f64), &[]).expect("third");
        match third.first_output() {
            Value::CharArray(ca) => {
                let text: String = ca.data.iter().collect();
                assert_eq!(text, "next\n");
            }
            other => panic!("expected char array, got {other:?}"),
        }

        test_support::fs::remove_file(&path).unwrap();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fgets_handles_crlf_newlines() {
        let _guard = registry_guard();
        registry::reset_for_tests();
        let path = unique_path("fgets_crlf");
        test_support::fs::write(&path, b"first line\r\nsecond\r\n").unwrap();
        let handle = fopen_path(&path);

        let eval = run_evaluate(&Value::Num(handle.fid as f64), &[]).expect("fgets");
        let outputs = eval.outputs();
        match &outputs[0] {
            Value::CharArray(ca) => {
                let text: String = ca.data.iter().collect();
                assert_eq!(text, "first line\r\n");
            }
            other => panic!("expected char array, got {other:?}"),
        }
        match &outputs[1] {
            Value::Tensor(t) => {
                assert_eq!(t.data, vec![13.0, 10.0]);
            }
            other => panic!("expected numeric tensor, got {other:?}"),
        }

        test_support::fs::remove_file(&path).unwrap();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fgets_decodes_latin1() {
        let _guard = registry_guard();
        registry::reset_for_tests();
        let path = unique_path("fgets_latin1");
        test_support::fs::write(&path, [0x48u8, 0x6f, 0x6c, 0x61, 0x20, 0xf1, b'\n']).unwrap();
        let eval = run_fopen(&[
            Value::from(path.to_string_lossy().to_string()),
            Value::from("r"),
            Value::from("native"),
            Value::from("latin1"),
        ])
        .expect("fopen");
        let open = eval.as_open().expect("open outputs");
        let fid = open.fid as i32;

        let read = run_evaluate(&Value::Num(fid as f64), &[]).expect("fgets");
        match read.first_output() {
            Value::CharArray(ca) => {
                let text: String = ca.data.iter().collect();
                assert_eq!(text, "Hola ñ\n");
            }
            other => panic!("expected char array, got {other:?}"),
        }

        let _ = registry::close(fid);
        test_support::fs::remove_file(&path).unwrap();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fgets_nchar_zero_returns_empty_char() {
        let _guard = registry_guard();
        registry::reset_for_tests();
        let path = unique_path("fgets_zero");
        test_support::fs::write(&path, "hello\n").unwrap();
        let handle = fopen_path(&path);

        let eval = run_evaluate(
            &Value::Num(handle.fid as f64),
            &[Value::Int(IntValue::I32(0))],
        )
        .expect("fgets");
        match eval.first_output() {
            Value::CharArray(ca) => {
                assert_eq!(ca.rows, 1);
                assert_eq!(ca.cols, 0);
                assert!(ca.data.is_empty());
            }
            other => panic!("expected empty char array, got {other:?}"),
        }

        test_support::fs::remove_file(&path).unwrap();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fgets_nchar_one_returns_empty_char() {
        let _guard = registry_guard();
        registry::reset_for_tests();
        let path = unique_path("fgets_one");
        test_support::fs::write(&path, "hello\n").unwrap();
        let handle = fopen_path(&path);

        let eval = run_evaluate(
            &Value::Num(handle.fid as f64),
            &[Value::Int(IntValue::I32(1))],
        )
        .expect("fgets");
        match eval.first_output() {
            Value::CharArray(ca) => {
                assert_eq!(ca.rows, 1);
                assert_eq!(ca.cols, 0);
                assert!(ca.data.is_empty());
            }
            other => panic!("expected empty char array, got {other:?}"),
        }

        test_support::fs::remove_file(&path).unwrap();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fgets_gathers_gpu_scalar_arguments() {
        let _guard = registry_guard();
        registry::reset_for_tests();
        let path = unique_path("fgets_gpu_args");
        test_support::fs::write(&path, b"abcdef\nextra").unwrap();
        let handle = fopen_path(&path);

        test_support::with_test_provider(|provider| {
            let fid_host = [handle.fid as f64];
            let fid_view = HostTensorView {
                data: &fid_host,
                shape: &[1, 1],
            };
            let fid_gpu = Value::GpuTensor(provider.upload(&fid_view).expect("upload fid"));

            let limit_host = [3.0f64];
            let limit_view = HostTensorView {
                data: &limit_host,
                shape: &[1, 1],
            };
            let limit_gpu = Value::GpuTensor(provider.upload(&limit_view).expect("upload limit"));

            let eval = run_evaluate(&fid_gpu, &[limit_gpu]).expect("fgets");
            match eval.first_output() {
                Value::CharArray(ca) => {
                    let text: String = ca.data.iter().collect();
                    assert_eq!(text, "ab");
                }
                other => panic!("expected char array, got {other:?}"),
            }
        });

        test_support::fs::remove_file(&path).unwrap();
    }
}
