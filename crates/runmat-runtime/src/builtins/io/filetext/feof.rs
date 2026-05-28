//! MATLAB-compatible `feof` builtin for RunMat.
//!
//! Provides end-of-file detection for file identifiers opened via `fopen`.
//! The implementation mirrors MATLAB semantics for host-side files and
//! integrates with the shared registry used by the other text I/O builtins.

use std::io::{ErrorKind, Seek, SeekFrom};

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::io::filetext::registry;
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const IDENTIFIER_TYPE_ERROR_DETAIL: &str = "file identifier must be a numeric scalar";
const BUILTIN_NAME: &str = "feof";

const FEOF_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "tf",
    ty: BuiltinParamType::LogicalArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "True when positioned at end-of-file.",
}];
const FEOF_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "fid",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "File identifier opened by fopen.",
}];
const FEOF_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "tf = feof(fid)",
    inputs: &FEOF_INPUTS,
    outputs: &FEOF_OUTPUT,
}];

const FEOF_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FEOF.INVALID_INPUT",
    identifier: Some("RunMat:feof:InvalidInput"),
    when: "Input identifier is malformed or out of range.",
    message: "feof: invalid file identifier input",
};
const FEOF_ERROR_INVALID_IDENTIFIER: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FEOF.INVALID_IDENTIFIER",
    identifier: Some("RunMat:feof:InvalidIdentifier"),
    when: "Identifier does not refer to an open file handle.",
    message: "feof: invalid file identifier. Use fopen to generate a valid file ID.",
};
const FEOF_ERROR_IO: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FEOF.IO",
    identifier: Some("RunMat:feof:IoFailure"),
    when: "File-position or length query fails.",
    message: "feof: file I/O query failed",
};
const FEOF_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FEOF.INTERNAL",
    identifier: None,
    when: "Internal control-flow conversion failed.",
    message: "feof: internal error",
};
const FEOF_ERRORS: [BuiltinErrorDescriptor; 4] = [
    FEOF_ERROR_INVALID_INPUT,
    FEOF_ERROR_INVALID_IDENTIFIER,
    FEOF_ERROR_IO,
    FEOF_ERROR_INTERNAL,
];
pub const FEOF_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &FEOF_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &FEOF_ERRORS,
};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::io::filetext::feof")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "feof",
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
    notes: "Host-only file I/O query; providers are not involved.",
};

fn feof_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn feof_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    feof_error_with_message(error.message, error)
}

fn feof_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    feof_error_with_message(format!("{}: {}", error.message, detail.as_ref()), error)
}

fn feof_error_with_source(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
    source: impl std::error::Error + Send + Sync + 'static,
) -> RuntimeError {
    let mut builder = build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .with_source(source);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn feof_error_with_source_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
    source: impl std::error::Error + Send + Sync + 'static,
) -> RuntimeError {
    feof_error_with_source(
        format!("{}: {}", error.message, detail.as_ref()),
        error,
        source,
    )
}

fn map_control_flow(err: RuntimeError) -> RuntimeError {
    let mut builder = build_runtime_error(format!("{BUILTIN_NAME}: {}", err.message()))
        .with_builtin(BUILTIN_NAME)
        .with_source(err);
    if let Some(identifier) = FEOF_ERROR_INTERNAL.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::io::filetext::feof")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "feof",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "File I/O queries are not eligible for fusion; metadata registered for completeness.",
};

#[runtime_builtin(
    name = "feof",
    category = "io/filetext",
    summary = "Query whether a file identifier is positioned at end-of-file.",
    keywords = "feof,end of file,io,file identifier",
    accel = "cpu",
    type_resolver(crate::builtins::io::type_resolvers::feof_type),
    descriptor(crate::builtins::io::filetext::feof::FEOF_DESCRIPTOR),
    builtin_path = "crate::builtins::io::filetext::feof"
)]
async fn feof_builtin(fid: Value) -> crate::BuiltinResult<Value> {
    let at_end = evaluate(&fid).await?;
    Ok(Value::Bool(at_end))
}

/// Evaluate the `feof` builtin without invoking the runtime dispatcher.
pub async fn evaluate(fid_value: &Value) -> BuiltinResult<bool> {
    let fid_host = gather_if_needed_async(fid_value)
        .await
        .map_err(map_control_flow)?;
    let fid = parse_fid(&fid_host)?;
    if fid < 0 {
        return Err(feof_error_with_detail(
            &FEOF_ERROR_INVALID_INPUT,
            "file identifier must be non-negative",
        ));
    }
    if fid < 3 {
        return Ok(false);
    }

    let handle =
        registry::take_handle(fid).ok_or_else(|| feof_error(&FEOF_ERROR_INVALID_IDENTIFIER))?;
    let mut guard = handle.lock().map_err(|_| {
        feof_error_with_detail(
            &FEOF_ERROR_INTERNAL,
            "failed to lock file handle (poisoned mutex)",
        )
    })?;
    let file = guard
        .as_mut()
        .ok_or_else(|| feof_error(&FEOF_ERROR_INVALID_IDENTIFIER))?;

    let position = file.stream_position().map_err(|err| {
        feof_error_with_source_detail(
            &FEOF_ERROR_IO,
            format!("failed to query file position: {err}"),
            err,
        )
    })?;

    let end_position = match file.seek(SeekFrom::End(0)) {
        Ok(pos) => pos,
        Err(err) => {
            if err.kind() == ErrorKind::Unsupported {
                let _ = file.seek(SeekFrom::Start(position));
                return Ok(false);
            }
            return Err(feof_error_with_source_detail(
                &FEOF_ERROR_IO,
                format!("failed to query file length: {err}"),
                err,
            ));
        }
    };

    if let Err(err) = file.seek(SeekFrom::Start(position)) {
        return Err(feof_error_with_source_detail(
            &FEOF_ERROR_IO,
            format!("failed to restore file position: {err}"),
            err,
        ));
    }

    Ok(position >= end_position)
}

fn parse_fid(value: &Value) -> BuiltinResult<i32> {
    match value {
        Value::Num(n) => parse_scalar_fid(*n),
        Value::Int(int) => {
            let v = int.to_f64();
            parse_scalar_fid(v)
        }
        Value::Tensor(t) => {
            if t.data.len() == 1 {
                parse_scalar_fid(t.data[0])
            } else {
                Err(feof_error_with_detail(
                    &FEOF_ERROR_INVALID_INPUT,
                    IDENTIFIER_TYPE_ERROR_DETAIL,
                ))
            }
        }
        Value::LogicalArray(la) if la.data.len() == 1 => {
            let v = if la.data[0] != 0 { 1.0 } else { 0.0 };
            parse_scalar_fid(v)
        }
        Value::LogicalArray(_) => Err(feof_error_with_detail(
            &FEOF_ERROR_INVALID_INPUT,
            IDENTIFIER_TYPE_ERROR_DETAIL,
        )),
        Value::Bool(b) => parse_scalar_fid(if *b { 1.0 } else { 0.0 }),
        _ => Err(feof_error_with_detail(
            &FEOF_ERROR_INVALID_INPUT,
            IDENTIFIER_TYPE_ERROR_DETAIL,
        )),
    }
}

fn parse_scalar_fid(value: f64) -> BuiltinResult<i32> {
    if !value.is_finite() {
        return Err(feof_error_with_detail(
            &FEOF_ERROR_INVALID_INPUT,
            "file identifier must be finite",
        ));
    }
    let rounded = value.round();
    if (rounded - value).abs() > f64::EPSILON {
        return Err(feof_error_with_detail(
            &FEOF_ERROR_INVALID_INPUT,
            "file identifier must be an integer",
        ));
    }
    if rounded < i32::MIN as f64 || rounded > i32::MAX as f64 {
        return Err(feof_error_with_detail(
            &FEOF_ERROR_INVALID_INPUT,
            "file identifier is out of range",
        ));
    }
    Ok(rounded as i32)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use crate::builtins::io::filetext::{fclose, fopen, fread, registry};
    use crate::RuntimeError;
    use runmat_accelerate_api::HostTensorView;
    use runmat_builtins::{Tensor, Value};
    use runmat_filesystem::File;
    use runmat_time::system_time_now;
    use std::io::Write;
    use std::path::PathBuf;
    use std::time::UNIX_EPOCH;

    fn unwrap_error_message(err: RuntimeError) -> String {
        err.message().to_string()
    }

    fn run_evaluate(fid_value: &Value) -> BuiltinResult<bool> {
        futures::executor::block_on(evaluate(fid_value))
    }

    fn run_fopen(args: &[Value]) -> BuiltinResult<fopen::FopenEval> {
        futures::executor::block_on(fopen::evaluate(args))
    }

    fn run_fread(fid_value: &Value, args: &[Value]) -> BuiltinResult<fread::FreadEval> {
        futures::executor::block_on(fread::evaluate(fid_value, args))
    }

    fn run_fclose(args: &[Value]) -> BuiltinResult<fclose::FcloseEval> {
        futures::executor::block_on(fclose::evaluate(args))
    }

    fn registry_guard() -> std::sync::MutexGuard<'static, ()> {
        registry::test_guard()
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn feof_descriptor_signature_covers_core_form() {
        let labels: Vec<&str> = FEOF_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"tf = feof(fid)"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn feof_returns_false_before_reading() {
        let _guard = registry_guard();
        registry::reset_for_tests();
        let path = unique_path("feof_false_before_read");
        {
            let mut file = File::create(&path).expect("create");
            file.write_all(b"abc").expect("write");
        }

        let open = run_fopen(&[
            Value::from(path.to_string_lossy().to_string()),
            Value::from("rb"),
        ])
        .expect("fopen");
        let fid = open.as_open().unwrap().fid as i32;

        let at_end = run_evaluate(&Value::Num(fid as f64)).expect("feof");
        assert!(!at_end);

        run_fclose(&[Value::Num(fid as f64)]).unwrap();
        test_support::fs::remove_file(path).unwrap();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn feof_returns_true_after_reading_to_end() {
        let _guard = registry_guard();
        registry::reset_for_tests();
        let path = unique_path("feof_true_after_read");
        {
            let mut file = File::create(&path).expect("create");
            file.write_all(&[1u8, 2, 3]).expect("write");
        }

        let open = run_fopen(&[
            Value::from(path.to_string_lossy().to_string()),
            Value::from("rb"),
        ])
        .expect("fopen");
        let fid = open.as_open().unwrap().fid as i32;

        // Read the entire file to advance the file position to EOF.
        let _ = run_fread(&Value::Num(fid as f64), &Vec::new()).expect("fread");

        let at_end = run_evaluate(&Value::Num(fid as f64)).expect("feof");
        assert!(at_end);

        run_fclose(&[Value::Num(fid as f64)]).unwrap();
        test_support::fs::remove_file(path).unwrap();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn feof_empty_file_is_true() {
        let _guard = registry_guard();
        registry::reset_for_tests();
        let path = unique_path("feof_empty_file");
        File::create(&path).expect("create empty");

        let open = run_fopen(&[
            Value::from(path.to_string_lossy().to_string()),
            Value::from("rb"),
        ])
        .expect("fopen");
        let fid = open.as_open().unwrap().fid as i32;

        let at_end = run_evaluate(&Value::Num(fid as f64)).expect("feof");
        assert!(at_end);

        run_fclose(&[Value::Num(fid as f64)]).unwrap();
        test_support::fs::remove_file(path).unwrap();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn feof_invalid_identifier_errors() {
        let _guard = registry_guard();
        registry::reset_for_tests();
        let err = unwrap_error_message(run_evaluate(&Value::Num(42.0)).unwrap_err());
        assert_eq!(err, FEOF_ERROR_INVALID_IDENTIFIER.message);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn feof_rejects_non_integer_identifier() {
        let _guard = registry_guard();
        registry::reset_for_tests();
        let err = unwrap_error_message(run_evaluate(&Value::Num(1.5)).unwrap_err());
        assert_eq!(
            err,
            format!(
                "{}: file identifier must be an integer",
                FEOF_ERROR_INVALID_INPUT.message
            )
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn feof_rejects_nan_identifier() {
        let _guard = registry_guard();
        registry::reset_for_tests();
        let err = unwrap_error_message(run_evaluate(&Value::Num(f64::NAN)).unwrap_err());
        assert_eq!(
            err,
            format!(
                "{}: file identifier must be finite",
                FEOF_ERROR_INVALID_INPUT.message
            )
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn feof_rejects_negative_identifier() {
        let _guard = registry_guard();
        registry::reset_for_tests();
        let err = unwrap_error_message(run_evaluate(&Value::Num(-1.0)).unwrap_err());
        assert_eq!(
            err,
            format!(
                "{}: file identifier must be non-negative",
                FEOF_ERROR_INVALID_INPUT.message
            )
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn feof_rejects_non_numeric_inputs() {
        let _guard = registry_guard();
        registry::reset_for_tests();
        let err = unwrap_error_message(run_evaluate(&Value::from("abc")).unwrap_err());
        assert_eq!(
            err,
            format!(
                "{}: {}",
                FEOF_ERROR_INVALID_INPUT.message, IDENTIFIER_TYPE_ERROR_DETAIL
            )
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn feof_accepts_scalar_tensor_identifier() {
        let _guard = registry_guard();
        registry::reset_for_tests();
        let path = unique_path("feof_tensor_identifier");
        {
            let mut file = File::create(&path).expect("create");
            file.write_all(b"data").expect("write");
        }

        let open = run_fopen(&[
            Value::from(path.to_string_lossy().to_string()),
            Value::from("rb"),
        ])
        .expect("fopen");
        let fid = open.as_open().unwrap().fid as f64;

        let tensor = Tensor::new(vec![fid], vec![1]).unwrap();
        let at_end = run_evaluate(&Value::Tensor(tensor)).expect("feof");
        assert!(!at_end);

        run_fclose(&[Value::Num(fid)]).unwrap();
        test_support::fs::remove_file(path).unwrap();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn feof_errors_on_closed_identifier() {
        let _guard = registry_guard();
        registry::reset_for_tests();
        let path = unique_path("feof_closed_identifier");
        {
            let mut file = File::create(&path).expect("create");
            file.write_all(b"x").expect("write");
        }

        let open = run_fopen(&[
            Value::from(path.to_string_lossy().to_string()),
            Value::from("rb"),
        ])
        .expect("fopen");
        let fid = open.as_open().unwrap().fid as f64;

        run_fclose(&[Value::Num(fid)]).unwrap();

        let err = unwrap_error_message(run_evaluate(&Value::Num(fid)).unwrap_err());
        assert_eq!(err, FEOF_ERROR_INVALID_IDENTIFIER.message);

        test_support::fs::remove_file(path).unwrap();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn feof_accepts_gpu_identifier_via_gather() {
        let _guard = registry_guard();
        registry::reset_for_tests();
        let path = unique_path("feof_gpu_identifier");
        {
            let mut file = File::create(&path).expect("create");
            file.write_all(b"xyz").expect("write");
        }

        let open = run_fopen(&[
            Value::from(path.to_string_lossy().to_string()),
            Value::from("rb"),
        ])
        .expect("fopen");
        let fid = open.as_open().unwrap().fid as f64;

        crate::builtins::common::test_support::with_test_provider(|provider| {
            let data = [fid];
            let shape = [1usize];
            let view = HostTensorView {
                data: &data,
                shape: &shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let value = Value::GpuTensor(handle.clone());

            let at_end = run_evaluate(&value).expect("feof");
            assert!(!at_end);

            provider.free(&handle).expect("free");
        });

        run_fclose(&[Value::Num(fid)]).unwrap();
        test_support::fs::remove_file(path).unwrap();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn feof_standard_identifier_returns_false() {
        let _guard = registry_guard();
        registry::reset_for_tests();
        let result = run_evaluate(&Value::Num(0.0)).expect("feof");
        assert!(!result);
    }

    fn unique_path(prefix: &str) -> PathBuf {
        let now = system_time_now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let filename = format!("{prefix}_{now}.tmp");
        std::env::temp_dir().join(filename)
    }
}
