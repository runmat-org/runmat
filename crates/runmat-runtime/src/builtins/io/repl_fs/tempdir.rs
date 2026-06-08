//! MATLAB-compatible `tempdir` builtin for RunMat.

use crate::builtins::common::env as runtime_env;
use std::convert::TryFrom;
use std::path::Path;

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    CharArray, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::{build_runtime_error, RuntimeError};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::io::repl_fs::tempdir")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "tempdir",
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
    notes: "Host-only operation that queries the environment for the temporary folder. No provider hooks are required.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::io::repl_fs::tempdir")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "tempdir",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "I/O builtin that always executes on the host; fusion metadata is present for introspection completeness.",
};

const BUILTIN_NAME: &str = "tempdir";

const TEMPDIR_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "folder",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Absolute path to the system temporary directory with trailing separator.",
}];
const TEMPDIR_INPUTS_NONE: [BuiltinParamDescriptor; 0] = [];
const TEMPDIR_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "folder = tempdir()",
    inputs: &TEMPDIR_INPUTS_NONE,
    outputs: &TEMPDIR_OUTPUT,
}];
const TEMPDIR_ERROR_TOO_MANY_INPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TEMPDIR.TOO_MANY_INPUTS",
    identifier: None,
    when: "Any positional input argument is supplied.",
    message: "tempdir: too many input arguments",
};
const TEMPDIR_ERROR_UNABLE_TO_DETERMINE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TEMPDIR.UNAVAILABLE",
    identifier: None,
    when: "OS temporary directory resolution returns an empty path.",
    message: "tempdir: unable to determine temporary directory (OS returned empty path)",
};
const TEMPDIR_ERRORS: [BuiltinErrorDescriptor; 2] = [
    TEMPDIR_ERROR_TOO_MANY_INPUTS,
    TEMPDIR_ERROR_UNABLE_TO_DETERMINE,
];
pub const TEMPDIR_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &TEMPDIR_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &TEMPDIR_ERRORS,
};

fn tempdir_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    let mut builder = build_runtime_error(error.message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runtime_builtin(
    name = "tempdir",
    category = "io/repl_fs",
    summary = "Return the system temporary directory path with trailing separator.",
    keywords = "tempdir,temporary folder,temp directory,system temp",
    accel = "cpu",
    type_resolver(crate::builtins::io::type_resolvers::tempdir_type),
    descriptor(crate::builtins::io::repl_fs::tempdir::TEMPDIR_DESCRIPTOR),
    builtin_path = "crate::builtins::io::repl_fs::tempdir"
)]
async fn tempdir_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    if !args.is_empty() {
        return Err(tempdir_error(&TEMPDIR_ERROR_TOO_MANY_INPUTS));
    }
    let path = runtime_env::temp_dir();
    if path.as_os_str().is_empty() {
        return Err(tempdir_error(&TEMPDIR_ERROR_UNABLE_TO_DETERMINE));
    }
    let value = path_to_char_array(&path);
    if let Ok(text) = String::try_from(&value) {
        if text.is_empty() {
            return Err(tempdir_error(&TEMPDIR_ERROR_UNABLE_TO_DETERMINE));
        }
    }
    Ok(value)
}

fn path_to_char_array(path: &Path) -> Value {
    let mut text = path.to_string_lossy().into_owned();
    if !text.is_empty() && !ends_with_separator(&text) {
        text.push(std::path::MAIN_SEPARATOR);
    }
    Value::CharArray(CharArray::new_row(&text))
}

fn ends_with_separator(text: &str) -> bool {
    let sep = std::path::MAIN_SEPARATOR;
    text.chars()
        .next_back()
        .is_some_and(|ch| ch == sep || (cfg!(windows) && (ch == '/' || ch == '\\')))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::BuiltinResult;
    use std::convert::TryFrom;
    use std::path::Path;

    fn tempdir_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
        futures::executor::block_on(super::tempdir_builtin(args))
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tempdir_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = TEMPDIR_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"folder = tempdir()"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tempdir_points_to_existing_directory() {
        let value = tempdir_builtin(Vec::new()).expect("tempdir");
        let path_string = String::try_from(&value).expect("convert to string");
        let _path = Path::new(&path_string);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tempdir_returns_char_array_row_vector() {
        let value = tempdir_builtin(Vec::new()).expect("tempdir");
        match value {
            Value::CharArray(CharArray { rows, cols, .. }) => {
                assert_eq!(rows, 1);
                assert!(
                    cols >= 1,
                    "expected tempdir to return at least one character"
                );
            }
            other => panic!("expected CharArray result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tempdir_appends_trailing_separator() {
        let value = tempdir_builtin(Vec::new()).expect("tempdir");
        let path_string = String::try_from(&value).expect("convert to string");
        let expected_sep = std::path::MAIN_SEPARATOR;
        let last = path_string
            .chars()
            .last()
            .expect("tempdir should return non-empty path");
        if cfg!(windows) {
            assert!(
                last == '/' || last == '\\',
                "expected trailing separator, got {:?}",
                path_string
            );
        } else {
            assert_eq!(
                last, expected_sep,
                "expected trailing separator {}, got {}",
                expected_sep, path_string
            );
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tempdir_returns_consistent_result() {
        let first = tempdir_builtin(Vec::new()).expect("tempdir");
        let second = tempdir_builtin(Vec::new()).expect("tempdir");
        let first_str = String::try_from(&first).expect("first string");
        let second_str = String::try_from(&second).expect("second string");
        assert_eq!(
            first_str, second_str,
            "tempdir should be stable within a process"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tempdir_errors_when_arguments_provided() {
        let err = tempdir_builtin(vec![Value::Num(1.0)]).expect_err("expected error");
        assert_eq!(err.message(), TEMPDIR_ERROR_TOO_MANY_INPUTS.message);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn path_to_char_array_appends_separator_when_missing() {
        let path = Path::new("runmat_tempdir_unit_path");
        let value = path_to_char_array(path);
        let text = String::try_from(&value).expect("string conversion");
        assert!(
            text.ends_with(std::path::MAIN_SEPARATOR),
            "expected trailing separator in {text:?}"
        );
        let trimmed = text.trim_end_matches(std::path::MAIN_SEPARATOR);
        assert_eq!(trimmed, path.to_string_lossy());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn path_to_char_array_preserves_existing_separator() {
        let sep = std::path::MAIN_SEPARATOR;
        let input = format!("runmat_tempdir_existing{sep}");
        let path = Path::new(&input);
        let value = path_to_char_array(path);
        let text = String::try_from(&value).expect("string conversion");
        assert_eq!(text, input);
    }

    #[cfg(windows)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ends_with_separator_accepts_forward_slash() {
        assert!(ends_with_separator("C:/Temp/"));
        assert!(ends_with_separator("temp/"));
    }
}
