//! MATLAB-compatible `fgetl` builtin for RunMat.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
};
use runmat_macros::runtime_builtin;
use runmat_value::{NumericDType, Value};

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::common::tensor;
use crate::builtins::io::filetext::{
    helpers::{bytes_to_char_array, read_text_line},
    registry,
};
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "fgetl";

const FGETL_INTEGER_ID_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "fgetl-integer-fileid",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "integer-class fgetl file identifiers are a RunMat extension",
    error_identifier: Some("RunMat:compatibility:FgetlIntegerIdExtension"),
};
const FGETL_SINGLE_ID_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "fgetl-single-fileid",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "single-precision fgetl file identifiers are a RunMat extension",
    error_identifier: Some("RunMat:compatibility:FgetlSingleIdExtension"),
};
const FGETL_RESIDENT_ID_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "fgetl-resident-fileid",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "provider-resident fgetl file identifiers are a RunMat extension",
    error_identifier: Some("RunMat:compatibility:FgetlResidentIdExtension"),
};
pub const FGETL_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    FGETL_INTEGER_ID_EXTENSION,
    FGETL_SINGLE_ID_EXTENSION,
    FGETL_RESIDENT_ID_EXTENSION,
];

const FGETL_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "fileID",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "MATLAB-compatible file identifiers are integer-valued doubles; all eight typed integer classes are one gated RunMat extension and are checked exactly against the registry identifier domain.",
    }];
pub const INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "tline = fgetl(integer_fileID)",
        inputs: &FGETL_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::ScalarOnly,
        notes: "The identifier is validated exactly before registry access; a successful read returns a host character row and end-of-file returns the double sentinel -1.",
    }];

const FGETL_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "tline",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Next line without terminator, or -1 at end-of-file.",
}];
const FGETL_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "fid",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "File identifier opened by fopen.",
}];
const FGETL_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "tline = fgetl(fid)",
    inputs: &FGETL_INPUTS,
    outputs: &FGETL_OUTPUT,
}];

const FGETL_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FGETL.INVALID_INPUT",
    identifier: Some("RunMat:fgetl:InvalidInput"),
    when: "Input argument count or identifier form is invalid.",
    message: "fgetl: invalid input arguments",
};
const FGETL_ERROR_INVALID_IDENTIFIER: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FGETL.INVALID_IDENTIFIER",
    identifier: Some("RunMat:fgetl:InvalidIdentifier"),
    when: "Identifier does not refer to a readable open file.",
    message: "fgetl: invalid file identifier. Use fopen to generate a valid file ID.",
};
const FGETL_ERROR_IO: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FGETL.IO",
    identifier: Some("RunMat:fgetl:IoFailure"),
    when: "Line read or decode fails.",
    message: "fgetl: file I/O failed",
};
const FGETL_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FGETL.INTERNAL",
    identifier: None,
    when: "Internal control-flow conversion failed.",
    message: "fgetl: internal error",
};
const FGETL_ERRORS: [BuiltinErrorDescriptor; 4] = [
    FGETL_ERROR_INVALID_INPUT,
    FGETL_ERROR_INVALID_IDENTIFIER,
    FGETL_ERROR_IO,
    FGETL_ERROR_INTERNAL,
];
pub const FGETL_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &FGETL_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &FGETL_ERRORS,
};

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

fn fgetl_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn fgetl_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    fgetl_error_with_message(error.message, error)
}

fn fgetl_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    fgetl_error_with_message(format!("{}: {}", error.message, detail.as_ref()), error)
}

fn map_control_flow(err: RuntimeError) -> RuntimeError {
    let mut builder = build_runtime_error(format!("{BUILTIN_NAME}: {}", err.message()))
        .with_builtin(BUILTIN_NAME)
        .with_source(err);
    if let Some(identifier) = FGETL_ERROR_INTERNAL.identifier {
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
    summary = "Read the next line without newline characters.",
    keywords = "fgetl,file,io,line,newline",
    accel = "cpu",
    type_resolver(crate::builtins::io::type_resolvers::fgetl_type),
    descriptor(crate::builtins::io::filetext::fgetl::FGETL_DESCRIPTOR),
    integer_capabilities(crate::builtins::io::filetext::fgetl::INTEGER_CAPABILITIES),
    extensions(crate::builtins::io::filetext::fgetl::FGETL_EXTENSIONS),
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
        return Err(fgetl_error_with_detail(
            &FGETL_ERROR_INVALID_INPUT,
            "too many input arguments",
        ));
    }

    preflight_fid(fid_value)?;
    let fid_host = gather_value(fid_value).await?;
    preflight_fid(&fid_host)?;
    let fid = parse_fid(&fid_host)?;
    if fid < 0 {
        return Err(fgetl_error_with_detail(
            &FGETL_ERROR_INVALID_INPUT,
            "file identifier must be non-negative",
        ));
    }
    if fid < 3 {
        return Err(fgetl_error_with_detail(
            &FGETL_ERROR_INVALID_INPUT,
            "standard input/output identifiers are not supported yet",
        ));
    }

    let info =
        registry::info_for(fid).ok_or_else(|| fgetl_error(&FGETL_ERROR_INVALID_IDENTIFIER))?;
    if !permission_allows_read(&info.permission) {
        return Err(fgetl_error_with_detail(
            &FGETL_ERROR_INVALID_IDENTIFIER,
            "file identifier is not open for reading",
        ));
    }
    let handle =
        registry::shared_handle(fid).ok_or_else(|| fgetl_error(&FGETL_ERROR_INVALID_IDENTIFIER))?;

    let mut guard = handle.lock().map_err(|_| {
        fgetl_error_with_detail(
            &FGETL_ERROR_INTERNAL,
            "failed to lock file handle (poisoned mutex)",
        )
    })?;
    let file = guard
        .as_mut()
        .ok_or_else(|| fgetl_error(&FGETL_ERROR_INVALID_IDENTIFIER))?;
    let read = read_text_line(file, None, BUILTIN_NAME)
        .map_err(|e| fgetl_error_with_detail(&FGETL_ERROR_IO, e.message()))?;
    drop(guard);
    if read.encountered_eof {
        registry::mark_eof_encountered(fid);
    }
    if read.eof_before_any {
        return Ok(FgetlEval::end_of_file());
    }

    let encoding = if info.encoding.trim().is_empty() {
        "UTF-8".to_string()
    } else {
        info.encoding.clone()
    };

    let line_len = read.data.len().saturating_sub(read.terminators.len());
    let line_value = bytes_to_char_array(&read.data[..line_len], &encoding, BUILTIN_NAME)
        .map_err(|e| fgetl_error_with_detail(&FGETL_ERROR_IO, e.message()))?;
    Ok(FgetlEval::new(line_value))
}

fn preflight_fid(value: &Value) -> BuiltinResult<()> {
    match value {
        Value::Num(_) => Ok(()),
        Value::Int(_) => crate::compatibility::ensure_builtin_extension_enabled(
            &FGETL_INTEGER_ID_EXTENSION,
            BUILTIN_NAME,
        ),
        Value::Tensor(tensor) if tensor.integer_storage().is_some() => {
            crate::compatibility::ensure_builtin_extension_enabled(
                &FGETL_INTEGER_ID_EXTENSION,
                BUILTIN_NAME,
            )
        }
        Value::Tensor(tensor) if tensor.numeric_dtype() == NumericDType::F32 => {
            crate::compatibility::ensure_builtin_extension_enabled(
                &FGETL_SINGLE_ID_EXTENSION,
                BUILTIN_NAME,
            )
        }
        Value::Tensor(_) => Ok(()),
        Value::GpuTensor(handle) => {
            crate::compatibility::ensure_builtin_extension_enabled(
                &FGETL_RESIDENT_ID_EXTENSION,
                BUILTIN_NAME,
            )?;
            if runmat_accelerate_api::handle_integer_type(handle).is_some() {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &FGETL_INTEGER_ID_EXTENSION,
                    BUILTIN_NAME,
                )?;
            } else if runmat_accelerate_api::handle_precision(handle)
                == Some(runmat_accelerate_api::ProviderPrecision::F32)
            {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &FGETL_SINGLE_ID_EXTENSION,
                    BUILTIN_NAME,
                )?;
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

async fn gather_value(value: &Value) -> BuiltinResult<Value> {
    gather_if_needed_async(value)
        .await
        .map_err(map_control_flow)
}

fn parse_fid(value: &Value) -> BuiltinResult<i32> {
    fn checked_f64_to_i32(n: f64) -> BuiltinResult<i32> {
        if n < i32::MIN as f64 || n > i32::MAX as f64 {
            return Err(fgetl_error_with_detail(
                &FGETL_ERROR_INVALID_INPUT,
                "file identifier is out of range",
            ));
        }
        Ok(n as i32)
    }

    match value {
        Value::Num(n) => {
            if !n.is_finite() {
                return Err(fgetl_error_with_detail(
                    &FGETL_ERROR_INVALID_INPUT,
                    "file identifier must be finite",
                ));
            }
            if (n.fract()).abs() > f64::EPSILON {
                return Err(fgetl_error_with_detail(
                    &FGETL_ERROR_INVALID_INPUT,
                    "file identifier must be an integer scalar",
                ));
            }
            checked_f64_to_i32(*n)
        }
        Value::Int(i) => i.try_to_i32().ok_or_else(|| {
            fgetl_error_with_detail(
                &FGETL_ERROR_INVALID_INPUT,
                "file identifier is out of range",
            )
        }),
        Value::Tensor(t) if tensor::is_scalar_tensor(t) => {
            if let Some(storage) = t.integer_storage() {
                let value = storage.value_at(0).expect("one-element integer storage");
                return value.try_to_i32().ok_or_else(|| {
                    fgetl_error_with_detail(
                        &FGETL_ERROR_INVALID_INPUT,
                        "file identifier is out of range",
                    )
                });
            }
            let n = tensor::tensor_value_f64(t, 0);
            if !n.is_finite() {
                return Err(fgetl_error_with_detail(
                    &FGETL_ERROR_INVALID_INPUT,
                    "file identifier must be finite",
                ));
            }
            if (n.fract()).abs() > f64::EPSILON {
                return Err(fgetl_error_with_detail(
                    &FGETL_ERROR_INVALID_INPUT,
                    "file identifier must be an integer scalar",
                ));
            }
            checked_f64_to_i32(n)
        }
        _ => Err(fgetl_error_with_detail(
            &FGETL_ERROR_INVALID_INPUT,
            "file identifier must be a numeric scalar",
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
    use runmat_value::{IntValue, IntegerStorage, Tensor};
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

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fgetl_descriptor_signature_covers_core_form() {
        let labels: Vec<&str> = FGETL_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"tline = fgetl(fid)"));
    }

    #[test]
    fn fgetl_integer_capability_covers_all_typed_file_identifiers() {
        assert_eq!(INTEGER_CAPABILITIES.len(), 1);
        assert_eq!(INTEGER_CAPABILITIES[0].inputs[0].classes.len(), 8);
        assert_eq!(
            INTEGER_CAPABILITIES[0].inputs[0].availability,
            BuiltinIntegerInputAvailability::RunMatOnly
        );
    }

    #[test]
    fn fgetl_typed_single_and_resident_identifiers_are_independently_gated() {
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let integer = preflight_fid(&Value::Int(IntValue::I32(3))).unwrap_err();
        assert_eq!(
            integer.identifier(),
            Some("RunMat:compatibility:FgetlIntegerIdExtension")
        );

        let single = Tensor::new_with_dtype(vec![3.0], vec![1, 1], NumericDType::F32)
            .expect("single identifier");
        let single = preflight_fid(&Value::Tensor(single)).unwrap_err();
        assert_eq!(
            single.identifier(),
            Some("RunMat:compatibility:FgetlSingleIdExtension")
        );

        test_support::with_test_provider(|provider| {
            let host = [3.0f64];
            let view = HostTensorView {
                data: &host,
                shape: &[1, 1],
            };
            let resident = Value::GpuTensor(provider.upload(&view).expect("upload identifier"));
            let resident = preflight_fid(&resident).unwrap_err();
            assert_eq!(
                resident.identifier(),
                Some("RunMat:compatibility:FgetlResidentIdExtension")
            );
        });
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
    fn parse_fid_rejects_values_outside_i32_range() {
        let too_large =
            unwrap_error_message(parse_fid(&Value::Num(i32::MAX as f64 + 1.0)).unwrap_err());
        assert_eq!(
            too_large,
            format!(
                "{}: file identifier is out of range",
                FGETL_ERROR_INVALID_INPUT.message
            )
        );

        let too_small =
            unwrap_error_message(parse_fid(&Value::Num(i32::MIN as f64 - 1.0)).unwrap_err());
        assert_eq!(
            too_small,
            format!(
                "{}: file identifier is out of range",
                FGETL_ERROR_INVALID_INPUT.message
            )
        );

        let int_too_large = unwrap_error_message(
            parse_fid(&Value::Int(IntValue::I64(i32::MAX as i64 + 1))).unwrap_err(),
        );
        assert_eq!(
            int_too_large,
            format!(
                "{}: file identifier is out of range",
                FGETL_ERROR_INVALID_INPUT.message
            )
        );

        let uint64_too_large =
            unwrap_error_message(parse_fid(&Value::Int(IntValue::U64(u64::MAX))).unwrap_err());
        assert_eq!(
            uint64_too_large,
            format!(
                "{}: file identifier is out of range",
                FGETL_ERROR_INVALID_INPUT.message
            )
        );

        let typed_tensor =
            Tensor::new_integer(IntegerStorage::U16(vec![7]), vec![1, 1]).expect("typed fid");
        assert_eq!(parse_fid(&Value::Tensor(typed_tensor)).unwrap(), 7);
        let typed_tensor_too_large =
            Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX]), vec![1, 1])
                .expect("typed fid");
        let typed_tensor_too_large =
            unwrap_error_message(parse_fid(&Value::Tensor(typed_tensor_too_large)).unwrap_err());
        assert_eq!(
            typed_tensor_too_large,
            format!(
                "{}: file identifier is out of range",
                FGETL_ERROR_INVALID_INPUT.message
            )
        );

        let tensor = Tensor::new(vec![i32::MAX as f64 + 1.0], vec![1, 1]).expect("tensor");
        let tensor_too_large = unwrap_error_message(parse_fid(&Value::Tensor(tensor)).unwrap_err());
        assert_eq!(
            tensor_too_large,
            format!(
                "{}: file identifier is out of range",
                FGETL_ERROR_INVALID_INPUT.message
            )
        );
    }

    #[test]
    fn fgetl_typed_identifier_tensors_ignore_poisoned_f64_mirrors() {
        let classes = [
            IntegerStorage::I8(vec![7]),
            IntegerStorage::I16(vec![7]),
            IntegerStorage::I32(vec![7]),
            IntegerStorage::I64(vec![7]),
            IntegerStorage::U8(vec![7]),
            IntegerStorage::U16(vec![7]),
            IntegerStorage::U32(vec![7]),
            IntegerStorage::U64(vec![7]),
        ];
        for storage in classes {
            let tensor = Tensor::new_integer(storage, vec![1, 1]).expect("typed fid");
            assert_eq!(parse_fid(&Value::Tensor(tensor)).unwrap(), 7);
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
    fn fgetl_consumes_lfcr_as_one_newline_sequence() {
        let _guard = registry_guard();
        registry::reset_for_tests();
        let path = unique_path("fgetl_lfcr");
        test_support::fs::write(&path, b"first\n\rsecond").unwrap();
        let handle = fopen_path(&path);

        let first = run_evaluate(&Value::Num(handle.fid as f64), &[]).expect("first");
        assert_eq!(char_text(first.first_output()), "first");
        let second = run_evaluate(&Value::Num(handle.fid as f64), &[]).expect("second");
        assert_eq!(char_text(second.first_output()), "second");

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
        assert_eq!(
            err,
            format!(
                "{}: file identifier is not open for reading",
                FGETL_ERROR_INVALID_IDENTIFIER.message
            )
        );
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
        assert_eq!(err, FGETL_ERROR_INVALID_IDENTIFIER.message);
        test_support::fs::remove_file(&path).unwrap();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fgetl_gathers_gpu_scalar_argument() {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
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
