//! MATLAB-compatible `feof` builtin for RunMat.
//!
//! Provides end-of-file detection for file identifiers opened via `fopen`.
//! The implementation mirrors MATLAB semantics for host-side files and
//! integrates with the shared registry used by the other text I/O builtins.

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
use crate::builtins::io::filetext::registry;
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const IDENTIFIER_TYPE_ERROR_DETAIL: &str = "file identifier must be a numeric scalar";
const BUILTIN_NAME: &str = "feof";

const FEOF_INTEGER_ID_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "feof-integer-fileid",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "integer-class feof file identifiers are a RunMat extension",
    error_identifier: Some("RunMat:compatibility:FeofIntegerIdExtension"),
};
const FEOF_LOGICAL_ID_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "feof-logical-fileid",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "logical feof file identifiers are a RunMat extension",
    error_identifier: Some("RunMat:compatibility:FeofLogicalIdExtension"),
};
const FEOF_RESIDENT_ID_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "feof-resident-fileid",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "provider-resident feof file identifiers are a RunMat extension",
    error_identifier: Some("RunMat:compatibility:FeofResidentIdExtension"),
};
pub const FEOF_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    FEOF_INTEGER_ID_EXTENSION,
    FEOF_LOGICAL_ID_EXTENSION,
    FEOF_RESIDENT_ID_EXTENSION,
];

const FEOF_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "fileID",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "MATLAB-compatible file IDs are integer-valued doubles; all eight typed integer classes are one gated RunMat extension and are checked exactly against i32 before registry access.",
    }];
pub const INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "tf = feof(integer_fileID)",
        inputs: &FEOF_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::Logical,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::ScalarOnly,
        notes: "The identifier is validated exactly before the registry query; resident and logical identifiers have independent RunMat-only gates, and the result is a host logical scalar.",
    }];

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
const FEOF_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FEOF.INTERNAL",
    identifier: None,
    when: "Internal control-flow conversion failed.",
    message: "feof: internal error",
};
const FEOF_ERRORS: [BuiltinErrorDescriptor; 3] = [
    FEOF_ERROR_INVALID_INPUT,
    FEOF_ERROR_INVALID_IDENTIFIER,
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
    summary = "Query end-of-file state for file identifiers.",
    keywords = "feof,end of file,io,file identifier",
    accel = "cpu",
    type_resolver(crate::builtins::io::type_resolvers::feof_type),
    descriptor(crate::builtins::io::filetext::feof::FEOF_DESCRIPTOR),
    integer_capabilities(crate::builtins::io::filetext::feof::INTEGER_CAPABILITIES),
    extensions(crate::builtins::io::filetext::feof::FEOF_EXTENSIONS),
    builtin_path = "crate::builtins::io::filetext::feof"
)]
async fn feof_builtin(fid: Value) -> crate::BuiltinResult<Value> {
    let at_end = evaluate(&fid).await?;
    Ok(Value::Bool(at_end))
}

/// Evaluate the `feof` builtin without invoking the runtime dispatcher.
pub async fn evaluate(fid_value: &Value) -> BuiltinResult<bool> {
    preflight_fid(fid_value)?;
    let fid_host = gather_if_needed_async(fid_value)
        .await
        .map_err(map_control_flow)?;
    preflight_fid(&fid_host)?;
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

    registry::eof_encountered(fid).ok_or_else(|| feof_error(&FEOF_ERROR_INVALID_IDENTIFIER))
}

fn preflight_fid(value: &Value) -> BuiltinResult<()> {
    match value {
        Value::Num(_) => Ok(()),
        Value::Int(_) => crate::compatibility::ensure_builtin_extension_enabled(
            &FEOF_INTEGER_ID_EXTENSION,
            BUILTIN_NAME,
        ),
        Value::Bool(_) | Value::LogicalArray(_) => {
            crate::compatibility::ensure_builtin_extension_enabled(
                &FEOF_LOGICAL_ID_EXTENSION,
                BUILTIN_NAME,
            )
        }
        Value::Tensor(tensor) if tensor.integer_storage().is_some() => {
            crate::compatibility::ensure_builtin_extension_enabled(
                &FEOF_INTEGER_ID_EXTENSION,
                BUILTIN_NAME,
            )
        }
        Value::Tensor(tensor) if tensor.numeric_dtype() == NumericDType::F64 => Ok(()),
        Value::Tensor(_) => Err(feof_error_with_detail(
            &FEOF_ERROR_INVALID_INPUT,
            "file identifier must be an integer-valued double scalar",
        )),
        Value::GpuTensor(handle) => {
            crate::compatibility::ensure_builtin_extension_enabled(
                &FEOF_RESIDENT_ID_EXTENSION,
                BUILTIN_NAME,
            )?;
            if runmat_accelerate_api::handle_integer_type(handle).is_some() {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &FEOF_INTEGER_ID_EXTENSION,
                    BUILTIN_NAME,
                )?;
            } else if runmat_accelerate_api::handle_is_logical(handle) {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &FEOF_LOGICAL_ID_EXTENSION,
                    BUILTIN_NAME,
                )?;
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

fn parse_fid(value: &Value) -> BuiltinResult<i32> {
    match value {
        Value::Num(n) => parse_scalar_fid(*n),
        Value::Int(int) => int.try_to_i32().ok_or_else(|| {
            feof_error_with_detail(&FEOF_ERROR_INVALID_INPUT, "file identifier is out of range")
        }),
        Value::Tensor(t) => {
            if tensor::is_scalar_tensor(t) {
                if let Some(storage) = t.integer_storage() {
                    let value = storage.value_at(0).expect("one-element integer storage");
                    return value.try_to_i32().ok_or_else(|| {
                        feof_error_with_detail(
                            &FEOF_ERROR_INVALID_INPUT,
                            "file identifier is out of range",
                        )
                    });
                }
                if t.numeric_dtype() != NumericDType::F64 {
                    return Err(feof_error_with_detail(
                        &FEOF_ERROR_INVALID_INPUT,
                        "file identifier must be an integer-valued double scalar",
                    ));
                }
                parse_scalar_fid(tensor::tensor_value_f64(t, 0))
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
    if value.fract() != 0.0 {
        return Err(feof_error_with_detail(
            &FEOF_ERROR_INVALID_INPUT,
            "file identifier must be an integer",
        ));
    }
    let rounded = value;
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
    use crate::builtins::io::filetext::{fclose, fgets, fopen, fread, registry};
    use crate::RuntimeError;
    use runmat_accelerate_api::HostTensorView;
    use runmat_filesystem::File;
    use runmat_time::system_time_now;
    use runmat_value::{IntValue, IntegerStorage, Tensor, Value};
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

    #[test]
    fn typed_file_identifier_parser_rejects_unrepresentable_uint64() {
        assert_eq!(parse_fid(&Value::Int(IntValue::U16(7))).unwrap(), 7);
        assert!(parse_fid(&Value::Int(IntValue::U64(u64::MAX))).is_err());

        let fid_tensor = Tensor::new_integer(IntegerStorage::U16(vec![7]), vec![1, 1])
            .expect("typed fid tensor");
        assert_eq!(parse_fid(&Value::Tensor(fid_tensor)).unwrap(), 7);
        let fid_too_large = Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX]), vec![1, 1])
            .expect("typed fid tensor");
        assert!(parse_fid(&Value::Tensor(fid_too_large)).is_err());

        let single = Tensor::new_with_dtype(vec![7.0], vec![1, 1], NumericDType::F32)
            .expect("single file identifier");
        assert!(preflight_fid(&Value::Tensor(single.clone())).is_err());
        assert!(parse_fid(&Value::Tensor(single)).is_err());

        assert!(parse_scalar_fid(f64::from_bits(1.0_f64.to_bits() + 1)).is_err());
    }

    #[test]
    fn feof_typed_identifier_tensors_ignore_poisoned_f64_mirrors() {
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
    fn feof_returns_true_after_a_read_encounters_end_of_file() {
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

        // An unbounded read probes once beyond the available bytes.
        let _ = run_fread(&Value::Num(fid as f64), &Vec::new()).expect("fread");

        let at_end = run_evaluate(&Value::Num(fid as f64)).expect("feof");
        assert!(at_end);

        run_fclose(&[Value::Num(fid as f64)]).unwrap();
        test_support::fs::remove_file(path).unwrap();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn feof_newly_opened_empty_file_is_false_until_a_read_encounters_eof() {
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
        assert!(!at_end);

        let _ = run_fread(&Value::Num(fid as f64), &Vec::new()).expect("fread");
        assert!(run_evaluate(&Value::Num(fid as f64)).expect("feof"));

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

            {
                let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
                let err = run_evaluate(&value).expect_err("resident ID must be gated");
                assert_eq!(
                    err.identifier(),
                    Some("RunMat:compatibility:FeofResidentIdExtension")
                );
            }
            {
                let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
                let at_end = run_evaluate(&value).expect("feof");
                assert!(!at_end);
            }

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

    #[test]
    fn feof_exact_end_requires_a_subsequent_read_and_frewind_clears_state() {
        let _guard = registry_guard();
        registry::reset_for_tests();
        let path = unique_path("feof_exact_end_then_rewind");
        {
            let mut file = File::create(&path).expect("create");
            file.write_all(b"abc").expect("write");
        }
        let open = run_fopen(&[
            Value::from(path.to_string_lossy().to_string()),
            Value::from("rb"),
        ])
        .expect("fopen");
        let fid = open.as_open().unwrap().fid;

        let _ = run_fread(&Value::Num(fid), &[Value::Num(3.0)]).expect("exact read");
        assert!(!run_evaluate(&Value::Num(fid)).expect("feof at exact end"));
        let _ = run_fread(&Value::Num(fid), &[Value::Num(1.0)]).expect("eof read");
        assert!(run_evaluate(&Value::Num(fid)).expect("feof after eof read"));

        futures::executor::block_on(crate::builtins::io::filetext::frewind::evaluate(
            &Value::Num(fid),
        ))
        .expect("frewind");
        assert!(!run_evaluate(&Value::Num(fid)).expect("feof after rewind"));

        run_fclose(&[Value::Num(fid)]).unwrap();
        test_support::fs::remove_file(path).unwrap();
    }

    #[test]
    fn fgets_marks_eof_when_final_newline_lookahead_reaches_end() {
        let _guard = registry_guard();
        registry::reset_for_tests();
        let path = unique_path("feof_fgets_state");
        {
            let mut file = File::create(&path).expect("create");
            file.write_all(b"line\n").expect("write");
        }
        let open = run_fopen(&[
            Value::from(path.to_string_lossy().to_string()),
            Value::from("r"),
        ])
        .expect("fopen");
        let fid = open.as_open().unwrap().fid;

        futures::executor::block_on(fgets::evaluate(&Value::Num(fid), &[]))
            .expect("newline-terminated line");
        assert!(run_evaluate(&Value::Num(fid)).expect("final newline lookahead reached eof"));

        run_fclose(&[Value::Num(fid)]).unwrap();
        test_support::fs::remove_file(path).unwrap();
    }

    #[test]
    fn feof_typed_and_logical_identifiers_are_independently_gated() {
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let integer = run_evaluate(&Value::Int(IntValue::I32(3))).unwrap_err();
        assert_eq!(
            integer.identifier(),
            Some("RunMat:compatibility:FeofIntegerIdExtension")
        );
        let logical = run_evaluate(&Value::Bool(true)).unwrap_err();
        assert_eq!(
            logical.identifier(),
            Some("RunMat:compatibility:FeofLogicalIdExtension")
        );
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
