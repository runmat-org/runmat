//! MATLAB-compatible `opentoline` builtin for editor navigation requests.

use std::path::PathBuf;

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    Tensor, Value,
};
use runmat_filesystem as vfs;
use runmat_macros::runtime_builtin;

use crate::builtins::common::path_search::{find_file_with_extensions, GENERAL_FILE_EXTENSIONS};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::common::tensor;
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "opentoline";

const INTEGER_LINE_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "opentoline-integer-line",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "opentoline with a native typed-integer line number is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:OpentolineIntegerLineExtension"),
};
const INTEGER_COLUMN_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "opentoline-integer-column",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "opentoline with a native typed-integer column number is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:OpentolineIntegerColumnExtension"),
};
const RESIDENT_POSITION_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "opentoline-resident-position",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "opentoline with an explicit GPU-resident line or column is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:OpentolineResidentPositionExtension"),
};

pub const EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    INTEGER_LINE_EXTENSION,
    INTEGER_COLUMN_EXTENSION,
    RESIDENT_POSITION_EXTENSION,
];

const INTEGER_LINE_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "line",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "[integer-audit-open] Public evidence establishes a positive line-number control but does not enumerate native typed classes; ordinary integer-valued double remains the compatibility form.",
    }];
const INTEGER_COLUMN_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "column",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "[integer-audit-open] Public evidence establishes an optional structural column position but does not enumerate native typed classes; ordinary integer-valued double remains the compatibility form.",
    }];

const fn position_capability(
    form: &'static str,
    inputs: &'static [BuiltinIntegerInputCapability],
) -> BuiltinIntegerCapabilityDescriptor {
    BuiltinIntegerCapabilityDescriptor {
        form,
        inputs,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::ScalarOnly,
        notes: "The exact positive position controls only editor navigation. Typed-class and explicit-residency gates run before gather, file resolution, or side effects; automatic residency remains transparent.",
    }
}

pub const INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    position_capability(
        "opentoline(filename, integer_line, ...)",
        &INTEGER_LINE_INPUTS,
    ),
    position_capability(
        "opentoline(filename, line, integer_column, ...)",
        &INTEGER_COLUMN_INPUTS,
    ),
];

const FILE_INPUT: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "filename",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "File to open in the editor.",
};

const LINE_INPUT: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "line",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "One-based line number.",
};

const COLUMN_INPUT: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "column",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Optional,
    default: Some("1"),
    description: "One-based column number.",
};

const SELECT_INPUT: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "option",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Editor navigation option such as 'select'.",
};

const TWO_INPUTS: [BuiltinParamDescriptor; 2] = [FILE_INPUT, LINE_INPUT];
const THREE_INPUTS: [BuiltinParamDescriptor; 3] = [FILE_INPUT, LINE_INPUT, COLUMN_INPUT];
const FOUR_INPUTS: [BuiltinParamDescriptor; 4] =
    [FILE_INPUT, LINE_INPUT, COLUMN_INPUT, SELECT_INPUT];

const SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "opentoline(filename, line)",
        inputs: &TWO_INPUTS,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "opentoline(filename, line, column)",
        inputs: &THREE_INPUTS,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "opentoline(filename, line, column, option)",
        inputs: &FOUR_INPUTS,
        outputs: &[],
    },
];

const ERROR_ARG_COUNT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.OPENTOLINE.ARG_COUNT",
    identifier: Some("RunMat:opentoline:ArgumentCount"),
    when: "The call does not provide two to four input arguments.",
    message: "opentoline: expected filename, line, optional column, and optional option",
};

const ERROR_FILENAME: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.OPENTOLINE.FILENAME",
    identifier: Some("RunMat:opentoline:InvalidFilename"),
    when: "The filename is not a string scalar or row character vector.",
    message: "opentoline: filename must be a character vector or string scalar",
};

const ERROR_POSITION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.OPENTOLINE.POSITION",
    identifier: Some("RunMat:opentoline:InvalidPosition"),
    when: "The line or column argument is not a positive integer scalar.",
    message: "opentoline: line and column must be positive integer scalars",
};

const ERROR_OPTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.OPENTOLINE.OPTION",
    identifier: Some("RunMat:opentoline:InvalidOption"),
    when: "The optional editor behavior argument is unsupported.",
    message: "opentoline: option must be 'select' when provided",
};

const ERROR_NOT_FOUND: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.OPENTOLINE.NOT_FOUND",
    identifier: Some("RunMat:opentoline:FileNotFound"),
    when: "No file matching the supplied name can be resolved.",
    message: "opentoline: file was not found",
};

const ERROR_OUTPUT_COUNT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.OPENTOLINE.OUTPUT_COUNT",
    identifier: Some("RunMat:opentoline:TooManyOutputs"),
    when: "The call requests one or more output arguments.",
    message: "opentoline: expected no output arguments",
};

const ERRORS: [BuiltinErrorDescriptor; 6] = [
    ERROR_ARG_COUNT,
    ERROR_FILENAME,
    ERROR_POSITION,
    ERROR_OPTION,
    ERROR_NOT_FOUND,
    ERROR_OUTPUT_COUNT,
];

pub const OPENTOLINE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::io::repl_fs::opentoline")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "opentoline",
    op_kind: GpuOpKind::Custom("io-opentoline"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Runs on the host. Explicit resident line/column controls are a gated RunMat extension and gather only after compatibility classification; resident filename/option values reject without provider access.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::io::repl_fs::opentoline")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "opentoline",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Editor navigation is a host-side side effect and is not eligible for fusion.",
};

#[runtime_builtin(
    name = "opentoline",
    category = "io/repl_fs",
    summary = "Open a file to a requested editor line in MATLAB-compatible code.",
    keywords = "opentoline,open,editor,file,line,column",
    accel = "cpu",
    sink = true,
    suppress_auto_output = true,
    type_resolver(crate::builtins::io::type_resolvers::opentoline_type),
    descriptor(crate::builtins::io::repl_fs::opentoline::OPENTOLINE_DESCRIPTOR),
    extensions(crate::builtins::io::repl_fs::opentoline::EXTENSIONS),
    integer_capabilities(crate::builtins::io::repl_fs::opentoline::INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::io::repl_fs::opentoline"
)]
async fn opentoline_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    if !(2..=4).contains(&args.len()) {
        return Err(opentoline_error(&ERROR_ARG_COUNT, ERROR_ARG_COUNT.message));
    }
    if requested_output_count() > 0 {
        return Err(opentoline_error(
            &ERROR_OUTPUT_COUNT,
            format!(
                "opentoline: expected no output arguments, got {}",
                requested_output_count()
            ),
        ));
    }

    gate_position_extensions(&args[1], &INTEGER_LINE_EXTENSION)?;
    if let Some(column) = args.get(2) {
        gate_position_extensions(column, &INTEGER_COLUMN_EXTENSION)?;
    }
    let filename = filename_arg(&args[0]).await?;
    if filename.is_empty() {
        return Err(opentoline_error(
            &ERROR_FILENAME,
            "opentoline: filename must not be empty",
        ));
    }
    let _path = resolve_file(&filename).await?;
    let _line = positive_integer_arg(&args[1], "line").await?;
    let _column = match args.get(2) {
        Some(value) => positive_integer_arg(value, "column").await?,
        None => 1,
    };
    if let Some(option) = args.get(3) {
        parse_option(option).await?;
    }

    Ok(empty_array())
}

fn gate_position_extensions(
    value: &Value,
    integer_extension: &BuiltinExtensionDescriptor,
) -> BuiltinResult<()> {
    if crate::builtins::common::validation::value_has_native_integer_class(value) {
        crate::compatibility::ensure_builtin_extension_enabled(integer_extension, BUILTIN_NAME)?;
    }
    if matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_is_explicit(handle))
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &RESIDENT_POSITION_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    Ok(())
}

async fn filename_arg(value: &Value) -> BuiltinResult<String> {
    match value {
        Value::String(text) => Ok(text.clone()),
        Value::CharArray(chars) if chars.rows == 1 => Ok(chars.data.iter().collect()),
        Value::StringArray(array) if array.data.len() == 1 => Ok(array.data[0].clone()),
        _ => Err(opentoline_error(&ERROR_FILENAME, ERROR_FILENAME.message)),
    }
}

async fn parse_option(value: &Value) -> BuiltinResult<()> {
    let option = text_arg(value, &ERROR_OPTION).await?;
    if option.eq_ignore_ascii_case("select") {
        Ok(())
    } else {
        Err(opentoline_error(
            &ERROR_OPTION,
            format!("opentoline: unsupported option '{option}'"),
        ))
    }
}

async fn text_arg(value: &Value, error: &'static BuiltinErrorDescriptor) -> BuiltinResult<String> {
    match value {
        Value::String(text) => Ok(text.clone()),
        Value::CharArray(chars) if chars.rows == 1 => Ok(chars.data.iter().collect()),
        Value::StringArray(array) if array.data.len() == 1 => Ok(array.data[0].clone()),
        _ => Err(opentoline_error(error, error.message)),
    }
}

async fn positive_integer_arg(value: &Value, label: &str) -> BuiltinResult<usize> {
    if let Some(text) = text_without_gather(value) {
        return parse_positive_integer_text(&text, label);
    }
    let gathered = gather_if_needed_async(value)
        .await
        .map_err(|err| opentoline_flow_error("opentoline", err))?;
    if let Value::Int(integer) = &gathered {
        return integer
            .try_to_usize()
            .filter(|position| *position > 0)
            .ok_or_else(|| {
                opentoline_error(
                    &ERROR_POSITION,
                    format!("opentoline: {label} must be a positive integer scalar"),
                )
            });
    }
    let Some(position) = (match gathered {
        Value::Num(value) => position_value_to_usize(value),
        Value::Tensor(tensor) if tensor::is_scalar_tensor(&tensor) => {
            position_tensor_to_usize(&tensor)
        }
        _ => {
            return Err(opentoline_error(
                &ERROR_POSITION,
                format!("opentoline: {label} must be a numeric scalar"),
            ))
        }
    }) else {
        return Err(opentoline_error(
            &ERROR_POSITION,
            format!("opentoline: {label} must be a positive integer scalar"),
        ));
    };
    Ok(position)
}

fn text_without_gather(value: &Value) -> Option<String> {
    match value {
        Value::String(text) => Some(text.clone()),
        Value::CharArray(chars) if chars.rows == 1 => Some(chars.data.iter().collect()),
        Value::StringArray(array) if array.data.len() == 1 => Some(array.data[0].clone()),
        _ => None,
    }
}

fn parse_positive_integer_text(text: &str, label: &str) -> BuiltinResult<usize> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Err(opentoline_error(
            &ERROR_POSITION,
            format!("opentoline: {label} must not be empty"),
        ));
    }
    let value = trimmed.parse::<usize>().map_err(|_| {
        opentoline_error(
            &ERROR_POSITION,
            format!("opentoline: {label} must be a positive integer scalar"),
        )
    })?;
    if value == 0 {
        return Err(opentoline_error(
            &ERROR_POSITION,
            format!("opentoline: {label} must be positive"),
        ));
    }
    Ok(value)
}

fn position_value_to_usize(value: f64) -> Option<usize> {
    if !value.is_finite() || value < 1.0 || value.fract().abs() > f64::EPSILON {
        return None;
    }
    if value > usize::MAX as f64 || (usize::BITS == 64 && value == usize::MAX as f64) {
        return None;
    }
    Some(value as usize)
}

fn position_tensor_to_usize(tensor: &Tensor) -> Option<usize> {
    if let Some(storage) = tensor.integer_storage() {
        return storage
            .value_at(0)
            .and_then(|value| value.try_to_usize())
            .filter(|position| *position > 0);
    }
    position_value_to_usize(tensor::tensor_value_f64(tensor, 0))
}

async fn resolve_file(name: &str) -> BuiltinResult<PathBuf> {
    let path = find_file_with_extensions(name, GENERAL_FILE_EXTENSIONS, BUILTIN_NAME)
        .await
        .map_err(|err| opentoline_error(&ERROR_NOT_FOUND, err))?
        .ok_or_else(|| {
            opentoline_error(
                &ERROR_NOT_FOUND,
                format!("opentoline: file was not found: {name}"),
            )
        })?;
    Ok(vfs::canonicalize_async(&path).await.unwrap_or(path))
}

fn requested_output_count() -> usize {
    crate::output_count::current_output_count()
        .or_else(crate::output_context::requested_output_count)
        .unwrap_or(0)
}

fn empty_array() -> Value {
    Value::Tensor(Tensor::zeros(vec![0, 0]))
}

fn opentoline_error(
    error: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn opentoline_flow_error(context: &str, err: RuntimeError) -> RuntimeError {
    let identifier = err.identifier().map(str::to_string);
    let mut builder = build_runtime_error(format!("{context}: {}", err.message()))
        .with_builtin(BUILTIN_NAME)
        .with_source(err);
    if let Some(identifier) = identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_builtins::IntValue;
    use tempfile::tempdir;

    fn call(args: Vec<Value>) -> BuiltinResult<Value> {
        block_on(opentoline_builtin(args))
    }

    #[test]
    fn opentoline_resolves_file_and_returns_empty_array() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("target.m");
        std::fs::write(&path, "a = 1;\nb = 2;\n").expect("write file");

        let result = call(vec![
            Value::String(path.to_string_lossy().to_string()),
            Value::Num(2.0),
        ])
        .expect("opentoline");

        assert_eq!(result, empty_array());
    }

    #[test]
    fn opentoline_typed_positions_preserve_integer_bounds() {
        assert_eq!(
            block_on(positive_integer_arg(&Value::Int(IntValue::U16(7)), "line")).unwrap(),
            7
        );
        assert!(block_on(positive_integer_arg(&Value::Int(IntValue::I8(-1)), "line")).is_err());
        assert!(block_on(positive_integer_arg(&Value::Int(IntValue::U8(0)), "column")).is_err());

        let large = 9_007_199_254_740_993_u64;
        let parsed = block_on(positive_integer_arg(
            &Value::Int(IntValue::U64(large)),
            "line",
        ));
        if usize::BITS == 64 {
            assert_eq!(parsed.unwrap(), large as usize);
        } else {
            assert!(parsed.is_err());
        }
    }

    #[test]
    fn opentoline_double_positions_reject_unrepresentable_platform_bounds() {
        let boundary = block_on(positive_integer_arg(&Value::Num(usize::MAX as f64), "line"));
        if usize::BITS == 64 {
            assert!(boundary.is_err());
        } else {
            assert_eq!(boundary.unwrap(), usize::MAX);
        }
        assert!(block_on(positive_integer_arg(
            &Value::Num((usize::MAX as f64) + 1.0),
            "line"
        ))
        .is_err());
    }

    #[test]
    fn opentoline_tensor_positions_read_integer_storage_exactly() {
        let line =
            Tensor::new_integer(runmat_builtins::IntegerStorage::U64(vec![9]), vec![1, 1]).unwrap();

        assert_eq!(
            block_on(positive_integer_arg(&Value::Tensor(line), "line")).unwrap(),
            9
        );

        let zero =
            Tensor::new_integer(runmat_builtins::IntegerStorage::U8(vec![0]), vec![1, 1]).unwrap();
        assert!(block_on(positive_integer_arg(&Value::Tensor(zero), "line")).is_err());
    }

    #[test]
    fn opentoline_typed_positions_are_independently_gated_before_file_lookup() {
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let line_error = call(vec![
            Value::String("definitely-missing.m".into()),
            Value::Int(IntValue::U16(7)),
        ])
        .expect_err("typed line must be gated");
        assert_eq!(
            line_error.identifier(),
            Some("RunMat:compatibility:OpentolineIntegerLineExtension")
        );

        let column_error = call(vec![
            Value::String("definitely-missing.m".into()),
            Value::Num(1.0),
            Value::Int(IntValue::U16(3)),
        ])
        .expect_err("typed column must be gated");
        assert_eq!(
            column_error.identifier(),
            Some("RunMat:compatibility:OpentolineIntegerColumnExtension")
        );
    }

    #[test]
    fn opentoline_resident_position_is_gated_without_provider_access() {
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let resident = Value::GpuTensor(
            runmat_accelerate_api::GpuTensorHandle {
                shape: vec![1, 1],
                device_id: u32::MAX,
                buffer_id: u64::MAX,
                descriptor: Default::default(),
            }
            .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit),
        );
        let error = call(vec![Value::String("definitely-missing.m".into()), resident])
            .expect_err("resident line must be gated");

        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:OpentolineResidentPositionExtension")
        );
    }

    #[test]
    fn opentoline_automatically_resident_double_position_gathers_in_strict_mode() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("target.m");
        std::fs::write(&path, "a = 1;\n").expect("write file");
        crate::builtins::common::test_support::with_test_provider(|provider| {
            let line = Tensor::new(vec![1.0], vec![1, 1]).expect("line");
            let handle = crate::builtins::common::gpu_helpers::upload_tensor(provider, &line)
                .expect("resident line");
            let handle =
                handle.with_provenance(runmat_accelerate_api::GpuHandleProvenance::Automatic);
            let result = {
                let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
                call(vec![
                    Value::String(path.to_string_lossy().into_owned()),
                    Value::GpuTensor(handle.clone()),
                ])
                .expect("automatic position gathers transparently")
            };
            assert_eq!(result, empty_array());
            runmat_accelerate_api::clear_handle_metadata(&handle);
        });
    }

    #[test]
    fn opentoline_accepts_column_and_select_option() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("target.m");
        std::fs::write(&path, "abcdef\n").expect("write file");

        let result = call(vec![
            Value::String(path.to_string_lossy().to_string()),
            Value::Num(1.0),
            Value::Num(4.0),
            Value::String("select".to_string()),
        ])
        .expect("opentoline");

        assert_eq!(result, empty_array());
    }

    #[test]
    fn opentoline_command_form_text_line_and_column_are_supported() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("target.m");
        std::fs::write(&path, "abcdef\n").expect("write file");

        let result = call(vec![
            Value::String(path.to_string_lossy().to_string()),
            Value::String("1".to_string()),
            Value::String("3".to_string()),
        ])
        .expect("opentoline");

        assert_eq!(result, empty_array());
    }

    #[test]
    fn opentoline_rejects_missing_file() {
        let err = call(vec![
            Value::String("definitely_missing_opentoline_target.m".to_string()),
            Value::Num(1.0),
        ])
        .expect_err("missing file");

        assert_eq!(err.identifier(), Some("RunMat:opentoline:FileNotFound"));
    }

    #[test]
    fn opentoline_rejects_invalid_positions_and_options() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("target.m");
        std::fs::write(&path, "abcdef\n").expect("write file");
        let filename = Value::String(path.to_string_lossy().to_string());

        let err = call(vec![filename.clone(), Value::Num(0.0)]).expect_err("invalid line");
        assert_eq!(err.identifier(), Some("RunMat:opentoline:InvalidPosition"));

        let err = call(vec![filename.clone(), Value::Num(1.5)]).expect_err("fractional line");
        assert_eq!(err.identifier(), Some("RunMat:opentoline:InvalidPosition"));

        let err = call(vec![
            filename,
            Value::Num(1.0),
            Value::Num(1.0),
            Value::String("reuse".to_string()),
        ])
        .expect_err("invalid option");
        assert_eq!(err.identifier(), Some("RunMat:opentoline:InvalidOption"));
    }

    #[test]
    fn opentoline_rejects_too_many_outputs() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("target.m");
        std::fs::write(&path, "abcdef\n").expect("write file");
        let _outputs = crate::output_count::push_output_count(Some(2));

        let err = call(vec![
            Value::String(path.to_string_lossy().to_string()),
            Value::Num(1.0),
        ])
        .expect_err("too many outputs");

        assert_eq!(err.identifier(), Some("RunMat:opentoline:TooManyOutputs"));
    }

    #[test]
    fn opentoline_descriptor_covers_core_forms() {
        let labels: Vec<&str> = OPENTOLINE_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"opentoline(filename, line)"));
        assert!(labels.contains(&"opentoline(filename, line, column)"));
        assert!(labels.contains(&"opentoline(filename, line, column, option)"));
    }
}
