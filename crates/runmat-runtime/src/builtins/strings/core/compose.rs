//! MATLAB-compatible `compose` builtin that formats data into text arrays.
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    CellArray, CharArray, StringArray, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::map_control_flow_with_builtin;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::strings::core::string::{
    extract_format_spec, format_from_spec, FormatSpecData,
};
use crate::builtins::strings::type_resolvers::string_array_type;
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::strings::core::compose")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "compose",
    op_kind: GpuOpKind::Custom("format"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Formatting always executes on the CPU; GPU tensors are gathered before substitution.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::strings::core::compose")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "compose",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Formatting builtin; not eligible for fusion and materialises host text arrays.",
};

const BUILTIN_NAME: &str = "compose";

pub const COMPOSE_RESIDENT_INPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "compose-resident-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "compose with resident array arguments is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:ComposeResidentInputExtension"),
    };

pub const COMPOSE_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [COMPOSE_RESIDENT_INPUT_EXTENSION];

const COMPOSE_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "A...",
    classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "All eight integer classes are formatted directly from native storage, including exact int64 and uint64 values above flintmax.",
}];

pub const COMPOSE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] = [BuiltinIntegerCapabilityDescriptor {
    form: "S = compose(formatSpec, integer_A...)",
    inputs: &COMPOSE_INTEGER_INPUTS,
    computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
    output_class: BuiltinIntegerOutputClassRule::NotApplicable,
    overflow: BuiltinIntegerOverflowRule::NotApplicable,
    backend: BuiltinIntegerBackendRule::HostOnly,
    overload: BuiltinIntegerOverloadKind::Multiple,
    notes: "Integer substitution is textual and exact; the format-spec family selects a host string array or cell array of character vectors. Resident arguments are a separately gated RunMat extension and gather before formatting.",
}];

const COMPOSE_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "S",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description:
        "Formatted string array or cell array of character vectors, selected by formatSpec.",
}];

const COMPOSE_INPUT_NO_ARGS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "formatSpec",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Format text or array returned in the corresponding string or cellstr family when no data arguments are supplied.",
}];

const COMPOSE_INPUT_WITH_ARGS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "formatSpec",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Format template text.",
    },
    BuiltinParamDescriptor {
        name: "A...",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Values substituted into formatSpec placeholders.",
    },
];

const COMPOSE_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "S = compose(formatSpec)",
        inputs: &COMPOSE_INPUT_NO_ARGS,
        outputs: &COMPOSE_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "S = compose(formatSpec, A...)",
        inputs: &COMPOSE_INPUT_WITH_ARGS,
        outputs: &COMPOSE_OUTPUT,
    },
];

const COMPOSE_ERROR_INVALID_FORMAT_SPEC: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.COMPOSE.INVALID_FORMAT_SPEC",
    identifier: Some("RunMat:compose:InvalidFormatSpec"),
    when: "formatSpec is not valid text input for compose formatting.",
    message: "compose: invalid formatSpec",
};

const COMPOSE_ERROR_ARGUMENT_MISMATCH: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.COMPOSE.ARGUMENT_MISMATCH",
    identifier: Some("RunMat:compose:ArgumentMismatch"),
    when: "Data arguments are not scalar or broadcast-compatible with formatSpec.",
    message: "compose: format data arguments must be scalars or match formatSpec size",
};

const COMPOSE_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.COMPOSE.INTERNAL",
    identifier: Some("RunMat:compose:InternalError"),
    when: "Internal string-array construction failed.",
    message: "compose: internal error",
};

const COMPOSE_ERROR_INVALID_DATA: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.COMPOSE.INVALID_DATA",
    identifier: Some("RunMat:compose:InvalidData"),
    when: "A data argument is outside compose's documented numeric, logical, character, or string domain.",
    message: "compose: unsupported data argument",
};

const COMPOSE_ERRORS: [BuiltinErrorDescriptor; 4] = [
    COMPOSE_ERROR_INVALID_FORMAT_SPEC,
    COMPOSE_ERROR_ARGUMENT_MISMATCH,
    COMPOSE_ERROR_INTERNAL,
    COMPOSE_ERROR_INVALID_DATA,
];

pub const COMPOSE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &COMPOSE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &COMPOSE_ERRORS,
};

fn compose_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    compose_error_with_message(error.message, error)
}

fn compose_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn remap_compose_flow(mut err: RuntimeError) -> RuntimeError {
    err = map_control_flow_with_builtin(err, BUILTIN_NAME);
    if let Some(message) = err.message.strip_prefix("string: ") {
        err.message = format!("compose: {message}");
        return err;
    }
    if !err.message.starts_with("compose: ") {
        err.message = format!("compose: {}", err.message);
    }
    err
}

#[runtime_builtin(
    name = "compose",
    category = "strings/core",
    summary = "Format values into text arrays using printf-style placeholders.",
    keywords = "compose,format,string array,gpu",
    accel = "sink",
    type_resolver(string_array_type),
    extensions(COMPOSE_EXTENSIONS),
    integer_capabilities(COMPOSE_INTEGER_CAPABILITIES),
    descriptor(crate::builtins::strings::core::compose::COMPOSE_DESCRIPTOR),
    builtin_path = "crate::builtins::strings::core::compose"
)]
async fn compose_builtin(format_spec: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    if !rest.is_empty() && matches!(format_spec, Value::Cell(_)) {
        return Err(compose_error_with_message(
            "compose: cellstr formatSpec is supported only by compose(formatSpec)",
            &COMPOSE_ERROR_INVALID_FORMAT_SPEC,
        ));
    }
    if rest.iter().any(|value| matches!(value, Value::Cell(_))) {
        return Err(compose_error_with_message(
            "compose: cell-array data arguments are outside the documented input domain",
            &COMPOSE_ERROR_INVALID_DATA,
        ));
    }
    if matches!(format_spec, Value::GpuTensor(_))
        || rest
            .iter()
            .any(|value| matches!(value, Value::GpuTensor(_)))
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &COMPOSE_RESIDENT_INPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    let char_format = matches!(format_spec, Value::CharArray(_) | Value::Cell(_));
    let format_value = gather_if_needed_async(&format_spec)
        .await
        .map_err(remap_compose_flow)?;
    let mut gathered_args = Vec::with_capacity(rest.len());
    for arg in rest {
        let gathered = gather_if_needed_async(&arg)
            .await
            .map_err(remap_compose_flow)?;
        gathered_args.push(gathered);
    }

    if gathered_args.is_empty() {
        let spec = extract_format_spec(format_value)
            .await
            .map_err(remap_compose_flow)?;
        let array = format_spec_data_to_string_array(spec)?;
        if char_format {
            return string_array_to_cellstr(array);
        }
        return Ok(Value::StringArray(array));
    }

    let formatted = format_from_spec(format_value, gathered_args)
        .await
        .map_err(remap_compose_flow)?;
    if char_format {
        return string_array_to_cellstr(formatted);
    }
    Ok(Value::StringArray(formatted))
}

fn string_array_to_cellstr(formatted: StringArray) -> BuiltinResult<Value> {
    let rows = formatted.shape.first().copied().unwrap_or(1);
    let cols = formatted.shape.get(1).copied().unwrap_or(1);
    let values = formatted
        .data
        .into_iter()
        .map(|text| Value::CharArray(CharArray::new_row(&text)))
        .collect();
    CellArray::new(values, rows, cols)
        .map(Value::Cell)
        .map_err(|_| compose_error(&COMPOSE_ERROR_INTERNAL))
}

fn format_spec_data_to_string_array(spec: FormatSpecData) -> BuiltinResult<StringArray> {
    let shape = if spec.shape.is_empty() {
        match spec.specs.len() {
            0 => vec![0, 0],
            1 => vec![1, 1],
            len => vec![len, 1],
        }
    } else {
        spec.shape
    };
    StringArray::new(spec.specs, shape).map_err(|_| compose_error(&COMPOSE_ERROR_INTERNAL))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use runmat_builtins::{IntValue, IntegerStorage, ResolveContext, Tensor, Type};

    fn compose_builtin(format_spec: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        futures::executor::block_on(super::compose_builtin(format_spec, rest))
    }

    fn error_message(err: crate::RuntimeError) -> String {
        err.message().to_string()
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn compose_scalar_numeric() {
        let result = compose_builtin(Value::from("Count %d"), vec![Value::Int(IntValue::I32(7))])
            .expect("compose");
        match result {
            Value::StringArray(sa) => {
                assert_eq!(sa.shape, vec![1, 1]);
                assert_eq!(sa.data, vec!["Count 7".to_string()]);
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn compose_broadcasts_scalar_spec() {
        let tensor = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
        let result = compose_builtin(Value::from("Item %0.0f"), vec![Value::Tensor(tensor)])
            .expect("compose");
        match result {
            Value::StringArray(sa) => {
                assert_eq!(sa.shape, vec![1, 2]);
                assert_eq!(sa.data, vec!["Item 1".to_string(), "Item 2".to_string()]);
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn compose_zero_arguments_returns_spec() {
        let spec = Value::StringArray(
            StringArray::new(vec!["alpha".into(), "beta".into()], vec![1, 2]).unwrap(),
        );
        let result = compose_builtin(spec, Vec::new()).expect("compose");
        match result {
            Value::StringArray(sa) => {
                assert_eq!(sa.shape, vec![1, 2]);
                assert_eq!(sa.data, vec!["alpha".to_string(), "beta".to_string()]);
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn compose_mismatched_lengths_errors() {
        let spec = Value::StringArray(
            StringArray::new(vec!["%d".into(), "%d".into()], vec![1, 2]).unwrap(),
        );
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
        let err = error_message(compose_builtin(spec, vec![Value::Tensor(tensor)]).unwrap_err());
        assert!(
            err.starts_with("compose: "),
            "expected compose prefix, got {err}"
        );
        assert!(
            err.contains("format data arguments must be scalars or match formatSpec size"),
            "unexpected error text: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn compose_gpu_argument() {
        test_support::with_test_provider(|provider| {
            let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
            let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.materialize_f64(),
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result =
                compose_builtin(Value::from("Value %0.0f"), vec![Value::GpuTensor(handle)])
                    .expect("compose");
            match result {
                Value::StringArray(sa) => {
                    assert_eq!(sa.shape, vec![1, 3]);
                    assert_eq!(
                        sa.data,
                        vec![
                            "Value 1".to_string(),
                            "Value 2".to_string(),
                            "Value 3".to_string()
                        ]
                    );
                }
                other => panic!("expected string array, got {other:?}"),
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn compose_wgpu_numeric_tensor_matches_cpu() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let tensor = Tensor::new(vec![1.25, 2.5, 3.75], vec![1, 3]).unwrap();
        let cpu = compose_builtin(
            Value::from("Value %0.2f"),
            vec![Value::Tensor(tensor.clone())],
        )
        .expect("cpu compose");
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.materialize_f64(),
            shape: &tensor.shape,
        };
        let provider = runmat_accelerate_api::provider().expect("wgpu provider");
        let handle = provider.upload(&view).expect("gpu upload");
        let gpu = compose_builtin(Value::from("Value %0.2f"), vec![Value::GpuTensor(handle)])
            .expect("gpu compose");
        match (cpu, gpu) {
            (Value::StringArray(expect), Value::StringArray(actual)) => {
                assert_eq!(actual.shape, expect.shape);
                assert_eq!(actual.data, expect.data);
            }
            other => panic!("unexpected results {other:?}"),
        }
    }

    #[test]
    fn compose_formats_all_integer_classes_and_wide_values_without_f64_materialization() {
        let cases = vec![
            (IntegerStorage::I8(vec![i8::MIN]), i8::MIN.to_string()),
            (IntegerStorage::I16(vec![i16::MIN]), i16::MIN.to_string()),
            (IntegerStorage::I32(vec![i32::MIN]), i32::MIN.to_string()),
            (IntegerStorage::I64(vec![i64::MIN]), i64::MIN.to_string()),
            (IntegerStorage::U8(vec![u8::MAX]), u8::MAX.to_string()),
            (IntegerStorage::U16(vec![u16::MAX]), u16::MAX.to_string()),
            (IntegerStorage::U32(vec![u32::MAX]), u32::MAX.to_string()),
            (IntegerStorage::U64(vec![u64::MAX]), u64::MAX.to_string()),
        ];
        for (storage, expected) in cases {
            let tensor = Tensor::new_integer(storage, vec![1, 1]).expect("integer tensor");
            let Value::StringArray(result) =
                compose_builtin(Value::from("%d"), vec![Value::Tensor(tensor)]).expect("compose")
            else {
                panic!("expected strings");
            };
            assert_eq!(result.data, vec![expected]);
        }
    }

    #[test]
    fn compose_resident_input_is_mode_gated_before_gather() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0], vec![1, 1]).expect("tensor");
            let handle = provider
                .upload(&runmat_accelerate_api::HostTensorView {
                    data: &tensor.materialize_f64(),
                    shape: &tensor.shape,
                })
                .expect("upload");
            let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
            let error = compose_builtin(Value::from("%d"), vec![Value::GpuTensor(handle)])
                .expect_err("resident compose is extension");
            assert_eq!(
                error.identifier(),
                COMPOSE_RESIDENT_INPUT_EXTENSION.error_identifier
            );
        });
    }

    #[test]
    fn compose_rejects_cell_data_before_nested_integer_or_resident_conversion() {
        let wide = Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX]), vec![1, 1])
            .expect("wide integer");
        let cell = runmat_builtins::CellArray::new(vec![Value::Tensor(wide)], 1, 1).expect("cell");
        let error = compose_builtin(Value::from("%d"), vec![Value::Cell(cell)])
            .expect_err("cell data is outside compose's public domain");
        assert_eq!(error.identifier(), COMPOSE_ERROR_INVALID_DATA.identifier);
    }

    #[test]
    fn compose_char_format_returns_cellstr_while_string_format_returns_string_array() {
        let char_spec = Value::CharArray(CharArray::new_row("%d"));
        let Value::Cell(cell) =
            compose_builtin(char_spec, vec![Value::Int(IntValue::I32(7))]).expect("cellstr")
        else {
            panic!("expected cellstr");
        };
        assert!(
            matches!(&cell.data[0], Value::CharArray(value) if value.data.iter().collect::<String>() == "7")
        );
        assert!(matches!(
            compose_builtin(Value::from("%d"), vec![Value::Int(IntValue::I32(7))])
                .expect("strings"),
            Value::StringArray(_)
        ));
        assert!(matches!(
            compose_builtin(Value::CharArray(CharArray::new_row("plain")), Vec::new())
                .expect("zero-data cellstr"),
            Value::Cell(_)
        ));
    }

    #[test]
    fn compose_cellstr_format_spec_is_unary_only() {
        let cell = CellArray::new(vec![Value::CharArray(CharArray::new_row("plain"))], 1, 1)
            .expect("cellstr");
        assert!(matches!(
            compose_builtin(Value::Cell(cell.clone()), Vec::new()).expect("unary cellstr"),
            Value::Cell(_)
        ));
        let error = compose_builtin(Value::Cell(cell), vec![Value::Num(1.0)])
            .expect_err("cellstr formatting is not public");
        assert_eq!(
            error.identifier(),
            COMPOSE_ERROR_INVALID_FORMAT_SPEC.identifier
        );
    }

    #[test]
    fn compose_type_is_string_array() {
        assert_eq!(
            string_array_type(&[Type::String], &ResolveContext::new(Vec::new())),
            Type::cell_of(Type::String)
        );
    }
}
