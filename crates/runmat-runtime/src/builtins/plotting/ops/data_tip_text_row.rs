//! MATLAB-compatible `dataTipTextRow` data-tip row constructor.

use std::cell::Cell;
use std::collections::HashMap;

use runmat_builtins::{
    Access, BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor,
    BuiltinExtensionDescriptor, BuiltinExtensionMode, BuiltinIntegerBackendRule,
    BuiltinIntegerCapabilityDescriptor, BuiltinIntegerComputationDomain,
    BuiltinIntegerInputAvailability, BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule,
    BuiltinIntegerOverflowRule, BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule,
    BuiltinOutputMode, BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType,
    BuiltinSignatureDescriptor, CharArray, ClassDef, HandleRef, MethodDef, ObjectInstance,
    PropertyDef, ResolveContext, StringArray, Type, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "dataTipTextRow";
const CLASS_NAME: &str = "matlab.graphics.datatip.DataTipTextRow";

thread_local! {
    static DATA_TIP_TEXT_ROW_CLASS_REGISTERED: Cell<bool> = const { Cell::new(false) };
}

const DATA_TIP_OUTPUT_ROW: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "row",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Data-tip text row object with Label, Value, and Format properties.",
}];

const DATA_TIP_INPUTS_LABEL_VALUE: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "label",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Display label for the data-tip row.",
    },
    BuiltinParamDescriptor {
        name: "value",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Value source, expression, or data vector shown by the row.",
    },
];

const DATA_TIP_INPUTS_LABEL_VALUE_FORMAT: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "label",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Display label for the data-tip row.",
    },
    BuiltinParamDescriptor {
        name: "value",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Value source, expression, or data vector shown by the row.",
    },
    BuiltinParamDescriptor {
        name: "format",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"auto\""),
        description: "Optional numeric, datetime, duration, or text display format.",
    },
];

const DATA_TIP_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "row = dataTipTextRow(label, value)",
        inputs: &DATA_TIP_INPUTS_LABEL_VALUE,
        outputs: &DATA_TIP_OUTPUT_ROW,
    },
    BuiltinSignatureDescriptor {
        label: "row = dataTipTextRow(label, value, format)",
        inputs: &DATA_TIP_INPUTS_LABEL_VALUE_FORMAT,
        outputs: &DATA_TIP_OUTPUT_ROW,
    },
];

const DATA_TIP_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DATATIPTEXTROW.INVALID_ARGUMENT",
    identifier: Some("RunMat:dataTipTextRow:InvalidArgument"),
    when: "The label/format text is not scalar, or the argument count is not supported.",
    message: "dataTipTextRow: invalid argument",
};

const DATA_TIP_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DATATIPTEXTROW.INTERNAL",
    identifier: Some("RunMat:dataTipTextRow:Internal"),
    when: "Internal data-tip row object construction fails.",
    message: "dataTipTextRow: internal error",
};

const DATA_TIP_ERRORS: [BuiltinErrorDescriptor; 2] =
    [DATA_TIP_ERROR_INVALID_ARGUMENT, DATA_TIP_ERROR_INTERNAL];

pub const DATA_TIP_TEXT_ROW_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &DATA_TIP_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &DATA_TIP_ERRORS,
};

pub const DATA_TIP_RESIDENT_VALUE_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "data-tip-text-row-resident-value",
        mode: BuiltinExtensionMode::RunMatOnly,
        description:
            "dataTipTextRow preserving a resident numeric Value source is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:DataTipTextRowResidentValueExtension"),
    };
pub const DATA_TIP_NONVECTOR_VALUE_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "data-tip-text-row-nonvector-value",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "dataTipTextRow with a nonvector numeric Value source is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:DataTipTextRowNonvectorValueExtension"),
    };
pub const DATA_TIP_TEXT_ROW_EXTENSIONS: [BuiltinExtensionDescriptor; 2] = [
    DATA_TIP_RESIDENT_VALUE_EXTENSION,
    DATA_TIP_NONVECTOR_VALUE_EXTENSION,
];

const DATA_TIP_INTEGER_VECTOR: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "value",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The public Value contract says vector without a per-class table. RunMat classifies integer vectors as an evidence-based inference and preserves authoritative storage structurally without conversion.",
    }];
const DATA_TIP_INTEGER_NONVECTOR: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "value",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Nonvector integer Value sources are preserved only behind the separate RunMat extension gate.",
    }];
const DATA_TIP_INTEGER_RESIDENT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "resident value",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Resident integer Value metadata is preserved without provider access only behind the separate resident-value extension gate.",
    }];

pub const DATA_TIP_TEXT_ROW_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    BuiltinIntegerCapabilityDescriptor { form: "row = dataTipTextRow(label, integer_vector, format?)", inputs: &DATA_TIP_INTEGER_VECTOR, computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific, output_class: BuiltinIntegerOutputClassRule::NotApplicable, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::HostOnly, overload: BuiltinIntegerOverloadKind::Multiple, notes: "Inferred vector support stores the exact integer Value object as metadata; formatting occurs later in the data-tip renderer." },
    BuiltinIntegerCapabilityDescriptor { form: "row = dataTipTextRow(label, integer_nonvector, format?)", inputs: &DATA_TIP_INTEGER_NONVECTOR, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::NotApplicable, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::HostOnly, overload: BuiltinIntegerOverloadKind::Multiple, notes: "RunMat-only nonvector metadata preservation is independently gated." },
    BuiltinIntegerCapabilityDescriptor { form: "row = dataTipTextRow(label, resident_integer, format?)", inputs: &DATA_TIP_INTEGER_RESIDENT, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::NotApplicable, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::GpuRestricted, overload: BuiltinIntegerOverloadKind::Multiple, notes: "RunMat-only resident metadata preservation performs no gather, upload, or provider dispatch." },
];

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::plotting::data_tip_text_row")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: BUILTIN_NAME,
    op_kind: GpuOpKind::Custom("data-tip-text-row"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::InheritInputs,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Constructs metadata only. Value inputs, including gpuArray handles, are stored without materializing or gathering their payloads.",
};

#[runmat_macros::register_fusion_spec(
    builtin_path = "crate::builtins::plotting::data_tip_text_row"
)]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: BUILTIN_NAME,
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Data-tip text row construction is a metadata sink and does not generate numeric fusion kernels.",
};

fn data_tip_text_row_type(_args: &[Type], _ctx: &ResolveContext) -> Type {
    Type::Unknown
}

fn error(descriptor: &'static BuiltinErrorDescriptor, detail: impl AsRef<str>) -> RuntimeError {
    let mut builder = build_runtime_error(format!("{}: {}", descriptor.message, detail.as_ref()))
        .with_builtin(BUILTIN_NAME);
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn ensure_class_registered() {
    DATA_TIP_TEXT_ROW_CLASS_REGISTERED.with(|registered| {
        if registered.get() {
            return;
        }

        let mut properties = HashMap::new();
        for name in ["Label", "Value", "Format"] {
            properties.insert(
                name.to_string(),
                PropertyDef {
                    name: name.to_string(),
                    is_static: false,
                    is_constant: false,
                    is_dependent: false,
                    get_access: Access::Public,
                    set_access: Access::Public,
                    default_value: None,
                },
            );
        }

        let methods: HashMap<String, MethodDef> = HashMap::new();
        runmat_builtins::register_class(ClassDef {
            name: CLASS_NAME.to_string(),
            parent: Some("handle".to_string()),
            properties,
            methods,
        });
        registered.set(true);
    });
}

fn scalar_text(value: &Value, name: &str) -> BuiltinResult<String> {
    match value {
        Value::String(text) => Ok(text.clone()),
        Value::CharArray(chars) => char_array_text(chars, name),
        Value::StringArray(array) => string_array_text(array, name),
        _ => Err(error(
            &DATA_TIP_ERROR_INVALID_ARGUMENT,
            format!("{name} must be a string scalar or character vector"),
        )),
    }
}

fn char_array_text(chars: &CharArray, name: &str) -> BuiltinResult<String> {
    if chars.rows > 1 {
        return Err(error(
            &DATA_TIP_ERROR_INVALID_ARGUMENT,
            format!("{name} must be a character vector"),
        ));
    }
    Ok(chars.data.iter().collect())
}

fn string_array_text(array: &StringArray, name: &str) -> BuiltinResult<String> {
    if array.data.len() != 1 {
        return Err(error(
            &DATA_TIP_ERROR_INVALID_ARGUMENT,
            format!("{name} must be a scalar string"),
        ));
    }
    Ok(array.data[0].clone())
}

#[runtime_builtin(
    name = "dataTipTextRow",
    category = "plotting",
    summary = "Create a data-tip text row object for plot data tips.",
    keywords = "dataTipTextRow,datatip,data cursor,plotting,graphics",
    type_resolver(data_tip_text_row_type),
    descriptor(crate::builtins::plotting::data_tip_text_row::DATA_TIP_TEXT_ROW_DESCRIPTOR),
    extensions(crate::builtins::plotting::data_tip_text_row::DATA_TIP_TEXT_ROW_EXTENSIONS),
    integer_capabilities(
        crate::builtins::plotting::data_tip_text_row::DATA_TIP_TEXT_ROW_INTEGER_CAPABILITIES
    ),
    builtin_path = "crate::builtins::plotting::data_tip_text_row"
)]
pub fn data_tip_text_row_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    if !(2..=3).contains(&args.len()) {
        return Err(error(
            &DATA_TIP_ERROR_INVALID_ARGUMENT,
            "expected label, value, and optional format",
        ));
    }

    ensure_class_registered();

    let label = scalar_text(&args[0], "label")?;
    validate_value_source(&args[1])?;
    let value = args[1].clone();
    let format = match args.get(2) {
        Some(value) => scalar_text(value, "format")?,
        None => "auto".to_string(),
    };

    let mut object = ObjectInstance::new(CLASS_NAME.to_string());
    object
        .properties
        .insert("Label".to_string(), Value::String(label));
    object.properties.insert("Value".to_string(), value);
    object
        .properties
        .insert("Format".to_string(), Value::String(format));
    let target = runmat_gc::gc_allocate(Value::Object(object)).map_err(|err| {
        error(
            &DATA_TIP_ERROR_INTERNAL,
            format!("failed to allocate data-tip row handle: {err}"),
        )
    })?;
    Ok(Value::HandleObject(HandleRef {
        class_name: CLASS_NAME.to_string(),
        target,
        valid: true,
    }))
}

fn vector_shape(shape: &[usize]) -> bool {
    matches!(shape, [] | [_] | [1, _] | [_, 1])
}

fn validate_value_source(value: &Value) -> BuiltinResult<()> {
    match value {
        Value::String(_) | Value::CharArray(_) => Ok(()),
        Value::StringArray(array) if array.data.len() == 1 => Ok(()),
        Value::FunctionHandle(_)
        | Value::ExternalFunctionHandle(_)
        | Value::MethodFunctionHandle(_)
        | Value::BoundFunctionHandle { .. } => Ok(()),
        Value::Num(_) | Value::Int(_) | Value::Bool(_) => Ok(()),
        Value::Tensor(tensor) => validate_host_vector_shape(&tensor.shape),
        Value::LogicalArray(array) => validate_host_vector_shape(&array.shape),
        Value::ComplexTensor(tensor) => validate_host_vector_shape(&tensor.shape),
        Value::Complex(_, _) => Ok(()),
        Value::GpuTensor(handle) => {
            crate::compatibility::ensure_builtin_extension_enabled(
                &DATA_TIP_RESIDENT_VALUE_EXTENSION,
                BUILTIN_NAME,
            )?;
            if !vector_shape(&handle.shape) {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &DATA_TIP_NONVECTOR_VALUE_EXTENSION,
                    BUILTIN_NAME,
                )?;
            }
            Ok(())
        }
        _ => Err(error(
            &DATA_TIP_ERROR_INVALID_ARGUMENT,
            "value must be text, a scalar/vector, or a function handle",
        )),
    }
}

fn validate_host_vector_shape(shape: &[usize]) -> BuiltinResult<()> {
    if vector_shape(shape) {
        return Ok(());
    }
    crate::compatibility::ensure_builtin_extension_enabled(
        &DATA_TIP_NONVECTOR_VALUE_EXTENSION,
        BUILTIN_NAME,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_accelerate_api::GpuTensorHandle;
    use runmat_builtins::{IntegerStorage, StringArray, Tensor};

    fn handle_payload(value: Value) -> ObjectInstance {
        let Value::HandleObject(handle) = value else {
            panic!("expected handle object");
        };
        assert_eq!(handle.class_name, CLASS_NAME);
        let target = runmat_gc::gc_clone_value(&handle.target).expect("clone handle target");
        let Value::Object(object) = target else {
            panic!("expected object handle target");
        };
        object
    }

    #[test]
    fn descriptor_signatures_cover_documented_forms() {
        let labels = DATA_TIP_TEXT_ROW_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect::<Vec<_>>();
        assert!(labels.contains(&"row = dataTipTextRow(label, value)"));
        assert!(labels.contains(&"row = dataTipTextRow(label, value, format)"));
    }

    #[test]
    fn creates_data_tip_row_object_with_label_value_and_default_format() {
        let value = Value::Tensor(Tensor::new_2d(vec![10.0, 20.0], 1, 2).unwrap());
        let row = data_tip_text_row_builtin(vec![Value::String("Amplitude".into()), value.clone()])
            .expect("dataTipTextRow");
        let object = handle_payload(row);
        assert_eq!(
            object.properties.get("Label"),
            Some(&Value::String("Amplitude".into()))
        );
        assert_eq!(object.properties.get("Value"), Some(&value));
        assert_eq!(
            object.properties.get("Format"),
            Some(&Value::String("auto".into()))
        );
    }

    #[test]
    fn accepts_char_and_scalar_string_array_text() {
        let label = Value::CharArray(CharArray::new_row("Speed"));
        let format = Value::StringArray(StringArray::new(vec!["%.2f".into()], vec![1, 1]).unwrap());
        let row = data_tip_text_row_builtin(vec![label, Value::String("YData".into()), format])
            .expect("dataTipTextRow");
        let object = handle_payload(row);
        assert_eq!(
            object.properties.get("Label"),
            Some(&Value::String("Speed".into()))
        );
        assert_eq!(
            object.properties.get("Value"),
            Some(&Value::String("YData".into()))
        );
        assert_eq!(
            object.properties.get("Format"),
            Some(&Value::String("%.2f".into()))
        );
    }

    #[test]
    fn rejects_non_scalar_text_inputs() {
        let label =
            Value::StringArray(StringArray::new(vec!["a".into(), "b".into()], vec![1, 2]).unwrap());
        let err = data_tip_text_row_builtin(vec![label, Value::Num(1.0)])
            .expect_err("expected scalar string error");
        assert_eq!(err.identifier(), DATA_TIP_ERROR_INVALID_ARGUMENT.identifier);
    }

    #[test]
    fn stores_gpu_value_without_gathering_or_replacing_handle() {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let handle = GpuTensorHandle {
            shape: vec![1, 3],
            device_id: 7,
            buffer_id: 42,
            descriptor: Default::default(),
        };
        let row = data_tip_text_row_builtin(vec![
            Value::String("GPU".into()),
            Value::GpuTensor(handle.clone()),
        ])
        .expect("dataTipTextRow");
        let object = handle_payload(row);
        assert!(matches!(
            object.properties.get("Value"),
            Some(Value::GpuTensor(stored))
                if stored.device_id == handle.device_id && stored.buffer_id == handle.buffer_id
        ));
    }

    #[test]
    fn preserves_registered_resident_value_without_provider_access() {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        crate::builtins::common::test_support::with_test_provider(|provider| {
            let handle = provider
                .upload(&runmat_accelerate_api::HostTensorView {
                    data: &[1.0, 2.0],
                    shape: &[1, 2],
                })
                .unwrap();
            provider.reset_telemetry();
            let object = handle_payload(
                data_tip_text_row_builtin(vec![
                    Value::String("GPU".into()),
                    Value::GpuTensor(handle.clone()),
                ])
                .unwrap(),
            );
            assert!(matches!(
                object.properties.get("Value"),
                Some(Value::GpuTensor(stored)) if stored == &handle
            ));
            assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
            let _ = provider.free(&handle);
        });
    }

    #[test]
    fn preserves_all_inferred_integer_vector_classes_structurally() {
        for storage in [
            IntegerStorage::I8(vec![1, 2]),
            IntegerStorage::I16(vec![1, 2]),
            IntegerStorage::I32(vec![1, 2]),
            IntegerStorage::I64(vec![1, 2]),
            IntegerStorage::U8(vec![1, 2]),
            IntegerStorage::U16(vec![1, 2]),
            IntegerStorage::U32(vec![1, 2]),
            IntegerStorage::U64(vec![1, 2]),
        ] {
            let value = Value::Tensor(Tensor::new_integer(storage.clone(), vec![1, 2]).unwrap());
            let object = handle_payload(
                data_tip_text_row_builtin(vec![Value::String("Integer".into()), value]).unwrap(),
            );
            let Some(Value::Tensor(stored)) = object.properties.get("Value") else {
                panic!("integer vector storage");
            };
            assert_eq!(stored.integer_storage(), Some(&storage));
        }
        assert_eq!(
            DATA_TIP_TEXT_ROW_INTEGER_CAPABILITIES[0].inputs[0]
                .classes
                .len(),
            8
        );
        assert_eq!(
            DATA_TIP_TEXT_ROW_INTEGER_CAPABILITIES[0].computation_domain,
            BuiltinIntegerComputationDomain::FunctionSpecific
        );
        assert_eq!(
            DATA_TIP_TEXT_ROW_INTEGER_CAPABILITIES[0].overload,
            BuiltinIntegerOverloadKind::Multiple
        );
    }

    #[test]
    fn resident_and_nonvector_value_extensions_are_gated_independently() {
        let strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let matrix = Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap());
        let error = data_tip_text_row_builtin(vec![Value::String("M".into()), matrix.clone()])
            .expect_err("nonvector gate");
        assert_eq!(
            error.identifier(),
            DATA_TIP_NONVECTOR_VALUE_EXTENSION.error_identifier
        );
        let handle = GpuTensorHandle {
            shape: vec![1, 2],
            device_id: u32::MAX,
            buffer_id: u64::MAX - 397,
            descriptor: Default::default(),
        };
        let error = data_tip_text_row_builtin(vec![
            Value::String("GPU".into()),
            Value::GpuTensor(handle.clone()),
        ])
        .expect_err("resident gate");
        assert_eq!(
            error.identifier(),
            DATA_TIP_RESIDENT_VALUE_EXTENSION.error_identifier
        );
        drop(strict);

        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        assert!(data_tip_text_row_builtin(vec![Value::String("M".into()), matrix]).is_ok());
        assert!(data_tip_text_row_builtin(vec![
            Value::String("GPU".into()),
            Value::GpuTensor(handle)
        ])
        .is_ok());
    }

    #[test]
    fn rejects_values_outside_documented_source_union() {
        let error = data_tip_text_row_builtin(vec![
            Value::String("Bad".into()),
            Value::Struct(runmat_builtins::StructValue::new()),
        ])
        .expect_err("unsupported source");
        assert_eq!(
            error.identifier(),
            DATA_TIP_ERROR_INVALID_ARGUMENT.identifier
        );
    }
}
