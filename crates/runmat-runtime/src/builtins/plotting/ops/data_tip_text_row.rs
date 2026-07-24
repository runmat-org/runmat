//! MATLAB-compatible `dataTipTextRow` data-tip row constructor.

use std::cell::Cell;
use std::collections::HashMap;

use runmat_builtins::{
    Access, BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    CharArray, ClassDef, HandleRef, MethodDef, ObjectInstance, PropertyDef, ResolveContext,
    StringArray, Type, Value,
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
        default: Some("\"\""),
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
    let value = args[1].clone();
    let format = match args.get(2) {
        Some(value) => scalar_text(value, "format")?,
        None => String::new(),
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

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_accelerate_api::GpuTensorHandle;
    use runmat_builtins::{StringArray, Tensor};

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
            Some(&Value::String(String::new()))
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
        let handle = GpuTensorHandle {
            shape: vec![1, 3],
            device_id: 7,
            buffer_id: 42,
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
}
