//! MATLAB-compatible `dataTipTextRow` data-tip row constructor.
use runmat_types::MemberAccess;

use std::collections::HashMap;

use runmat_builtins::BuiltinErrorDescriptor;
#[cfg(test)]
use runmat_builtins::{BuiltinIntegerComputationDomain, BuiltinIntegerOverloadKind};

use runmat_macros::runtime_builtin;
#[cfg(test)]
use runmat_value::ObjectInstance;
use runmat_value::{CharArray, HandleRef, StringArray, Value};

use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "dataTipTextRow";
const CLASS_IDENTITY: runmat_types::StaticClassIdentity =
    runmat_types::StaticClassIdentity::new("matlab.graphics.datatip.DataTipTextRow");

static DATA_TIP_TEXT_ROW_CLASS_REGISTERED: crate::class_registry::ClassRegistration =
    crate::class_registry::ClassRegistration::new(CLASS_IDENTITY);

pub use runmat_builtins::DATA_TIP_ERROR_INVALID_ARGUMENT;

pub use runmat_builtins::DATA_TIP_ERROR_INTERNAL;

#[cfg(test)]
use runmat_builtins::DATA_TIP_TEXT_ROW_DESCRIPTOR;

pub use runmat_builtins::DATA_TIP_NONVECTOR_VALUE_EXTENSION;
pub use runmat_builtins::DATA_TIP_RESIDENT_VALUE_EXTENSION;

#[cfg(test)]
use runmat_builtins::DATA_TIP_TEXT_ROW_INTEGER_CAPABILITIES;

fn error(descriptor: &'static BuiltinErrorDescriptor, detail: impl AsRef<str>) -> RuntimeError {
    let mut builder = build_runtime_error(format!("{}: {}", descriptor.message, detail.as_ref()))
        .with_builtin(BUILTIN_NAME);
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn ensure_class_registered() {
    DATA_TIP_TEXT_ROW_CLASS_REGISTERED.ensure(|| {
        let mut properties = HashMap::new();
        for name in ["Label", "Value", "Format"] {
            properties.insert(
                name.into(),
                crate::class_registry::RuntimeProperty {
                    name: name.into(),
                    is_static: false,
                    is_constant: false,
                    is_dependent: false,
                    get_access: MemberAccess::Public,
                    set_access: MemberAccess::Public,
                    default_value: None,
                },
            );
        }

        let methods: HashMap<runmat_types::MethodName, crate::class_registry::RuntimeMethod> =
            HashMap::new();
        crate::class_registry::register_builtin_class(
            CLASS_IDENTITY,
            crate::class_registry::RuntimeClass {
                events: Default::default(),
                name: CLASS_IDENTITY.into(),
                parent: Some(runmat_types::standard::HANDLE.into()),
                properties,
                methods,
            },
        );
    });
}

fn scalar_text(value: &Value, name: &str) -> BuiltinResult<String> {
    match value {
        Value::String(runmat_value::StringElement::Text(text)) => Ok(text.as_str().to_owned()),
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
    runmat_value::decode_utf16_code_units(chars.data()).map_err(|source| {
        let mut error = error(&DATA_TIP_ERROR_INVALID_ARGUMENT, source.to_string());
        error.source = Some(Box::new(source));
        error
    })
}

fn string_array_text(array: &StringArray, name: &str) -> BuiltinResult<String> {
    if array.data().len() != 1 {
        return Err(error(
            &DATA_TIP_ERROR_INVALID_ARGUMENT,
            format!("{name} must be a scalar string"),
        ));
    }
    array.data()[0].as_str().map(str::to_owned).ok_or_else(|| {
        error(
            &DATA_TIP_ERROR_INVALID_ARGUMENT,
            format!("{name} must be a scalar string"),
        )
    })
}

#[runtime_builtin(
    name = "dataTipTextRow",
    context = true,
    binding_variant = "default",
    builtin_path = "crate::builtins::plotting::data_tip_text_row"
)]
pub fn data_tip_text_row_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
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

    let fields = [
        ("Label", Value::string(label)),
        ("Value", value),
        ("Format", Value::string(format)),
    ];
    let object = crate::execution::object::construct_object_from_borrowed_fields(
        runtime.as_ref(),
        CLASS_IDENTITY.display_name(),
        fields.iter().map(|(name, value)| (*name, value)),
    )?;
    let (object, _roots) = object.into_value_and_lease();
    let Value::Object(object) = object else {
        unreachable!("data-tip constructor publishes an object")
    };
    let class_name = object.class_name_owner().clone();
    let target = runmat_gc::gc_allocate(Value::Object(object)).map_err(|err| {
        error(
            &DATA_TIP_ERROR_INTERNAL,
            format!("failed to allocate data-tip row handle: {err}"),
        )
    })?;
    Ok(Value::HandleObject(HandleRef {
        class_name,
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
        Value::StringArray(array) if array.data().len() == 1 && !array.data()[0].is_missing() => {
            Ok(())
        }
        Value::FunctionHandle(_)
        | Value::ExternalFunctionHandle(_)
        | Value::MethodFunctionHandle(_)
        | Value::BoundFunctionHandle(_)
        | Value::ScopedFunctionHandle(_) => Ok(()),
        Value::Num(_) | Value::Int(_) | Value::Bool(_) => Ok(()),
        Value::Tensor(tensor) => validate_host_vector_shape(tensor.shape()),
        Value::LogicalArray(array) => validate_host_vector_shape(&array.shape),
        Value::ComplexTensor(tensor) => validate_host_vector_shape(tensor.shape()),
        Value::Complex(_, _) => Ok(()),
        Value::GpuTensor(handle) => {
            crate::compatibility::ensure_builtin_extension_enabled(
                &DATA_TIP_RESIDENT_VALUE_EXTENSION,
                BUILTIN_NAME,
            )?;
            if !vector_shape(handle.shape()) {
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
    use runmat_value::{IntegerStorage, StringArray, Tensor};

    fn data_tip_text_row_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
        super::data_tip_text_row_builtin(
            None,
            crate::call::BuiltinArguments::from_borrowed(&args, None)?,
        )
    }

    fn handle_payload(value: Value) -> ObjectInstance {
        let Value::HandleObject(handle) = value else {
            panic!("expected handle object");
        };
        assert!(handle.class_name.is(CLASS_IDENTITY));
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
            object.properties().get("Label"),
            Some(&Value::String("Amplitude".into()))
        );
        assert_eq!(object.properties().get("Value"), Some(&value));
        assert_eq!(
            object.properties().get("Format"),
            Some(&Value::String("auto".into()))
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn preserves_low_rank_real_and_complex_value_sources() {
        let values = [
            Value::Tensor(Tensor::new(vec![7.0], Vec::new()).unwrap()),
            Value::ComplexTensor(
                runmat_value::ComplexTensor::new(vec![(1.0, 2.0), (3.0, 4.0)], vec![2]).unwrap(),
            ),
        ];
        for value in values {
            let row = data_tip_text_row_builtin(vec![Value::String("Value".into()), value.clone()])
                .expect("low-rank value source");
            let object = handle_payload(row);
            assert_eq!(object.properties().get("Value"), Some(&value));
        }
    }

    #[test]
    fn accepts_char_and_scalar_string_array_text() {
        let label = Value::CharArray(CharArray::new_row("Speed"));
        let format = Value::StringArray(StringArray::new(vec!["%.2f".into()], vec![1, 1]).unwrap());
        let row = data_tip_text_row_builtin(vec![label, Value::String("YData".into()), format])
            .expect("dataTipTextRow");
        let object = handle_payload(row);
        assert_eq!(
            object.properties().get("Label"),
            Some(&Value::String("Speed".into()))
        );
        assert_eq!(
            object.properties().get("Value"),
            Some(&Value::String("YData".into()))
        );
        assert_eq!(
            object.properties().get("Format"),
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
        let handle = runmat_accelerate_api::test_support::TestAllocationOwner::new()
            .allocate_zeroed(
                vec![1, 3],
                runmat_accelerate_api::NumericElementType::F64,
                runmat_accelerate_api::GpuTensorStorage::Real,
            )
            .unwrap()
            .handle;
        let row = data_tip_text_row_builtin(vec![
            Value::String("GPU".into()),
            Value::GpuTensor(handle.clone()),
        ])
        .expect("dataTipTextRow");
        let object = handle_payload(row);
        assert!(matches!(
            object.properties().get("Value"),
            Some(Value::GpuTensor(stored))
                if stored.shares_allocation_with(&handle)
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
                object.properties().get("Value"),
                Some(Value::GpuTensor(stored)) if stored == &handle
            ));
            assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
            let _ = handle.release();
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
            let Some(Value::Tensor(stored)) = object.properties().get("Value") else {
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
        let handle = runmat_accelerate_api::test_support::TestAllocationOwner::new()
            .allocate_zeroed(
                vec![1, 2],
                runmat_accelerate_api::NumericElementType::F64,
                runmat_accelerate_api::GpuTensorStorage::Real,
            )
            .unwrap()
            .handle;
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
            Value::Struct(runmat_value::StructValue::new()),
        ])
        .expect_err("unsupported source");
        assert_eq!(
            error.identifier(),
            DATA_TIP_ERROR_INVALID_ARGUMENT.identifier
        );
    }
}
