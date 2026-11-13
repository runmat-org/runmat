//! MATLAB-compatible `ismethod` builtin backed by RunMat class metadata.
pub use runmat_builtins::{
    ISMETHOD_DESCRIPTOR, ISMETHOD_ERROR_NAME_INVALID, ISMETHOD_ERROR_TOO_MANY_OUTPUTS,
};
use runmat_types::MemberAccess;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};
use runmat_builtins::BuiltinErrorDescriptor;
use runmat_macros::runtime_builtin;
use runmat_value::{Listener, MException, Value};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::introspection::ismethod")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "ismethod",
    op_kind: GpuOpKind::Custom("metadata"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::InheritInputs,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Metadata predicate that inspects host class registry entries and returns a host logical scalar; no GPU kernels or gathers are required.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::introspection::ismethod")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "ismethod",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Not eligible for fusion planning; ismethod executes on the host and produces a logical scalar.",
};

const BUILTIN_NAME: &str = "ismethod";

#[runtime_builtin(
    name = "ismethod",
    binding_variant = "default",
    builtin_path = "crate::builtins::introspection::ismethod"
)]
fn ismethod_builtin(receiver: Value, method_designator: Value) -> crate::BuiltinResult<Value> {
    crate::output_count::validate_bounded_outputs(
        &ISMETHOD_DESCRIPTOR,
        &ISMETHOD_ERROR_TOO_MANY_OUTPUTS,
        "ismethod",
    )?;
    let method_name = parse_method_name(&method_designator)?;
    Ok(Value::Bool(value_has_public_method(
        &receiver,
        &method_name,
    )))
}

fn parse_method_name(value: &Value) -> BuiltinResult<String> {
    match value {
        Value::String(runmat_value::StringElement::Text(s)) => Ok(s.as_str().to_owned()),
        Value::StringArray(sa) if sa.data().len() == 1 && !sa.data()[0].is_missing() => Ok(sa
            .data()[0]
            .as_str()
            .expect("nonmissing scalar text")
            .to_owned()),
        Value::CharArray(ca) if ca.rows <= 1 && ca.shape().iter().skip(2).all(|n| *n == 1) => {
            super::text::decode_utf16_with_error(ca.data(), || {
                ismethod_error(&ISMETHOD_ERROR_NAME_INVALID)
            })
        }
        _ => Err(ismethod_error(&ISMETHOD_ERROR_NAME_INVALID)),
    }
}

fn ismethod_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    let mut builder = build_runtime_error(error.message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn value_has_public_method(receiver: &Value, method_name: &str) -> bool {
    if method_name.is_empty() {
        return false;
    }

    let Some(class_name) = receiver_class_name(receiver) else {
        return false;
    };

    crate::class_registry::lookup_method(&class_name, &runmat_types::MethodName::from(method_name))
        .is_some_and(|(method, _owner)| matches!(method.access, MemberAccess::Public))
}

fn receiver_class_name(receiver: &Value) -> Option<runmat_types::ClassIdentity> {
    match receiver {
        Value::ObjectArray(array) => Some(array.class_name().clone()),
        Value::Object(object) => Some(object.class_name().clone()),
        Value::HandleObject(handle) => Some(handle.class_name.as_identity().clone()),
        Value::GpuTensor(handle) if runmat_accelerate_api::handle_is_explicit(handle) => {
            Some(runmat_types::standard::GPU_ARRAY.into())
        }
        Value::GpuTensor(_) => None,
        Value::Listener(Listener { .. }) => Some(runmat_types::standard::EVENT_LISTENER.into()),
        Value::ClassRef(_) => None,
        Value::MException(MException { .. }) => Some(runmat_types::standard::MEXCEPTION.into()),
        Value::String(_) | Value::StringArray(_) | Value::CharArray(_) => None,
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use runmat_accelerate_api::HostTensorView;
    use runmat_value::{
        CharArray, HandleRef, IntValue, IntegerStorage, ObjectInstance, StringArray, Tensor,
    };
    use std::collections::HashMap;

    #[test]
    fn method_designators_reject_extra_pages_and_requested_outputs() {
        let pages =
            CharArray::from_unicode_scalars_with_shape(vec!['a', 'b'], vec![1, 1, 2]).unwrap();
        assert!(parse_method_name(&Value::CharArray(pages)).is_err());
        let names = StringArray::new(vec!["a".into(), "b".into()], vec![1, 1, 2]).unwrap();
        assert!(parse_method_name(&Value::StringArray(names)).is_err());
        let _outputs = crate::output_count::push_output_count(Some(2)).unwrap();
        let error = ismethod_builtin(Value::Num(0.0), Value::string("method")).unwrap_err();
        assert_eq!(error.identifier(), Some("RunMat:ismethod:TooManyOutputs"));
    }

    fn unique_class_name(label: &str) -> String {
        format!(
            "Ismethod{}{}",
            label,
            std::thread::current().name().unwrap_or("thread")
        )
    }

    fn method(
        name: &str,
        access: MemberAccess,
        is_static: bool,
    ) -> crate::class_registry::RuntimeMethod {
        crate::class_registry::RuntimeMethod {
            name: name.into(),
            is_static,
            is_abstract: false,
            is_sealed: false,
            access,
            function_name: format!("{name}_impl"),
            implicit_class_argument: None,
            role: runmat_types::ClassMethodRole::Ordinary,
        }
    }

    fn register_pair(label: &str) -> (String, String) {
        let parent = unique_class_name(&format!("{label}Parent"));
        let child = unique_class_name(&format!("{label}Child"));

        let mut parent_methods = HashMap::new();
        parent_methods.insert(
            "inherited".into(),
            method("inherited", MemberAccess::Public, false),
        );
        parent_methods.insert(
            "hidden".into(),
            method("hidden", MemberAccess::Private, false),
        );

        crate::class_registry::register_class(crate::class_registry::RuntimeClass {
            events: Default::default(),
            name: parent.clone().into(),
            parent: None,
            properties: HashMap::new(),
            methods: parent_methods,
        });

        let mut child_methods = HashMap::new();
        child_methods.insert("run".into(), method("run", MemberAccess::Public, false));
        child_methods.insert("make".into(), method("make", MemberAccess::Public, true));
        child_methods.insert(
            "secret".into(),
            method("secret", MemberAccess::Protected, false),
        );

        crate::class_registry::register_class(crate::class_registry::RuntimeClass {
            events: Default::default(),
            name: child.clone().into(),
            parent: Some(parent.clone().into()),
            properties: HashMap::new(),
            methods: child_methods,
        });

        (parent, child)
    }

    fn object(class_name: &str) -> Value {
        Value::Object(ObjectInstance::new(class_name.to_string()))
    }

    fn handle_object(class_name: &str) -> Value {
        let target =
            runmat_gc::gc_allocate(Value::Object(ObjectInstance::new(class_name.to_string())))
                .expect("gc allocation");
        Value::HandleObject(HandleRef {
            class_name: class_name.into(),
            target,
            valid: true,
        })
    }

    fn call(receiver: Value, method_name: Value) -> bool {
        let Value::Bool(result) = ismethod_builtin(receiver, method_name).expect("ismethod") else {
            panic!("ismethod should return bool");
        };
        result
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ismethod_finds_public_instance_static_and_inherited_methods() {
        let (_parent, child) = register_pair("Public");
        let receiver = object(&child);

        assert!(call(
            receiver.clone(),
            Value::String("run".to_string().into())
        ));
        assert!(call(
            receiver.clone(),
            Value::String("make".to_string().into())
        ));
        assert!(call(
            receiver,
            Value::String("inherited".to_string().into())
        ));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ismethod_respects_public_visibility() {
        let (_parent, child) = register_pair("MemberAccess");
        let receiver = object(&child);

        assert!(!call(
            receiver.clone(),
            Value::String("secret".to_string().into())
        ));
        assert!(!call(receiver, Value::String("hidden".to_string().into())));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ismethod_supports_handle_objects_and_scalar_text_forms() {
        let (_parent, child) = register_pair("Handle");
        let receiver = handle_object(&child);

        assert!(call(
            receiver.clone(),
            Value::CharArray(CharArray::new_row("inherited"))
        ));
        assert!(call(
            receiver,
            Value::StringArray(
                StringArray::new(vec!["run".to_string()], vec![1, 1]).expect("string scalar")
            )
        ));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ismethod_rejects_class_name_text_and_class_refs_as_receivers() {
        let (_parent, child) = register_pair("ClassName");

        assert!(!call(
            Value::String(child.clone().into()),
            Value::String("run".to_string().into())
        ));
        assert!(!call(
            Value::CharArray(CharArray::new_row(&child)),
            Value::String("make".to_string().into())
        ));
        assert!(!call(
            Value::StringArray(
                StringArray::new(vec![child.clone()], vec![1, 1]).expect("class scalar")
            ),
            Value::String("inherited".to_string().into())
        ));
        assert!(!call(
            Value::ClassRef(child.into()),
            Value::String("run".to_string().into())
        ));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ismethod_uses_production_gpuarray_metadata_without_gather() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
            let view = HostTensorView {
                data: &tensor.materialize_f64(),
                shape: tensor.shape(),
            };
            let handle = provider.upload(&view).expect("upload");
            let caller_alias = handle.clone();
            let original = handle.wire_descriptor();
            let lease = handle.allocation_lease().downgrade();
            provider.reset_telemetry();
            assert!(!call(
                Value::GpuTensor(handle.clone()),
                Value::from("gather")
            ));
            let explicit = handle
                .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
                .expect("explicit immutable view");
            drop(handle);
            let handle = explicit;
            for method in [
                "arrayfun",
                "existsOnGPU",
                "gather",
                "isgpuarray",
                "isUnderlyingType",
                "ndims",
                "pagefun",
                "size",
                "underlyingType",
            ] {
                assert!(call(Value::GpuTensor(handle.clone()), Value::from(method)));
            }
            assert!(!call(
                Value::GpuTensor(handle),
                Value::from("notARealGpuArrayMethod")
            ));
            assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
            assert_eq!(caller_alias.wire_descriptor(), original);
            assert!(lease.acquire().is_some());
            drop(caller_alias);
            assert!(lease.acquire().is_none());
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ismethod_returns_false_for_missing_empty_and_nonobject_values() {
        let (_parent, child) = register_pair("Negative");

        assert!(!call(
            object(&child),
            Value::String("missing".to_string().into())
        ));
        assert!(!call(object(&child), Value::String(String::new().into())));
        assert!(!call(
            object(&child),
            Value::String(" run ".to_string().into())
        ));
        assert!(!call(
            Value::Num(1.0),
            Value::String("run".to_string().into())
        ));
        assert!(!call(
            Value::String("MissingClassForIsmethod".to_string().into()),
            Value::String("run".to_string().into())
        ));
    }

    #[test]
    fn ismethod_reports_false_for_all_fundamental_integer_classes() {
        for value in [
            IntValue::I8(-1),
            IntValue::I16(-2),
            IntValue::I32(-3),
            IntValue::I64(i64::MIN),
            IntValue::U8(1),
            IntValue::U16(2),
            IntValue::U32(3),
            IntValue::U64(u64::MAX),
        ] {
            assert_eq!(
                ismethod_builtin(Value::Int(value), Value::from("run"))
                    .expect("fundamental integer has no public method"),
                Value::Bool(false)
            );
        }
    }

    #[test]
    fn ismethod_only_exposes_gpuarray_wrapper_for_explicit_residency() {
        test_support::with_test_provider(|provider| {
            for storage in [
                IntegerStorage::I8(vec![-1]),
                IntegerStorage::I16(vec![-2]),
                IntegerStorage::I32(vec![-3]),
                IntegerStorage::I64(vec![i64::MIN]),
                IntegerStorage::U8(vec![1]),
                IntegerStorage::U16(vec![2]),
                IntegerStorage::U32(vec![3]),
                IntegerStorage::U64(vec![u64::MAX]),
            ] {
                let tensor = Tensor::new_integer(storage, vec![1, 1]).unwrap();
                let handle = crate::builtins::common::gpu_helpers::upload_tensor(provider, &tensor)
                    .expect("upload integer");
                let caller_alias = handle.clone();
                let original = handle.wire_descriptor();
                let lease = handle.allocation_lease().downgrade();
                provider.reset_telemetry();
                assert_eq!(
                    ismethod_builtin(Value::GpuTensor(handle.clone()), Value::from("gather"))
                        .unwrap(),
                    Value::Bool(false)
                );
                let explicit = handle
                    .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
                    .expect("explicit immutable view");
                drop(handle);
                let handle = explicit;
                assert_eq!(
                    ismethod_builtin(Value::GpuTensor(handle.clone()), Value::from("gather"))
                        .unwrap(),
                    Value::Bool(true)
                );
                assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
                assert_eq!(caller_alias.wire_descriptor(), original);
                handle.release().expect("release explicit view");
                assert!(lease.acquire().is_some());
                drop(caller_alias);
                assert!(lease.acquire().is_none());
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ismethod_rejects_nonscalar_method_name() {
        let (_parent, child) = register_pair("Invalid");
        let err = ismethod_builtin(
            object(&child),
            Value::StringArray(
                StringArray::new(vec!["run".to_string(), "make".to_string()], vec![1, 2])
                    .expect("string array"),
            ),
        )
        .expect_err("non-scalar string array should fail");

        assert_eq!(
            err.identifier.as_deref(),
            Some("RunMat:IsMethodNameInvalid")
        );
    }
}
