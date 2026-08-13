//! MATLAB-compatible `ismethod` builtin backed by RunMat class metadata.
use runmat_types::MemberAccess;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::introspection::type_resolvers::ismethod_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
};
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

const ISMETHOD_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "tf",
    ty: BuiltinParamType::LogicalArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "True when the object has the requested public method.",
}];

const ISMETHOD_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "obj",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Object, class name, class reference, or gpuArray value to inspect.",
    },
    BuiltinParamDescriptor {
        name: "method_name",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Method name as a string scalar or row character vector.",
    },
];

const ISMETHOD_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "tf = ismethod(obj, method_name)",
    inputs: &ISMETHOD_INPUTS,
    outputs: &ISMETHOD_OUTPUT,
}];

const BUILTIN_NAME: &str = "ismethod";

const ISMETHOD_ERROR_NAME_INVALID: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ISMETHOD.NAME_INVALID",
    identifier: Some("RunMat:IsMethodNameInvalid"),
    when: "The method-name argument is not a string scalar or row character vector.",
    message: "ismethod: method name must be a string scalar or character vector",
};

const ISMETHOD_ERRORS: [BuiltinErrorDescriptor; 1] = [ISMETHOD_ERROR_NAME_INVALID];

pub const ISMETHOD_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &ISMETHOD_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ISMETHOD_ERRORS,
};

#[runtime_builtin(
    name = "ismethod",
    category = "introspection",
    summary = "Test whether an object has a public method.",
    keywords = "ismethod,method,classdef,object,handle,inheritance",
    accel = "metadata",
    type_resolver(ismethod_type),
    descriptor(crate::builtins::introspection::ismethod::ISMETHOD_DESCRIPTOR),
    builtin_path = "crate::builtins::introspection::ismethod"
)]
fn ismethod_builtin(receiver: Value, method_designator: Value) -> crate::BuiltinResult<Value> {
    let method_name = parse_method_name(&method_designator)?;
    Ok(Value::Bool(value_has_public_method(
        &receiver,
        &method_name,
    )))
}

fn parse_method_name(value: &Value) -> BuiltinResult<String> {
    match value {
        Value::String(s) => Ok(s.clone()),
        Value::StringArray(sa) if sa.rows == 1 && sa.cols == 1 && !sa.data.is_empty() => {
            Ok(sa.data[0].clone())
        }
        Value::CharArray(ca) if ca.rows <= 1 => Ok(ca.data.iter().collect::<String>()),
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

    let class_name = match receiver_class_name(receiver) {
        Some(class_name) if !class_name.is_empty() => class_name,
        _ => return false,
    };

    crate::class_registry::lookup_method(&class_name, method_name)
        .is_some_and(|(method, _owner)| matches!(method.access, MemberAccess::Public))
}

fn receiver_class_name(receiver: &Value) -> Option<String> {
    match receiver {
        Value::ObjectArray(array) => Some(array.class_name().to_string()),
        Value::Object(object) => Some(object.class_name.clone()),
        Value::HandleObject(handle) => {
            if handle.class_name.is_empty() {
                Some("handle".to_string())
            } else {
                Some(handle.class_name.clone())
            }
        }
        Value::GpuTensor(_) => Some("gpuArray".to_string()),
        Value::Listener(Listener { .. }) => Some("event.listener".to_string()),
        Value::ClassRef(class_name) => Some(class_name.clone()),
        Value::MException(MException { .. }) => Some("MException".to_string()),
        Value::String(class_name) => Some(class_name.clone()),
        Value::StringArray(sa) if sa.rows == 1 && sa.cols == 1 && !sa.data.is_empty() => {
            Some(sa.data[0].clone())
        }
        Value::CharArray(ca) if ca.rows <= 1 => Some(ca.data.iter().collect::<String>()),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use runmat_accelerate_api::HostTensorView;
    use runmat_value::{CharArray, HandleRef, ObjectInstance, StringArray, Tensor};
    use std::collections::HashMap;

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
            name: name.to_string(),
            is_static,
            is_abstract: false,
            is_sealed: false,
            access,
            function_name: format!("{name}_impl"),
            implicit_class_argument: None,
        }
    }

    fn register_pair(label: &str) -> (String, String) {
        let parent = unique_class_name(&format!("{label}Parent"));
        let child = unique_class_name(&format!("{label}Child"));

        let mut parent_methods = HashMap::new();
        parent_methods.insert(
            "inherited".to_string(),
            method("inherited", MemberAccess::Public, false),
        );
        parent_methods.insert(
            "hidden".to_string(),
            method("hidden", MemberAccess::Private, false),
        );

        crate::class_registry::register_class(crate::class_registry::RuntimeClass {
            name: parent.clone(),
            parent: None,
            properties: HashMap::new(),
            methods: parent_methods,
        });

        let mut child_methods = HashMap::new();
        child_methods.insert(
            "run".to_string(),
            method("run", MemberAccess::Public, false),
        );
        child_methods.insert(
            "make".to_string(),
            method("make", MemberAccess::Public, true),
        );
        child_methods.insert(
            "secret".to_string(),
            method("secret", MemberAccess::Protected, false),
        );

        crate::class_registry::register_class(crate::class_registry::RuntimeClass {
            name: child.clone(),
            parent: Some(parent.clone()),
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
            class_name: class_name.to_string(),
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

        assert!(call(receiver.clone(), Value::String("run".to_string())));
        assert!(call(receiver.clone(), Value::String("make".to_string())));
        assert!(call(receiver, Value::String("inherited".to_string())));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ismethod_respects_public_visibility() {
        let (_parent, child) = register_pair("MemberAccess");
        let receiver = object(&child);

        assert!(!call(receiver.clone(), Value::String("secret".to_string())));
        assert!(!call(receiver, Value::String("hidden".to_string())));
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
    fn ismethod_supports_class_name_text_and_class_refs() {
        let (_parent, child) = register_pair("ClassName");

        assert!(call(
            Value::String(child.clone()),
            Value::String("run".to_string())
        ));
        assert!(call(
            Value::CharArray(CharArray::new_row(&child)),
            Value::String("make".to_string())
        ));
        assert!(call(
            Value::StringArray(
                StringArray::new(vec![child.clone()], vec![1, 1]).expect("class scalar")
            ),
            Value::String("inherited".to_string())
        ));
        assert!(call(
            Value::ClassRef(child),
            Value::String("run".to_string())
        ));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ismethod_uses_gpuarray_class_metadata_without_gather() {
        let mut methods = HashMap::new();
        methods.insert(
            "__runmat_ismethod_marker".to_string(),
            method("__runmat_ismethod_marker", MemberAccess::Public, false),
        );
        crate::class_registry::register_class(crate::class_registry::RuntimeClass {
            name: "gpuArray".to_string(),
            parent: None,
            properties: HashMap::new(),
            methods,
        });

        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
            let view = HostTensorView {
                data: &tensor.materialize_f64(),
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");

            assert!(call(
                Value::GpuTensor(handle),
                Value::String("__runmat_ismethod_marker".to_string())
            ));
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ismethod_returns_false_for_missing_empty_and_nonobject_values() {
        let (_parent, child) = register_pair("Negative");

        assert!(!call(object(&child), Value::String("missing".to_string())));
        assert!(!call(object(&child), Value::String(String::new())));
        assert!(!call(object(&child), Value::String(" run ".to_string())));
        assert!(!call(Value::Num(1.0), Value::String("run".to_string())));
        assert!(!call(
            Value::String("MissingClassForIsmethod".to_string()),
            Value::String("run".to_string())
        ));
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
