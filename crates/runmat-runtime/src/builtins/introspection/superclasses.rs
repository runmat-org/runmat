//! MATLAB-compatible `superclasses` builtin backed by RunMat class metadata.

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::introspection::class::class_name_for_value;
use crate::builtins::introspection::type_resolvers::superclasses_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};
use runmat_builtins::{
    superclass_chain, BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor,
    BuiltinOutputMode, BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType,
    BuiltinSignatureDescriptor, CellArray, CharArray, Value,
};
use runmat_macros::runtime_builtin;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::introspection::superclasses")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "superclasses",
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
    notes: "Metadata-only class hierarchy query. Explicit gpuArray inputs use wrapper metadata, while internally auto-resident values use their underlying host class; neither path downloads payload data.",
};

#[runmat_macros::register_fusion_spec(
    builtin_path = "crate::builtins::introspection::superclasses"
)]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "superclasses",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Not eligible for fusion planning; superclasses executes on the host and returns cell metadata.",
};

const SUPERCLASSES_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "s",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Cell row of visible superclass names as character vectors.",
}];

const SUPERCLASSES_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "obj_or_class",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Object instance or class name string/character vector.",
}];

const SUPERCLASSES_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "s = superclasses(obj_or_class)",
    inputs: &SUPERCLASSES_INPUTS,
    outputs: &SUPERCLASSES_OUTPUT,
}];

const BUILTIN_NAME: &str = "superclasses";

const SUPERCLASSES_ERROR_CLASS_INVALID: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SUPERCLASSES.CLASS_INVALID",
    identifier: Some("RunMat:SuperclassesClassInvalid"),
    when: "The argument is neither an object nor a class-name string scalar/row character vector.",
    message: "superclasses: input must be an object or class name",
};

const SUPERCLASSES_ERROR_CLASS_UNKNOWN: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SUPERCLASSES.CLASS_UNKNOWN",
    identifier: Some("RunMat:SuperclassesClassUnknown"),
    when: "The requested class is not present in RunMat class metadata.",
    message: "superclasses: class metadata not found",
};

const SUPERCLASSES_ERRORS: [BuiltinErrorDescriptor; 2] = [
    SUPERCLASSES_ERROR_CLASS_INVALID,
    SUPERCLASSES_ERROR_CLASS_UNKNOWN,
];

pub const SUPERCLASSES_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SUPERCLASSES_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &SUPERCLASSES_ERRORS,
};

#[runtime_builtin(
    name = "superclasses",
    category = "introspection",
    summary = "Return visible superclass names for a class or object.",
    keywords = "superclasses,classdef,inheritance,object,introspection",
    accel = "metadata",
    type_resolver(superclasses_type),
    descriptor(crate::builtins::introspection::superclasses::SUPERCLASSES_DESCRIPTOR),
    builtin_path = "crate::builtins::introspection::superclasses"
)]
fn superclasses_builtin(value: Value) -> crate::BuiltinResult<Value> {
    let class_name = requested_class_name(&value)?;
    let supers = if let Some(chain) = superclass_chain(&class_name) {
        chain
    } else if known_leaf_class_without_registered_metadata(&class_name) {
        Vec::new()
    } else {
        return Err(superclasses_error_with_detail(
            &SUPERCLASSES_ERROR_CLASS_UNKNOWN,
            class_name,
        ));
    };

    let cols = supers.len();
    let cells = supers
        .into_iter()
        .map(|name| Value::CharArray(CharArray::new_row(&name)))
        .collect::<Vec<_>>();
    Ok(Value::Cell(
        CellArray::new(cells, 1, cols).map_err(superclasses_message_error)?,
    ))
}

fn requested_class_name(value: &Value) -> BuiltinResult<String> {
    match value {
        Value::String(class_name) => Ok(class_name.clone()),
        Value::StringArray(sa) if sa.data.len() == 1 => Ok(sa.data[0].clone()),
        Value::CharArray(ca) if ca.rows <= 1 => Ok(ca.data.iter().collect::<String>()),
        Value::ClassRef(class_name) => Ok(class_name.clone()),
        Value::Object(_)
        | Value::HandleObject(_)
        | Value::GpuTensor(_)
        | Value::Listener(_)
        | Value::MException(_) => Ok(class_name_for_value(value)),
        _ => Err(superclasses_error(&SUPERCLASSES_ERROR_CLASS_INVALID)),
    }
}

fn known_leaf_class_without_registered_metadata(class_name: &str) -> bool {
    matches!(
        class_name,
        "double"
            | "single"
            | "logical"
            | "char"
            | "string"
            | "cell"
            | "struct"
            | "function_handle"
            | "handle"
            | "gpuArray"
            | "meta.class"
            | "MException"
            | "event.listener"
            | "sparse"
            | "sym"
            | "int8"
            | "uint8"
            | "int16"
            | "uint16"
            | "int32"
            | "uint32"
            | "int64"
            | "uint64"
    )
}

fn superclasses_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    let mut builder = build_runtime_error(error.message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn superclasses_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: String,
) -> RuntimeError {
    let mut builder =
        build_runtime_error(format!("{}: {}", error.message, detail)).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn superclasses_message_error(message: String) -> RuntimeError {
    build_runtime_error(format!("superclasses: {message}"))
        .with_builtin(BUILTIN_NAME)
        .build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use runmat_accelerate_api::HostTensorView;
    use runmat_builtins::{
        register_class, Access, CharArray, ClassDef, HandleRef, MethodDef, ObjectInstance,
        StringArray, Tensor,
    };
    use std::collections::HashMap;

    fn method(name: &str) -> MethodDef {
        MethodDef {
            name: name.to_string(),
            is_static: false,
            is_abstract: false,
            is_sealed: false,
            access: Access::Public,
            function_name: format!("{name}_impl"),
            implicit_class_argument: None,
        }
    }

    fn unique_class_name(label: &str) -> String {
        format!(
            "Superclasses{}{}",
            label,
            std::thread::current().name().unwrap_or("thread")
        )
    }

    fn register_hierarchy(label: &str) -> (String, String, String) {
        let grand = unique_class_name(&format!("{label}Grand"));
        let parent = unique_class_name(&format!("{label}Parent"));
        let child = unique_class_name(&format!("{label}Child"));

        register_class(ClassDef {
            name: grand.clone(),
            parent: None,
            properties: HashMap::new(),
            methods: HashMap::from([("root".to_string(), method("root"))]),
        });
        register_class(ClassDef {
            name: parent.clone(),
            parent: Some(grand.clone()),
            properties: HashMap::new(),
            methods: HashMap::from([("mid".to_string(), method("mid"))]),
        });
        register_class(ClassDef {
            name: child.clone(),
            parent: Some(parent.clone()),
            properties: HashMap::new(),
            methods: HashMap::from([("leaf".to_string(), method("leaf"))]),
        });

        (grand, parent, child)
    }

    fn cell_strings(value: Value) -> Vec<String> {
        let Value::Cell(cell) = value else {
            panic!("superclasses should return cell");
        };
        cell.data
            .into_iter()
            .map(|value| match value {
                Value::CharArray(chars) => chars.data.iter().collect::<String>(),
                other => panic!("expected char superclass name, got {other:?}"),
            })
            .collect()
    }

    fn call(value: Value) -> Vec<String> {
        cell_strings(superclasses_builtin(value).expect("superclasses"))
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn superclasses_returns_parent_chain_for_class_name_text() {
        let (grand, parent, child) = register_hierarchy("ClassName");

        assert_eq!(
            call(Value::String(child.clone())),
            vec![parent.clone(), grand.clone()]
        );
        assert_eq!(
            call(Value::CharArray(CharArray::new_row(&child))),
            vec![parent.clone(), grand.clone()]
        );
        assert_eq!(
            call(Value::StringArray(
                StringArray::new(vec![child], vec![1, 1]).expect("string scalar")
            )),
            vec![parent, grand]
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn superclasses_returns_parent_chain_for_objects_and_class_refs() {
        let (grand, parent, child) = register_hierarchy("Object");
        let target = runmat_gc::gc_allocate(Value::Object(ObjectInstance::new(child.clone())))
            .expect("gc allocation");

        assert_eq!(
            call(Value::Object(ObjectInstance::new(child.clone()))),
            vec![parent.clone(), grand.clone()]
        );
        assert_eq!(
            call(Value::HandleObject(HandleRef {
                class_name: child.clone(),
                target,
                valid: true,
            })),
            vec![parent.clone(), grand.clone()]
        );
        assert_eq!(call(Value::ClassRef(child)), vec![parent, grand]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn superclasses_returns_empty_cell_for_leaf_builtin_classes() {
        assert!(call(Value::from("char")).is_empty());
        assert!(call(Value::Object(ObjectInstance::new("double".to_string()))).is_empty());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn superclasses_distinguishes_automatic_and_explicit_residency_without_gather() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 2.0], vec![2, 1]).expect("tensor");
            let view = HostTensorView {
                data: &tensor.materialize_f64(),
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            assert!(call(Value::GpuTensor(handle.clone())).is_empty());
            runmat_accelerate_api::mark_handle_explicit(&handle);
            assert!(call(Value::GpuTensor(handle)).is_empty());
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn superclasses_handles_known_runtime_leaf_classes_without_registry_metadata() {
        assert!(call(Value::GpuTensor(test_support::with_test_provider(
            |provider| {
                let tensor = Tensor::new(vec![1.0], vec![1, 1]).expect("tensor");
                let view = HostTensorView {
                    data: &tensor.materialize_f64(),
                    shape: &tensor.shape,
                };
                provider.upload(&view).expect("upload")
            }
        )))
        .is_empty());
        assert!(call(Value::HandleObject(HandleRef {
            class_name: String::new(),
            target: runmat_gc::gc_allocate(Value::Num(0.0)).expect("gc allocation"),
            valid: true,
        }))
        .is_empty());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn superclasses_errors_for_invalid_or_unknown_class_inputs() {
        let invalid = superclasses_builtin(Value::Num(1.0)).expect_err("numeric is invalid");
        assert_eq!(
            invalid.identifier.as_deref(),
            Some("RunMat:SuperclassesClassInvalid")
        );

        let invalid_string_array = superclasses_builtin(Value::StringArray(
            StringArray::new(
                vec!["ClassA".to_string(), "ClassB".to_string()],
                vec![1, 1, 2],
            )
            .expect("string array"),
        ))
        .expect_err("nonscalar string array is invalid");
        assert_eq!(
            invalid_string_array.identifier.as_deref(),
            Some("RunMat:SuperclassesClassInvalid")
        );

        let unknown =
            superclasses_builtin(Value::from("MissingRunMatClass")).expect_err("missing class");
        assert_eq!(
            unknown.identifier.as_deref(),
            Some("RunMat:SuperclassesClassUnknown")
        );
    }
}
