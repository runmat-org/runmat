//! MATLAB-compatible `metaclass` builtin backed by RunMat class metadata.

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::introspection::class::class_name_for_value;
use crate::builtins::introspection::type_resolvers::metaclass_type;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor, Value,
};
use runmat_macros::runtime_builtin;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::introspection::metaclass")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "metaclass",
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
    notes: "Metadata-only predicate. RunMat reads runtime class metadata and returns a host meta-class reference without gathering gpuArray buffers.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::introspection::metaclass")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "metaclass",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes:
        "Not eligible for fusion planning; metaclass executes on the host and produces metadata.",
};

const METACLASS_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "mc",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Meta-class reference for the input value's runtime class.",
}];

const METACLASS_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "object",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Object or value whose runtime class metadata is requested.",
}];

const METACLASS_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "mc = metaclass(object)",
    inputs: &METACLASS_INPUTS,
    outputs: &METACLASS_OUTPUT,
}];

const METACLASS_ERRORS: [BuiltinErrorDescriptor; 0] = [];

pub const METACLASS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &METACLASS_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &METACLASS_ERRORS,
};

#[runtime_builtin(
    name = "metaclass",
    category = "introspection",
    summary = "Return metadata for the runtime class of a value.",
    keywords = "metaclass,meta.class,classdef,object,introspection",
    accel = "metadata",
    type_resolver(metaclass_type),
    descriptor(crate::builtins::introspection::metaclass::METACLASS_DESCRIPTOR),
    builtin_path = "crate::builtins::introspection::metaclass"
)]
fn metaclass_builtin(value: Value) -> crate::BuiltinResult<Value> {
    Ok(Value::ClassRef(class_name_for_value(&value)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use runmat_accelerate_api::HostTensorView;
    use runmat_builtins::{
        CellArray, CharArray, HandleRef, Listener, MException, ObjectInstance, StringArray, Tensor,
    };

    fn handle_target() -> runmat_gc::GcHandle {
        runmat_gc::gc_allocate(Value::Num(0.0)).expect("gc allocation")
    }

    fn call(value: Value) -> String {
        let Value::ClassRef(class_name) = metaclass_builtin(value).expect("metaclass") else {
            panic!("metaclass should return ClassRef");
        };
        class_name
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn metaclass_reports_object_and_handle_runtime_classes() {
        assert_eq!(
            call(Value::Object(ObjectInstance::new("pkg.Point".to_string()))),
            "pkg.Point"
        );
        assert_eq!(
            call(Value::HandleObject(HandleRef {
                class_name: "pkg.HandlePoint".to_string(),
                target: handle_target(),
                valid: true,
            })),
            "pkg.HandlePoint"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn metaclass_reports_runtime_classes_for_text_values() {
        assert_eq!(call(Value::String("Point".to_string())), "string");
        assert_eq!(call(Value::CharArray(CharArray::new_row("Point"))), "char");
        assert_eq!(
            call(Value::StringArray(
                StringArray::new(vec!["Point".to_string()], vec![1, 1]).expect("string array")
            )),
            "string"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn metaclass_reports_core_runtime_classes() {
        assert_eq!(call(Value::Num(1.0)), "double");
        assert_eq!(call(Value::Bool(true)), "logical");
        assert_eq!(
            call(Value::Cell(
                CellArray::new(vec![Value::Num(1.0)], 1, 1).expect("cell")
            )),
            "cell"
        );
        assert_eq!(call(Value::ClassRef("Point".to_string())), "meta.class");
        assert_eq!(
            call(Value::MException(MException::new(
                "RunMat:Test".to_string(),
                "failure".to_string()
            ))),
            "MException"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn metaclass_reports_listener_and_gpuarray_without_gather() {
        assert_eq!(
            call(Value::Listener(Listener {
                id: 7,
                target: handle_target(),
                target_class_name: "pkg.Target".to_string(),
                event_name: "Changed".to_string(),
                callback: handle_target(),
                enabled: true,
                valid: true,
            })),
            "event.listener"
        );

        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 2.0], vec![2, 1]).expect("tensor");
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            assert_eq!(call(Value::GpuTensor(handle)), "gpuArray");
        });
    }
}
