//! MATLAB-compatible `isobject` builtin.

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::introspection::type_resolvers::isobject_type;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor, Value,
};
use runmat_macros::runtime_builtin;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::introspection::isobject")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "isobject",
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
    notes: "Metadata predicate; gpuArray inputs stay resident and return logical true.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::introspection::isobject")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "isobject",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Object/type predicate that does not participate in fusion planning.",
};

const ISOBJECT_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "tf",
    ty: BuiltinParamType::LogicalArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "True when input is a MATLAB object.",
}];

const ISOBJECT_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input value to inspect.",
}];

const ISOBJECT_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "tf = isobject(A)",
    inputs: &ISOBJECT_INPUTS,
    outputs: &ISOBJECT_OUTPUT,
}];

const ISOBJECT_ERRORS: [BuiltinErrorDescriptor; 0] = [];

pub const ISOBJECT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &ISOBJECT_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ISOBJECT_ERRORS,
};

#[runtime_builtin(
    name = "isobject",
    category = "introspection",
    summary = "Return true when a value is a MATLAB object.",
    keywords = "isobject,object,handle,metaclass,type checking",
    accel = "metadata",
    type_resolver(isobject_type),
    descriptor(crate::builtins::introspection::isobject::ISOBJECT_DESCRIPTOR),
    builtin_path = "crate::builtins::introspection::isobject"
)]
fn isobject_builtin(value: Value) -> crate::BuiltinResult<Value> {
    Ok(Value::Bool(isobject_value(&value)))
}

pub(crate) fn isobject_value(value: &Value) -> bool {
    matches!(
        value,
        Value::Object(_)
            | Value::HandleObject(_)
            | Value::Listener(_)
            | Value::ClassRef(_)
            | Value::MException(_)
            | Value::GpuTensor(_)
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use runmat_accelerate_api::HostTensorView;
    use runmat_builtins::{
        CellArray, CharArray, Closure, HandleRef, Listener, MException, ObjectInstance,
        StructValue, Tensor,
    };

    fn handle_target() -> runmat_gc::GcHandle {
        runmat_gc::gc_allocate(Value::Num(0.0)).expect("gc allocation")
    }

    #[test]
    fn runmat_object_variants_report_true() {
        let object = Value::Object(ObjectInstance::new("pkg.Point".to_string()));
        let handle = Value::HandleObject(HandleRef {
            class_name: "pkg.PointHandle".to_string(),
            target: handle_target(),
            valid: true,
        });
        let listener = Value::Listener(Listener {
            id: 1,
            target: handle_target(),
            target_class_name: "pkg.Target".to_string(),
            event_name: "Changed".to_string(),
            callback: handle_target(),
            enabled: true,
            valid: true,
        });
        let class_ref = Value::ClassRef("pkg.Point".to_string());
        let exception = Value::MException(MException::new(
            "RunMat:Test".to_string(),
            "failure".to_string(),
        ));

        for value in [object, handle, listener, class_ref, exception] {
            assert_eq!(
                isobject_builtin(value).expect("isobject"),
                Value::Bool(true)
            );
        }
    }

    #[test]
    fn non_object_values_report_false() {
        let mut st = StructValue::new();
        st.insert("x", Value::Num(1.0));
        let values = vec![
            Value::Num(1.0),
            Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).expect("tensor")),
            Value::Bool(true),
            Value::String("text".to_string()),
            Value::CharArray(CharArray::new_row("text")),
            Value::Cell(CellArray::new(vec![Value::Num(1.0)], 1, 1).expect("cell")),
            Value::Struct(st),
            Value::FunctionHandle("sin".to_string()),
            Value::Closure(Closure {
                function_name: "anon".to_string(),
                bound_function: None,
                captures: Vec::new(),
            }),
        ];

        for value in values {
            assert_eq!(
                isobject_builtin(value).expect("isobject"),
                Value::Bool(false)
            );
        }
    }

    #[test]
    fn gpu_tensor_reports_true_without_gathering() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 2.0], vec![1, 2]).expect("tensor");
            let view = HostTensorView {
                data: &tensor.materialize_f64(),
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            assert_eq!(
                isobject_builtin(Value::GpuTensor(handle)).expect("isobject"),
                Value::Bool(true)
            );
        });
    }
}
