//! Stability test for SISO transfer-function models.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
};
use runmat_builtins::{BuiltinIntegerAuditDescriptor, BuiltinIntegerAuditKind};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::control::tf_model::TfModel;
use crate::builtins::control::type_resolvers::isstable_type;
use crate::BuiltinResult;

const ISSTABLE_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "tf",
    ty: BuiltinParamType::LogicalArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "True when all continuous poles are in the open left-half plane or all discrete poles are inside the unit circle.",
}];
const ISSTABLE_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "sys",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "SISO tf model.",
}];
const ISSTABLE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "tf = isstable(sys)",
    inputs: &ISSTABLE_INPUTS,
    outputs: &ISSTABLE_OUTPUT,
}];
const ISSTABLE_ERRORS: [BuiltinErrorDescriptor; 3] = [
    BuiltinErrorDescriptor {
        code: "RM.ISSTABLE.INVALID_MODEL",
        identifier: Some("RunMat:isstable:InvalidModel"),
        when: "Input system is not a valid SISO tf object.",
        message: "isstable: invalid model",
    },
    BuiltinErrorDescriptor {
        code: "RM.ISSTABLE.UNSUPPORTED_MODEL",
        identifier: Some("RunMat:isstable:UnsupportedModel"),
        when: "Model form is unsupported.",
        message: "isstable: unsupported model",
    },
    BuiltinErrorDescriptor {
        code: "RM.ISSTABLE.INTERNAL",
        identifier: Some("RunMat:isstable:Internal"),
        when: "Pole calculation failed.",
        message: "isstable: internal error",
    },
];
pub const ISSTABLE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &ISSTABLE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ISSTABLE_ERRORS,
};
pub const ISSTABLE_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "isstable requires a dynamic-system model; fundamental integer host or resident values are invalid-domain inputs and reject before payload or provider access.",
};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::control::isstable")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "isstable",
    op_kind: GpuOpKind::Custom("control-stability"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "isstable analyzes host-side transfer-function metadata.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::control::isstable")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "isstable",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "isstable returns scalar model metadata and is not fused.",
};

#[runtime_builtin(
    name = "isstable",
    category = "control",
    summary = "Test stability of SISO transfer-function models.",
    keywords = "isstable,control system,stability,poles,tf",
    type_resolver(isstable_type),
    descriptor(crate::builtins::control::isstable::ISSTABLE_DESCRIPTOR),
    integer_audit(crate::builtins::control::isstable::ISSTABLE_INTEGER_AUDIT),
    builtin_path = "crate::builtins::control::isstable"
)]
async fn isstable_builtin(sys: Value) -> BuiltinResult<Value> {
    if value_has_integer_storage(&sys) {
        return Err(crate::build_runtime_error(
            "isstable: input must be a dynamic-system model, not an integer value",
        )
        .with_builtin("isstable")
        .with_identifier(
            ISSTABLE_ERRORS[0]
                .identifier
                .expect("isstable invalid-model descriptor identifier"),
        )
        .build()
        .into());
    }
    let model = TfModel::from_value_async(sys, "isstable").await?;
    Ok(Value::Bool(model.is_stable()?))
}

fn value_has_integer_storage(value: &Value) -> bool {
    matches!(value, Value::Int(_))
        || matches!(value, Value::Tensor(tensor) if tensor.integer_storage().is_some())
        || matches!(value, Value::ComplexTensor(tensor) if tensor.integer_storage().is_some())
        || matches!(value, Value::SparseTensor(tensor) if tensor.integer_storage().is_some())
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_integer_type(handle).is_some())
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_value::Tensor;
    use runmat_value::{IntValue, IntegerStorage};

    #[test]
    fn stable_continuous_model_returns_true() {
        let sys = block_on(crate::call_builtin_async(
            "tf",
            &[
                Value::Num(1.0),
                Value::Tensor(Tensor::new(vec![1.0, 3.0, 2.0], vec![1, 3]).unwrap()),
            ],
        ))
        .expect("tf");
        assert_eq!(
            block_on(isstable_builtin(sys)).expect("isstable"),
            Value::Bool(true)
        );
    }

    #[test]
    fn all_integer_classes_are_invalid_model_inputs() {
        for integer in [
            IntValue::I8(-1),
            IntValue::I16(-2),
            IntValue::I32(-3),
            IntValue::I64(i64::MIN),
            IntValue::U8(1),
            IntValue::U16(2),
            IntValue::U32(3),
            IntValue::U64(u64::MAX),
        ] {
            let error = block_on(isstable_builtin(Value::Int(integer)))
                .expect_err("integer is not a system model");
            assert_eq!(
                error.identifier.as_deref(),
                Some("RunMat:isstable:InvalidModel")
            );
        }
    }

    #[test]
    fn resident_integer_rejects_before_model_gather() {
        let tensor = Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX]), vec![1, 1])
            .expect("integer tensor");
        crate::builtins::common::test_support::with_test_provider(|provider| {
            let handle = crate::builtins::common::gpu_helpers::upload_tensor(provider, &tensor)
                .expect("upload integer");
            let error = block_on(isstable_builtin(Value::GpuTensor(handle)))
                .expect_err("resident integer is not a system model");
            assert_eq!(
                error.identifier.as_deref(),
                Some("RunMat:isstable:InvalidModel")
            );
        });
    }
}
