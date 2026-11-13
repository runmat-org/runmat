//! Zero extraction for SISO transfer-function control models.

use runmat_builtins::ZERO_ERRORS;
use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::control::tf_model::{output_complex_column, TfModel};
use crate::BuiltinResult;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::control::zero")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "zero",
    op_kind: GpuOpKind::Custom("control-zeros"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "zero computes roots from host-side transfer-function metadata.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::control::zero")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "zero",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "zero is model analysis and is not fused.",
};

#[runtime_builtin(
    name = "zero",
    binding_variant = "default",
    builtin_path = "crate::builtins::control::zero"
)]
async fn zero_builtin(sys: Value) -> BuiltinResult<Value> {
    if crate::builtins::common::validation::value_has_native_integer_class(&sys) {
        return Err(crate::build_runtime_error(
            "zero: input must be a dynamic-system model, not an integer value",
        )
        .with_builtin("zero")
        .with_identifier(
            ZERO_ERRORS[0]
                .identifier
                .expect("zero invalid-model descriptor identifier"),
        )
        .build()
        .into());
    }
    let model = TfModel::from_value_async(sys, "zero").await?;
    output_complex_column(model.zeros()?, "zero")
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_value::{ObjectInstance, Tensor};

    #[test]
    fn unrelated_resident_property_is_not_downloaded() {
        let unrelated = Tensor::new(vec![5.0, 6.0], vec![1, 2]).unwrap();
        crate::builtins::common::test_support::with_test_provider(|provider| {
            let resident =
                crate::builtins::common::gpu_helpers::upload_tensor(provider, &unrelated)
                    .expect("upload unrelated property");
            let mut model =
                ObjectInstance::new(crate::builtins::control::tf_model::TF_CLASS.to_string());
            model.replace_property_unbounded("Numerator".into(), Value::Num(2.0));
            model.replace_property_unbounded("Denominator".into(), Value::Num(1.0));
            model.replace_property_unbounded("Ts".into(), Value::Num(0.0));
            model.replace_property_unbounded("InputDelay".into(), Value::Num(0.0));
            model.replace_property_unbounded("OutputDelay".into(), Value::Num(0.0));
            model.replace_property_unbounded("Unrelated".into(), Value::GpuTensor(resident));
            provider.reset_telemetry();
            let Value::Tensor(zeros) =
                block_on(zero_builtin(Value::Object(model))).expect("static gain")
            else {
                panic!("expected real empty column");
            };
            assert_eq!(zeros.shape(), &[0, 1]);
            assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
        });
    }

    #[test]
    fn zero_returns_roots_of_numerator() {
        let sys = block_on(crate::call_builtin_async(
            "tf",
            &[
                Value::Tensor(Tensor::new(vec![1.0, 3.0, 2.0], vec![1, 3]).unwrap()),
                Value::Tensor(Tensor::new(vec![1.0, 4.0], vec![1, 2]).unwrap()),
            ],
        ))
        .expect("tf");
        let Value::Tensor(zeros) = sys
            .with_owned(|sys| block_on(zero_builtin(sys)))
            .expect("zero")
        else {
            panic!("expected real zeros");
        };
        assert_eq!(zeros.shape(), &[2, 1]);
        assert!(zeros
            .materialize_f64()
            .iter()
            .any(|z| (*z + 1.0).abs() < 1.0e-8));
        assert!(zeros
            .materialize_f64()
            .iter()
            .any(|z| (*z + 2.0).abs() < 1.0e-8));
    }

    #[test]
    fn zero_returns_complex_conjugate_roots() {
        let sys = block_on(crate::call_builtin_async(
            "tf",
            &[
                Value::Tensor(Tensor::new(vec![1.0, 0.0, 1.0], vec![1, 3]).unwrap()),
                Value::Tensor(Tensor::new(vec![1.0, 1.0], vec![1, 2]).unwrap()),
            ],
        ))
        .expect("tf");
        let Value::ComplexTensor(zeros) = sys
            .with_owned(|sys| block_on(zero_builtin(sys)))
            .expect("zero")
        else {
            panic!("expected complex zeros");
        };
        assert_eq!(zeros.shape(), &[2, 1]);
        assert!(zeros
            .materialize_f64()
            .iter()
            .all(|(re, _)| re.abs() < 1.0e-8));
        assert!(zeros
            .materialize_f64()
            .iter()
            .any(|(_, im)| (*im - 1.0).abs() < 1.0e-8));
        assert!(zeros
            .materialize_f64()
            .iter()
            .any(|(_, im)| (*im + 1.0).abs() < 1.0e-8));
    }

    #[test]
    fn zero_static_gain_returns_empty_column() {
        let sys = block_on(crate::call_builtin_async(
            "tf",
            &[Value::Num(5.0), Value::Num(2.0)],
        ))
        .expect("tf");
        let Value::Tensor(zeros) = sys
            .with_owned(|sys| block_on(zero_builtin(sys)))
            .expect("zero")
        else {
            panic!("expected real empty column");
        };
        assert_eq!(zeros.shape(), &[0, 1]);
        assert!(zeros.materialize_f64().is_empty());
    }
}
