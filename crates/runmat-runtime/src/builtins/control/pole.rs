//! Pole extraction for transfer-function and state-space control models.

use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::control::tf_model::{
    control_error, output_complex_column, ss_poles_from_object_async, TfModel, SS_CLASS, TF_CLASS,
};
use crate::BuiltinResult;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::control::pole")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "pole",
    op_kind: GpuOpKind::Custom("control-poles"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "pole computes roots or state-matrix eigenvalues from host-side model metadata.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::control::pole")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "pole",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "pole is model analysis and is not fused.",
};

#[runtime_builtin(
    name = "pole",
    binding_variant = "default",
    builtin_path = "crate::builtins::control::pole"
)]
async fn pole_builtin(sys: Value) -> BuiltinResult<Value> {
    let poles = match sys {
        Value::Object(object) if object.is_class(TF_CLASS) => {
            TfModel::from_value_async(Value::Object(object), "pole")
                .await?
                .poles()?
        }
        Value::Object(object) if object.is_class(SS_CLASS) => {
            ss_poles_from_object_async(&object, "pole").await?.0
        }
        Value::Object(object) => {
            return Err(control_error(
                "pole",
                "RunMat:pole:UnsupportedModel",
                format!(
                    "pole: unsupported model class '{}'; supported classes are tf and ss",
                    object.class_name()
                ),
            ));
        }
        other => {
            return Err(control_error(
                "pole",
                "RunMat:pole:InvalidModel",
                format!("pole: expected a tf or ss object, got {other:?}"),
            ));
        }
    };
    output_complex_column(poles, "pole")
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_value::{ObjectInstance, Tensor};

    #[test]
    fn state_space_preflights_properties_before_resident_download() {
        let a = Tensor::new(vec![-2.0], vec![1, 1]).unwrap();
        crate::builtins::common::test_support::with_test_provider(|provider| {
            let resident = crate::builtins::common::gpu_helpers::upload_tensor(provider, &a)
                .expect("upload state matrix");
            let mut incomplete = ObjectInstance::new(SS_CLASS.to_string());
            incomplete.replace_property_unbounded("A".into(), Value::GpuTensor(resident.clone()));
            provider.reset_telemetry();
            let error = block_on(pole_builtin(Value::Object(incomplete))).expect_err("missing Ts");
            assert_eq!(error.identifier(), Some("RunMat:pole:InvalidModel"));
            assert_eq!(provider.telemetry_snapshot().download_bytes, 0);

            let mut complete = ObjectInstance::new(SS_CLASS.to_string());
            complete.replace_property_unbounded("A".into(), Value::GpuTensor(resident));
            complete.replace_property_unbounded("Ts".into(), Value::Num(0.0));
            provider.reset_telemetry();
            let Value::Tensor(poles) =
                block_on(pole_builtin(Value::Object(complete))).expect("resident A")
            else {
                panic!("expected real poles");
            };
            assert_eq!(poles.shape(), &[1, 1]);
            assert!((poles.materialize_f64()[0] + 2.0).abs() < 1e-8);
            assert!(provider.telemetry_snapshot().download_bytes > 0);
        });
    }

    #[test]
    fn pole_returns_roots_of_denominator() {
        let sys = block_on(crate::call_builtin_async(
            "tf",
            &[
                Value::Num(1.0),
                Value::Tensor(Tensor::new(vec![1.0, 3.0, 2.0], vec![1, 3]).unwrap()),
            ],
        ))
        .expect("tf");
        let Value::Tensor(poles) = sys
            .with_owned(|sys| block_on(pole_builtin(sys)))
            .expect("pole")
        else {
            panic!("expected real poles");
        };
        assert_eq!(poles.shape(), &[2, 1]);
        assert!(poles
            .materialize_f64()
            .iter()
            .any(|p| (*p + 1.0).abs() < 1.0e-8));
        assert!(poles
            .materialize_f64()
            .iter()
            .any(|p| (*p + 2.0).abs() < 1.0e-8));
    }

    #[test]
    fn pole_returns_repeated_roots_of_denominator() {
        let sys = block_on(crate::call_builtin_async(
            "tf",
            &[
                Value::Num(1.0),
                Value::Tensor(Tensor::new(vec![1.0, 2.0, 1.0], vec![1, 3]).unwrap()),
            ],
        ))
        .expect("tf");
        let Value::Tensor(poles) = sys
            .with_owned(|sys| block_on(pole_builtin(sys)))
            .expect("pole")
        else {
            panic!("expected real poles");
        };
        assert_eq!(poles.shape(), &[2, 1]);
        assert!(poles
            .materialize_f64()
            .iter()
            .all(|p| (*p + 1.0).abs() < 1.0e-8));
    }

    #[test]
    fn pole_returns_complex_conjugate_roots() {
        let sys = block_on(crate::call_builtin_async(
            "tf",
            &[
                Value::Num(1.0),
                Value::Tensor(Tensor::new(vec![1.0, 0.0, 1.0], vec![1, 3]).unwrap()),
            ],
        ))
        .expect("tf");
        let Value::ComplexTensor(poles) = sys
            .with_owned(|sys| block_on(pole_builtin(sys)))
            .expect("pole")
        else {
            panic!("expected complex poles");
        };
        assert_eq!(poles.shape(), &[2, 1]);
        assert!(poles
            .materialize_f64()
            .iter()
            .all(|(re, _)| re.abs() < 1.0e-8));
        assert!(poles
            .materialize_f64()
            .iter()
            .any(|(_, im)| (*im - 1.0).abs() < 1.0e-8));
        assert!(poles
            .materialize_f64()
            .iter()
            .any(|(_, im)| (*im + 1.0).abs() < 1.0e-8));
    }

    #[test]
    fn pole_uses_state_matrix_eigenvalues_for_ss() {
        let sys = block_on(crate::call_builtin_async(
            "ss",
            &[
                Value::Tensor(Tensor::new(vec![0.0, -4.0, 1.0, -0.5], vec![2, 2]).unwrap()),
                Value::Tensor(Tensor::new(vec![0.0, 1.0], vec![2, 1]).unwrap()),
                Value::Tensor(Tensor::new(vec![1.0, 0.0], vec![1, 2]).unwrap()),
                Value::Num(0.0),
            ],
        ))
        .expect("ss");
        let Value::ComplexTensor(poles) = sys
            .with_owned(|sys| block_on(pole_builtin(sys)))
            .expect("pole")
        else {
            panic!("expected complex poles");
        };
        assert_eq!(poles.shape(), &[2, 1]);
        assert!(poles
            .materialize_f64()
            .iter()
            .all(|(re, _)| (*re + 0.25).abs() < 1.0e-8));
        assert!(poles
            .materialize_f64()
            .iter()
            .any(|(_, im)| (*im - 1.984313483298443).abs() < 1.0e-8));
        assert!(poles
            .materialize_f64()
            .iter()
            .any(|(_, im)| (*im + 1.984313483298443).abs() < 1.0e-8));
    }
}
