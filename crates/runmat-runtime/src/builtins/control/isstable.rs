//! Stability test for SISO transfer-function models.

use runmat_builtins::{ISSTABLE_ERROR_INVALID_MODEL, ISSTABLE_ERROR_UNSUPPORTED_MODEL};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::control::tf_model::{TfModel, TF_CLASS};
use crate::BuiltinResult;

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
    binding_variant = "default",
    builtin_path = "crate::builtins::control::isstable"
)]
async fn isstable_builtin(sys: Value) -> BuiltinResult<Value> {
    // Reject invalid top-level model classes before recursively gathering any
    // provider-resident values from an object's properties.
    match &sys {
        Value::Object(object) if object.is_class(TF_CLASS) => {}
        Value::Object(object) => {
            return Err(crate::build_runtime_error(format!(
                "isstable: unsupported model class '{}'; only SISO tf objects are supported",
                object.class_name()
            ))
            .with_builtin("isstable")
            .with_identifier(
                ISSTABLE_ERROR_UNSUPPORTED_MODEL
                    .identifier
                    .expect("isstable unsupported-model descriptor identifier"),
            )
            .build()
            .into());
        }
        _ => {
            return Err(
                crate::build_runtime_error("isstable: input must be a SISO tf object")
                    .with_builtin("isstable")
                    .with_identifier(
                        ISSTABLE_ERROR_INVALID_MODEL
                            .identifier
                            .expect("isstable invalid-model descriptor identifier"),
                    )
                    .build()
                    .into(),
            );
        }
    }
    let model = TfModel::from_value_async(sys, "isstable").await?;
    Ok(Value::Bool(model.is_stable()?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_value::{IntValue, IntegerStorage};
    use runmat_value::{ObjectInstance, Tensor};

    fn raw_tf(numerator: Value, denominator: Value) -> ObjectInstance {
        let mut object = ObjectInstance::new(TF_CLASS.to_string());
        object.replace_property_unbounded("Numerator".into(), numerator);
        object.replace_property_unbounded("Denominator".into(), denominator);
        object.replace_property_unbounded("Ts".into(), Value::Num(0.0));
        object.replace_property_unbounded("InputDelay".into(), Value::Num(0.0));
        object.replace_property_unbounded("OutputDelay".into(), Value::Num(0.0));
        object
    }

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
            sys.with_owned(|sys| block_on(isstable_builtin(sys)))
                .expect("isstable"),
            Value::Bool(true)
        );
    }

    #[test]
    fn unstable_marginal_discrete_and_static_models_follow_pole_region() {
        fn model(denominator: Vec<f64>, sample_time: Option<f64>) -> crate::execution::RootedValue {
            let mut arguments = vec![
                Value::Num(1.0),
                Value::Tensor(
                    Tensor::new(denominator.clone(), vec![1, denominator.len()]).unwrap(),
                ),
            ];
            if let Some(sample_time) = sample_time {
                arguments.push(Value::Num(sample_time));
            }
            block_on(crate::call_builtin_async("tf", &arguments)).expect("tf")
        }

        for (sys, expected) in [
            (model(vec![1.0, -1.0], None), false),
            (model(vec![1.0, 0.0], None), false),
            (model(vec![1.0, -0.5], Some(1.0)), true),
            (model(vec![1.0, -1.0], Some(1.0)), false),
            (model(vec![1.0], None), true),
        ] {
            assert_eq!(
                sys.with_owned(|sys| block_on(isstable_builtin(sys)))
                    .unwrap(),
                Value::Bool(expected)
            );
        }
    }

    #[test]
    fn near_boundary_pole_obeys_current_control_model_tolerance() {
        let sys = block_on(crate::call_builtin_async(
            "tf",
            &[
                Value::Num(1.0),
                Value::Tensor(Tensor::new(vec![1.0, 1.0e-13], vec![1, 2]).unwrap()),
            ],
        ))
        .expect("tf");
        assert_eq!(
            sys.with_owned(|sys| block_on(isstable_builtin(sys)))
                .unwrap(),
            Value::Bool(false)
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

    #[test]
    fn resident_float_rejects_as_invalid_model_before_gather() {
        let tensor = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
        crate::builtins::common::test_support::with_test_provider(|provider| {
            let handle = crate::builtins::common::gpu_helpers::upload_tensor(provider, &tensor)
                .expect("upload float");
            provider.reset_telemetry();
            let error = block_on(isstable_builtin(Value::GpuTensor(handle)))
                .expect_err("resident numeric input is not a system model");
            assert_eq!(
                error.identifier.as_deref(),
                Some("RunMat:isstable:InvalidModel")
            );
            assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
        });
    }

    #[test]
    fn nested_resident_coefficients_gather_only_for_valid_tf_model() {
        let denominator = Tensor::new(vec![1.0, 1.0], vec![1, 2]).unwrap();
        crate::builtins::common::test_support::with_test_provider(|provider| {
            let resident =
                crate::builtins::common::gpu_helpers::upload_tensor(provider, &denominator)
                    .expect("upload denominator");
            let mut valid = ObjectInstance::new(TF_CLASS.to_string());
            valid.replace_property_unbounded("Numerator".into(), Value::Num(1.0));
            valid.replace_property_unbounded(
                "Denominator".into(),
                Value::GpuTensor(resident.clone()),
            );
            valid.replace_property_unbounded("Ts".into(), Value::Num(0.0));
            valid.replace_property_unbounded("InputDelay".into(), Value::Num(0.0));
            valid.replace_property_unbounded("OutputDelay".into(), Value::Num(0.0));
            provider.reset_telemetry();
            assert_eq!(
                block_on(isstable_builtin(Value::Object(valid))).unwrap(),
                Value::Bool(true)
            );
            assert!(provider.telemetry_snapshot().download_bytes > 0);

            let mut unsupported = ObjectInstance::new("ss");
            unsupported.replace_property_unbounded(
                "Denominator".into(),
                Value::GpuTensor(resident.clone()),
            );
            provider.reset_telemetry();
            let error = block_on(isstable_builtin(Value::Object(unsupported)))
                .expect_err("unsupported object class");
            assert_eq!(
                error.identifier.as_deref(),
                Some("RunMat:isstable:UnsupportedModel")
            );
            assert_eq!(provider.telemetry_snapshot().download_bytes, 0);

            // Exercise the shared parser directly: other callers need the
            // same class-before-gather behavior as isstable's entrypoint.
            let mut unsupported = ObjectInstance::new("ss");
            unsupported.replace_property_unbounded(
                "Denominator".into(),
                Value::GpuTensor(resident.clone()),
            );
            provider.reset_telemetry();
            let error = block_on(TfModel::from_value_async(
                Value::Object(unsupported),
                "isstable",
            ))
            .expect_err("shared parser rejects unsupported class");
            assert_eq!(
                error.identifier.as_deref(),
                Some("RunMat:isstable:UnsupportedModel")
            );
            assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
        });
    }

    #[test]
    fn unrelated_resident_property_is_not_gathered() {
        let tensor = Tensor::new(vec![3.0, 4.0], vec![1, 2]).unwrap();
        crate::builtins::common::test_support::with_test_provider(|provider| {
            let resident =
                crate::builtins::common::gpu_helpers::upload_tensor(provider, &tensor).unwrap();
            let mut model = raw_tf(Value::Num(1.0), Value::Num(1.0));
            model.replace_property_unbounded("Unrelated".into(), Value::GpuTensor(resident));
            provider.reset_telemetry();
            assert_eq!(
                block_on(isstable_builtin(Value::Object(model))).unwrap(),
                Value::Bool(true)
            );
            assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
        });
    }

    #[test]
    fn missing_model_property_rejects_before_required_resident_gather() {
        let tensor = Tensor::new(vec![1.0, 1.0], vec![1, 2]).unwrap();
        crate::builtins::common::test_support::with_test_provider(|provider| {
            let resident =
                crate::builtins::common::gpu_helpers::upload_tensor(provider, &tensor).unwrap();
            let mut model = raw_tf(Value::Num(1.0), Value::GpuTensor(resident));
            crate::builtins::common::test_support::remove_object_property_unbounded(
                &mut model, "Ts",
            );
            provider.reset_telemetry();
            let error = block_on(isstable_builtin(Value::Object(model)))
                .expect_err("missing required model property");
            assert_eq!(
                error.identifier.as_deref(),
                Some("RunMat:isstable:InvalidModel")
            );
            assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
        });
    }

    #[test]
    fn forged_nonfinite_coefficients_are_invalid_models() {
        for (numerator, denominator) in [
            (Value::Num(f64::NAN), Value::Num(1.0)),
            (Value::Num(1.0), Value::Num(f64::NAN)),
            (Value::Num(1.0), Value::Num(f64::INFINITY)),
            (Value::Num(1.0), Value::Num(f64::NEG_INFINITY)),
        ] {
            let error = block_on(isstable_builtin(Value::Object(raw_tf(
                numerator,
                denominator,
            ))))
            .expect_err("nonfinite coefficient is not a valid tf model");
            assert_eq!(
                error.identifier.as_deref(),
                Some("RunMat:isstable:InvalidModel")
            );
        }
    }

    #[test]
    fn shared_parser_errors_remain_declared_for_malformed_models() {
        let mut bad_scalar = raw_tf(Value::Num(1.0), Value::Num(1.0));
        bad_scalar.replace_property_unbounded("Ts".into(), Value::String("invalid".into()));
        let error = block_on(isstable_builtin(Value::Object(bad_scalar))).unwrap_err();
        assert_eq!(
            error.identifier.as_deref(),
            Some("RunMat:isstable:InvalidArgument")
        );

        let mut bad_variable = raw_tf(Value::Num(1.0), Value::Num(1.0));
        bad_variable.replace_property_unbounded("Variable".into(), Value::String("invalid".into()));
        let error = block_on(isstable_builtin(Value::Object(bad_variable))).unwrap_err();
        assert_eq!(
            error.identifier.as_deref(),
            Some("RunMat:tf:InvalidVariable")
        );

        let mut bad_pairing = raw_tf(Value::Num(1.0), Value::Num(1.0));
        bad_pairing.replace_property_unbounded("Variable".into(), Value::String("z".into()));
        let error = block_on(isstable_builtin(Value::Object(bad_pairing))).unwrap_err();
        assert_eq!(
            error.identifier.as_deref(),
            Some("RunMat:isstable:InvalidModel")
        );
    }
}
