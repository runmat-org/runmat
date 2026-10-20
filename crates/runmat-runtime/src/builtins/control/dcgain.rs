//! DC gain for SISO transfer-function models.

use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::control::tf_model::{output_complex_scalar, TfModel};
use crate::BuiltinResult;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::control::dcgain")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "dcgain",
    op_kind: GpuOpKind::Custom("control-dc-gain"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "dcgain evaluates host-side transfer-function metadata.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::control::dcgain")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "dcgain",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "dcgain is scalar model analysis and is not fused.",
};

#[runtime_builtin(
    name = "dcgain",
    binding_variant = "default",
    builtin_path = "crate::builtins::control::dcgain"
)]
async fn dcgain_builtin(sys: Value) -> BuiltinResult<Value> {
    let model = TfModel::from_value_async(sys, "dcgain").await?;
    Ok(output_complex_scalar(model.dc_gain()?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::control::tf_model::{SS_CLASS, TF_CLASS};
    use futures::executor::block_on;
    use runmat_builtins::{BuiltinIntegerInputAvailability, DCGAIN_INTEGER_CAPABILITIES};
    use runmat_value::{ComplexTensor, IntValue, IntegerStorage, ObjectInstance, Tensor};

    fn raw_tf(numerator: Value, denominator: Value) -> ObjectInstance {
        let mut model = ObjectInstance::new(TF_CLASS.to_string());
        model.replace_property_unbounded("Numerator".into(), numerator);
        model.replace_property_unbounded("Denominator".into(), denominator);
        model.replace_property_unbounded("Ts".into(), Value::Num(0.0));
        model.replace_property_unbounded("InputDelay".into(), Value::Num(0.0));
        model.replace_property_unbounded("OutputDelay".into(), Value::Num(0.0));
        model
    }

    #[test]
    fn nested_resident_coefficients_gather_through_owner_after_preflight() {
        let denominator = Tensor::new(vec![1.0, 3.0], vec![1, 2]).unwrap();
        crate::builtins::common::test_support::with_test_provider(|provider| {
            let resident =
                crate::builtins::common::gpu_helpers::upload_tensor(provider, &denominator)
                    .expect("upload denominator");
            let valid = raw_tf(Value::Num(2.0), Value::GpuTensor(resident.clone()));
            provider.reset_telemetry();
            let Value::Num(gain) =
                block_on(dcgain_builtin(Value::Object(valid))).expect("resident tf")
            else {
                panic!("real scalar gain");
            };
            assert!((gain - 2.0 / 3.0).abs() < 1e-12);
            assert!(provider.telemetry_snapshot().download_bytes > 0);

            let mut unsupported = ObjectInstance::new(SS_CLASS.to_string());
            unsupported.replace_property_unbounded(
                "Denominator".into(),
                Value::GpuTensor(resident.clone()),
            );
            provider.reset_telemetry();
            let error = block_on(dcgain_builtin(Value::Object(unsupported)))
                .expect_err("unsupported model");
            assert_eq!(error.identifier(), Some("RunMat:dcgain:UnsupportedModel"));
            assert_eq!(provider.telemetry_snapshot().download_bytes, 0);

            let mut incomplete = raw_tf(Value::Num(2.0), Value::GpuTensor(resident));
            crate::builtins::common::test_support::remove_object_property_unbounded(
                &mut incomplete,
                "Ts",
            );
            provider.reset_telemetry();
            let error =
                block_on(dcgain_builtin(Value::Object(incomplete))).expect_err("missing property");
            assert_eq!(error.identifier(), Some("RunMat:dcgain:InvalidModel"));
            assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
        });
    }

    #[test]
    fn unrelated_resident_property_is_not_gathered() {
        let tensor = Tensor::new(vec![3.0, 4.0], vec![1, 2]).unwrap();
        crate::builtins::common::test_support::with_test_provider(|provider| {
            let resident = crate::builtins::common::gpu_helpers::upload_tensor(provider, &tensor)
                .expect("upload");
            let mut model = raw_tf(Value::Num(2.0), Value::Num(3.0));
            model.replace_property_unbounded("Unrelated".into(), Value::GpuTensor(resident));
            provider.reset_telemetry();
            let Value::Num(gain) = block_on(dcgain_builtin(Value::Object(model))).expect("gain")
            else {
                panic!("real scalar gain");
            };
            assert!((gain - 2.0 / 3.0).abs() < 1e-12);
            assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
        });
    }

    #[test]
    fn tiny_nonzero_imaginary_gain_is_preserved() {
        let numerator =
            ComplexTensor::new(vec![(1.0, 1.0e-13)], vec![1, 1]).expect("complex coefficient");
        let model = raw_tf(Value::ComplexTensor(numerator), Value::Num(1.0));
        let Value::Complex(real, imaginary) =
            block_on(dcgain_builtin(Value::Object(model))).expect("complex gain")
        else {
            panic!("complex scalar gain");
        };
        assert_eq!(real, 1.0);
        assert_eq!(imaginary, 1.0e-13);
    }

    #[test]
    fn continuous_dcgain_evaluates_at_zero() {
        let sys = block_on(crate::call_builtin_async(
            "tf",
            &[
                Value::Num(2.0),
                Value::Tensor(Tensor::new(vec![1.0, 3.0], vec![1, 2]).unwrap()),
            ],
        ))
        .expect("tf");
        let Value::Num(gain) = sys
            .with_owned(|sys| block_on(dcgain_builtin(sys)))
            .expect("dcgain")
        else {
            panic!("expected scalar gain");
        };
        assert!((gain - 2.0 / 3.0).abs() < 1.0e-12);
    }

    #[test]
    fn dcgain_returns_infinity_for_continuous_and_discrete_integrators() {
        for (numerator, denominator, sample_time, expected_negative) in [
            (
                Value::Num(2.0),
                Value::Tensor(Tensor::new(vec![1.0, 0.0], vec![1, 2]).unwrap()),
                None,
                false,
            ),
            (
                Value::Num(-2.0),
                Value::Tensor(Tensor::new(vec![1.0, 0.0], vec![1, 2]).unwrap()),
                None,
                true,
            ),
            (
                Value::Num(3.0),
                Value::Tensor(Tensor::new(vec![1.0, -1.0], vec![1, 2]).unwrap()),
                Some(0.1),
                false,
            ),
        ] {
            let mut arguments = vec![numerator, denominator];
            if let Some(sample_time) = sample_time {
                arguments.push(Value::Num(sample_time));
            }
            let sys = block_on(crate::call_builtin_async("tf", &arguments)).expect("tf");
            let Value::Num(gain) = sys
                .with_owned(|sys| block_on(dcgain_builtin(sys)))
                .expect("dcgain")
            else {
                panic!("expected real infinite gain");
            };
            assert!(gain.is_infinite());
            assert_eq!(gain.is_sign_negative(), expected_negative);
        }
    }

    #[test]
    fn dcgain_rejects_integer_sys_without_conflating_integer_coefficients() {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        for sys in [
            Value::Int(IntValue::I64(i64::MAX)),
            Value::Tensor(
                Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX]), vec![1, 1]).unwrap(),
            ),
        ] {
            let error = block_on(dcgain_builtin(sys)).expect_err("integer sys rejection");
            assert_eq!(error.identifier(), Some("RunMat:dcgain:InvalidModel"));
        }

        let sys = block_on(crate::call_builtin_async(
            "tf",
            &[
                Value::Int(IntValue::U8(2)),
                Value::Tensor(
                    Tensor::new_integer(IntegerStorage::I16(vec![1, 3]), vec![1, 2]).unwrap(),
                ),
            ],
        ))
        .expect("tf with integer coefficients");
        let Value::Num(gain) = sys
            .with_owned(|sys| block_on(dcgain_builtin(sys)))
            .expect("dcgain model object")
        else {
            panic!("real scalar gain");
        };
        assert!((gain - 2.0 / 3.0).abs() < 1.0e-12);
    }

    #[test]
    fn dcgain_resident_nonobject_rejects_before_provider_access() {
        let allocation = runmat_accelerate_api::test_support::TestAllocationOwner::new()
            .allocate_zeroed(
                vec![1, 1],
                runmat_accelerate_api::NumericElementType::U64,
                runmat_accelerate_api::GpuTensorStorage::Real,
            )
            .unwrap();
        assert!(
            crate::builtins::common::gpu_helpers::exact_provider_for_handle(&allocation.handle)
                .is_none()
        );
        let handle = allocation.handle.clone();
        let error = block_on(dcgain_builtin(Value::GpuTensor(handle)))
            .expect_err("resident nonobject rejection");
        assert_eq!(error.identifier(), Some("RunMat:dcgain:InvalidModel"));
        assert_eq!(allocation.probe.retirements(), 0);
        allocation.handle.release().unwrap();
        assert_eq!(allocation.probe.retirements(), 1);
    }

    #[test]
    fn dcgain_integer_capability_is_an_object_boundary_rejection() {
        assert_eq!(DCGAIN_INTEGER_CAPABILITIES.len(), 1);
        assert!(DCGAIN_INTEGER_CAPABILITIES[0].inputs[0].classes.is_empty());
        assert_eq!(
            DCGAIN_INTEGER_CAPABILITIES[0].inputs[0].availability,
            BuiltinIntegerInputAvailability::Rejected
        );
    }
}
