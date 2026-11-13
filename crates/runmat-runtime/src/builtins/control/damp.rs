//! Damping ratio and natural-frequency analysis for control models.

use num_complex::Complex64;
use runmat_builtins::{
    BuiltinErrorDescriptor, DAMP_ERROR_INTERNAL, DAMP_ERROR_INVALID_MODEL, DAMP_ERROR_OUTPUT_COUNT,
    DAMP_ERROR_UNSUPPORTED_MODEL,
};
use runmat_macros::runtime_builtin;
use runmat_value::{Tensor, Value};

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::control::tf_model::{
    control_error, output_complex_column, ss_poles_from_object_async, TfModel, TfSampleDomain, EPS,
    SS_CLASS, TF_CLASS,
};
use crate::BuiltinResult;

const BUILTIN_NAME: &str = "damp";

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::control::damp")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "damp",
    op_kind: GpuOpKind::Custom("control-damping"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "damp computes pole-derived control metadata on the host.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::control::damp")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "damp",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: true,
    notes: "damp is model analysis and is not fused.",
};

#[runtime_builtin(
    name = "damp",
    binding_variant = "default",
    builtin_path = "crate::builtins::control::damp"
)]
async fn damp_builtin(sys: Value) -> BuiltinResult<Value> {
    let output_count = crate::output_count::current_output_count();
    if output_count.is_some_and(|count| count > 3) {
        return Err(damp_error(
            &DAMP_ERROR_OUTPUT_COUNT,
            "damp: at most three outputs are supported",
        ));
    }
    let eval = DampingEval::from_value(sys).await?;
    match output_count {
        Some(0) => Ok(Value::output_list(Vec::new())),
        Some(1) => Ok(Value::output_list(vec![eval.wn_value()?])),
        Some(2) => Ok(Value::output_list(vec![
            eval.wn_value()?,
            eval.zeta_value()?,
        ])),
        Some(3) => Ok(Value::output_list(vec![
            eval.wn_value()?,
            eval.zeta_value()?,
            eval.poles_value()?,
        ])),
        Some(_) => unreachable!("output count validated before model analysis"),
        None => eval.wn_value(),
    }
}

#[derive(Clone, Debug)]
struct DampingEval {
    modes: Vec<DampingMode>,
}

#[derive(Clone, Copy, Debug)]
struct DampingMode {
    pole: Complex64,
    wn: f64,
    zeta: f64,
}

impl DampingEval {
    async fn from_value(value: Value) -> BuiltinResult<Self> {
        match value {
            Value::Object(object) if object.is_class(TF_CLASS) => {
                let model = TfModel::from_value_async(Value::Object(object), BUILTIN_NAME).await?;
                if model.sample_domain() == TfSampleDomain::DiscreteUnspecified {
                    return Err(damp_error(&DAMP_ERROR_UNSUPPORTED_MODEL,
                        "damp: discrete model requires a known positive sample time for physical natural frequency"));
                }
                Ok(Self::from_poles(model.poles()?, model.sample_time()))
            }
            Value::Object(object) if object.is_class(SS_CLASS) => {
                let (poles, sample_time) =
                    ss_poles_from_object_async(&object, BUILTIN_NAME).await?;
                Ok(Self::from_poles(poles, sample_time))
            }
            Value::Object(object) => Err(damp_error(
                &DAMP_ERROR_UNSUPPORTED_MODEL,
                format!(
                    "damp: unsupported model class '{}'; supported classes are tf and ss",
                    object.class_name()
                ),
            )),
            other => Err(damp_error(
                &DAMP_ERROR_INVALID_MODEL,
                format!("damp: expected a tf or ss object, got {other:?}"),
            )),
        }
    }

    fn from_poles(poles: Vec<Complex64>, sample_time: f64) -> Self {
        let mut modes = poles
            .into_iter()
            .map(|pole| {
                let is_zero = pole.re == 0.0 && pole.im == 0.0;
                let is_one = pole.re == 1.0 && pole.im == 0.0;
                let is_infinite = pole.re.is_infinite() || pole.im.is_infinite();
                if sample_time > 0.0 {
                    if is_zero {
                        return DampingMode {
                            pole,
                            wn: f64::INFINITY,
                            zeta: 1.0,
                        };
                    }
                    if is_one {
                        return DampingMode {
                            pole,
                            wn: 0.0,
                            zeta: -1.0,
                        };
                    }
                    if is_infinite {
                        return DampingMode {
                            pole,
                            wn: f64::INFINITY,
                            zeta: -1.0,
                        };
                    }
                } else if is_zero {
                    return DampingMode {
                        pole,
                        wn: 0.0,
                        zeta: -1.0,
                    };
                } else if is_infinite {
                    return DampingMode {
                        pole,
                        wn: f64::INFINITY,
                        zeta: -1.0,
                    };
                }
                let equivalent = if sample_time > 0.0 {
                    pole.ln() / sample_time
                } else {
                    pole
                };
                let wn = equivalent.norm();
                let zeta = if wn <= EPS {
                    f64::NAN
                } else {
                    -equivalent.re / wn
                };
                DampingMode { pole, wn, zeta }
            })
            .collect::<Vec<_>>();
        modes.sort_by(compare_modes);
        Self { modes }
    }

    fn wn_value(&self) -> BuiltinResult<Value> {
        real_column(self.modes.iter().map(|mode| mode.wn).collect())
    }

    fn zeta_value(&self) -> BuiltinResult<Value> {
        real_column(self.modes.iter().map(|mode| mode.zeta).collect())
    }

    fn poles_value(&self) -> BuiltinResult<Value> {
        output_complex_column(
            self.modes.iter().map(|mode| mode.pole).collect(),
            BUILTIN_NAME,
        )
    }
}

fn compare_modes(lhs: &DampingMode, rhs: &DampingMode) -> std::cmp::Ordering {
    lhs.wn
        .total_cmp(&rhs.wn)
        .then_with(|| lhs.zeta.total_cmp(&rhs.zeta))
        .then_with(|| lhs.pole.re.total_cmp(&rhs.pole.re))
        .then_with(|| lhs.pole.im.total_cmp(&rhs.pole.im))
}

fn real_column(data: Vec<f64>) -> BuiltinResult<Value> {
    let rows = data.len();
    Tensor::new(data, vec![rows, 1])
        .map(Value::Tensor)
        .map_err(|err| {
            damp_error(
                &DAMP_ERROR_INTERNAL,
                format!("damp: failed to build output tensor: {err}"),
            )
        })
}

fn damp_error(
    descriptor: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
) -> crate::RuntimeError {
    control_error(
        BUILTIN_NAME,
        descriptor
            .identifier
            .expect("damp error descriptor identifier"),
        message,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_builtins::{
        BuiltinIntegerInputAvailability, DAMP_DESCRIPTOR, DAMP_INTEGER_CAPABILITIES,
    };

    fn tf(numerator: Value, denominator: Value, rest: &[Value]) -> crate::execution::RootedValue {
        let mut args = vec![numerator, denominator];
        args.extend_from_slice(rest);
        block_on(crate::call_builtin_async("tf", &args)).expect("tf")
    }

    fn ss(a: Value, b: Value, c: Value, d: Value, rest: &[Value]) -> crate::execution::RootedValue {
        let mut args = vec![a, b, c, d];
        args.extend_from_slice(rest);
        block_on(crate::call_builtin_async("ss", &args)).expect("ss")
    }

    fn damp_outputs(sys: Value, count: usize) -> Vec<Value> {
        let _guard =
            crate::output_count::push_output_count(Some(count)).expect("output-count scope");
        let Value::OutputList(outputs) = block_on(damp_builtin(sys)).expect("damp") else {
            panic!("expected output list");
        };
        crate::sequence::into_unbounded_output_values(outputs).expect("ordinary damp outputs")
    }

    fn tensor_data(value: &Value) -> Vec<f64> {
        let Value::Tensor(tensor) = value else {
            panic!("expected tensor, got {value:?}");
        };
        tensor.materialize_f64()
    }

    #[test]
    fn damp_descriptor_signatures_cover_multi_output_forms() {
        let labels: Vec<&str> = DAMP_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"wn = damp(sys)"));
        assert!(labels.contains(&"[wn, zeta] = damp(sys)"));
        assert!(labels.contains(&"[wn, zeta, p] = damp(sys)"));
    }

    #[test]
    fn damp_tf_second_order_returns_frequency_ratio_and_poles() {
        let sys = tf(
            Value::Num(1.0),
            Value::Tensor(Tensor::new(vec![100.0, 50.0, 400.0], vec![1, 3]).unwrap()),
            &[],
        );

        let outputs = sys.with_owned(|sys| damp_outputs(sys, 3));
        let wn = tensor_data(&outputs[0]);
        let zeta = tensor_data(&outputs[1]);
        assert_eq!(wn.len(), 2);
        assert_eq!(zeta.len(), 2);
        for value in wn {
            assert!((value - 2.0).abs() < 1.0e-8);
        }
        for value in zeta {
            assert!((value - 0.125).abs() < 1.0e-8);
        }
        match &outputs[2] {
            Value::ComplexTensor(poles) => {
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
            other => panic!("expected complex poles, got {other:?}"),
        }
    }

    #[test]
    fn damp_tf_first_order_returns_real_pole() {
        let sys = tf(
            Value::Num(1.0),
            Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap()),
            &[],
        );

        let outputs = sys.with_owned(|sys| damp_outputs(sys, 3));
        assert!((tensor_data(&outputs[0])[0] - 2.0).abs() < 1.0e-12);
        assert!((tensor_data(&outputs[1])[0] - 1.0).abs() < 1.0e-12);
        assert_eq!(tensor_data(&outputs[2]), &[-2.0]);
    }

    #[test]
    fn damp_discrete_tf_uses_log_equivalent_poles() {
        let sys = tf(
            Value::Num(1.0),
            Value::Tensor(Tensor::new(vec![1.0, -0.5], vec![1, 2]).unwrap()),
            &[Value::Num(0.1)],
        );

        let outputs = sys.with_owned(|sys| damp_outputs(sys, 3));
        assert!((tensor_data(&outputs[0])[0] - 6.931471805599453).abs() < 1.0e-10);
        assert!((tensor_data(&outputs[1])[0] - 1.0).abs() < 1.0e-12);
        assert!((tensor_data(&outputs[2])[0] - 0.5).abs() < 1.0e-12);
    }

    #[test]
    fn damp_discrete_zero_pole_reports_deadbeat_damping() {
        let report = DampingEval::from_poles(vec![Complex64::new(0.0, 0.0)], 0.1);
        assert_eq!(report.modes.len(), 1);
        assert!(report.modes[0].wn.is_infinite());
        assert_eq!(report.modes[0].zeta, 1.0);
    }

    #[test]
    fn damp_matches_r2023b_zero_and_infinite_pole_semantics() {
        fn mode_for(report: &DampingEval, pole: Complex64) -> &DampingMode {
            report
                .modes
                .iter()
                .find(|mode| mode.pole == pole)
                .expect("pole must retain its wn/zeta pair")
        }

        let zero = Complex64::new(0.0, 0.0);
        let one = Complex64::new(1.0, 0.0);
        let infinite = Complex64::new(f64::INFINITY, 0.0);
        let continuous = DampingEval::from_poles(vec![zero, infinite], 0.0);
        let continuous_zero = mode_for(&continuous, zero);
        assert_eq!((continuous_zero.wn, continuous_zero.zeta), (0.0, -1.0));
        let continuous_infinite = mode_for(&continuous, infinite);
        assert!(continuous_infinite.wn.is_infinite());
        assert_eq!(continuous_infinite.zeta, -1.0);

        let discrete = DampingEval::from_poles(vec![one, zero, infinite], 0.1);
        let discrete_one = mode_for(&discrete, one);
        assert_eq!((discrete_one.wn, discrete_one.zeta), (0.0, -1.0));
        let discrete_zero = mode_for(&discrete, zero);
        assert!(discrete_zero.wn.is_infinite());
        assert_eq!(discrete_zero.zeta, 1.0);
        let discrete_infinite = mode_for(&discrete, infinite);
        assert!(discrete_infinite.wn.is_infinite());
        assert_eq!(discrete_infinite.zeta, -1.0);
    }

    #[test]
    fn damp_pole_output_uses_shared_near_real_tolerance() {
        let near_real = DampingEval::from_poles(vec![Complex64::new(-2.0, EPS / 2.0)], 0.0);
        assert!(matches!(near_real.poles_value().unwrap(), Value::Tensor(_)));
        let complex = DampingEval::from_poles(vec![Complex64::new(-2.0, EPS * 2.0)], 0.0);
        assert!(matches!(
            complex.poles_value().unwrap(),
            Value::ComplexTensor(_)
        ));
    }

    #[test]
    fn damp_integer_capability_marks_object_input_inapplicable() {
        assert_eq!(DAMP_INTEGER_CAPABILITIES.len(), 1);
        assert!(DAMP_INTEGER_CAPABILITIES[0].inputs[0].classes.is_empty());
        assert_eq!(
            DAMP_INTEGER_CAPABILITIES[0].inputs[0].availability,
            BuiltinIntegerInputAvailability::Rejected
        );
        for value in [
            runmat_value::IntValue::I8(1),
            runmat_value::IntValue::I16(1),
            runmat_value::IntValue::I32(1),
            runmat_value::IntValue::I64(1),
            runmat_value::IntValue::U8(1),
            runmat_value::IntValue::U16(1),
            runmat_value::IntValue::U32(1),
            runmat_value::IntValue::U64(1),
        ] {
            let error = block_on(damp_builtin(Value::Int(value))).expect_err("object input only");
            assert_eq!(error.identifier(), Some("RunMat:damp:InvalidModel"));
        }
        let allocation = runmat_accelerate_api::test_support::TestAllocationOwner::new()
            .allocate_zeroed(
                vec![1, 1],
                runmat_accelerate_api::NumericElementType::F64,
                runmat_accelerate_api::GpuTensorStorage::Real,
            )
            .unwrap();
        assert!(
            crate::builtins::common::gpu_helpers::exact_provider_for_handle(&allocation.handle)
                .is_none()
        );
        let resident = Value::GpuTensor(allocation.handle.clone());
        let error = block_on(damp_builtin(resident)).expect_err("resident object input rejection");
        assert_eq!(error.identifier(), Some("RunMat:damp:InvalidModel"));
        assert_eq!(allocation.probe.retirements(), 0);
        allocation.handle.release().unwrap();
        assert_eq!(allocation.probe.retirements(), 1);
    }

    #[test]
    fn damp_rejects_registered_resident_input_without_provider_access() {
        crate::builtins::common::test_support::with_test_provider(|provider| {
            let handle = provider
                .upload(&runmat_accelerate_api::HostTensorView {
                    data: &[1.0],
                    shape: &[1, 1],
                })
                .unwrap();
            provider.reset_telemetry();
            let error = block_on(damp_builtin(Value::GpuTensor(handle.clone())))
                .expect_err("dynamic-system object required");
            assert_eq!(error.identifier(), Some("RunMat:damp:InvalidModel"));
            assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
            handle.allocation_lease().validate_live().unwrap();
            let physical = handle.allocation_lease().downgrade();
            handle.release().unwrap();
            assert!(physical.acquire().is_none());
        });
    }

    #[test]
    fn damp_ss_uses_state_matrix_eigenvalues() {
        let sys = ss(
            Value::Tensor(Tensor::new(vec![0.0, -4.0, 1.0, -0.5], vec![2, 2]).unwrap()),
            Value::Tensor(Tensor::new(vec![0.0, 1.0], vec![2, 1]).unwrap()),
            Value::Tensor(Tensor::new(vec![1.0, 0.0], vec![1, 2]).unwrap()),
            Value::Num(0.0),
            &[],
        );

        let outputs = sys.with_owned(|sys| damp_outputs(sys, 3));
        for value in tensor_data(&outputs[0]) {
            assert!((value - 2.0).abs() < 1.0e-8);
        }
        for value in tensor_data(&outputs[1]) {
            assert!((value - 0.125).abs() < 1.0e-8);
        }
        assert!(matches!(&outputs[2], Value::ComplexTensor(tensor) if tensor.shape() == [2, 1]));
    }

    #[test]
    fn damp_gathers_only_consumed_tf_properties() {
        crate::builtins::common::test_support::with_test_provider(|provider| {
            let model = tf(
                Value::Num(1.0),
                Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap()),
                &[],
            );
            model.with_owned(|value| {
                let mut object = match value {
                    Value::Object(object) => object,
                    _ => panic!("tf constructor must return an object"),
                };
                let denominator = provider
                    .upload(&runmat_accelerate_api::HostTensorView {
                        data: &[1.0, 2.0],
                        shape: &[1, 2],
                    })
                    .unwrap();
                let unused = provider
                    .upload(&runmat_accelerate_api::HostTensorView {
                        data: &[9.0],
                        shape: &[1, 1],
                    })
                    .unwrap();
                object.replace_property_unbounded(
                    "Denominator".into(),
                    Value::GpuTensor(denominator),
                );
                object.replace_property_unbounded("Unused".into(), Value::GpuTensor(unused));
                provider.reset_telemetry();

                let outputs = damp_outputs(Value::Object(object), 3);
                assert_eq!(tensor_data(&outputs[0]), vec![2.0]);
                assert_eq!(tensor_data(&outputs[1]), vec![1.0]);
                assert_eq!(tensor_data(&outputs[2]), vec![-2.0]);
                assert_eq!(provider.telemetry_snapshot().download_bytes, 2 * 8);
            });
        });
    }

    #[test]
    fn damp_gathers_only_consumed_ss_properties() {
        crate::builtins::common::test_support::with_test_provider(|provider| {
            let model = ss(
                Value::Tensor(Tensor::new(vec![-2.0], vec![1, 1]).unwrap()),
                Value::Num(1.0),
                Value::Num(1.0),
                Value::Num(0.0),
                &[],
            );
            model.with_owned(|value| {
                let mut object = match value {
                    Value::Object(object) => object,
                    _ => panic!("ss constructor must return an object"),
                };
                let a = provider
                    .upload(&runmat_accelerate_api::HostTensorView {
                        data: &[-2.0],
                        shape: &[1, 1],
                    })
                    .unwrap();
                let unused_b = provider
                    .upload(&runmat_accelerate_api::HostTensorView {
                        data: &[9.0],
                        shape: &[1, 1],
                    })
                    .unwrap();
                object.replace_property_unbounded("A".into(), Value::GpuTensor(a));
                object.replace_property_unbounded("B".into(), Value::GpuTensor(unused_b));
                provider.reset_telemetry();

                let outputs = damp_outputs(Value::Object(object), 3);
                assert_eq!(tensor_data(&outputs[0]), vec![2.0]);
                assert_eq!(tensor_data(&outputs[1]), vec![1.0]);
                assert_eq!(tensor_data(&outputs[2]), vec![-2.0]);
                assert_eq!(provider.telemetry_snapshot().download_bytes, 8);
            });
        });
    }

    #[test]
    fn damp_rejects_missing_ss_state_before_resident_sample_time_access() {
        crate::builtins::common::test_support::with_test_provider(|provider| {
            let mut object = runmat_value::ObjectInstance::new(SS_CLASS.to_string());
            let ts = provider
                .upload(&runmat_accelerate_api::HostTensorView {
                    data: &[0.1],
                    shape: &[1, 1],
                })
                .unwrap();
            object.replace_property_unbounded("Ts".into(), Value::GpuTensor(ts));
            provider.reset_telemetry();

            let error = block_on(damp_builtin(Value::Object(object))).expect_err("missing A");
            assert_eq!(error.identifier(), Some("RunMat:damp:InvalidModel"));
            assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
        });
    }

    #[test]
    fn damp_rejects_invalid_host_state_before_resident_sample_time_access() {
        crate::builtins::common::test_support::with_test_provider(|provider| {
            let mut object = runmat_value::ObjectInstance::new(SS_CLASS.to_string());
            object.replace_property_unbounded(
                "A".into(),
                Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap()),
            );
            let ts = provider
                .upload(&runmat_accelerate_api::HostTensorView {
                    data: &[0.1],
                    shape: &[1, 1],
                })
                .unwrap();
            object.replace_property_unbounded("Ts".into(), Value::GpuTensor(ts));
            provider.reset_telemetry();

            let error = block_on(damp_builtin(Value::Object(object)))
                .expect_err("non-square A must reject before sample-time gather");
            assert_eq!(error.identifier(), Some("RunMat:damp:InvalidModel"));
            assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
        });
    }

    #[test]
    fn damp_output_count_error_precedes_model_validation() {
        let _guard = crate::output_count::push_output_count(Some(4)).expect("output-count scope");
        let error = block_on(damp_builtin(Value::Num(1.0)))
            .expect_err("too many outputs must reject before model analysis");
        assert_eq!(error.identifier(), Some("RunMat:damp:OutputCount"));
    }

    #[test]
    fn damp_output_count_error_precedes_nested_provider_access() {
        crate::builtins::common::test_support::with_test_provider(|provider| {
            let model = tf(
                Value::Num(1.0),
                Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap()),
                &[],
            );
            model.with_owned(|value| {
                let mut object = match value {
                    Value::Object(object) => object,
                    _ => panic!("tf constructor must return an object"),
                };
                let denominator = provider
                    .upload(&runmat_accelerate_api::HostTensorView {
                        data: &[1.0, 2.0],
                        shape: &[1, 2],
                    })
                    .unwrap();
                object.replace_property_unbounded(
                    "Denominator".into(),
                    Value::GpuTensor(denominator),
                );
                provider.reset_telemetry();
                let _guard =
                    crate::output_count::push_output_count(Some(4)).expect("output-count scope");
                let error = block_on(damp_builtin(Value::Object(object)))
                    .expect_err("output count must reject before nested gather");
                assert_eq!(error.identifier(), Some("RunMat:damp:OutputCount"));
                assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
            });
        });
    }

    #[test]
    fn damp_rejects_more_than_three_outputs() {
        let sys = tf(
            Value::Num(1.0),
            Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap()),
            &[],
        );

        let _guard = crate::output_count::push_output_count(Some(4)).expect("output-count scope");
        let err = sys
            .with_owned(|sys| block_on(damp_builtin(sys)))
            .expect_err("too many outputs should fail");
        assert_eq!(err.identifier(), Some("RunMat:damp:OutputCount"));
    }
}
