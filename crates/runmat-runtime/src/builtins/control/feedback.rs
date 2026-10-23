//! MATLAB-compatible SISO `feedback` interconnection for transfer functions.

use runmat_builtins::{
    BuiltinErrorDescriptor, BuiltinExtensionDescriptor, FEEDBACK_ERRORS, INTEGER_GAIN_EXTENSION,
    INTEGER_SIGN_EXTENSION, LOGICAL_NUMERIC_EXTENSION, RESIDENT_NUMERIC_EXTENSION,
    SCALAR_FORWARD_EXTENSION, SCALAR_SYSTEMS_EXTENSION, SINGLE_NUMERIC_EXTENSION,
    SINGLE_SYSTEM_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::control::tf_model::{
    preflight_model_operand, scalar_f64, two_models_ordered, TF_CLASS,
};
use crate::{BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "feedback";

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::control::feedback")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "feedback",
    op_kind: GpuOpKind::Custom("control-feedback-interconnection"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "SISO transfer-function interconnection runs on host-side metadata.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::control::feedback")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "feedback",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "feedback creates a transfer-function object and terminates numeric fusion chains.",
};

fn feedback_error(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = crate::build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runtime_builtin(
    name = "feedback",
    binding_variant = "default",
    builtin_path = "crate::builtins::control::feedback"
)]
async fn feedback_builtin(
    sys1: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    if rest.len() > 2 {
        return Err(feedback_error(
            "feedback: expected feedback(sys1), feedback(sys1, sys2), or feedback(sys1, sys2, sign)",
            &FEEDBACK_ERRORS[0],
        ));
    }
    ensure_extensions_enabled(&sys1, rest.view())?;
    let sys2 = rest.first().cloned().unwrap_or(Value::Num(1.0));
    preflight_model_operand(&sys1, BUILTIN_NAME)?;
    preflight_model_operand(&sys2, BUILTIN_NAME)?;
    let sign = match rest.get(1) {
        Some(value) => {
            let gathered = crate::dispatcher::gather_if_needed_async(value).await?;
            scalar_f64(&gathered, "sign", BUILTIN_NAME)?
        }
        None => -1.0,
    };
    if sign != -1.0 && sign != 1.0 {
        return Err(feedback_error(
            "feedback: sign must be -1 or +1",
            &FEEDBACK_ERRORS[2],
        ));
    }

    let (forward, feedback_path) = two_models_ordered(sys1, sys2, BUILTIN_NAME).await?;
    forward
        .feedback(&feedback_path, sign)?
        .to_value(BUILTIN_NAME)
}

fn ensure_extensions_enabled(
    sys1: &Value,
    rest: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<()> {
    if rest.is_empty() {
        ensure_extension(&SINGLE_SYSTEM_EXTENSION)?;
    }

    let sys2 = rest.first();
    let sys1_is_scalar = is_scalar_system_value(sys1);
    if sys1_is_scalar && sys2.is_some_and(is_scalar_system_value) {
        ensure_extension(&SCALAR_SYSTEMS_EXTENSION)?;
    }
    if sys1_is_scalar {
        ensure_extension(&SCALAR_FORWARD_EXTENSION)?;
    }

    if is_typed_integer_value(sys1) || sys2.is_some_and(is_typed_integer_value) {
        ensure_extension(&INTEGER_GAIN_EXTENSION)?;
    }
    if rest.get(1).is_some_and(is_typed_integer_value) {
        ensure_extension(&INTEGER_SIGN_EXTENSION)?;
    }
    if is_logical_value(sys1) || rest.iter().any(is_logical_value) {
        ensure_extension(&LOGICAL_NUMERIC_EXTENSION)?;
    }
    if is_resident_value(sys1)
        || rest.iter().any(is_resident_value)
        || std::iter::once(sys1)
            .chain(rest.iter())
            .any(has_consumed_resident_model_property)
    {
        ensure_extension(&RESIDENT_NUMERIC_EXTENSION)?;
    }
    if is_single_numeric_value(sys1) || rest.iter().any(is_single_numeric_value) {
        ensure_extension(&SINGLE_NUMERIC_EXTENSION)?;
    }
    Ok(())
}

fn has_consumed_resident_model_property(value: &Value) -> bool {
    let Value::Object(object) = value else {
        return false;
    };
    object.is_class(TF_CLASS)
        && [
            "Numerator",
            "Denominator",
            "Ts",
            "InputDelay",
            "OutputDelay",
            "Variable",
        ]
        .into_iter()
        .filter_map(|name| object.properties().get(name))
        .any(is_resident_value)
}

fn ensure_extension(extension: &'static BuiltinExtensionDescriptor) -> BuiltinResult<()> {
    crate::compatibility::ensure_builtin_extension_enabled(extension, BUILTIN_NAME)
}

fn is_scalar_system_value(value: &Value) -> bool {
    match value {
        Value::Num(_) | Value::Int(_) | Value::Bool(_) | Value::Complex(_, _) => true,
        Value::Tensor(tensor) => tensor.len() == 1,
        Value::ComplexTensor(tensor) => tensor.len() == 1,
        Value::LogicalArray(logical) => logical.data.len() == 1,
        Value::GpuTensor(handle) => handle.shape().iter().all(|&dimension| dimension == 1),
        _ => false,
    }
}

fn is_typed_integer_value(value: &Value) -> bool {
    matches!(value, Value::Int(_))
        || matches!(value, Value::Tensor(tensor) if tensor.integer_storage().is_some())
        || matches!(value, Value::ComplexTensor(tensor) if tensor.integer_storage().is_some())
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_integer_type(handle).is_some())
}

fn is_single_numeric_value(value: &Value) -> bool {
    matches!(value, Value::Tensor(tensor) if tensor.numeric_dtype() == runmat_value::NumericDType::F32)
        || matches!(value, Value::ComplexTensor(tensor) if tensor.numeric_dtype() == runmat_value::NumericDType::F32)
}

fn is_logical_value(value: &Value) -> bool {
    matches!(value, Value::Bool(_) | Value::LogicalArray(_))
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_is_logical(handle))
}

fn is_resident_value(value: &Value) -> bool {
    matches!(value, Value::GpuTensor(_))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::control::tf_model::{SS_CLASS, TF_CLASS};
    use futures::executor::block_on;
    use runmat_accelerate_api::{HostIntegerDataView, HostIntegerTensorView};
    use runmat_builtins::{BuiltinIntegerComputationDomain, FEEDBACK_INTEGER_CAPABILITIES};
    use runmat_value::{
        ComplexTensor, IntValue, IntegerComplexStorage, IntegerStorage, LogicalArray, NumericDType,
        ObjectInstance, Tensor,
    };

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
    fn ordered_model_parser_gathers_only_valid_required_resident_coefficients() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let denominator = Tensor::new(vec![1.0, 3.0], vec![1, 2]).unwrap();
        crate::builtins::common::test_support::with_test_provider(|provider| {
            let resident =
                crate::builtins::common::gpu_helpers::upload_tensor(provider, &denominator)
                    .expect("upload denominator");
            let valid = raw_tf(Value::Num(2.0), Value::GpuTensor(resident.clone()));
            provider.reset_telemetry();
            let result = block_on(feedback_builtin(
                Value::Object(valid),
                crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(1.0)]),
            ))
            .expect("resident tf");
            assert_eq!(coeff(&result, "Denominator"), vec![1.0, 5.0]);
            assert!(provider.telemetry_snapshot().download_bytes > 0);

            let mut unsupported = ObjectInstance::new(SS_CLASS.to_string());
            unsupported.replace_property_unbounded(
                "Denominator".into(),
                Value::GpuTensor(resident.clone()),
            );
            provider.reset_telemetry();
            let error = block_on(feedback_builtin(
                Value::Object(unsupported),
                crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(1.0)]),
            ))
            .expect_err("unsupported object");
            assert_eq!(error.identifier(), Some("RunMat:feedback:UnsupportedModel"));
            assert_eq!(provider.telemetry_snapshot().download_bytes, 0);

            let mut incomplete = raw_tf(Value::Num(2.0), Value::GpuTensor(resident.clone()));
            crate::builtins::common::test_support::remove_object_property_unbounded(
                &mut incomplete,
                "Ts",
            );
            provider.reset_telemetry();
            let error = block_on(feedback_builtin(
                Value::Object(incomplete),
                crate::builtins::common::test_support::builtin_arguments(vec![Value::GpuTensor(
                    resident,
                )]),
            ))
            .expect_err("missing property before second operand gather");
            assert_eq!(error.identifier(), Some("RunMat:feedback:InvalidModel"));
            assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
        });
    }

    #[test]
    fn nested_resident_coefficients_are_gated_before_provider_access() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let allocation = runmat_accelerate_api::test_support::TestAllocationOwner::new()
            .allocate_zeroed(
                vec![1, 1],
                runmat_accelerate_api::NumericElementType::F64,
                runmat_accelerate_api::GpuTensorStorage::Real,
            )
            .unwrap();
        let model = raw_tf(Value::Num(2.0), Value::GpuTensor(allocation.handle));
        let error = block_on(feedback_builtin(
            Value::Object(model),
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(1.0)]),
        ))
        .expect_err("nested resident admission");
        assert_eq!(
            error.identifier(),
            RESIDENT_NUMERIC_EXTENSION.error_identifier
        );
    }

    #[test]
    fn invalid_host_denominator_precedes_resident_numerator_download() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        crate::builtins::common::test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![2.0], vec![1, 1]).unwrap();
            let resident =
                crate::builtins::common::gpu_helpers::upload_tensor(provider, &tensor).unwrap();
            let model = raw_tf(Value::GpuTensor(resident), Value::String("bad".into()));
            provider.reset_telemetry();
            let error = block_on(feedback_builtin(
                Value::Object(model),
                crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(1.0)]),
            ))
            .expect_err("invalid denominator");
            assert_eq!(error.identifier(), Some("RunMat:feedback:InvalidModel"));
            assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
        });
    }

    #[test]
    fn unrelated_resident_model_property_is_not_gathered() {
        let unrelated = Tensor::new(vec![3.0, 4.0], vec![1, 2]).unwrap();
        crate::builtins::common::test_support::with_test_provider(|provider| {
            let resident =
                crate::builtins::common::gpu_helpers::upload_tensor(provider, &unrelated)
                    .expect("upload unrelated");
            let mut model = raw_tf(
                Value::Num(2.0),
                Value::Tensor(Tensor::new(vec![1.0, 3.0], vec![1, 2]).unwrap()),
            );
            model.replace_property_unbounded("Unrelated".into(), Value::GpuTensor(resident));
            provider.reset_telemetry();
            let result = block_on(feedback_builtin(
                Value::Object(model),
                crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(1.0)]),
            ))
            .expect("feedback");
            assert_eq!(coeff(&result, "Denominator"), vec![1.0, 5.0]);
            assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
        });
    }

    fn tf(num: Vec<f64>, den: Vec<f64>) -> crate::execution::RootedValue {
        block_on(crate::call_builtin_async(
            "tf",
            &[
                Value::Tensor(Tensor::new(num.clone(), vec![1, num.len()]).unwrap()),
                Value::Tensor(Tensor::new(den.clone(), vec![1, den.len()]).unwrap()),
            ],
        ))
        .expect("tf")
    }

    fn coeff(value: &Value, field: &str) -> Vec<f64> {
        let Value::Object(object) = value else {
            panic!("expected tf object");
        };
        let Value::Tensor(tensor) = object.properties().get(field).expect(field) else {
            panic!("expected tensor");
        };
        tensor.materialize_f64().clone()
    }

    fn feedback_with_tf(
        model: crate::execution::RootedValue,
        rest: Vec<Value>,
    ) -> BuiltinResult<crate::execution::RootedValue> {
        model.try_map_owned(
            |model| {
                block_on(feedback_builtin(
                    model,
                    crate::builtins::common::test_support::builtin_arguments(rest),
                ))
            },
            |error| error.into_runtime_error(),
        )
    }

    fn assign_numerator(
        model: crate::execution::RootedValue,
        numerator: Value,
    ) -> crate::execution::RootedValue {
        model
            .with_owned(|model| {
                block_on(crate::object::resolve::store_member(
                    model,
                    "Numerator".into(),
                    numerator,
                    None,
                    |_, _| {},
                ))
            })
            .expect("public tf coefficient property assignment")
    }

    #[test]
    fn mutated_model_coefficients_cannot_round_in_either_feedback_position() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let numerator = Value::Tensor(
            Tensor::new_integer(IntegerStorage::U64(vec![9_007_199_254_740_993]), vec![1, 1])
                .unwrap(),
        );
        for (sys1, sys2) in [
            (
                assign_numerator(tf(vec![1.0], vec![1.0]), numerator.clone()),
                tf(vec![1.0], vec![1.0]),
            ),
            (
                tf(vec![1.0], vec![1.0]),
                assign_numerator(tf(vec![1.0], vec![1.0]), numerator),
            ),
        ] {
            let error = sys1
                .with_owned(|sys1| {
                    sys2.with_owned(|sys2| {
                        block_on(feedback_builtin(
                            sys1,
                            crate::builtins::common::test_support::builtin_arguments(vec![sys2]),
                        ))
                    })
                })
                .expect_err("model coefficient must not use scalar-gain rounding");
            assert_eq!(error.identifier(), Some("RunMat:feedback:InvalidModel"));
        }
    }

    #[test]
    fn resident_mutated_coefficients_gather_once_after_host_preflight() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        crate::builtins::common::test_support::with_test_provider(|provider| {
            for (coefficient, expected_valid) in [
                (9_007_199_254_740_992_u64, true),
                (9_007_199_254_740_993_u64, false),
            ] {
                let resident = provider
                    .upload_integer(&HostIntegerTensorView {
                        data: HostIntegerDataView::U64(std::slice::from_ref(&coefficient)),
                        shape: &[1, 1],
                    })
                    .expect("resident integer coefficient");
                let model = assign_numerator(tf(vec![1.0], vec![1.0]), Value::GpuTensor(resident));
                provider.reset_telemetry();
                let result = model.with_owned(|model| {
                    block_on(feedback_builtin(
                        model,
                        crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(
                            0.0,
                        )]),
                    ))
                });
                if expected_valid {
                    assert_eq!(
                        coeff(&result.expect("exact coefficient"), "Numerator"),
                        vec![coefficient as f64]
                    );
                } else {
                    assert_eq!(
                        result.unwrap_err().identifier(),
                        Some("RunMat:feedback:InvalidModel")
                    );
                }
                assert_eq!(provider.telemetry_snapshot().download_bytes, 8);
            }

            let resident = provider
                .upload_integer(&HostIntegerTensorView {
                    data: HostIntegerDataView::U64(&[1]),
                    shape: &[1, 1],
                })
                .expect("resident coefficient");
            let model = assign_numerator(tf(vec![1.0], vec![1.0]), Value::GpuTensor(resident));
            let error = model.with_owned(|value| {
                let Value::Object(mut object) = value else {
                    unreachable!()
                };
                object.replace_property_unbounded("Denominator".into(), Value::from("invalid"));
                provider.reset_telemetry();
                block_on(feedback_builtin(
                    Value::Object(object),
                    crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(0.0)]),
                ))
                .expect_err("host denominator preflight")
            });
            assert_eq!(error.identifier(), Some("RunMat:feedback:InvalidModel"));
            assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
        });
    }

    #[test]
    fn unity_negative_feedback_forms_closed_loop() {
        let g = tf(vec![2.0], vec![1.0, 3.0]);
        let out = g
            .with_owned(|g| {
                block_on(feedback_builtin(
                    g,
                    crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(1.0)]),
                ))
            })
            .expect("feedback");
        assert_eq!(coeff(&out, "Numerator"), vec![2.0]);
        assert_eq!(coeff(&out, "Denominator"), vec![1.0, 5.0]);
    }

    #[test]
    fn positive_feedback_subtracts_loop_gain() {
        let g = tf(vec![2.0], vec![1.0, 3.0]);
        let out = g
            .with_owned(|g| {
                block_on(feedback_builtin(
                    g,
                    crate::builtins::common::test_support::builtin_arguments(vec![
                        Value::Num(1.0),
                        Value::Num(1.0),
                    ]),
                ))
            })
            .expect("feedback");
        assert_eq!(coeff(&out, "Numerator"), vec![2.0]);
        assert_eq!(coeff(&out, "Denominator"), vec![1.0, 1.0]);
    }

    #[test]
    fn integer_metadata_covers_gain_and_sign_for_all_classes() {
        assert_eq!(FEEDBACK_INTEGER_CAPABILITIES.len(), 2);
        assert_eq!(FEEDBACK_INTEGER_CAPABILITIES[0].inputs.len(), 2);
        assert_eq!(FEEDBACK_INTEGER_CAPABILITIES[0].inputs[0].classes.len(), 8);
        assert_eq!(FEEDBACK_INTEGER_CAPABILITIES[0].inputs[1].classes.len(), 8);
        assert_eq!(FEEDBACK_INTEGER_CAPABILITIES[1].inputs[0].classes.len(), 8);
        assert_eq!(
            FEEDBACK_INTEGER_CAPABILITIES[0].computation_domain,
            BuiltinIntegerComputationDomain::FloatingPoint
        );
        assert_eq!(
            FEEDBACK_INTEGER_CAPABILITIES[1].computation_domain,
            BuiltinIntegerComputationDomain::Structural
        );
    }

    #[test]
    fn integer_gain_and_sign_support_all_eight_classes_in_runmat_mode() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let gains = [
            IntValue::I8(1),
            IntValue::I16(1),
            IntValue::I32(1),
            IntValue::I64(1),
            IntValue::U8(1),
            IntValue::U16(1),
            IntValue::U32(1),
            IntValue::U64(1),
        ];
        for gain in gains {
            let out = feedback_with_tf(
                tf(vec![2.0], vec![1.0, 3.0]),
                vec![Value::Int(gain.clone()), Value::Int(gain)],
            )
            .expect("typed-integer gain and sign");
            assert_eq!(coeff(&out, "Denominator"), vec![1.0, 1.0]);
        }
    }

    #[test]
    fn integer_tensor_gain_uses_explicit_binary64_boundary() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let wide = (1_u64 << 53) + 1;
        let gain = Tensor::new_integer(IntegerStorage::U64(vec![wide]), vec![1, 1])
            .expect("integer scalar tensor");
        let out = feedback_with_tf(tf(vec![1.0], vec![1.0]), vec![Value::Tensor(gain)])
            .expect("wide integer gain");
        assert_eq!(coeff(&out, "Denominator"), vec![(wide as f64) + 1.0]);
    }

    #[test]
    fn integer_sign_keeps_the_exact_plus_or_minus_one_contract() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let negative = feedback_with_tf(
            tf(vec![2.0], vec![1.0, 3.0]),
            vec![Value::Num(1.0), Value::Int(IntValue::I64(-1))],
        )
        .expect("negative integer sign");
        assert_eq!(coeff(&negative, "Denominator"), vec![1.0, 5.0]);

        let error = feedback_with_tf(
            tf(vec![2.0], vec![1.0, 3.0]),
            vec![Value::Num(1.0), Value::Int(IntValue::U64(0))],
        )
        .expect_err("zero is not a feedback sign");
        assert_eq!(error.identifier(), FEEDBACK_ERRORS[2].identifier);
    }

    #[test]
    fn logical_numeric_inputs_are_a_separate_runmat_extension() {
        let g = tf(vec![2.0], vec![1.0, 3.0]);
        {
            let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
            let error = block_on(feedback_builtin(
                g.value().clone(),
                crate::builtins::common::test_support::builtin_arguments(vec![Value::Bool(true)]),
            ))
            .expect_err("strict mode rejects logical gain");
            assert_eq!(
                error.identifier(),
                LOGICAL_NUMERIC_EXTENSION.error_identifier
            );
        }
        {
            let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
            let logical = LogicalArray::new(vec![1], vec![1, 1]).expect("logical scalar");
            let out = g
                .try_map_owned(
                    |g| {
                        block_on(feedback_builtin(
                            g,
                            crate::builtins::common::test_support::builtin_arguments(vec![
                                Value::LogicalArray(logical),
                            ]),
                        ))
                    },
                    |error| error.into_runtime_error(),
                )
                .expect("logical gain");
            assert_eq!(coeff(&out, "Denominator"), vec![1.0, 5.0]);
        }
    }

    #[test]
    fn runmat_only_forms_have_stable_independent_guards() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);

        let documented = feedback_with_tf(
            tf(vec![2.0], vec![1.0, 3.0]),
            vec![Value::Num(1.0), Value::Num(-1.0)],
        )
        .expect("tf plus ordinary double gain and sign remains admitted");
        assert_eq!(coeff(&documented, "Denominator"), vec![1.0, 5.0]);

        let error =
            feedback_with_tf(tf(vec![1.0], vec![1.0]), vec![]).expect_err("one-argument form");
        assert_eq!(error.identifier(), SINGLE_SYSTEM_EXTENSION.error_identifier);

        let gain = tf(vec![1.0], vec![1.0]);
        let error = gain
            .with_owned(|gain| {
                block_on(feedback_builtin(
                    Value::Num(1.0),
                    crate::builtins::common::test_support::builtin_arguments(vec![gain]),
                ))
            })
            .expect_err("scalar forward form");
        assert_eq!(
            error.identifier(),
            SCALAR_FORWARD_EXTENSION.error_identifier
        );

        let error = block_on(feedback_builtin(
            Value::Num(1.0),
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(1.0)]),
        ))
        .expect_err("scalar-scalar form");
        assert_eq!(
            error.identifier(),
            SCALAR_SYSTEMS_EXTENSION.error_identifier
        );

        let error = feedback_with_tf(tf(vec![1.0], vec![1.0]), vec![Value::Int(IntValue::I16(1))])
            .expect_err("integer gain");
        assert_eq!(error.identifier(), INTEGER_GAIN_EXTENSION.error_identifier);

        let error = feedback_with_tf(
            tf(vec![1.0], vec![1.0]),
            vec![Value::Num(1.0), Value::Int(IntValue::I8(-1))],
        )
        .expect_err("integer sign");
        assert_eq!(error.identifier(), INTEGER_SIGN_EXTENSION.error_identifier);

        let complex_integer = ComplexTensor::new_integer(
            IntegerComplexStorage::new(IntegerStorage::I16(vec![1]), IntegerStorage::I16(vec![0]))
                .expect("paired integer storage"),
            vec![1, 1],
        )
        .expect("complex integer scalar");
        let error = feedback_with_tf(
            tf(vec![1.0], vec![1.0]),
            vec![Value::ComplexTensor(complex_integer)],
        )
        .expect_err("complex integer gain");
        assert_eq!(error.identifier(), INTEGER_GAIN_EXTENSION.error_identifier);

        let single = Tensor::new_with_dtype(vec![1.0], vec![1, 1], NumericDType::F32)
            .expect("single scalar");
        let error = feedback_with_tf(tf(vec![1.0], vec![1.0]), vec![Value::Tensor(single)])
            .expect_err("single gain");
        assert_eq!(
            error.identifier(),
            SINGLE_NUMERIC_EXTENSION.error_identifier
        );
    }

    #[test]
    fn resident_gate_runs_before_provider_access() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
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
        let error =
            feedback_with_tf(tf(vec![1.0], vec![1.0]), vec![resident]).expect_err("resident gate");
        assert_eq!(
            error.identifier(),
            RESIDENT_NUMERIC_EXTENSION.error_identifier
        );
        assert_eq!(allocation.probe.retirements(), 0);
        allocation.handle.release().unwrap();
        assert_eq!(allocation.probe.retirements(), 1);
    }
}
