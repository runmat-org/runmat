//! SISO transfer-function operator bindings. Model parsing and arithmetic live in `tf_model`.

use runmat_builtins::INTEGER_COEFFICIENT_EXTENSION;
use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::builtins::control::tf_model::{
    control_error, preflight_model_operand, scalar_f64, two_models_ordered, TfModel,
};
use crate::{dispatcher, BuiltinResult};

async fn binary(
    lhs: Value,
    rhs: Value,
    operation: fn(&TfModel, &TfModel) -> BuiltinResult<TfModel>,
) -> BuiltinResult<Value> {
    if !matches!(lhs, Value::Object(_)) && !matches!(rhs, Value::Object(_)) {
        return Err(control_error(
            "tf",
            "RunMat:tf:InvalidArgument",
            "tf: an operator method requires a tf model operand",
        ));
    }
    for operand in [&lhs, &rhs] {
        if !matches!(operand, Value::Object(_))
            && crate::builtins::common::validation::value_has_native_integer_class(operand)
        {
            crate::compatibility::ensure_builtin_extension_enabled(
                &INTEGER_COEFFICIENT_EXTENSION,
                "tf",
            )?;
        }
    }
    let (lhs, rhs) = two_models_ordered(lhs, rhs, "tf").await?;
    operation(&lhs, &rhs)?.to_value("tf")
}

pub(super) fn integer_exponent(value: &Value) -> BuiltinResult<i64> {
    if let Some(integer) = crate::builtins::common::tensor::scalar_integer_value(value) {
        return integer.try_to_i64().ok_or_else(|| {
            control_error(
                "tf",
                "RunMat:tf:InvalidExponent",
                "tf: exponent exceeds integer range",
            )
        });
    }
    let exponent = scalar_f64(value, "exponent", "tf")?;
    if !exponent.is_finite() || exponent.fract() != 0.0 {
        return Err(control_error(
            "tf",
            "RunMat:tf:InvalidExponent",
            "tf: transfer-function powers require an integer scalar exponent",
        ));
    }
    // i64::MAX rounds to 2^63 in f64. The upper bound is exclusive; the
    // representable lower bound -2^63 is valid; powi uses unsigned magnitude.
    if exponent < i64::MIN as f64 || exponent >= i64::MAX as f64 {
        return Err(control_error(
            "tf",
            "RunMat:tf:InvalidExponent",
            "tf: exponent exceeds integer range",
        ));
    }
    Ok(exponent as i64)
}

async fn power(lhs: Value, rhs: Value) -> BuiltinResult<Value> {
    preflight_model_operand(&lhs, "tf")?;
    let host_exponent = match &rhs {
        Value::GpuTensor(handle) => {
            let scalar = handle
                .shape()
                .iter()
                .try_fold(1usize, |count, &dim| count.checked_mul(dim))
                == Some(1);
            if !scalar
                || handle.physical().storage() != runmat_accelerate_api::GpuTensorStorage::Real
            {
                return Err(control_error(
                    "tf",
                    "RunMat:tf:InvalidExponent",
                    "tf: exponent must be a real numeric scalar",
                ));
            }
            crate::builtins::common::validation::validate_resident_metadata(&rhs)?;
            None
        }
        _ => Some(integer_exponent(&rhs)?),
    };
    let sys = TfModel::from_value_async(lhs, "tf").await?;
    let exponent = match host_exponent {
        Some(exponent) => exponent,
        None => integer_exponent(&dispatcher::gather_if_needed_async(&rhs).await?)?,
    };
    sys.powi(exponent)?.to_value("tf")
}

#[runtime_builtin(
    name = "tf.plus",
    binding_variant = "default",
    builtin_path = "crate::builtins::control::tf::methods"
)]
pub(super) async fn tf_plus(lhs: Value, rhs: Value) -> BuiltinResult<Value> {
    binary(lhs, rhs, TfModel::add).await
}

#[runtime_builtin(
    name = "tf.minus",
    binding_variant = "default",
    builtin_path = "crate::builtins::control::tf::methods"
)]
pub(super) async fn tf_minus(lhs: Value, rhs: Value) -> BuiltinResult<Value> {
    binary(lhs, rhs, TfModel::sub).await
}

#[runtime_builtin(
    name = "tf.uplus",
    binding_variant = "default",
    builtin_path = "crate::builtins::control::tf::methods"
)]
pub(super) async fn tf_uplus(sys: Value) -> BuiltinResult<Value> {
    TfModel::from_value_async(sys, "tf").await?.to_value("tf")
}

#[runtime_builtin(
    name = "tf.uminus",
    binding_variant = "default",
    builtin_path = "crate::builtins::control::tf::methods"
)]
pub(super) async fn tf_uminus(sys: Value) -> BuiltinResult<Value> {
    TfModel::from_value_async(sys, "tf")
        .await?
        .neg()?
        .to_value("tf")
}

#[runtime_builtin(
    name = "tf.times",
    binding_variant = "default",
    builtin_path = "crate::builtins::control::tf::methods"
)]
pub(super) async fn tf_times(lhs: Value, rhs: Value) -> BuiltinResult<Value> {
    binary(lhs, rhs, TfModel::mul).await
}

#[runtime_builtin(
    name = "tf.mtimes",
    binding_variant = "default",
    builtin_path = "crate::builtins::control::tf::methods"
)]
pub(super) async fn tf_mtimes(lhs: Value, rhs: Value) -> BuiltinResult<Value> {
    binary(lhs, rhs, TfModel::mul).await
}

#[runtime_builtin(
    name = "tf.rdivide",
    binding_variant = "default",
    builtin_path = "crate::builtins::control::tf::methods"
)]
pub(super) async fn tf_rdivide(lhs: Value, rhs: Value) -> BuiltinResult<Value> {
    binary(lhs, rhs, TfModel::div).await
}

#[runtime_builtin(
    name = "tf.mrdivide",
    binding_variant = "default",
    builtin_path = "crate::builtins::control::tf::methods"
)]
pub(super) async fn tf_mrdivide(lhs: Value, rhs: Value) -> BuiltinResult<Value> {
    binary(lhs, rhs, TfModel::div).await
}

#[runtime_builtin(
    name = "tf.ldivide",
    binding_variant = "default",
    builtin_path = "crate::builtins::control::tf::methods"
)]
pub(super) async fn tf_ldivide(lhs: Value, rhs: Value) -> BuiltinResult<Value> {
    binary(rhs, lhs, TfModel::div).await
}

#[runtime_builtin(
    name = "tf.mldivide",
    binding_variant = "default",
    builtin_path = "crate::builtins::control::tf::methods"
)]
pub(super) async fn tf_mldivide(lhs: Value, rhs: Value) -> BuiltinResult<Value> {
    binary(rhs, lhs, TfModel::div).await
}

#[runtime_builtin(
    name = "tf.power",
    binding_variant = "default",
    builtin_path = "crate::builtins::control::tf::methods"
)]
pub(super) async fn tf_power(lhs: Value, rhs: Value) -> BuiltinResult<Value> {
    power(lhs, rhs).await
}

#[runtime_builtin(
    name = "tf.mpower",
    binding_variant = "default",
    builtin_path = "crate::builtins::control::tf::methods"
)]
pub(super) async fn tf_mpower(lhs: Value, rhs: Value) -> BuiltinResult<Value> {
    power(lhs, rhs).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_value::{IntValue, IntegerStorage, Tensor};

    fn model(value: f64) -> Value {
        block_on(super::super::tf_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::Num(value),
                Value::Num(1.0),
            ]),
        ))
        .expect("scalar transfer function")
    }

    fn coefficients(value: &Value, name: &str) -> Vec<f64> {
        let Value::Object(object) = value else {
            panic!("tf object expected")
        };
        let Value::Tensor(tensor) = &object.properties()[name] else {
            panic!("real coefficients expected")
        };
        tensor.materialize_f64().clone()
    }

    fn assign_numerator(model: Value, numerator: Value) -> crate::execution::RootedValue {
        block_on(crate::object::resolve::store_member(
            model,
            "Numerator".into(),
            numerator,
            None,
            |_, _| {},
        ))
        .expect("public tf coefficient property assignment")
    }

    #[test]
    fn twelve_siso_bindings_preserve_operand_orientation() {
        let a = model(6.0);
        let b = model(2.0);
        let binary = [
            (block_on(tf_plus(a.clone(), b.clone())), 8.0),
            (block_on(tf_minus(a.clone(), b.clone())), 4.0),
            (block_on(tf_times(a.clone(), b.clone())), 12.0),
            (block_on(tf_mtimes(a.clone(), b.clone())), 12.0),
            (block_on(tf_rdivide(a.clone(), b.clone())), 3.0),
            (block_on(tf_mrdivide(a.clone(), b.clone())), 3.0),
            (block_on(tf_ldivide(b.clone(), a.clone())), 3.0),
            (block_on(tf_mldivide(b.clone(), a.clone())), 3.0),
        ];
        for (result, expected) in binary {
            let actual = result.expect("operator result");
            let numerator = coefficients(&actual, "Numerator")[0];
            let denominator = coefficients(&actual, "Denominator")[0];
            assert_eq!(numerator / denominator, expected);
        }
        assert_eq!(
            coefficients(&block_on(tf_uplus(a.clone())).unwrap(), "Numerator"),
            vec![6.0]
        );
        assert_eq!(
            coefficients(&block_on(tf_uminus(a.clone())).unwrap(), "Numerator"),
            vec![-6.0]
        );
        for result in [
            block_on(tf_power(b.clone(), Value::Num(3.0))),
            block_on(tf_mpower(b.clone(), Value::Num(3.0))),
        ] {
            assert_eq!(coefficients(&result.unwrap(), "Numerator"), vec![8.0]);
        }
    }

    #[test]
    fn typed_integer_exponent_is_decoded_without_floating_point() {
        let exact = Value::Tensor(
            Tensor::new_integer(IntegerStorage::U64(vec![9_007_199_254_740_993]), vec![1, 1])
                .unwrap(),
        );
        assert_eq!(integer_exponent(&exact).unwrap(), 9_007_199_254_740_993);
        let overflow = Value::Tensor(
            Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX]), vec![1, 1]).unwrap(),
        );
        assert_eq!(
            integer_exponent(&overflow).unwrap_err().identifier(),
            Some("RunMat:tf:InvalidExponent")
        );
        assert_eq!(
            integer_exponent(&Value::Num(1.5)).unwrap_err().identifier(),
            Some("RunMat:tf:InvalidExponent")
        );
        assert_eq!(
            integer_exponent(&Value::Int(IntValue::I64(i64::MIN))).unwrap(),
            i64::MIN
        );
        for result in [
            block_on(tf_power(model(-1.0), Value::Int(IntValue::I64(i64::MIN)))),
            block_on(tf_mpower(model(-1.0), Value::Int(IntValue::I64(i64::MIN)))),
        ] {
            let result = result.expect("minimum signed exponent");
            assert_eq!(coefficients(&result, "Numerator"), vec![1.0]);
            assert_eq!(coefficients(&result, "Denominator"), vec![1.0]);
        }
    }

    #[test]
    fn wide_integer_scalar_gain_rounds_at_double_coefficient_boundary() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let gain = Value::Tensor(
            Tensor::new_integer(IntegerStorage::U64(vec![9_007_199_254_740_993]), vec![1, 1])
                .unwrap(),
        );
        let expected = 9_007_199_254_740_992.0;
        for result in [
            block_on(tf_plus(model(0.0), gain.clone())),
            block_on(tf_plus(gain, model(0.0))),
        ] {
            let result = result.expect("wide scalar gain remains admissible");
            assert_eq!(coefficients(&result, "Numerator"), vec![expected]);
            assert_eq!(coefficients(&result, "Denominator"), vec![1.0]);
        }
    }

    #[test]
    fn mutated_integer_coefficients_keep_constructor_exactness_in_both_orders() {
        let wide = Value::Tensor(
            Tensor::new_integer(IntegerStorage::U64(vec![9_007_199_254_740_993]), vec![1, 1])
                .unwrap(),
        );
        {
            let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
            let mutated = assign_numerator(model(1.0), wide.clone());
            for result in [
                block_on(tf_plus(mutated.value().clone(), Value::Num(1.0))),
                mutated.with_owned(|mutated| block_on(tf_plus(Value::Num(1.0), mutated))),
            ] {
                let error = result.expect_err("model coefficient cannot round as a scalar gain");
                assert_eq!(error.identifier(), Some("RunMat:tf:InvalidCoefficients"));
            }
            let exact = assign_numerator(
                model(1.0),
                Value::Tensor(
                    Tensor::new_integer(
                        IntegerStorage::U64(vec![9_007_199_254_740_992]),
                        vec![1, 1],
                    )
                    .unwrap(),
                ),
            );
            assert_eq!(
                coefficients(
                    &exact
                        .with_owned(|exact| block_on(tf_plus(exact, Value::Num(0.0))))
                        .unwrap(),
                    "Numerator"
                ),
                vec![9_007_199_254_740_992.0],
            );
        }
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let mutated = assign_numerator(model(1.0), wide);
        let error = mutated
            .with_owned(|mutated| block_on(tf_plus(mutated, Value::Num(1.0))))
            .unwrap_err();
        assert_eq!(
            error.identifier(),
            INTEGER_COEFFICIENT_EXTENSION.error_identifier
        );
    }

    #[test]
    fn unary_and_power_bindings_reject_numeric_receivers() {
        for result in [
            block_on(tf_uplus(Value::Num(2.0))),
            block_on(tf_uminus(Value::Num(2.0))),
            block_on(tf_power(Value::Num(2.0), Value::Num(3.0))),
            block_on(tf_mpower(Value::Num(2.0), Value::Num(3.0))),
        ] {
            assert_eq!(
                result.unwrap_err().identifier(),
                Some("RunMat:tf:InvalidModel")
            );
        }
    }

    #[test]
    fn invalid_host_gain_and_exponent_preflight_before_resident_download() {
        crate::builtins::common::test_support::with_test_provider(|provider| {
            let resident = crate::builtins::common::gpu_helpers::upload_tensor(
                provider,
                &Tensor::new(vec![2.0], vec![1, 1]).unwrap(),
            )
            .unwrap();
            let mut sys = model(2.0);
            let Value::Object(object) = &mut sys else {
                unreachable!()
            };
            object
                .replace_property_unbounded("Numerator".into(), Value::GpuTensor(resident.clone()));
            provider.reset_telemetry();
            let error = block_on(tf_plus(sys, Value::from("bad"))).unwrap_err();
            assert_eq!(error.identifier(), Some("RunMat:tf:InvalidArgument"));
            assert_eq!(provider.telemetry_snapshot().download_bytes, 0);

            let mut sys = model(2.0);
            let Value::Object(object) = &mut sys else {
                unreachable!()
            };
            object
                .replace_property_unbounded("Numerator".into(), Value::GpuTensor(resident.clone()));
            provider.reset_telemetry();
            let error = block_on(tf_plus(sys, Value::Num(f64::INFINITY))).unwrap_err();
            assert_eq!(error.identifier(), Some("RunMat:tf:InvalidCoefficients"));
            assert_eq!(provider.telemetry_snapshot().download_bytes, 0);

            let mut sys = model(2.0);
            let Value::Object(object) = &mut sys else {
                unreachable!()
            };
            object.replace_property_unbounded("Numerator".into(), Value::GpuTensor(resident));
            provider.reset_telemetry();
            let error = block_on(tf_power(sys, Value::Num(1.5))).unwrap_err();
            assert_eq!(error.identifier(), Some("RunMat:tf:InvalidExponent"));
            assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
        });
    }

    #[test]
    fn resident_exponent_shape_is_rejected_before_model_download() {
        crate::builtins::common::test_support::with_test_provider(|provider| {
            let numerator = crate::builtins::common::gpu_helpers::upload_tensor(
                provider,
                &Tensor::new(vec![2.0], vec![1, 1]).unwrap(),
            )
            .unwrap();
            let exponent = crate::builtins::common::gpu_helpers::upload_tensor(
                provider,
                &Tensor::new(vec![2.0, 3.0], vec![2, 1]).unwrap(),
            )
            .unwrap();
            let mut sys = model(2.0);
            let Value::Object(object) = &mut sys else {
                unreachable!()
            };
            object.replace_property_unbounded("Numerator".into(), Value::GpuTensor(numerator));
            provider.reset_telemetry();
            for result in [
                block_on(tf_power(sys.clone(), Value::GpuTensor(exponent.clone()))),
                block_on(tf_mpower(sys.clone(), Value::GpuTensor(exponent.clone()))),
            ] {
                let error = result.unwrap_err();
                assert_eq!(error.identifier(), Some("RunMat:tf:InvalidExponent"));
                assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
            }

            let complex = runmat_accelerate_api::test_support::TestAllocationOwner::new()
                .allocate_zeroed(
                    vec![1, 1],
                    runmat_accelerate_api::NumericElementType::F64,
                    runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved,
                )
                .unwrap();
            let error = block_on(tf_power(
                sys.clone(),
                Value::GpuTensor(complex.handle.clone()),
            ))
            .unwrap_err();
            assert_eq!(error.identifier(), Some("RunMat:tf:InvalidExponent"));
            assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
            complex.handle.release().unwrap();

            let valid_exponent = crate::builtins::common::gpu_helpers::upload_tensor(
                provider,
                &Tensor::new(vec![3.0], vec![1, 1]).unwrap(),
            )
            .unwrap();
            provider.reset_telemetry();
            let result = block_on(tf_power(sys, Value::GpuTensor(valid_exponent)))
                .expect("valid resident exponent");
            assert_eq!(coefficients(&result, "Numerator"), vec![8.0]);
            assert_eq!(provider.telemetry_snapshot().download_bytes, 16);
        });
    }

    #[test]
    fn integer_gain_extension_gate_precedes_resident_download() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        crate::builtins::common::test_support::with_test_provider(|provider| {
            let resident = crate::builtins::common::gpu_helpers::upload_tensor(
                provider,
                &Tensor::new(vec![2.0], vec![1, 1]).unwrap(),
            )
            .unwrap();
            let gain = Value::Tensor(
                Tensor::new_integer(IntegerStorage::U16(vec![3]), vec![1, 1]).unwrap(),
            );
            let mut sys = model(2.0);
            let Value::Object(object) = &mut sys else {
                unreachable!()
            };
            object.replace_property_unbounded("Numerator".into(), Value::GpuTensor(resident));
            provider.reset_telemetry();
            let error = block_on(tf_mtimes(sys, gain)).unwrap_err();
            assert_eq!(
                error.identifier(),
                INTEGER_COEFFICIENT_EXTENSION.error_identifier
            );
            assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
        });
    }

    #[test]
    fn admitted_resident_property_is_downloaded_once_by_its_owner() {
        crate::builtins::common::test_support::with_test_provider(|provider| {
            let resident = crate::builtins::common::gpu_helpers::upload_tensor(
                provider,
                &Tensor::new(vec![2.0], vec![1, 1]).unwrap(),
            )
            .unwrap();
            let mut sys = model(2.0);
            let Value::Object(object) = &mut sys else {
                unreachable!()
            };
            object.replace_property_unbounded("Numerator".into(), Value::GpuTensor(resident));
            provider.reset_telemetry();
            let result = block_on(tf_plus(sys, model(3.0))).expect("resident model operand");
            assert_eq!(coefficients(&result, "Numerator"), vec![5.0]);
            assert_eq!(provider.telemetry_snapshot().download_bytes, 8);
        });
    }

    #[test]
    fn unpolled_method_future_has_no_provider_effect() {
        crate::builtins::common::test_support::with_test_provider(|provider| {
            let resident = crate::builtins::common::gpu_helpers::upload_tensor(
                provider,
                &Tensor::new(vec![2.0], vec![1, 1]).unwrap(),
            )
            .unwrap();
            let mut sys = model(2.0);
            let Value::Object(object) = &mut sys else {
                unreachable!()
            };
            object.replace_property_unbounded("Numerator".into(), Value::GpuTensor(resident));
            provider.reset_telemetry();
            drop(tf_mtimes(sys, model(3.0)));
            assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
        });
    }
}
