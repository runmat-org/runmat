//! Semantic-only selected chain. The separate compiled-image fixture must
//! still authenticate its original source transcript and five publications.

use super::*;
use runmat_runtime::builtins::common::spec::FusionSignEvidence;
use runmat_types::NumericClass;
use runmat_value::Tensor;

#[path = "sign_chain/regressions.rs"]
mod regressions;

fn chain() -> FusionGroupPlan {
    let mut source = plan(
        vec![
            FusionOp::Primitive {
                op: PrimitiveOp::ElemPow,
                inputs: vec![0, 3],
                output: Some(4),
            },
            FusionOp::Builtin {
                origin: Default::default(),
                name: "mean".into(),
                inputs: vec![4, 7, 8],
                output: Some(6),
            },
            FusionOp::Primitive {
                op: PrimitiveOp::Add,
                inputs: vec![6, 1],
                output: Some(10),
            },
            cast("sqrt", 10, 12),
            cast("single", 12, 14),
            FusionOp::Builtin {
                origin: Default::default(),
                name: "max".into(),
                inputs: vec![14, 17],
                output: Some(16),
            },
            FusionOp::Primitive {
                op: PrimitiveOp::ElemPow,
                inputs: vec![16, 20],
                output: Some(19),
            },
        ],
        19,
    );
    source.const_values.insert(3, Value::Num(2.0));
    source.const_values.insert(
        7,
        Value::Tensor(Tensor::new(vec![2.0, 3.0], vec![1, 2]).unwrap()),
    );
    source
        .const_values
        .insert(8, Value::String("native".into()));
    source.const_values.insert(
        17,
        Value::Tensor(Tensor::from_f32(vec![0.0], vec![1, 1]).unwrap()),
    );
    source.const_values.insert(
        20,
        Value::Tensor(Tensor::from_f32(vec![1.8], vec![1, 1]).unwrap()),
    );
    source
}

fn input() -> Value {
    let mut values = vec![0.0_f32; 40];
    values[0] = -0.0;
    values[1] = f32::NAN;
    values[2] = f32::MAX;
    values[3] = -2.0;
    Value::Tensor(Tensor::from_f32(values, vec![2, 4, 5]).unwrap())
}

fn epsilon() -> Value {
    Value::Tensor(Tensor::from_f32(vec![1.0e-6], vec![1, 1]).unwrap())
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn ordered_sign_chain_keeps_power_receipts_and_requires_exact_source() {
    runmat_gc::gc_test_context(|| {
        let source = chain();
        let program = resolve_policy(
            &source,
            &[input(), epsilon()],
            ProviderPrecision::F64,
            PrecisionPolicy::Exact,
        )
        .expect("owner-admitted semantic chain");
        for id in [4, 6, 10, 12, 14] {
            assert_eq!(
                program.sign.get(&id),
                Some(&FusionSignEvidence::NonnegativeOrNaN {
                    class: NumericClass::Single
                }),
                "value {id}"
            );
        }
        assert_eq!(program.floating_class(14).unwrap(), NumericClass::Single);
        assert_eq!(
            program.power_arithmetic(4).unwrap().exponent_scalar,
            Some(
                runmat_runtime::builtins::common::spec::FusionScalarControl::DoubleBits(
                    2.0_f64.to_bits()
                )
            )
        );
        assert_eq!(
            program.power_arithmetic(19).unwrap().exponent_scalar,
            Some(
                runmat_runtime::builtins::common::spec::FusionScalarControl::SingleBits(
                    1.8_f32.to_bits()
                )
            )
        );
        assert_eq!(
            program.evidence.get(&19),
            Some(&runmat_accelerate_api::NumericalDomainEvidence::RealPower(
                runmat_accelerate_api::PowerResultDomain::Real
            ))
        );
        let mut changed_source = source.clone();
        changed_source.const_values.insert(3, Value::Num(4.0));
        assert!(program.require_source_plan(&changed_source).is_err());
        let mut changed_class = source.clone();
        changed_class.const_values.insert(
            3,
            Value::Tensor(Tensor::from_f32(vec![2.0], vec![1, 1]).unwrap()),
        );
        assert!(program.require_source_plan(&changed_class).is_err());
        let mut changed_axes = source.clone();
        changed_axes.const_values.insert(
            7,
            Value::Tensor(Tensor::new(vec![2.0, 4.0], vec![1, 2]).unwrap()),
        );
        assert!(program.require_source_plan(&changed_axes).is_err());
        let same_numeric_different_epsilon_class = Value::Num(f64::from(1.0e-6_f32));
        let changed = resolve_policy(
            &source,
            &[input(), same_numeric_different_epsilon_class],
            ProviderPrecision::F64,
            PrecisionPolicy::Exact,
        )
        .expect("generic typed Add theorem permits an independently selected Double epsilon");
        assert_ne!(
            program.cache_identity().unwrap(),
            changed.cache_identity().unwrap()
        );
    });
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn changed_control_negative_epsilon_complex_and_empty_fail_closed() {
    runmat_gc::gc_test_context(|| {
        let source = chain();
        for exponent in [3.0, 1.8, f64::NAN, f64::INFINITY] {
            let mut changed = source.clone();
            changed.const_values.insert(3, Value::Num(exponent));
            assert!(
                resolve_policy(
                    &changed,
                    &[input(), epsilon()],
                    ProviderPrecision::F64,
                    PrecisionPolicy::Exact
                )
                .is_err(),
                "{exponent:?}"
            );
        }
        let negative = Value::Tensor(Tensor::from_f32(vec![-1.0e-6], vec![1, 1]).unwrap());
        assert!(resolve_policy(
            &source,
            &[input(), negative],
            ProviderPrecision::F64,
            PrecisionPolicy::Exact
        )
        .is_err());
        assert!(resolve_policy(
            &source,
            &[input(), Value::Complex(1.0e-6, 0.0)],
            ProviderPrecision::F64,
            PrecisionPolicy::Exact
        )
        .is_err());
        assert!(resolve_policy(
            &source,
            &[Value::Complex(-2.0, 0.0), epsilon()],
            ProviderPrecision::F64,
            PrecisionPolicy::Exact
        )
        .is_err());
        let mut unknown_sign_producer = source.clone();
        unknown_sign_producer.operations[0] = FusionOp::Primitive {
            op: PrimitiveOp::ElemMul,
            inputs: vec![0, 0],
            output: Some(4),
        };
        assert!(resolve_policy(
            &unknown_sign_producer,
            &[input(), epsilon()],
            ProviderPrecision::F64,
            PrecisionPolicy::Exact
        )
        .is_err());
        let empty = Value::Tensor(Tensor::from_f32(vec![], vec![2, 0, 5]).unwrap());
        assert!(resolve_policy(
            &source,
            &[empty, epsilon()],
            ProviderPrecision::F64,
            PrecisionPolicy::Exact
        )
        .is_err());
    });
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn unsigned_mean_and_sqrt_keep_their_ordinary_unknown_result() {
    runmat_gc::gc_test_context(|| {
        let mut mean_source = plan(
            vec![FusionOp::Builtin {
                origin: Default::default(),
                name: "mean".into(),
                inputs: vec![0, 7, 8],
                output: Some(6),
            }],
            6,
        );
        mean_source.const_values.insert(
            7,
            Value::Tensor(Tensor::new(vec![2.0, 3.0], vec![1, 2]).unwrap()),
        );
        mean_source
            .const_values
            .insert(8, Value::String("native".into()));
        let mean = resolve_policy(
            &mean_source,
            &[input(), epsilon()],
            ProviderPrecision::F64,
            PrecisionPolicy::Exact,
        )
        .expect("unsigned ordinary MEAN remains eligible");
        assert_eq!(mean.sign.get(&6), None);

        let sqrt_source = plan(vec![cast("sqrt", 0, 2)], 2);
        let sqrt = resolve_policy(
            &sqrt_source,
            &[input(), epsilon()],
            ProviderPrecision::F64,
            PrecisionPolicy::Exact,
        )
        .expect("unsigned SQRT retains catalog inference");
        assert_eq!(sqrt.sign.get(&2), None);
        assert!(sqrt
            .fact(2)
            .unwrap()
            .numeric()
            .is_none_or(|numeric| numeric.domain != runmat_types::NumericDomain::Real));
    });
}
