use super::*;
use crate::builtins::control::tf_model::TfOptions;
use num_complex::Complex64;

fn real(values: &[f64]) -> Vec<Complex64> {
    values
        .iter()
        .copied()
        .map(|value| Complex64::new(value, 0.0))
        .collect()
}

fn reduce_model(model: &TfModel, tolerance: Value) -> TfModel {
    let output = futures::executor::block_on(minreal_builtin(
        None,
        model.to_value(NAME).expect("serialize input model"),
        crate::builtins::common::test_support::builtin_arguments(vec![tolerance]),
    ))
    .expect("execute minreal builtin");
    futures::executor::block_on(TfModel::from_value_async(output, NAME))
        .expect("decode reduced model")
}

#[test]
fn near_pair_and_conjugate_pair_cancel_without_changing_real_model_metadata() {
    let mut near = TfModel::new(
        real(&[1.0, 3.00000001, 2.00000002]),
        real(&[1.0, 4.0, 3.0]),
        TfOptions {
            variable: "z".into(),
            sample_time: 0.1,
        },
    )
    .expect("near-pair model");
    near.input_delay = 0.2;
    near.output_delay = 0.3;
    let near = reduce_model(
        &near,
        Value::Tensor(runmat_value::Tensor::from_f32(vec![1.0e-6], vec![1, 1]).unwrap()),
    );
    assert_eq!(near.numerator.len(), 2);
    assert_eq!(near.denominator.len(), 2);
    assert!((near.numerator[1].re - 2.0).abs() < 1.0e-6);
    assert!((near.denominator[1].re - 3.0).abs() < 1.0e-6);
    assert_eq!(near.sample_time(), 0.1);
    assert_eq!(near.input_delay, 0.2);
    assert_eq!(near.output_delay, 0.3);
    assert!(near
        .numerator
        .iter()
        .chain(&near.denominator)
        .all(|value| value.im == 0.0));

    // (s²+2s+2)(s+3) / (s²+2s+2)(s+4) must cancel
    // both conjugate roots together and retain real coefficients.
    let pair = TfModel::new(
        real(&[1.0, 5.0, 8.0, 6.0]),
        real(&[1.0, 6.0, 10.0, 8.0]),
        TfOptions::default(),
    )
    .expect("conjugate-pair model");
    let pair = reduce_model(&pair, Value::Num(1.0e-6));
    assert_eq!(pair.numerator.len(), 2);
    assert_eq!(pair.denominator.len(), 2);
    assert!(pair
        .numerator
        .iter()
        .chain(&pair.denominator)
        .all(|value| value.im == 0.0));
    assert!((pair.numerator[1].re - 3.0).abs() < 1.0e-6);
    assert!((pair.denominator[1].re - 4.0).abs() < 1.0e-6);

    // Cancel only (s+2). Reconstructing from surviving roots and treating
    // tiny imaginary parts as real would erase this genuine s²+1e-18.
    let tiny_pair = TfModel::new(
        real(&[1.0, 2.0, 1.0e-18, 2.0e-18]),
        real(&[1.0, 5.0, 6.0]),
        TfOptions::default(),
    )
    .expect("tiny-conjugate model");
    let tiny_pair = reduce_model(&tiny_pair, Value::Num(f64::EPSILON.sqrt()));
    assert_eq!(tiny_pair.numerator.len(), 3);
    assert!((tiny_pair.numerator[2].re - 1.0e-18).abs() < 1.0e-25);
    assert_eq!(tiny_pair.denominator.len(), 2);
    assert!((tiny_pair.denominator[1].re - 3.0).abs() < 1.0e-8);

    // Zero gain has no observable poles, even with an inverse discrete basis.
    let mut zero = TfModel::new(
        real(&[0.0]),
        real(&[1.0, 2.0, 3.0]),
        TfOptions {
            variable: "z^-1".into(),
            sample_time: 0.1,
        },
    )
    .expect("zero-gain model");
    zero.input_delay = 0.2;
    zero.output_delay = 0.3;
    let zero = reduce_model(&zero, Value::Num(f64::EPSILON.sqrt()));
    assert_eq!(zero.numerator, real(&[0.0]));
    assert_eq!(zero.denominator, real(&[1.0]));
    assert_eq!(zero.variable.as_str(), "z^-1");
    assert_eq!(zero.sample_time(), 0.1);
    assert_eq!(zero.input_delay, 0.2);
    assert_eq!(zero.output_delay, 0.3);
}
