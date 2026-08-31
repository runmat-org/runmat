#[path = "support/mod.rs"]
mod test_helpers;

use runmat_value::Value;
use test_helpers::execute_source;

#[test]
fn compiled_log2_dissection_receives_two_requested_outputs() {
    let values =
        execute_source("x=[0 1 -3 Inf NaN]; [f,e]=log2(x); s=single([1 -3]); [sf,se]=log2(s);")
            .expect("compiled log2 dissection");

    let Value::Tensor(fraction) = &values[1] else {
        panic!("expected double fraction tensor");
    };
    let Value::Tensor(exponent) = &values[2] else {
        panic!("expected double exponent tensor");
    };
    assert_eq!(fraction.shape, vec![1, 5]);
    assert_eq!(
        fraction.materialize_f64()[..4],
        [0.0, 0.5, -0.75, f64::INFINITY]
    );
    assert!(fraction.materialize_f64()[4].is_nan());
    assert_eq!(exponent.materialize_f64(), vec![0.0, 1.0, 2.0, 0.0, 0.0]);

    let Value::Tensor(single_fraction) = &values[4] else {
        panic!("expected single fraction tensor");
    };
    let Value::Tensor(single_exponent) = &values[5] else {
        panic!("expected single exponent tensor");
    };
    assert_eq!(single_fraction.as_f32_slice(), Some(&[0.5, -0.75][..]));
    assert_eq!(single_exponent.as_f32_slice(), Some(&[1.0, 2.0][..]));
}

#[test]
fn compiled_log2_dissection_enforces_the_current_complex_boundary() {
    let error = execute_source("[f,e]=log2(complex(2));")
        .expect_err("the current compatibility pin rejects complex dissection");
    assert_eq!(error.identifier(), Some("RunMat:log2:ComplexDissection"));
}
