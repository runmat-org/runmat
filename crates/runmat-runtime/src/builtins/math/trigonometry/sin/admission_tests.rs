use futures::executor::block_on;
use runmat_value::Value;

// Exact applicability: every fixed-output trigonometric/angle identity.
// pol2cart has form-dependent multiple outputs and owns separate admission tests.
const UNARY: &[&str] = &[
    "acos", "acosh", "asin", "asinh", "atan", "atanh", "cos", "cosd", "cosh", "cospi", "sin",
    "sind", "sinh", "sinpi", "tan", "tand", "tanh", "deg2rad", "rad2deg",
];

#[test]
fn trigonometric_excess_outputs_precede_input_evaluation() {
    for &name in UNARY.iter().chain(std::iter::once(&"atan2")) {
        let mut inputs = vec![Value::String("invalid numeric input".into())];
        if name == "atan2" {
            inputs.push(Value::Num(1.0));
        }
        let error = block_on(crate::dispatcher::call_builtin_async_with_outputs(
            name, &inputs, 2,
        ))
        .expect_err(name);
        assert_eq!(
            error.identifier(),
            Some(format!("RunMat:{name}:TooManyOutputs").as_str()),
            "{name}: {error}"
        );
    }
}

#[test]
fn trigonometric_fixed_forms_preserve_discarded_and_single_output_calls() {
    for &name in UNARY.iter().chain(std::iter::once(&"atan2")) {
        let mut inputs = vec![Value::Num(0.5)];
        if name == "atan2" {
            inputs.push(Value::Num(1.0));
        }
        for outputs in [0, 1] {
            block_on(crate::dispatcher::call_builtin_async_with_outputs(
                name, &inputs, outputs,
            ))
            .unwrap_or_else(|error| panic!("{name}/{outputs}: {error}"));
        }
    }
}
