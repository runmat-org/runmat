//! A nonsymmetric weighted loss exposes a missing inverse presentation map.
use super::*;

fn unformatted_input(data: Vec<f64>, dtype: NumericDType) -> Value {
    object(
        "dlarray",
        vec![
            ("Data", values(data, vec![2, 3, 4], dtype)),
            ("Format", Value::String(String::new().into())),
            ("Labels", Value::String(String::new().into())),
        ],
    )
    .expect("fixture object")
}

fn predict(net: &Value, x: Value) -> Value {
    let Value::Object(network) = net else {
        panic!("network")
    };
    block_on(model::predict_deep_learning_object(
        network.clone(),
        x,
        vec![],
    ))
    .unwrap()
    .remove(0)
}

#[test]
fn rank_changing_predict_inverts_observation_major_axes_in_the_shared_reverse() {
    let data: Vec<f64> = (0..24).map(|index| index as f64 * 0.2 - 1.5).collect();
    let weights = [0.3, -0.7, 1.1, -1.3, 0.5, 2.0];
    for dtype in [NumericDType::F32, NumericDType::F64] {
        let net = network(K::GlobalAveragePooling1D, dtype);
        let analytic = {
            let _activation = TapeActivation::enter(Rc::new(RefCell::new(Tape::new())));
            let x = annotate_dlarray_value(unformatted_input(data.clone(), dtype)).unwrap();
            let y = predict(&net, x.clone());
            assert_eq!(tensor(&y).shape(), &[3, 2]);
            assert_eq!(tensor(&y).numeric_dtype(), dtype);
            let loss = dlarray_sum_builtin(
                dlarray_times_builtin(y, values(weights.to_vec(), vec![3, 2], dtype)).unwrap(),
                crate::builtins::common::test_support::builtin_arguments(vec![Value::String(
                    "all".into(),
                )]),
            )
            .unwrap();
            let gradient = block_on(dlgradient_direct(
                loss,
                crate::builtins::common::test_support::builtin_arguments(vec![x]),
            ))
            .unwrap();
            assert_eq!(tensor(&gradient).shape(), &[2, 3, 4]);
            assert_eq!(tensor(&gradient).numeric_dtype(), dtype);
            tensor(&gradient).materialize_f64()
        };
        // Source is CBT; presented output is BC. Every time sample gets
        // its own channel/batch weight divided by the four pooled times.
        for (index, actual) in analytic.iter().enumerate() {
            let channel = index % 2;
            let batch = (index / 2) % 3;
            let expected = weights[batch + channel * 3] / 4.0;
            assert!((actual - expected).abs() < 1.0e-7);
        }
        let step = if dtype == NumericDType::F32 {
            0.001
        } else {
            0.00001
        };
        let tolerance = if dtype == NumericDType::F32 {
            0.003
        } else {
            0.00001
        };
        let objective = |data| {
            tensor(&predict(&net, unformatted_input(data, dtype)))
                .materialize_f64()
                .iter()
                .zip(weights)
                .map(|(value, weight)| value * weight)
                .sum::<f64>()
        };
        for index in 0..data.len() {
            let mut upper = data.clone();
            upper[index] += step;
            let mut lower = data.clone();
            lower[index] -= step;
            let expected = (objective(upper) - objective(lower)) / (2.0 * step);
            assert!((analytic[index] - expected).abs() < tolerance);
        }
    }
}
