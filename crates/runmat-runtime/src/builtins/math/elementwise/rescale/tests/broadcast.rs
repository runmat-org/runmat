use runmat_value::Value;

use super::super::{broadcast::OperandBroadcast, rescale_builtin};
use super::support::{assert_close, tensor, values};

#[test]
fn operand_indexing_appends_trailing_singletons() {
    let operand = OperandBroadcast::new(&[2, 1], 3);
    assert_eq!(
        (0..6)
            .map(|linear| operand.index(linear, &[2, 1, 3]))
            .collect::<Vec<_>>(),
        vec![0, 1, 0, 1, 0, 1]
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn explicit_rank_zero_bounds_preserve_scalarization_and_rank_one_output() {
    let scalar = |value| tensor(vec![value], Vec::new());
    let bounds = || {
        vec![
            scalar(0.0),
            scalar(1.0),
            Value::from("InputMin"),
            scalar(0.0),
            Value::from("InputMax"),
            scalar(10.0),
        ]
    };

    let rank_zero = futures::executor::block_on(rescale_builtin(scalar(5.0), bounds()))
        .expect("rank-zero input and bounds remain scalar");
    assert!(matches!(rank_zero, Value::Num(value) if value == 0.5));

    let rank_one =
        futures::executor::block_on(rescale_builtin(tensor(vec![0.0, 10.0], vec![2]), bounds()))
            .expect("rank-one input broadcasts against rank-zero bounds");
    let (data, shape, _) = values(rank_one);
    assert_eq!(shape, vec![2]);
    assert_close(&data, &[0.0, 1.0]);
}

#[tokio::test]
async fn output_and_input_bounds_broadcast_together() {
    let result = rescale_builtin(
        tensor(vec![0.4, 0.5, 0.9, 0.2, -4.0, -5.0, 9.0, 1.0], vec![4, 2]),
        vec![
            tensor(vec![0.0, -1.0], vec![1, 2]),
            Value::Num(1.0),
            Value::from("InputMin"),
            tensor(vec![0.2, -5.0], vec![1, 2]),
            Value::from("InputMax"),
            tensor(vec![0.9, 9.0], vec![1, 2]),
        ],
    )
    .await
    .expect("broadcast bounds");
    assert_close(
        &values(result).0,
        &[
            2.0 / 7.0,
            3.0 / 7.0,
            1.0,
            0.0,
            -6.0 / 7.0,
            -1.0,
            1.0,
            -1.0 / 7.0,
        ],
    );
}
