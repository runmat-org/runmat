use super::*;
use runmat_value::{IntValue, Tensor};

#[test]
fn fixed_width_controls_are_gated_without_gating_documented_double_controls() {
    let input = Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap());
    let _mode = crate::compatibility::push_runmat_extensions_enabled(false);
    let typed = call(input.clone(), vec![Value::Int(IntValue::U8(2))]).unwrap_err();
    assert_eq!(
        typed.identifier(),
        runmat_builtins::GROUPCOUNTS_INTEGER_CONTROL_EXTENSION.error_identifier
    );
    assert!(call(input, vec![Value::Num(2.0)]).is_ok());
}
