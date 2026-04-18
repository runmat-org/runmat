use runmat_builtins::Value;
use runmat_vm::indexing::selectors::{
    indices_from_value_linear, selector_from_value_dim, SliceSelector,
};

#[test]
fn linear_false_bool_index_is_empty() {
    let indices = futures::executor::block_on(indices_from_value_linear(&Value::Bool(false), 4))
        .expect("false logical index should be empty");
    assert!(indices.is_empty());
}

#[test]
fn linear_true_bool_index_selects_first() {
    let indices = futures::executor::block_on(indices_from_value_linear(&Value::Bool(true), 4))
        .expect("true logical index should select first element");
    assert_eq!(indices, vec![1]);
}

#[test]
fn dim_false_bool_selector_is_empty() {
    let selector = futures::executor::block_on(selector_from_value_dim(&Value::Bool(false), 4))
        .expect("false logical selector should be empty");
    match selector {
        SliceSelector::Indices(indices) => assert!(indices.is_empty()),
        SliceSelector::Scalar(_) | SliceSelector::Colon | SliceSelector::LinearIndices { .. } => {
            panic!("expected empty indices selector")
        }
    }
}
