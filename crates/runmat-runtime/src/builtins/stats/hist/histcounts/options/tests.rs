use super::*;
use runmat_value::IntegerStorage;

#[test]
fn integer_edge_admission_preserves_exact_boundaries() {
    let input =
        Value::Tensor(Tensor::new_integer(IntegerStorage::U16(vec![2, 4, 6]), vec![1, 3]).unwrap());
    match classify_bin_argument(&input).unwrap() {
        BinArgument::Edges(display, exact) => {
            assert_eq!(display, vec![2.0, 4.0, 6.0]);
            assert_eq!(
                exact,
                Some(vec![IntValue::U16(2), IntValue::U16(4), IntValue::U16(6)])
            );
        }
        other => panic!("expected exact edge vector, got {other:?}"),
    }
}
