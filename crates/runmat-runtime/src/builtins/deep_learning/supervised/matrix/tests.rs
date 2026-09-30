use super::*;
use runmat_value::{LogicalArray, NumericDType};

#[test]
fn supervised_host_matrix_reads_single_and_logical_storage() {
    let single = Value::Tensor(
        Tensor::new_with_dtype(vec![1.5, -2.0], vec![2, 1], NumericDType::F32).unwrap(),
    );
    let parsed = numeric_matrix_ref(&single, "trainNetwork", "X").unwrap();
    assert_eq!((parsed.rows, parsed.cols), (2, 1));
    assert_eq!(parsed.data, vec![1.5, -2.0]);

    let logical = Value::LogicalArray(LogicalArray::new(vec![1, 0], vec![1, 2]).unwrap());
    let parsed = numeric_matrix_ref(&logical, "trainNetwork", "X").unwrap();
    assert_eq!((parsed.rows, parsed.cols), (1, 2));
    assert_eq!(parsed.data, vec![1.0, 0.0]);
}
