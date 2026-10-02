use super::*;
use runmat_value::{IntValue, IntegerStorage};

#[test]
fn scalar_group_label_reads_typed_integer_storage_without_double_mirror() {
    let tensor = Tensor::new_integer(IntegerStorage::U16(vec![42]), vec![1, 1]).unwrap();
    let Some(label) = scalar_group_label(&Value::Tensor(tensor)).unwrap() else {
        panic!("expected exact integer label");
    };
    assert!(matches!(
        label.as_slice(),
        [KeyAtom::Integer(IntValue::U16(42))]
    ));
}
