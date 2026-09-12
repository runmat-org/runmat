use runmat_value::{IntegerStorage, Tensor};

use super::tensor_char_codes_to_string;

#[test]
fn reads_typed_integer_storage_exactly() {
    let storages = [
        IntegerStorage::I8(vec![82, 77]),
        IntegerStorage::I16(vec![82, 77]),
        IntegerStorage::I32(vec![82, 77]),
        IntegerStorage::I64(vec![82, 77]),
        IntegerStorage::U8(vec![82, 77]),
        IntegerStorage::U16(vec![82, 77]),
        IntegerStorage::U32(vec![82, 77]),
        IntegerStorage::U64(vec![82, 77]),
    ];
    for storage in storages {
        let tensor = Tensor::new_integer(storage, vec![1, 2]).expect("tensor");
        assert_eq!(tensor_char_codes_to_string(&tensor).as_deref(), Some("RM"));
    }

    let negative = Tensor::new_integer(IntegerStorage::I16(vec![-1]), vec![1, 1]).expect("tensor");
    assert!(tensor_char_codes_to_string(&negative).is_none());

    let invalid =
        Tensor::new_integer(IntegerStorage::U32(vec![0x11_0000]), vec![1, 1]).expect("tensor");
    assert!(tensor_char_codes_to_string(&invalid).is_none());
}
