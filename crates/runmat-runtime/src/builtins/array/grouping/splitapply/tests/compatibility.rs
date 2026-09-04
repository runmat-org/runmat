use super::*;
use runmat_value::{IntegerStorage, Tensor};

#[test]
fn fixed_width_group_numbers_are_gated_and_data_remains_exact() {
    let data = Value::Tensor(
        Tensor::new_integer(IntegerStorage::U64(vec![9, 7, 11, 8]), vec![4, 1]).unwrap(),
    );
    let groups = Value::Tensor(
        Tensor::new_integer(IntegerStorage::U8(vec![1, 2, 1, 2]), vec![4, 1]).unwrap(),
    );
    {
        let _mode = crate::compatibility::push_runmat_extensions_enabled(false);
        let error = call("max", data.clone(), vec![groups.clone()])
            .expect_err("fixed-width G must be gated");
        assert_eq!(
            error.identifier(),
            runmat_builtins::SPLITAPPLY_INTEGER_GROUP_EXTENSION.error_identifier
        );
    }
    let _mode = crate::compatibility::push_runmat_extensions_enabled(true);
    let output = call("max", data, vec![groups]).unwrap();
    let Value::Tensor(output) = output else {
        panic!("expected integer output")
    };
    assert_eq!(
        output.integer_storage(),
        Some(&IntegerStorage::U64(vec![11, 8]))
    );
}
