use runmat_value::{CharArray, Value};

pub(super) fn path_list(folders: &[String]) -> Value {
    Value::CharArray(CharArray::new_row(&super::super::path_list::join(folders)))
}
