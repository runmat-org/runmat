use runmat_value::Value;

use super::FieldOrder;

pub(super) fn resolve(argument: &Value) -> crate::BuiltinResult<FieldOrder> {
    if let Some(names) = super::reference::parse(argument)? {
        return if names.is_empty() {
            Ok(FieldOrder(names))
        } else {
            Err(super::super::error::empty_struct_array())
        };
    }
    if let Some(names) = super::names::parse(argument)? {
        return if names.is_empty() {
            Ok(FieldOrder(names))
        } else {
            Err(super::super::error::no_fields())
        };
    }
    if let Some(count) = super::permutation::element_count(argument) {
        return if count == 0 {
            Ok(FieldOrder(Vec::new()))
        } else {
            Err(super::super::error::no_fields())
        };
    }
    Err(super::super::error::invalid_order())
}
