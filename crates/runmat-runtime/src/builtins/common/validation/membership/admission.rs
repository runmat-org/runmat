//! Content-independent admission precedes every membership download.
use super::{invalid_argument_error, BuiltinResult, CellLeaves, Value};

pub(super) fn shape_and_kind(value: &Value) -> BuiltinResult<()> {
    for leaf in CellLeaves::new(value) {
        match leaf {
            Value::Num(_)
            | Value::Int(_)
            | Value::Bool(_)
            | Value::Complex(_, _)
            | Value::String(_)
            | Value::Tensor(_)
            | Value::ComplexTensor(_)
            | Value::SparseTensor(_)
            | Value::LogicalArray(_)
            | Value::StringArray(_)
            | Value::GpuTensor(_) => {}
            Value::CharArray(chars) if chars.rows == 1 => {}
            _ => {
                return Err(
                    invalid_argument_error("mustBeMember", "unsupported member value type").into(),
                )
            }
        }
    }
    Ok(())
}

pub(super) fn resident_metadata(value: &Value) -> BuiltinResult<()> {
    for leaf in CellLeaves::new(value) {
        super::super::validate_resident_metadata(leaf)?;
    }
    Ok(())
}
