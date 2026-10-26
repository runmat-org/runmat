//! Admit a semantic genome's returned coordinates through the enclosing
//! Pattern numeric policy; the caller retains the semantic result's roots.
use super::super::super::{invalid, numeric, ScalarCooperation, NUMERIC};
use crate::BuiltinResult;
use runmat_value::{NumericDType, Tensor, Value};

pub(super) async fn coordinates(
    value: Value,
    rows: usize,
    columns: usize,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Value> {
    let count = rows
        .checked_mul(columns)
        .ok_or_else(|| invalid("genetic projection extent overflow"))?;
    let value = numeric::prepare(
        value,
        "genetic projection",
        numeric::Geometry::Elements(count),
        &NUMERIC,
        cooperation,
    )
    .await?
    .ok_or_else(|| invalid("genetic genomes must project to real coordinate rows"))?;
    let shape = numeric::shape(&value)
        .ok_or_else(|| invalid("genetic coordinate geometry is unavailable"))?;
    if shape != [rows, columns]
        && !(rows == 1 && shape.iter().filter(|&&axis| axis != 1).count() <= 1)
    {
        return Err(invalid("genetic genomes must project to coordinate rows"));
    }
    match value {
        Value::Tensor(tensor) if tensor.numeric_dtype() == NumericDType::F64 => {
            Ok(Value::Tensor(tensor))
        }
        value @ Value::Num(_) => Ok(value),
        value => {
            let numeric::Numeric::Real { values, .. } =
                numeric::materialize_raw(value, "genetic projection", &NUMERIC, cooperation)
                    .await?
            else {
                return Err(invalid("genetic coordinates must be finite real values"));
            };
            Ok(Value::Tensor(
                Tensor::new(values, vec![rows, columns]).map_err(invalid)?,
            ))
        }
    }
}
