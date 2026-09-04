mod categorical;
mod levels;
mod numeric;
mod temporal;
mod text;

use runmat_value::{NumericScalar, Value};

use crate::BuiltinResult;

use super::super::keys::{GroupIndex, KeyAtom, KeyOrder};
use super::error;

pub(super) struct GroupingInput {
    pub(super) value: Value,
    pub(super) rows: Vec<Option<Vec<KeyAtom>>>,
    pub(super) order: KeyOrder,
    pub(super) categorical: bool,
}

impl GroupingInput {
    pub(super) fn prepare(value: Value) -> BuiltinResult<Self> {
        match value {
            Value::Tensor(value) => numeric::tensor(value),
            Value::Num(value) => Ok(Self::scalar(Value::Num(value), KeyAtom::Number(value))),
            Value::Int(value) => Ok(Self::scalar(
                Value::Int(value.clone()),
                KeyAtom::Integer(value),
            )),
            Value::LogicalArray(value) => numeric::logical(value),
            Value::Bool(value) => Ok(Self::scalar(Value::Bool(value), KeyAtom::Logical(value))),
            Value::StringArray(value) => text::strings(value),
            Value::String(value) => Ok(text::scalar(value)),
            Value::Cell(value) => text::cellstr(value),
            Value::CharArray(value) => text::characters(value),
            Value::Object(value) if value.is_class(runmat_types::standard::CATEGORICAL) => {
                categorical::prepare(value)
            }
            Value::Object(value) if value.is_class(runmat_types::standard::DATETIME) => {
                temporal::datetime(value)
            }
            Value::Object(value) if value.is_class(runmat_types::standard::DURATION) => {
                temporal::duration(value)
            }
            Value::SparseTensor(_) => Err(error::invalid("grp2idx: sparse input is not supported")),
            Value::Complex(_, _) | Value::ComplexTensor(_) => {
                Err(error::invalid("grp2idx: complex input is not supported"))
            }
            other => Err(error::invalid(format!(
                "grp2idx: unsupported grouping input {other:?}"
            ))),
        }
    }

    pub(super) fn index(&self) -> BuiltinResult<GroupIndex> {
        GroupIndex::build(self.rows.clone(), self.order.clone()).map_err(error::invalid)
    }

    pub(super) fn levels(&self, index: &GroupIndex) -> BuiltinResult<Value> {
        levels::build(self, index)
    }

    pub(super) fn new(value: Value, rows: Vec<Option<Vec<KeyAtom>>>, order: KeyOrder) -> Self {
        Self {
            value,
            rows,
            order,
            categorical: false,
        }
    }

    fn scalar(value: Value, atom: KeyAtom) -> Self {
        Self::new(value, vec![Some(vec![atom])], KeyOrder::Sorted)
    }
}

pub(super) fn numeric_atom(value: NumericScalar) -> Option<Vec<KeyAtom>> {
    match value {
        NumericScalar::F64(value) if value.is_nan() => None,
        NumericScalar::F64(value) => Some(vec![KeyAtom::Number(value)]),
        NumericScalar::F32(value) if value.is_nan() => None,
        NumericScalar::F32(value) => Some(vec![KeyAtom::Number(f64::from(value))]),
        value => value
            .into_int_value()
            .map(|value| vec![KeyAtom::Integer(value)]),
    }
}

pub(super) fn text_atom(value: &str) -> Option<Vec<KeyAtom>> {
    (!crate::builtins::strings::common::is_missing_string(value))
        .then(|| vec![KeyAtom::Text(value.into())])
}

pub(super) fn ensure_vector(shape: &[usize], kind: &str) -> BuiltinResult<()> {
    if shape.iter().filter(|dimension| **dimension > 1).count() <= 1 {
        Ok(())
    } else {
        Err(error::invalid(format!(
            "grp2idx: {kind} input must be a vector"
        )))
    }
}
