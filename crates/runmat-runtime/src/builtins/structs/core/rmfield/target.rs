use runmat_value::{CellArray, StructValue, Value};

pub(super) struct StructArray {
    pub(super) elements: Vec<StructValue>,
    shape: Vec<usize>,
}

pub(super) enum Target {
    Scalar(StructValue),
    Array(StructArray),
}

impl Target {
    pub(super) fn parse(value: Value) -> crate::BuiltinResult<Self> {
        match value {
            Value::Struct(structure) => Ok(Self::Scalar(structure)),
            Value::Cell(array) => StructArray::from_cell(array).map(Self::Array),
            other => Err(super::error::invalid_target(&other)),
        }
    }

    pub(super) fn into_value(self) -> crate::BuiltinResult<Value> {
        match self {
            Self::Scalar(structure) => Ok(Value::Struct(structure)),
            Self::Array(array) => array.into_cell().map(Value::Cell),
        }
    }
}

impl StructArray {
    fn from_cell(array: CellArray) -> crate::BuiltinResult<Self> {
        let mut elements = Vec::with_capacity(array.data.len());
        for value in array.data {
            let Value::Struct(structure) = value else {
                return Err(super::error::invalid_target_kind());
            };
            elements.push(structure);
        }
        Ok(Self {
            elements,
            shape: array.shape,
        })
    }

    pub(super) fn into_cell(self) -> crate::BuiltinResult<CellArray> {
        let values = self.elements.into_iter().map(Value::Struct).collect();
        CellArray::new_with_shape(values, self.shape).map_err(super::error::rebuild)
    }
}
