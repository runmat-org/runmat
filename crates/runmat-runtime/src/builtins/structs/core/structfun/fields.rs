use runmat_value::{StructValue, Value};

use super::error;

pub(super) struct Fields {
    structure: StructValue,
}

impl Fields {
    pub(super) fn parse(value: Value) -> crate::BuiltinResult<Self> {
        let Value::Struct(structure) = value else {
            return Err(error::not_scalar(
                "structfun: second input must be a scalar struct",
            ));
        };
        Ok(Self { structure })
    }

    pub(super) fn iter(&self) -> impl ExactSizeIterator<Item = (&str, &Value)> + Clone {
        self.structure.fields.iter()
    }

    pub(super) fn len(&self) -> usize {
        self.structure.fields.len()
    }
}
