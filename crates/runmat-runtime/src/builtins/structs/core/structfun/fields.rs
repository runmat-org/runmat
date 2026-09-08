use runmat_value::{StructValue, Value};

use super::error;

pub(super) struct Fields {
    pub(super) names: Vec<String>,
    pub(super) values: Vec<Value>,
}

impl Fields {
    pub(super) fn parse(value: Value) -> crate::BuiltinResult<Self> {
        let Value::Struct(structure) = value else {
            return Err(error::not_scalar(
                "structfun: second input must be a scalar struct",
            ));
        };
        Ok(Self::from_structure(structure))
    }

    fn from_structure(structure: StructValue) -> Self {
        let mut names = Vec::with_capacity(structure.fields.len());
        let mut values = Vec::with_capacity(structure.fields.len());
        for (name, value) in structure.fields {
            names.push(name);
            values.push(value);
        }
        Self { names, values }
    }
}
