//! Admitted scalar-optimizer controls after one public-key decoding pass.

use runmat_builtins::OptimOptionField;
use runmat_value::{StructValue, Value};

#[derive(Default)]
pub(crate) struct ScalarOptionFields<'a> {
    tol_x: Option<&'a Value>,
    max_iter: Option<&'a Value>,
    max_fun_evals: Option<&'a Value>,
    display: Option<&'a Value>,
}

impl<'a> ScalarOptionFields<'a> {
    pub(crate) fn from_struct(options: Option<&'a StructValue>) -> Self {
        let mut fields = Self::default();
        if let Some(options) = options {
            for (name, value) in &options.fields {
                match OptimOptionField::parse(name) {
                    Some(OptimOptionField::TolX) => fields.tol_x.get_or_insert(value),
                    Some(OptimOptionField::MaxIter) => fields.max_iter.get_or_insert(value),
                    Some(OptimOptionField::MaxFunEvals) => {
                        fields.max_fun_evals.get_or_insert(value)
                    }
                    Some(OptimOptionField::Display) => fields.display.get_or_insert(value),
                    _ => continue,
                };
            }
        }
        fields
    }

    pub(crate) fn get(&self, field: OptimOptionField) -> Option<&'a Value> {
        match field {
            OptimOptionField::TolX => self.tol_x,
            OptimOptionField::MaxIter => self.max_iter,
            OptimOptionField::MaxFunEvals => self.max_fun_evals,
            OptimOptionField::Display => self.display,
            _ => None,
        }
    }
}
