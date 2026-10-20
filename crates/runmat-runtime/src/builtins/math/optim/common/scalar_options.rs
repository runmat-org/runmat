//! Admitted scalar-optimizer controls after one public-key decoding pass.

use runmat_builtins::OptimOptionField;
use runmat_value::{StructValue, Value};

mod preparation;
pub(crate) use preparation::prepare_solver_options;

/// One prepared immutable generation and its original managed root ownership.
pub(crate) struct PreparedSolverOptions(crate::execution::RootedValue);

impl PreparedSolverOptions {
    pub(crate) fn structure(&self) -> &StructValue {
        let Value::Struct(value) = self.0.value() else {
            unreachable!("solver option preparation publishes a structure")
        };
        value
    }
}

#[derive(Clone, Copy)]
struct ScalarOptionBinding<'a> {
    name: &'a str,
    value: &'a Value,
}

#[derive(Default)]
pub(crate) struct ScalarOptionFields<'a> {
    tol_x: Option<ScalarOptionBinding<'a>>,
    tol_fun: Option<ScalarOptionBinding<'a>>,
    max_iter: Option<ScalarOptionBinding<'a>>,
    max_fun_evals: Option<ScalarOptionBinding<'a>>,
    display: Option<ScalarOptionBinding<'a>>,
    algorithm: Option<ScalarOptionBinding<'a>>,
    specify_gradient: Option<ScalarOptionBinding<'a>>,
}

impl<'a> ScalarOptionFields<'a> {
    pub(crate) fn from_struct(options: Option<&'a StructValue>) -> Self {
        let mut fields = Self::default();
        if let Some(options) = options {
            for (name, value) in &options.fields {
                let binding = ScalarOptionBinding { name, value };
                match OptimOptionField::parse(name) {
                    Some(OptimOptionField::TolX) => fields.tol_x.get_or_insert(binding),
                    Some(OptimOptionField::TolFun) => fields.tol_fun.get_or_insert(binding),
                    Some(OptimOptionField::MaxIter) => fields.max_iter.get_or_insert(binding),
                    Some(OptimOptionField::MaxFunEvals) => {
                        fields.max_fun_evals.get_or_insert(binding)
                    }
                    Some(OptimOptionField::Display) => fields.display.get_or_insert(binding),
                    Some(OptimOptionField::Algorithm) => fields.algorithm.get_or_insert(binding),
                    Some(OptimOptionField::SpecifyObjectiveGradient) => {
                        fields.specify_gradient.get_or_insert(binding)
                    }
                    _ => continue,
                };
            }
        }
        fields
    }

    pub(crate) fn get(&self, field: OptimOptionField) -> Option<&'a Value> {
        self.binding(field).map(|binding| binding.value)
    }

    fn binding(&self, field: OptimOptionField) -> Option<ScalarOptionBinding<'a>> {
        match field {
            OptimOptionField::TolX => self.tol_x,
            OptimOptionField::TolFun => self.tol_fun,
            OptimOptionField::MaxIter => self.max_iter,
            OptimOptionField::MaxFunEvals => self.max_fun_evals,
            OptimOptionField::Display => self.display,
            OptimOptionField::Algorithm => self.algorithm,
            OptimOptionField::SpecifyObjectiveGradient => self.specify_gradient,
        }
    }

    pub(crate) fn real(
        &self,
        builtin: &str,
        field: OptimOptionField,
        default: f64,
    ) -> crate::BuiltinResult<f64> {
        super::option_f64_value(builtin, self.get(field), field.name(), default)
    }

    pub(crate) fn count(
        &self,
        builtin: &str,
        field: OptimOptionField,
        default: usize,
    ) -> crate::BuiltinResult<usize> {
        super::option_usize_value(builtin, self.get(field), field.name(), default)
    }

    pub(crate) fn text(
        &self,
        field: OptimOptionField,
        default: &str,
    ) -> crate::BuiltinResult<String> {
        super::option_string_value(self.get(field), field.name(), default)
    }

    pub(crate) fn count_product_default(
        &self,
        builtin: &str,
        field: OptimOptionField,
        factor: usize,
        base: usize,
    ) -> crate::BuiltinResult<usize> {
        if self.get(field).is_some() {
            return self.count(builtin, field, 0);
        }
        factor.checked_mul(base).ok_or_else(|| {
            super::optim_error(
                builtin,
                format!("{builtin}: default {} exceeds supported size", field.name()),
            )
        })
    }
}
