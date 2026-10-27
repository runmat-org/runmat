//! Admitted scalar-optimizer controls after one public-key decoding pass.

use runmat_builtins::OptimOptionField;
use runmat_value::{StructValue, Value};
#[cfg(test)]
#[path = "scalar_options/fields_tests.rs"]
mod fields_tests;

mod preparation;
pub(crate) use preparation::{
    prepare_fmincon_options, prepare_modeled_fminunc_options, prepare_modeled_residual_options,
    prepare_multiobjective_options, prepare_solver_options, switch_value,
};

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
    order: usize,
}

pub(crate) struct ScalarOptionFields<'a> {
    bindings: [Option<ScalarOptionBinding<'a>>; OptimOptionField::ALL.len()],
}
impl Default for ScalarOptionFields<'_> {
    fn default() -> Self {
        Self {
            bindings: std::array::from_fn(|_| None),
        }
    }
}
fn index(field: OptimOptionField) -> usize {
    OptimOptionField::ALL
        .iter()
        .position(|&candidate| candidate == field)
        .expect("every typed field belongs to ALL")
}

impl<'a> ScalarOptionFields<'a> {
    pub(crate) fn switch(
        &self,
        builtin: &str,
        field: OptimOptionField,
    ) -> crate::BuiltinResult<bool> {
        self.get(field)
            .map(|value| preparation::switch_value(builtin, field.name(), value))
            .transpose()
            .map(|value| value.unwrap_or(false))
    }
    pub(crate) fn from_struct(options: Option<&'a StructValue>) -> Self {
        Self::from_decoded(options, OptimOptionField::parse)
    }
    pub(crate) fn from_struct_for_solver(
        options: Option<&'a StructValue>,
        solver: runmat_builtins::OptimSolver,
    ) -> Self {
        Self::from_decoded(options, |name| {
            OptimOptionField::parse_for_solver(name, solver)
        })
    }
    fn from_decoded(
        options: Option<&'a StructValue>,
        decode: impl Fn(&str) -> Option<OptimOptionField>,
    ) -> Self {
        let mut fields = Self::default();
        if let Some(options) = options {
            for (order, (name, value)) in options.fields.iter().enumerate() {
                if matches!(value,Value::Tensor(value)if value.is_empty()) {
                    continue;
                }
                if let Some(field) = decode(name) {
                    fields.bindings[index(field)].get_or_insert(ScalarOptionBinding {
                        name,
                        value,
                        order,
                    });
                }
            }
        }
        fields
    }
    pub(crate) fn get(&self, field: OptimOptionField) -> Option<&'a Value> {
        self.binding(field).map(|binding| binding.value)
    }
    pub(crate) fn exact(&self, field: OptimOptionField) -> Option<&'a Value> {
        self.binding_exact(field).map(|binding| binding.value)
    }
    fn binding_exact(&self, field: OptimOptionField) -> Option<ScalarOptionBinding<'a>> {
        self.bindings[index(field)]
    }
    fn binding_for(
        &self,
        solver: runmat_builtins::OptimSolver,
        field: OptimOptionField,
    ) -> Option<ScalarOptionBinding<'a>> {
        if matches!(
            solver,
            runmat_builtins::OptimSolver::Fminunc
                | runmat_builtins::OptimSolver::Fmincon
                | runmat_builtins::OptimSolver::Fseminf
                | runmat_builtins::OptimSolver::Fminimax
                | runmat_builtins::OptimSolver::Fgoalattain
                | runmat_builtins::OptimSolver::Lsqnonlin
        ) {
            self.binding_exact(field)
        } else {
            self.binding(field)
        }
    }
    fn binding(&self, field: OptimOptionField) -> Option<ScalarOptionBinding<'a>> {
        if field == OptimOptionField::TolFun {
            // Preserve the historical first-binding semantics for existing
            // solver profiles. fminunc's algorithm-aware consumer uses exact
            // semantic fields and checks conflicting legacy/modern controls.
            [
                OptimOptionField::TolFun,
                OptimOptionField::OptimalityTolerance,
                OptimOptionField::FunctionTolerance,
            ]
            .into_iter()
            .filter_map(|field| self.binding_exact(field))
            .min_by_key(|binding| binding.order)
        } else {
            self.binding_exact(field)
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
