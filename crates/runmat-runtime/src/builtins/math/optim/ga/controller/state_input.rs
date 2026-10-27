//! Only fields the solver consumes are admitted. Extra user state/captures are
//! retained opaquely, never recursively gathered or used as numerical proof.
use super::super::{
    common, invalid, numeric,
    options::{admission, Owner},
    ScalarCooperation,
};
use super::frame::Flags;
use crate::{execution::RootedValue, BuiltinResult};
use runmat_builtins::{OptimGeneticPopulation, OptimGeneticStateField as Field};
use runmat_value::Value;

#[derive(Clone, Copy)]
pub(super) enum Mode {
    Raw,
    Penalty,
    Augmented,
}
pub(super) struct State<'a> {
    root: &'a RootedValue,
}
impl<'a> State<'a> {
    pub fn new(root: &'a RootedValue) -> BuiltinResult<Self> {
        if !matches!(root.value(), Value::Struct(_)) {
            return Err(invalid("observer state must be a scalar structure"));
        }
        Ok(Self { root })
    }
    pub fn get(&self, field: Field) -> Option<&'a Value> {
        let Value::Struct(value) = self.root.value() else {
            unreachable!("admitted state structure");
        };
        value.fields.get(field.name())
    }
    pub fn require(&self, field: Field) -> BuiltinResult<&'a Value> {
        self.get(field)
            .ok_or_else(|| invalid(format!("observer state requires {}", field.name())))
    }
    pub fn preflight(
        &self,
        kind: OptimGeneticPopulation,
        mode: Mode,
        rows: Option<usize>,
        cooperation: &ScalarCooperation,
    ) -> BuiltinResult<()> {
        let population = self.require(Field::Population)?;
        if kind != OptimGeneticPopulation::Custom {
            numeric::preflight(
                population,
                "state.Population",
                common::constrained_numeric::Geometry::MatrixAny,
                cooperation.runtime(),
            )?;
        }
        super::score_value::metadata(self.require(Field::Score)?, rows, cooperation)?;
        if let Some(best) = self.get(Field::Best) {
            numeric::preflight(
                best,
                "state.Best",
                common::constrained_numeric::Geometry::Vector,
                cooperation.runtime(),
            )?;
        }
        if matches!(mode, Mode::Penalty) {
            super::score_value::metadata(self.require(Field::Fitness)?, rows, cooperation)?;
        }
        for field in [Field::EvalElites, Field::HaveDuplicates] {
            if let Some(value) = self.get(field) {
                admission::numeric_metadata(
                    Owner::Solver,
                    admission::Role::Flag,
                    value,
                    cooperation.runtime(),
                )?;
                if !matches!(value, Value::GpuTensor(_)) {
                    boolean(value)?;
                }
            }
        }
        Ok(())
    }
    pub async fn flags(
        &self,
        previous: &Flags,
        cooperation: &ScalarCooperation,
    ) -> BuiltinResult<Flags> {
        let mut flags = Flags {
            eval_elites: previous.eval_elites,
            duplicates: previous.duplicates,
        };
        for (field, target) in [
            (Field::EvalElites, &mut flags.eval_elites),
            (Field::HaveDuplicates, &mut flags.duplicates),
        ] {
            if let Some(value) = self.get(field) {
                let value = crate::dispatcher::gather_if_needed_in_context_async(
                    value,
                    cooperation.runtime(),
                )
                .await?;
                *target = boolean(&value)?;
            }
        }
        Ok(flags)
    }
    pub fn root(
        &self,
        field: Field,
        cooperation: &ScalarCooperation,
    ) -> BuiltinResult<RootedValue> {
        RootedValue::in_optional_context(self.require(field)?.clone(), cooperation.runtime())
    }
}
fn boolean(value: &Value) -> BuiltinResult<bool> {
    if !matches!(
        value,
        Value::Bool(_) | Value::LogicalArray(_) | Value::Num(_) | Value::Int(_) | Value::Tensor(_)
    ) {
        return Err(invalid(
            "state flags require logical or numeric zero/one scalar",
        ));
    }
    super::super::options::switch(Owner::Solver, value)
}

pub(super) fn same_numeric(first: &Value, second: &Value) -> bool {
    match (first, second) {
        (Value::Tensor(a), Value::Tensor(b)) => {
            a.shape() == b.shape()
                && a.numeric_dtype() == runmat_value::NumericDType::F64
                && b.numeric_dtype() == runmat_value::NumericDType::F64
                && a.as_f64_slice().map(|value| value.as_ptr())
                    == b.as_f64_slice().map(|value| value.as_ptr())
        }
        (Value::ComplexTensor(a), Value::ComplexTensor(b)) => {
            a.shape() == b.shape()
                && a.numeric_dtype() == runmat_value::NumericDType::F64
                && b.numeric_dtype() == runmat_value::NumericDType::F64
                && a.as_f64_slice().map(|value| value.as_ptr())
                    == b.as_f64_slice().map(|value| value.as_ptr())
        }
        _ => false,
    }
}
