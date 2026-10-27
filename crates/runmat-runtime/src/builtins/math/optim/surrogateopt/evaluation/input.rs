//! One actual callback argument and its returned-generation decoder.
//! Serial, vectorized and accepted worker calls share this owned point header.
use super::{invalid, observation, Completed, DoubleArray, Form, Point, ScalarCooperation};
use crate::{
    execution::{RootedValue, RootedValueSequence},
    BuiltinResult,
};
use runmat_value::{Tensor, Value};
use std::num::NonZeroUsize;

pub(in super::super) struct Input {
    argument: RootedValue,
    points: DoubleArray,
    rows: NonZeroUsize,
    variables: usize,
    form: Form,
    completed: Vec<Completed>,
}
impl Input {
    pub(in super::super) async fn new(
        coordinates: Vec<f64>,
        variables: usize,
        form: Form,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Self> {
        cooperation.completed(0).await?;
        let rows = match form {
            Form::Point => NonZeroUsize::MIN,
            Form::Vectorized(rows) => rows,
            Form::Initial(_) => {
                return Err(invalid(
                    "initial measurement decoding is not a callback argument route",
                ))
            }
        };
        let length = rows
            .get()
            .checked_mul(variables)
            .ok_or_else(|| invalid("objective batch extent overflow"))?;
        if variables == 0 || coordinates.len() != length {
            return Err(invalid(
                "objective point changed its original variable extent",
            ));
        }
        for value in &coordinates {
            if !value.is_finite() {
                return Err(invalid("objective point contains a nonfinite coordinate"));
            }
            cooperation.completed(1).await?;
        }
        crate::context::host_allocation::require_unbounded_workspace(
            cooperation.runtime(),
            super::super::NAME,
            "finite host-memory execution has no complete surrogate callback resource bound",
        )?;
        // Allocate result-row metadata before invoking the selected callback.
        let completed = runmat_value::exact_capacity_vec(rows.get()).map_err(|source| {
            crate::build_runtime_error("surrogateopt: completed-row metadata allocation failed")
                .with_builtin(super::super::NAME)
                .with_source(source)
                .build()
        })?;
        let (argument, points) = {
            let runtime = cooperation.runtime();
            let _entry = runtime.map(crate::context::RuntimeContext::enter);
            let tensor = Tensor::new(coordinates, vec![rows.get(), variables])
                .map_err(|detail| invalid(format!("invalid objective point storage: {detail}")))?;
            let points = DoubleArray::owned_point(tensor.clone())?;
            let argument = RootedValue::in_optional_context(Value::Tensor(tensor), runtime)?;
            (argument, points)
        };
        Ok(Self {
            argument,
            points,
            rows,
            variables,
            form,
            completed,
        })
    }
    pub(in super::super) fn argument(&self) -> &RootedValue {
        &self.argument
    }
    pub(in super::super) fn scalar(&self) -> bool {
        matches!(self.form, Form::Point)
    }
    pub(in super::super) async fn same_point(
        &self,
        point: &[f64],
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<bool> {
        if !self.scalar() || point.len() != self.variables {
            return Ok(false);
        }
        for (column, &value) in point.iter().enumerate() {
            if self.points.get(column) != value {
                return Ok(false);
            }
            cooperation.completed(1).await?;
        }
        Ok(true)
    }
    pub(in super::super) async fn finish(
        mut self,
        returned: RootedValueSequence,
        tolerance: f64,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Vec<Completed>> {
        let observations = observation::decode(returned, self.form, tolerance, cooperation).await?;
        if observations.len() != self.rows.get() {
            return Err(invalid("objective decoder changed its actual row count"));
        }
        for (row, observation) in observations.into_iter().enumerate() {
            let point =
                Point::batch_row(self.points.clone(), row, self.rows.get(), self.variables)?;
            self.completed.push(Completed { point, observation });
            cooperation.completed(1).await?;
        }
        // Original input roots span dispatch AND returned-payload decoding.
        drop(self.argument);
        Ok(self.completed)
    }
}
