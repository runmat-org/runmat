//! One owned generation, with no numeric mirror of a custom representation.
use super::super::callbacks::{take_direct, Direct, Invocation};
use super::super::{common, invalid, numeric, ScalarCooperation, POLICY};
use crate::{execution::RootedValue, BuiltinResult};
use common::population::projection::{Argument, Arguments, Prepared};
use runmat_builtins::OptimGeneticPopulation;
use runmat_value::{NumericDType, Tensor, Value};
use std::ops::ControlFlow;

pub(in super::super) enum Generation {
    Numeric(common::population::Population),
    Custom {
        value: super::custom::Custom,
        variables: usize,
    },
}
impl Generation {
    pub async fn from_result(
        value: RootedValue,
        members: usize,
        variables: usize,
        kind: OptimGeneticPopulation,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Self> {
        Self::from_result_with(value, members, variables, kind, &mut Direct, cooperation)
            .await
            .map(take_direct)
    }
    pub async fn from_result_with<I: Invocation>(
        value: RootedValue,
        members: usize,
        variables: usize,
        kind: OptimGeneticPopulation,
        invocation: &mut I,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<ControlFlow<I::Stop, Self>> {
        if kind == OptimGeneticPopulation::Custom {
            return Ok(
                match super::custom::Custom::from_result(value, members, invocation, cooperation)
                    .await?
                {
                    ControlFlow::Continue(value) => {
                        ControlFlow::Continue(Self::Custom { value, variables })
                    }
                    ControlFlow::Break(stop) => ControlFlow::Break(stop),
                },
            );
        }
        let tensor = value
            .with_owned_async(|value| async {
                numeric_tensor(value, members, variables, cooperation).await
            })
            .await?;
        let population = common::population::Population::new(POLICY, tensor, members, variables)?;
        for &value in population
            .positions
            .as_f64_slice()
            .expect("double population")
        {
            if !value.is_finite()
                || kind == OptimGeneticPopulation::BitString && value != 0. && value != 1.
            {
                return Err(invalid(
                    "population coordinates must be finite; bitstrings require exact zero or one",
                ));
            }
            cooperation.completed(1).await?;
        }
        Ok(ControlFlow::Continue(Self::Numeric(population)))
    }
    pub fn variables(&self) -> usize {
        match self {
            Self::Numeric(value) => value.variables,
            Self::Custom { variables, .. } => *variables,
        }
    }
    pub fn value(&self) -> Value {
        match self {
            Self::Numeric(value) => value.value(),
            Self::Custom { value, .. } => value.value().clone(),
        }
    }
    pub fn numeric(&self) -> Option<&common::population::Population> {
        match self {
            Self::Numeric(value) => Some(value),
            _ => None,
        }
    }
    pub async fn point_with<I: Invocation>(
        &self,
        row: usize,
        invocation: &mut I,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<ControlFlow<I::Stop, RootedValue>> {
        let prepared = self.prepare_projection(&[row], cooperation).await?;
        let argument = match invocation.projection(prepared, cooperation).await? {
            ControlFlow::Continue(argument) => argument,
            ControlFlow::Break(stop) => return Ok(ControlFlow::Break(stop)),
        };
        Ok(ControlFlow::Continue(match argument {
            Argument::Rooted(value) => value,
            Argument::Source(value) => {
                RootedValue::in_optional_context(value, cooperation.runtime())?
            }
        }))
    }
    pub async fn replace_with<I: Invocation>(
        &mut self,
        rows: &[usize],
        children: Generation,
        invocation: &mut I,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<ControlFlow<I::Stop>> {
        if rows.len() != children.members() || self.variables() != children.variables() {
            return Err(invalid(
                "offspring extent differs from selected destination",
            ));
        }
        match (self, children) {
            (Self::Numeric(destination), Self::Numeric(children)) => {
                destination.writable(cooperation).await?;
                for column in 0..destination.variables {
                    for (child, &row) in rows.iter().enumerate() {
                        if row >= destination.members {
                            return Err(invalid("offspring destination is outside population"));
                        }
                        let value = children.get(child, column);
                        destination
                            .positions
                            .try_as_f64_slice_mut_unique()
                            .expect("writable population")[row + destination.members * column] =
                            value;
                        cooperation.completed(1).await?;
                    }
                }
                Ok(ControlFlow::Continue(()))
            }
            (
                Self::Custom {
                    value: destination, ..
                },
                Self::Custom {
                    value: children, ..
                },
            ) => {
                destination
                    .replace_rows(rows, &children, invocation, cooperation)
                    .await
            }
            _ => Err(invalid("an operator changed PopulationType")),
        }
    }
}
impl Arguments for Generation {
    fn known_coordinates(&self) -> Option<common::population::projection::Coordinates<'_>> {
        let (tensor, members, variables) = match self {
            Self::Numeric(population) => (
                &population.positions,
                population.members,
                population.variables,
            ),
            Self::Custom { value, variables } => match value.value() {
                Value::Tensor(tensor) if tensor.shape() == [value.members(), *variables] => {
                    (tensor, value.members(), *variables)
                }
                _ => return None,
            },
        };
        Some(common::population::projection::Coordinates {
            values: tensor.as_f64_slice()?,
            members,
            variables,
        })
    }
    fn numeric_rows(&self) -> Option<&common::population::Population> {
        self.numeric()
    }
    fn members(&self) -> usize {
        match self {
            Self::Numeric(value) => value.members,
            Self::Custom { value, .. } => value.members(),
        }
    }
    async fn prepare_projection<'a>(
        &'a self,
        rows: &[usize],
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Prepared<'a>> {
        match self {
            Self::Numeric(value) => value.prepare_projection(rows, cooperation).await,
            Self::Custom { value, .. } => value.prepare_projection(rows, cooperation).await,
        }
    }
}
pub(in super::super) async fn numeric_tensor(
    value: Value,
    rows: usize,
    columns: usize,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Tensor> {
    let Some(value) = common::constrained_numeric::prepare(
        value,
        "population",
        common::constrained_numeric::Geometry::Matrix(rows, columns),
        &numeric::INPUT_POLICY,
        cooperation,
    )
    .await?
    else {
        return Err(invalid("numeric population requires a real matrix"));
    };
    match value {
        Value::Tensor(value) if value.numeric_dtype() == NumericDType::F64 => Ok(value),
        value => {
            let common::constrained_numeric::Numeric::Real { values, .. } =
                common::constrained_numeric::materialize_raw(
                    value,
                    "population",
                    &numeric::INPUT_POLICY,
                    cooperation,
                )
                .await?
            else {
                return Err(invalid("numeric population requires a real matrix"));
            };
            Tensor::new(values, vec![rows, columns]).map_err(invalid)
        }
    }
}
