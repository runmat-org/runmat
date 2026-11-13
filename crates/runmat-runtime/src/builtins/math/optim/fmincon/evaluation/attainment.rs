//! Typed vector-objective integration into the shared constrained oracle.
//! Original callbacks never see the extra smooth attainment coordinate.
use super::{objective::Decoded, Evaluation, EvaluationStop, Evaluator, Point};
use crate::builtins::math::optim::common::multiobjective::{
    formulation::{capacity, Formulation, Specification},
    invalid, jacobian, Kind,
};
use crate::BuiltinResult;

pub(in crate::builtins::math::optim::fmincon) enum Binding {
    Unbound {
        kind: Kind,
        specification: Specification,
    },
    Bound(Formulation),
}
pub(in crate::builtins::math::optim::fmincon) struct Attainment {
    pub binding: Binding,
    pub merit: runmat_builtins::OptimMeritFunction,
}
impl Binding {
    pub fn kind(&self) -> Kind {
        match self {
            Self::Unbound { kind, .. } => *kind,
            Self::Bound(value) => value.kind,
        }
    }
}
pub(in crate::builtins::math::optim::fmincon) struct ObjectiveValues {
    pub values: Vec<f64>,
    pub shape: Vec<usize>,
}

impl Evaluator<'_> {
    pub async fn reuses_generation(
        &mut self,
        current: &[f64],
        trial: &[f64],
    ) -> BuiltinResult<bool> {
        if self.attainment.is_none() || current.len() != trial.len() || current.is_empty() {
            return Ok(false);
        }
        for (&current, &trial) in current[..current.len() - 1]
            .iter()
            .zip(&trial[..trial.len() - 1])
        {
            if current != trial {
                return Ok(false);
            }
            self.cooperation.completed(1).await?;
        }
        Ok(trial.last().is_some_and(|value| value.is_finite()))
    }

    /// Rebuild only the exact affine gamma contributions in an owned proposed
    /// original-row stage. The original evaluated Point remains borrowed.
    pub async fn stage_gamma(
        &mut self,
        point: &Point,
        gamma: f64,
        values: &mut super::super::model::ConstraintValues,
    ) -> BuiltinResult<bool> {
        let Some(Attainment {
            binding: Binding::Bound(formulation),
            ..
        }) = &self.attainment
        else {
            return Ok(false);
        };
        let payload = point
            .objectives
            .as_ref()
            .expect("bound original objective generation");
        if values.roles.len() != values.rhs.len() {
            return Err(invalid(
                formulation.kind,
                "proposed original row counts are inconsistent",
            ));
        }
        formulation.validate_extent(&payload.values)?;
        for (index, &role) in values.roles.iter().enumerate() {
            if let super::super::model::Role::InequalityNonlinear(row) = role {
                if let Some(row) = formulation.rows.get(row) {
                    let residual = row.direction.coefficient()
                        * (payload.values[row.objective] - row.goal)
                        - row.weight * gamma;
                    if !residual.is_finite() {
                        return Ok(false);
                    }
                    values.rhs[index] = -residual;
                }
            }
            self.cooperation.completed(1).await?;
        }
        Ok(true)
    }

    /// Only a validated original-x cache hit may retain an evaluated Point.
    /// The same proposed stage then supplies its new affine residuals.
    pub async fn accept_retained(
        &mut self,
        current: &[f64],
        trial: &[f64],
        point: &mut Point,
        values: &super::super::model::Linearization,
    ) -> BuiltinResult<()> {
        let kind = self.multiobjective_kind().ok_or_else(|| {
            super::super::invalid("retained point requires typed attainment mode")
        })?;
        if !self.reuses_generation(current, trial).await? {
            return Err(invalid(
                kind,
                "retained point does not match the proposed original-x generation",
            ));
        }
        if values.roles.len() != values.b.len() {
            return Err(invalid(
                kind,
                "retained original row counts are inconsistent",
            ));
        }
        for (&rhs, &role) in values.b.iter().zip(&values.roles) {
            if let super::super::model::Role::InequalityNonlinear(row) = role {
                let Some(value) = point.inequality.get_mut(row) else {
                    return Err(invalid(
                        kind,
                        "retained nonlinear row identity is inconsistent",
                    ));
                };
                *value = -rhs;
            }
            self.cooperation.completed(1).await?;
        }
        point.f = *trial.last().expect("validated attainment coordinate");
        Ok(())
    }

    pub fn callback_point<'a>(&self, x: &'a [f64]) -> BuiltinResult<&'a [f64]> {
        if let Some(binding) = &self.attainment {
            x.split_last()
                .map(|(_, x)| x)
                .ok_or_else(|| invalid(binding.binding.kind(), "missing attainment coordinate"))
        } else {
            Ok(x)
        }
    }
    pub fn original_variables(&self, variables: usize) -> usize {
        if self.attainment.is_some() {
            variables.saturating_sub(1)
        } else {
            variables
        }
    }
    pub fn attainment_derivative(&self, row: usize, equality: bool) -> f64 {
        if equality {
            return 0.0;
        }
        match &self.attainment {
            Some(Attainment {
                binding: Binding::Bound(formulation),
                ..
            }) => formulation.rows.get(row).map_or(0.0, |row| -row.weight),
            _ => unreachable!("attainment column requires a bound vector objective"),
        }
    }

    pub(super) async fn finish_objective(
        &mut self,
        x: &[f64],
        objective: Decoded,
        mut point: Point,
    ) -> BuiltinResult<Evaluation> {
        match objective {
            Decoded::Scalar {
                value, gradient, ..
            } => {
                point.f = value;
                point.gradient = gradient;
            }
            Decoded::Vector {
                values,
                shape,
                jacobian: objective_jacobian,
            } => {
                let Some(Attainment { binding, .. }) = &mut self.attainment else {
                    unreachable!("typed objective shape selection")
                };
                if let Binding::Unbound {
                    kind,
                    specification,
                } = binding
                {
                    let formulation = specification
                        .bind(*kind, values.len(), self.cooperation)
                        .await?;
                    *binding = Binding::Bound(formulation);
                }
                let Binding::Bound(formulation) = binding else {
                    unreachable!("bound objective identities")
                };
                let gamma = *x.last().expect("validated attainment coordinate");
                let Some(mut generated) = formulation
                    .residuals(&values, gamma, self.cooperation)
                    .await?
                else {
                    return Ok(Err(EvaluationStop::Undefined));
                };
                generated
                    .try_reserve(point.inequality.len())
                    .map_err(|source| {
                        crate::build_runtime_error(format!(
                            "{}: epigraph row allocation failed",
                            formulation.kind.name()
                        ))
                        .with_builtin(formulation.kind.name())
                        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
                        .with_source(source)
                        .build()
                    })?;
                for value in point.inequality {
                    generated.push(value);
                    self.cooperation.completed(1).await?;
                }
                point.inequality = generated;
                let mut gradient = capacity(x.len(), formulation.kind)?;
                for index in 0..x.len() {
                    gradient.push(if index + 1 == x.len() { 1.0 } else { 0.0 });
                    self.cooperation.completed(1).await?;
                }
                point.gradient = Some(gradient);
                point.inequality_jacobian = match (objective_jacobian, point.inequality_jacobian) {
                    (Some(objective), Some(original)) => {
                        let Some(generated) =
                            formulation.jacobian(objective, self.cooperation).await?
                        else {
                            return Ok(Err(EvaluationStop::Undefined));
                        };
                        Some(
                            jacobian::stack(
                                generated,
                                original,
                                formulation.kind,
                                self.cooperation,
                            )
                            .await?,
                        )
                    }
                    _ => None,
                };
                if let Some(matrix) = point.equality_jacobian.as_mut() {
                    // Existing original equality Jacobians are unchanged in x.
                    // The added column contains no entries, not a dense identity.
                    matrix.colptr.try_reserve(1).map_err(|source| {
                        crate::build_runtime_error(format!(
                            "{}: equality column allocation failed",
                            formulation.kind.name()
                        ))
                        .with_builtin(formulation.kind.name())
                        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
                        .with_source(source)
                        .build()
                    })?;
                    matrix.colptr.push(matrix.nzval.len());
                    matrix.n = matrix.n.checked_add(1).ok_or_else(|| {
                        invalid(formulation.kind, "equality column extent overflow")
                    })?;
                }
                point.f = gamma;
                point.objectives = Some(ObjectiveValues { values, shape });
            }
        }
        Ok(Ok(point))
    }
}
