//! Typed globalization/model hooks; the ordinary scalar oracle is unchanged.
use super::super::{
    model::{ConstraintValues, Linearization},
    multiobjective::merit::{self, Terms},
};
use super::{Attainment, Binding, Evaluator, Point};
use crate::builtins::math::optim::common::multiobjective::Kind;
use crate::BuiltinResult;

impl Evaluator<'_> {
    pub fn multiobjective_kind(&self) -> Option<Kind> {
        self.attainment.as_ref().map(|value| value.binding.kind())
    }
    pub fn uses_multiobjective_merit(&self) -> bool {
        self.attainment
            .as_ref()
            .is_some_and(|value| value.merit == runmat_builtins::OptimMeritFunction::Multiobjective)
    }
    pub async fn merit_model(
        &mut self,
        point: &Point,
        original: &Linearization,
        activity: &[f64],
        direction: &[f64],
    ) -> BuiltinResult<Option<(Terms, Terms)>> {
        let Some(Attainment {
            binding: Binding::Bound(formulation),
            ..
        }) = &self.attainment
        else {
            unreachable!("multiobjective merit requires a bound objective")
        };
        let Some(current) = merit::current(
            formulation,
            point,
            &original.b,
            &original.roles,
            original.equalities,
            self.cooperation,
        )
        .await?
        else {
            return Ok(None);
        };
        let Some(predicted) = merit::predicted(
            formulation,
            point,
            original,
            activity,
            direction,
            self.cooperation,
        )
        .await?
        else {
            return Ok(None);
        };
        Ok(Some((current, predicted)))
    }
    pub async fn trial_merit(
        &mut self,
        point: &Point,
        gamma: f64,
        values: &ConstraintValues,
    ) -> BuiltinResult<Option<Terms>> {
        if self.uses_multiobjective_merit() {
            let Some(Attainment {
                binding: Binding::Bound(formulation),
                ..
            }) = &self.attainment
            else {
                unreachable!("bound multiobjective merit")
            };
            return merit::current(
                formulation,
                point,
                &values.rhs,
                &values.roles,
                values.equalities,
                self.cooperation,
            )
            .await;
        }
        let mut violation = 0.0;
        for (row, &rhs) in values.rhs.iter().enumerate() {
            violation += if row < values.equalities {
                rhs.abs()
            } else {
                (-rhs).max(0.0)
            };
            if !violation.is_finite() {
                return Ok(None);
            }
            self.cooperation.completed(1).await?;
        }
        Ok(Some(Terms {
            objective: if self.attainment.is_some() {
                gamma
            } else {
                point.f
            },
            violation,
        }))
    }
    pub async fn reported_objective(&mut self, point: &Point) -> BuiltinResult<Option<f64>> {
        if self.uses_multiobjective_merit() {
            let Some(Attainment {
                binding: Binding::Bound(formulation),
                ..
            }) = &self.attainment
            else {
                unreachable!("bound multiobjective objective")
            };
            return formulation
                .worst(
                    &point.objectives.as_ref().expect("vector payload").values,
                    self.cooperation,
                )
                .await;
        }
        Ok(Some(point.f))
    }
    pub async fn seed_attainment(
        &mut self,
        x: &mut [f64],
        point: &mut Point,
    ) -> BuiltinResult<bool> {
        let Some(Attainment {
            binding: Binding::Bound(formulation),
            ..
        }) = &self.attainment
        else {
            return Ok(true);
        };
        let values = &point
            .objectives
            .as_ref()
            .expect("evaluated vector payload")
            .values;
        let Some(gamma) = formulation.worst(values, self.cooperation).await? else {
            return Ok(false);
        };
        let Some(residuals) = formulation
            .residuals(values, gamma, self.cooperation)
            .await?
        else {
            return Ok(false);
        };
        for (destination, value) in point.inequality.iter_mut().zip(residuals) {
            *destination = value;
            self.cooperation.completed(1).await?;
        }
        *x.last_mut().expect("attainment coordinate") = gamma;
        point.f = gamma;
        Ok(true)
    }
    pub async fn initial_model(
        &mut self,
        variables: usize,
    ) -> BuiltinResult<clarabel::algebra::CscMatrix<f64>> {
        let mut model =
            super::super::hessian::initial(variables, self.policy.owner, self.cooperation).await?;
        if self.attainment.is_some() {
            // The smooth epigraph objective and every generated constraint are
            // affine in gamma. This chosen PSD model has exact zero curvature
            // there; no arbitrary small positive diagonal is manufactured.
            *model.nzval.last_mut().expect("nonempty augmented model") = 0.0;
        }
        Ok(model)
    }
}
