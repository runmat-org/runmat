//! Original quadratic targets/checker on the single shared working-set controller.
use super::{quadratic, Linearization};
use crate::builtins::math::optim::common::{working_set as shared, ScalarCooperation};
use crate::BuiltinResult;
use clarabel::{algebra::CscMatrix, solver::SupportedConeT};
use runmat_builtins::BuiltinCatalogIdentity;
struct Target<'a> {
    hessian: &'a CscMatrix<f64>,
    gradient: &'a [f64],
    model: &'a Linearization,
    tolerance: f64,
    owner: BuiltinCatalogIdentity,
}
impl shared::Target for Target<'_> {
    async fn equality(
        &mut self,
        a: &CscMatrix<f64>,
        b: &[f64],
        _x: &[f64],
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<Option<shared::EqualityTarget>> {
        let cones = [SupportedConeT::ZeroConeT(b.len())];
        let selected = quadratic::Constraints {
            a,
            b,
            equalities: b.len(),
            cones: &cones,
        };
        Ok(
            match quadratic::solve(
                self.hessian,
                self.gradient,
                &selected,
                self.tolerance,
                self.owner,
                c,
            )
            .await?
            {
                quadratic::Result::Step(step) => {
                    Some(shared::EqualityTarget::Point(shared::Step {
                        direction: step.direction,
                        multipliers: step.multipliers,
                    }))
                }
                _ => None,
            },
        )
    }
    async fn checked(
        &mut self,
        _rows: shared::Rows<'_>,
        x: &[f64],
        z: &[f64],
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<bool> {
        quadratic::checked(
            self.hessian,
            self.gradient,
            &quadratic::Constraints::from(self.model),
            x,
            z,
            self.tolerance,
            self.owner,
            c,
        )
        .await
    }
}
pub(super) async fn minimize(
    hessian: &CscMatrix<f64>,
    gradient: &[f64],
    model: &Linearization,
    x: Vec<f64>,
    limits: super::Limits,
    owner: BuiltinCatalogIdentity,
    c: &mut ScalarCooperation,
) -> BuiltinResult<quadratic::Result> {
    let default_limit = gradient
        .len()
        .checked_add(model.b.len())
        .and_then(|extent| extent.checked_mul(2))
        .and_then(|extent| extent.checked_add(1))
        .ok_or_else(|| super::super::invalid("active-set pivot extent overflow"))?
        .max(quadratic::LOCAL_ITERATIONS as usize);
    let mut target = Target {
        hessian,
        gradient,
        model,
        tolerance: limits.tolerance,
        owner,
    };
    let result = shared::minimize(
        &mut target,
        shared::Rows {
            a: &model.a,
            b: &model.b,
            equalities: model.equalities,
        },
        x,
        shared::Limits {
            feasibility: limits.tolerance,
            dual: limits.tolerance,
            pivots: limits.iterations.unwrap_or(default_limit),
        },
        owner,
        c,
    )
    .await?;
    Ok(match result {
        shared::Outcome::Optimal(step) => quadratic::Result::Step(quadratic::Step {
            direction: step.direction,
            multipliers: step.multipliers,
        }),
        shared::Outcome::Numerical { .. }
        | shared::Outcome::Stopped { .. }
        | shared::Outcome::Unbounded { .. } => quadratic::Result::NumericalDifficulty,
    })
}
