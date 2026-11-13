//! Quasirandom/LHS construction with original-domain GSS feasibility and
//! current-phase interpolation-site uniqueness, including accepted inputs.
use super::{
    campaign::Campaign,
    construction::Sampler,
    dispatch::Dispatcher,
    domain::BoxGeometry,
    gss::{Geometry, Proposal},
    invalid,
    parameters::Parameters,
    proposal::{self, Prepared},
};
use crate::{
    builtins::math::optim::{
        common::{feasible_cone::Basis, solver_timing::SolverTiming, ScalarCooperation},
        intlinprog::feasibility::Domain,
    },
    BuiltinResult,
};
pub(super) struct Construction {
    sampler: Option<Sampler>,
    attempts: usize,
    allowance: usize,
}
pub(super) struct Controls<'a> {
    pub parameters: &'a Parameters,
    pub clock: &'a SolverTiming,
    pub remaining: usize,
    pub anchor: Option<&'a [f64]>,
    pub linear_tolerance: f64,
}
pub(super) enum Next {
    Point(Vec<f64>),
    NoNewSites,
    LinearInfeasible,
    TimeLimit,
}
pub(super) struct Owners<'a, 'source> {
    pub campaign: &'a Campaign,
    pub dispatcher: &'a Dispatcher<'source>,
    pub domain: &'a mut Domain,
    pub geometry: &'a BoxGeometry,
    pub gss: &'a Geometry,
}
impl Construction {
    pub(super) fn new(minimum: usize) -> BuiltinResult<Self> {
        // RunMat's finite proposal-search allowance bounds failed numerical
        // attempts, not expensive evaluations or a global feasibility proof.
        let allowance = minimum
            .checked_mul(20)
            .ok_or_else(|| invalid("construction attempt extent overflow"))?
            .max(1000);
        Ok(Self {
            sampler: None,
            attempts: 0,
            allowance,
        })
    }
    pub(super) async fn next(
        &mut self,
        owners: Owners<'_, '_>,
        controls: Controls<'_>,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Next> {
        let phase = owners.campaign.phase();
        while self.attempts < self.allowance {
            cooperation.completed(0).await?;
            if controls.parameters.time.reached(controls.clock.elapsed()) {
                return Ok(Next::TimeLimit);
            }
            if controls.remaining == 0 {
                return Ok(Next::NoNewSites);
            }
            if self.sampler.is_none() {
                let view = owners.geometry.view(owners.domain)?;
                self.sampler = Some(
                    Sampler::new(
                        view.variables(),
                        view.dimensions(),
                        controls
                            .parameters
                            .minimum_surrogate
                            .min(controls.remaining),
                        cooperation,
                    )
                    .await?,
                );
            }
            let Some(unit) = self
                .sampler
                .as_mut()
                .expect("construction sequence")
                .next(cooperation)
                .await?
            else {
                // A depleted Latin block needs a new randomized block after
                // integer/linear or NaN rejection, not source replay.
                self.sampler = None;
                self.attempts += 1;
                continue;
            };
            self.attempts += 1;
            let point = {
                let view = owners.geometry.view(owners.domain)?;
                if view.linear().rows() != 0 && controls.anchor.is_some() {
                    let anchor = controls
                        .anchor
                        .expect("selected feasible construction anchor");
                    let cone = owners
                        .gss
                        .tangent_in(&view, anchor, 1.0, &view, cooperation)
                        .await?;
                    let count = cone.len(Basis::Maximal)?;
                    if count == 0 {
                        return Ok(Next::NoNewSites);
                    }
                    let Some(ray) = cone
                        .oriented_direction(
                            (self.attempts - 1) % count,
                            Basis::Maximal,
                            false,
                            cooperation,
                        )
                        .await?
                    else {
                        continue;
                    };
                    let radius = unit[0];
                    if radius == 0.0 {
                        continue;
                    }
                    match owners
                        .gss
                        .candidate(
                            &view,
                            anchor,
                            &ray,
                            radius,
                            controls.linear_tolerance,
                            cooperation,
                        )
                        .await?
                    {
                        Proposal::Point { mut point, .. } => {
                            for &column in view.integers() {
                                let (lower, upper) = view.effective_bounds(column);
                                point[column] = point[column].round().max(lower).min(upper);
                                cooperation.completed(1).await?;
                            }
                            point
                        }
                        Proposal::Blocked | Proposal::Unrepresentable => continue,
                    }
                } else {
                    let mut point =
                        runmat_value::exact_capacity_vec(view.variables()).map_err(|source| {
                            crate::build_runtime_error(
                                "surrogateopt: construction point allocation failed",
                            )
                            .with_builtin(super::NAME)
                            .with_source(source)
                            .build()
                        })?;
                    point.resize(view.variables(), 0.0);
                    view.denormalize_into(&unit, &mut point, cooperation)
                        .await?;
                    point
                }
            };
            let point = match proposal::prepare(
                point,
                owners.domain,
                controls.linear_tolerance,
                cooperation,
            )
            .await?
            {
                Prepared::Point(point) => point,
                Prepared::LinearInfeasible => return Ok(Next::LinearInfeasible),
                Prepared::TimeLimit => return Ok(Next::TimeLimit),
                Prepared::Unverified => continue,
            };
            if proposal::already_evaluated(&point, owners.campaign.history(), phase, cooperation)
                .await?
                || owners
                    .dispatcher
                    .contains_point(&point, phase, cooperation)
                    .await?
            {
                continue;
            }
            return Ok(Next::Point(point));
        }
        Ok(Next::NoNewSites)
    }
}
