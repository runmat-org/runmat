//! All four adaptive sampler families feed one original-domain candidate
//! owner. Modeled points carry no expensive observations or callback counts.
use super::{
    affine,
    campaign::Campaign,
    candidates::{Added, Builder, Candidates},
    crossover::Parents,
    domain::BoxGeometry,
    frame::{self, Frame, Rings},
    gss::{Geometry, Proposal},
    invalid,
    merit::Phase,
    parameters::Parameters,
    sampling::{self, Kind, Metric},
};
use crate::builtins::math::optim::common::retained_control::Retained;
use crate::{
    builtins::math::optim::{
        common::{feasible_cone::Basis, solver_timing::SolverTiming, ScalarCooperation},
        intlinprog::feasibility::Domain,
    },
    BuiltinResult,
};
pub(super) struct Request<'a> {
    pub kind: Kind,
    pub phase: Phase,
    pub anchor: &'a [f64],
    pub scale: f64,
    pub parameters: &'a Parameters,
    pub clock: &'a SolverTiming,
    pub linear_tolerance: f64,
}
pub(super) enum Generated<'a> {
    Ready(Candidates<'a>),
    Blocked,
    LinearInfeasible,
    TimeLimit,
    Unrepresentable,
}
enum Stream<'a> {
    Random,
    Crossover(Parents<'a>),
    Rings(Retained<Rings<'a>>),
}
pub(super) async fn generate<'domain>(
    request: Request<'_>,
    campaign: &Campaign,
    domain: &'domain mut Domain,
    geometry: &'domain BoxGeometry,
    gss: &Geometry,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Generated<'domain>> {
    let (dimensions, target, span, tangent) = {
        let view = geometry.view(domain)?;
        let target = request
            .parameters
            .samples(request.kind, view.dimensions(), request.scale)?;
        let metric = Metric::new(&view, request.scale)?;
        let span = if matches!(request.kind, Kind::Orthogonal | Kind::Coordinate)
            && view.linear().equalities() != 0
        {
            let Some(span) = affine::span_in(&view, &metric, cooperation).await? else {
                return Ok(Generated::Unrepresentable);
            };
            Some(span)
        } else {
            None
        };
        // GSS inward generators supplement the actual sampler on constrained
        // faces. Without them an arbitrary orthogonal frame can have every
        // ray outside a narrow pointed tangent cone. This constrained policy
        // does not replace the QR/random/crossover source family.
        let tangent = if view.linear().rows() != 0 && !matches!(request.kind, Kind::Crossover) {
            Some(
                gss.tangent_in(&view, request.anchor, 1.0, &metric, cooperation)
                    .await?,
            )
        } else {
            None
        };
        (view.dimensions(), target, span, tangent)
    };
    let mut stream = match request.kind {
        Kind::Random => Stream::Random,
        Kind::Crossover => {
            let Some(parents) =
                Parents::new(campaign.history(), request.phase, cooperation).await?
            else {
                return Ok(Generated::Blocked);
            };
            Stream::Crossover(parents)
        }
        Kind::Orthogonal | Kind::Coordinate => {
            let kind = if request.kind == Kind::Orthogonal {
                frame::Kind::RandomOrthogonal
            } else {
                frame::Kind::Coordinate
            };
            let Some(frame) = Frame::new(kind, dimensions, span.as_ref(), cooperation).await?
            else {
                return Ok(Generated::Unrepresentable);
            };
            Stream::Rings(Retained::new(
                Rings::new(frame, 1.0)?,
                cooperation,
                super::CONTROL_STORAGE,
            )?)
        }
    };
    let period = span
        .as_ref()
        .map_or(dimensions, |span| span.rank())
        .checked_mul(2)
        .and_then(|n| n.checked_add(2))
        .ok_or_else(|| invalid("adaptive sampling period overflow"))?;
    let supplement = tangent
        .as_ref()
        .map_or(Ok(0), |cone| cone.len(Basis::Maximal))?;
    let cycle = period
        .checked_add(supplement)
        .ok_or_else(|| invalid("constrained sampler period overflow"))?;
    let attempts = target
        .checked_mul(20)
        .ok_or_else(|| invalid("candidate attempt extent overflow"))?
        .max(1000);
    let mut builder = Builder::new(
        domain,
        geometry,
        target,
        request.linear_tolerance,
        cooperation,
    )?;
    let mut ray = runmat_value::exact_capacity_vec(dimensions).map_err(|source| {
        crate::build_runtime_error("surrogateopt: sampler ray allocation failed")
            .with_builtin(super::NAME)
            .with_source(source)
            .build()
    })?;
    ray.resize(dimensions, 0.0);
    let mut last_step = 1.0;
    for attempt in 0..attempts {
        cooperation.completed(0).await?;
        if request.parameters.time.reached(request.clock.elapsed()) {
            return Ok(Generated::TimeLimit);
        }
        let point = {
            let view = builder.domain()?;
            let metric = Metric::new(&view, request.scale)?;
            let position = attempt % cycle;
            if position >= period && supplement != 0 {
                let Some(direction) = tangent
                    .as_ref()
                    .expect("supplemented cone")
                    .oriented_direction(position - period, Basis::Maximal, false, cooperation)
                    .await?
                else {
                    continue;
                };
                for (slot, value) in ray.iter_mut().zip(direction) {
                    *slot = value;
                    cooperation.completed(1).await?;
                }
                metric.box_ray_into(&mut ray, cooperation).await?;
                match gss
                    .candidate(
                        &view,
                        request.anchor,
                        &ray,
                        last_step,
                        request.linear_tolerance,
                        cooperation,
                    )
                    .await?
                {
                    Proposal::Point { point, .. } => Some(point),
                    Proposal::Blocked | Proposal::Unrepresentable => None,
                }
            } else {
                match &mut stream {
                    Stream::Crossover(parents) => {
                        parents
                            .select(cooperation)
                            .await?
                            .point(&view, cooperation)
                            .await?
                    }
                    Stream::Random => {
                        let point = sampling::random_point(
                            &view,
                            request.anchor,
                            request.scale,
                            cooperation,
                        )
                        .await?;
                        if view.linear().rows() == 0 {
                            Some(point)
                        } else {
                            for (axis, slot) in ray.iter_mut().enumerate() {
                                *slot = view.unit_coordinate(&point, axis)
                                    - view.unit_coordinate(request.anchor, axis);
                                cooperation.completed(1).await?;
                            }
                            match gss
                                .candidate(
                                    &view,
                                    request.anchor,
                                    &ray,
                                    1.0,
                                    request.linear_tolerance,
                                    cooperation,
                                )
                                .await?
                            {
                                Proposal::Point { point, .. } => Some(point),
                                Proposal::Blocked | Proposal::Unrepresentable => None,
                            }
                        }
                    }
                    Stream::Rings(rings) => {
                        let Some(step) = rings.next_into(&mut ray, cooperation).await? else {
                            break;
                        };
                        last_step = step;
                        metric.box_ray_into(&mut ray, cooperation).await?;
                        match gss
                            .candidate(
                                &view,
                                request.anchor,
                                &ray,
                                step,
                                request.linear_tolerance,
                                cooperation,
                            )
                            .await?
                        {
                            Proposal::Point { point, .. } => Some(point),
                            Proposal::Blocked | Proposal::Unrepresentable => None,
                        }
                    }
                }
            }
        };
        let Some(mut point) = point else {
            continue;
        };
        {
            let view = builder.domain()?;
            for &column in view.integers() {
                let (lower, upper) = view.effective_bounds(column);
                point[column] = point[column].round().max(lower).min(upper);
                cooperation.completed(1).await?;
            }
        }
        match builder.push(point, cooperation).await? {
            Added::Point => {}
            Added::LinearInfeasible => return Ok(Generated::LinearInfeasible),
            Added::TimeLimit => return Ok(Generated::TimeLimit),
            Added::Unverified => {}
        }
        if builder.len() == target {
            break;
        }
    }
    if builder.len() == 0 {
        Ok(Generated::Blocked)
    } else {
        builder.finish().map(Generated::Ready)
    }
}
