//! Typed classic versus nonuniform frame; no alternate identity or direction math.
mod classic;
use super::super::{
    evaluation::Point,
    invalid,
    linear::{Linear, Prefix, ProposalStop},
    mads,
    mesh::Mesh,
    nups,
    scaling::Scaling,
};
use super::Acceptance;
use crate::builtins::math::optim::common::ScalarCooperation;
use crate::BuiltinResult;
use classic::ClassicFrame;
use runmat_builtins::OptimPatternPoll;
#[derive(Clone, Copy)]
pub(super) enum DirectionId {
    Previous,
    Generated(usize),
}
pub(in crate::builtins::math::optim::patternsearch) enum History {
    Previous,
    New(Vec<f64>),
}
pub(super) enum Frame<'a> {
    Classic(ClassicFrame<'a>),
    Nonuniform { model: nups::Frame<'a>, step: f64 },
}
impl<'a> Frame<'a> {
    pub async fn new(
        base: &Point,
        geometry: &Linear,
        mesh: &'a Mesh,
        scaling: &'a Scaling,
        method: OptimPatternPoll,
        binding: f64,
        acceptance: Acceptance,
        refinements: &mut Option<mads::Refinements>,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Option<Self>> {
        Ok(ClassicFrame::new(
            base,
            geometry,
            mesh,
            scaling,
            method,
            binding,
            acceptance,
            refinements,
            cooperation,
        )
        .await?
        .map(Self::Classic))
    }
    pub fn nonuniform(model: nups::Frame<'a>, step: f64) -> BuiltinResult<Self> {
        if !step.is_finite() || step <= 0.0 {
            return Err(invalid("invalid nonuniform frame step"));
        }
        Ok(Self::Nonuniform { model, step })
    }
    pub fn count(&self) -> BuiltinResult<usize> {
        match self {
            Self::Classic(frame) => frame.count(),
            Self::Nonuniform { model, .. } => model.count(),
        }
    }
    pub fn rotatable(&self) -> bool {
        match self {
            Self::Classic(frame) => frame.rotatable(),
            Self::Nonuniform { .. } => false,
        }
    }
    pub fn measure(&self) -> f64 {
        match self {
            Self::Classic(frame) => frame.measure(),
            Self::Nonuniform { step, .. } => *step,
        }
    }
    pub async fn candidate(
        &self,
        geometry: &Linear,
        base: &[f64],
        ray: &[f64],
        tolerance: f64,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Result<Prefix, ProposalStop>> {
        match self {
            Self::Classic(frame) => {
                frame
                    .candidate(geometry, base, ray, tolerance, cooperation)
                    .await
            }
            Self::Nonuniform { step, .. } => {
                geometry
                    .candidate_with_step(base, ray, *step, tolerance, cooperation)
                    .await
            }
        }
    }
    pub async fn direction(
        &self,
        index: usize,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Option<Vec<f64>>> {
        match self {
            Self::Classic(frame) => frame.direction(index, cooperation).await,
            Self::Nonuniform { model, .. } => model.direction(index, cooperation).await,
        }
    }
    pub async fn history(
        &self,
        id: DirectionId,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<History> {
        match self {
            Self::Classic(frame) => frame.history(id, cooperation).await,
            Self::Nonuniform { .. } => Err(invalid(
                "nonuniform polling has no Success direction history",
            )),
        }
    }
}
/// Compare only actual checked points: the same prefix FMA recipe, not an
/// approximate angular match or ordinal belonging to an older cone.
pub(super) async fn same_point(
    point: &[f64],
    origin: &[f64],
    ray: &[f64],
    length: f64,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<bool> {
    if point.len() != origin.len() || ray.len() != origin.len() {
        return Err(invalid("previous direction extent changed"));
    }
    for ((&point, &origin), &ray) in point.iter().zip(origin).zip(ray) {
        if point.to_bits() != length.mul_add(ray, origin).to_bits() {
            return Ok(false);
        }
        cooperation.completed(1).await?;
    }
    Ok(true)
}
