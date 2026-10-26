//! Nonuniform phase/polarity ownership and retained random orthogonal frames.
mod cycle;
mod frame;
mod transition;
#[cfg(test)]
mod tests;
use super::{
    evaluation::Point,
    linear::Linear,
    mesh::{self, Mesh},
    poll,
    scaling::Scaling,
};
use crate::builtins::math::linalg::ops::sparse_factor::orthogonal_frame::OrthogonalFrame;
use crate::{builtins::math::optim::common::ScalarCooperation, BuiltinResult};
pub(super) use frame::Frame;
use runmat_builtins::OptimPatternAlgorithm;
pub(super) use transition::{relative_decrease, RELATIVE_DECREASE};

pub(super) struct Driver {
    algorithm: OptimPatternAlgorithm,
    cycle: cycle::Cycle,
    transitions: transition::Transitions,
    random: Option<OrthogonalFrame>,
    coverage: [u8; 2],
    complete_cone: bool,
    last_covered: bool,
    configuration: (f64, f64),
}
impl Driver {
    pub fn new(
        algorithm: OptimPatternAlgorithm,
        equalities: bool,
        mesh: f64,
        binding: f64,
        tolerance: f64,
    ) -> BuiltinResult<Self> {
        Ok(Self {
            algorithm,
            cycle: cycle::Cycle::new(algorithm, equalities)?,
            transitions: transition::Transitions::new(mesh)?,
            random: None,
            coverage: [0; 2],
            complete_cone: false,
            last_covered: false,
            configuration: (binding, tolerance),
        })
    }
    pub fn algorithm(&self) -> OptimPatternAlgorithm {
        self.algorithm
    }
    pub fn configure(&mut self, binding: f64, tolerance: f64, scale_changed: bool) {
        if self.configuration != (binding, tolerance) || scale_changed {
            self.invalidate();
        }
        self.configuration = (binding, tolerance);
    }
    pub fn invalidate(&mut self) {
        self.coverage = [0; 2];
        self.last_covered = false;
    }
    pub fn covered(&self) -> bool {
        self.last_covered
    }
    pub async fn frame<'a>(
        &'a mut self,
        point: &Point,
        geometry: &Linear,
        mesh: f64,
        scaling: &'a Scaling,
        binding: f64,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Option<Frame<'a>>> {
        let phase = self.cycle.phase();
        let positive = self.cycle.positive();
        self.last_covered = false;
        if positive {
            self.coverage[phase.index()] = 0;
        }
        if phase == cycle::Phase::Orthogonal && positive {
            let (runtime, budget) = cooperation.runtime_budget();
            self.random =
                OrthogonalFrame::random(point.x.len(), runtime, super::IDENTITY, budget).await?;
            self.coverage[phase.index()] = 0;
        }
        let orientation = match phase {
            cycle::Phase::Gps => frame::Orientation::Gps,
            cycle::Phase::Orthogonal => match self.random.as_ref() {
                Some(frame) => frame::Orientation::Random(frame),
                None => return Ok(None),
            },
        };
        let Some(frame) = Frame::new(
            point,
            geometry,
            mesh,
            scaling,
            orientation,
            positive,
            binding,
            cooperation,
        )
        .await?
        else {
            return Ok(None);
        };
        self.complete_cone = frame.spans_pair();
        Ok(Some(frame))
    }
    /// Observations are only from the current completed poll, never from a
    /// search skipped into this cycle or a partially evaluated direction set.
    pub fn completed(
        &mut self,
        improved: Option<poll::MeritChange>,
        points: usize,
        mesh: &mut Mesh,
    ) -> BuiltinResult<mesh::Update> {
        let phase = self.cycle.phase();
        let positive = self.cycle.positive();
        let factor = if let Some(change) = improved {
            self.invalidate();
            self.transitions
                .successful(mesh.size(), change.before, change.after)?
        } else {
            self.coverage[phase.index()] |= if self.complete_cone {
                3
            } else if positive {
                1
            } else {
                2
            };
            self.transitions.unsuccessful()
        };
        let covered = self.coverage[phase.index()] == 3;
        self.last_covered = covered;
        self.cycle.advance(Some(match improved {
            Some(change) => cycle::Observation::Improved { change, points },
            None => cycle::Observation::NoImprovement { points },
        }))?;
        Ok(mesh.nonuniform_update(factor, covered))
    }
    pub fn searched(&mut self) -> BuiltinResult<()> {
        self.invalidate();
        self.transitions.interrupted();
        self.cycle.advance(None)
    }
    pub fn unrepresentable(&mut self) -> BuiltinResult<bool> {
        let index = self.cycle.phase().index();
        self.coverage[index] |= if self.complete_cone {
            3
        } else if self.cycle.positive() {
            1
        } else {
            2
        };
        self.cycle
            .advance(Some(cycle::Observation::NoImprovement { points: 0 }))?;
        self.last_covered = self.coverage[index] == 3;
        Ok(self.last_covered)
    }
}
