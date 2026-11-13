//! A model update boundary over append-only measured history. Pending batch
//! completions may append rows without silently changing a frozen interpolant.
use super::{BoxView, Generation, History, Quality, Training, TrainingOwner, View};
use crate::builtins::math::optim::common::ScalarCooperation;
use crate::BuiltinResult;

pub(in super::super) struct Snapshot {
    identity: TrainingOwner,
    generation: Generation,
    extent: usize,
    rows: Vec<usize>,
}
pub(in super::super) struct SnapshotView<'a, 'domain> {
    snapshot: &'a Snapshot,
    values: View<'a, 'domain>,
}
impl Snapshot {
    pub(in super::super) async fn capture(
        history: &History,
        rows: Vec<usize>,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Self> {
        let mut previous = None;
        for &row in &rows {
            if row >= history.len()
                || previous.is_some_and(|before| before >= row)
                || history.trials[row].observation.quality != Quality::Finite
            {
                return Err(super::invalid("invalid measured model snapshot rows"));
            }
            previous = Some(row);
            cooperation.completed(1).await?;
        }
        Ok(Self {
            identity: history.identity.clone(),
            generation: history.generation,
            extent: history.len(),
            rows,
        })
    }
    pub(in super::super) async fn view<'a, 'domain>(
        &'a self,
        history: &'a History,
        axes: &'a [usize],
        domain: &'a BoxView<'domain>,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<SnapshotView<'a, 'domain>> {
        let view = SnapshotView {
            snapshot: self,
            values: View {
                history,
                rows: &self.rows,
                axes,
                domain,
            },
        };
        if !view.valid() {
            return Err(super::invalid(
                "model snapshot lost its original measured history",
            ));
        }
        let mut previous = None;
        for &axis in axes {
            if axis >= domain.dimensions() || previous.is_some_and(|before| before >= axis) {
                return Err(super::invalid("model snapshot coordinate extent mismatch"));
            }
            previous = Some(axis);
            cooperation.completed(1).await?;
        }
        Ok(view)
    }
}
impl Training for SnapshotView<'_, '_> {
    fn owner(&self) -> &TrainingOwner {
        &self.snapshot.identity
    }
    fn valid(&self) -> bool {
        self.snapshot
            .identity
            .same_as(&self.values.history.identity)
            && self.values.history.len() >= self.snapshot.extent
            && self.values.history.generation.0 >= self.snapshot.generation.0
            && self.values.valid()
    }
    fn generation(&self) -> Generation {
        self.snapshot.generation
    }
    fn rows(&self) -> &[usize] {
        self.values.rows()
    }
    fn dimensions(&self) -> usize {
        self.values.dimensions()
    }
    fn affine_axes(&self) -> Option<&[usize]> {
        self.values.affine_axes()
    }
    fn lanes(&self) -> usize {
        self.values.lanes()
    }
    fn coordinate(&self, row: usize, axis: usize) -> f64 {
        self.values.coordinate(row, axis)
    }
    fn measurement(&self, row: usize, lane: usize) -> f64 {
        self.values.measurement(row, lane)
    }
}
