//! An over-defined sketch stays visible. When the solver reports a conflict the sketch feature
//! fails as before (its `SketchSolveFailed` error, and every feature that reads it), but sketch
//! mode still draws it from its written coordinates with the conflict set marked in the
//! over-defined color. The entry lives apart from the solved sketches, so no
//! feature builds on it.

use super::*;
use crate::sketch_solver::ExplicitSketchSolver;

impl<'a, 'b> Run<'a, 'b> {
    /// Keeps `sketch` for display after a failed solve: its written geometry, status
    /// `over_defined` and the constraints in the conflict set.
    pub(super) fn keep_over_defined_sketch(
        &mut self,
        id: &FeatureId,
        sketch: &crate::model::sketch::Sketch,
        frame: Frame,
        conflicting: &[usize],
    ) {
        let Ok(mut shown) = ExplicitSketchSolver.solve(sketch, &self.ev, self.values) else {
            return;
        };
        shown.status = crate::model::sketch::SketchSolveStatus::OverDefined;
        shown.conflicting = conflicting.to_vec();
        self.st.over_defined_sketches.insert(id.clone(), (shown, frame));
    }
}
