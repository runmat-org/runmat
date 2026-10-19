use super::{
    error, resource, DelaunayPointSetError, DelaunayPointSetErrorKind, DelaunayPointSetOptions,
};
use runmat_meshing_core::{
    quality::predicate::{orient2d, orient3d, PredicateSign},
    MeshingCancellationSignal,
};
pub(super) struct HullWork<'a> {
    options: DelaunayPointSetOptions,
    cancellation: &'a dyn MeshingCancellationSignal,
    predicates: u64,
}

impl<'a> HullWork<'a> {
    pub(super) fn new(
        options: DelaunayPointSetOptions,
        cancellation: &'a dyn MeshingCancellationSignal,
    ) -> Self {
        Self {
            options,
            cancellation,
            predicates: 0,
        }
    }

    pub(super) fn check_cancellation(&self) -> Result<(), DelaunayPointSetError> {
        if self.cancellation.is_cancelled() {
            return Err(error(DelaunayPointSetErrorKind::Cancelled, "cancelled"));
        }
        Ok(())
    }

    fn checkpoint(&mut self) -> Result<(), DelaunayPointSetError> {
        self.predicates = self
            .predicates
            .checked_add(1)
            .ok_or_else(|| resource("convex-hull predicate count overflow"))?;
        if self.predicates > self.options.insertion.maximum_predicate_evaluations {
            return Err(resource("convex-hull predicate limit exceeded"));
        }
        if self
            .predicates
            .is_multiple_of(self.options.insertion.topology.cancellation_check_interval)
            && self.cancellation.is_cancelled()
        {
            return Err(error(DelaunayPointSetErrorKind::Cancelled, "cancelled"));
        }
        Ok(())
    }

    pub(super) fn orient2d(
        &mut self,
        points: [[f64; 2]; 3],
    ) -> Result<PredicateSign, DelaunayPointSetError> {
        self.checkpoint()?;
        orient2d(points).map_err(|failure| predicate_error("planar", failure))
    }

    pub(super) fn orient3d(
        &mut self,
        points: [[f64; 3]; 4],
    ) -> Result<PredicateSign, DelaunayPointSetError> {
        self.checkpoint()?;
        orient3d(points).map_err(|failure| predicate_error("spatial", failure))
    }
}

fn predicate_error(kind: &str, failure: impl std::fmt::Debug) -> DelaunayPointSetError {
    error(
        DelaunayPointSetErrorKind::InvalidTopology,
        format!("convex-hull {kind} predicate rejected admitted coordinates: {failure:?}"),
    )
}
