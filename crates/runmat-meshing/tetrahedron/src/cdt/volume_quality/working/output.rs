//! Move cell evidence out of the search owner before canonical topology admission.
//! These cells have no publication authority until complete handoff verification.
use super::*;

pub(in crate::cdt) struct QualityCells(Vec<DelaunayTetrahedronQuality>);

impl WorkingVolumeQuality {
    pub(in crate::cdt) fn into_cells(
        self,
        options: DelaunayVolumeQualityOptions,
        cancellation: &dyn MeshingCancellationSignal,
    ) -> Result<QualityCells, DelaunayVolumeQualityError> {
        checkpoint(0, options, cancellation)?;
        if self.cells.len() as u64 > options.maximum_tetrahedra {
            return Err(error(
                DelaunayVolumeQualityErrorKind::ResourceLimit,
                None,
                "canonical quality cells exceed the current inventory limit",
            ));
        }
        assert!(
            self.journals.is_empty(),
            "cannot publish speculative quality"
        );
        let Self {
            cells,
            shape,
            refinement,
            journals,
            baseline_originals,
            comparison_ready: _,
            repair_floor: _,
            last_scoped_update: _,
        } = self;
        // Rankings, rollback originals and entry geometry cannot participate in
        // the next phase. Release them before topology and output validation.
        drop((shape, refinement, journals, baseline_originals));
        let mut tetrahedra = Vec::with_capacity(cells.len());
        for (index, entry) in cells.into_values().enumerate() {
            checkpoint(index as u64, options, cancellation)?;
            tetrahedra.push(entry.quality);
        }
        Ok(QualityCells(tetrahedra))
    }
}

impl QualityCells {
    pub(in crate::cdt) fn summarize(
        self,
        topology: &DelaunayVolumeTopology,
        options: DelaunayVolumeQualityOptions,
        cancellation: &dyn MeshingCancellationSignal,
    ) -> Result<DelaunayVolumeQuality, DelaunayVolumeQualityError> {
        summary::summarize(topology, self.0, options, cancellation)
    }
}

#[cfg(test)]
mod tests;
