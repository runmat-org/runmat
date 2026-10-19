//! Topology and complete affected quality share the same joint relocation trial.
use super::*;
use crate::cdt::mutation::CavityRelocation;
impl WorkingTrial<'_, '_> {
    pub(in crate::cdt) fn relocate_cavity(
        &mut self,
        change: CavityRelocation,
        options: impl Into<VolumeStateOptions>,
        cancellation: &dyn MeshingCancellationSignal,
    ) -> Result<(), VolumeStateError> {
        let options = options.into();
        let removed = change
            .removed
            .iter()
            .map(|cell| self.topology.cell_key(*cell))
            .collect::<Vec<_>>();
        let cells = self
            .topology
            .relocate_closed_cavity(
                change,
                self.protected_nodes,
                self.prepared.protected_facets(),
                options.insertion.topology,
                cancellation,
            )
            .map_err(mutation_error)?;
        self.prepared
            .replace_quality(
                &mut self.quality,
                &removed,
                &cells,
                &*self.topology,
                cancellation,
            )
            .map_err(quality_error)
    }
}
