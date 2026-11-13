//! Constrained insertion uses the shared insertion owner. Its checked edit
//! trace supplies the exact changed quality inventory, including legalization.
use super::*;

impl WorkingTrial<'_, '_> {
    pub(in crate::cdt) fn insert_constrained(
        &mut self,
        node: DelaunayVolumeNode,
        options: impl Into<VolumeStateOptions>,
        cancellation: &dyn MeshingCancellationSignal,
    ) -> Result<(), VolumeStateError> {
        let options = options.into();
        let trace_start = self.topology.active_edits().len();
        crate::cdt::insertion::insert_delaunay_volume_node_into_mutation_with_prepared_barriers_from_hint(
            &mut self.topology, node, self.prepared.protected_facets(), false,
            None, options.insertion, cancellation,
        ).map_err(insertion_error)?;
        self.update_insertion_quality(trace_start, cancellation)
    }

    pub(in crate::cdt) fn insert_prepared(
        &mut self,
        insertion: crate::cdt::insertion::PreparedInsertion,
        options: impl Into<VolumeStateOptions>,
        cancellation: &dyn MeshingCancellationSignal,
    ) -> Result<(), VolumeStateError> {
        let options = options.into();
        let trace_start = self.topology.active_edits().len();
        crate::cdt::insertion::apply_prepared_delaunay_volume_node_insertion(
            &mut self.topology,
            insertion,
            self.prepared.protected_facets(),
            true,
            options.insertion,
            cancellation,
        )
        .map_err(insertion_error)?;
        self.update_insertion_quality(trace_start, cancellation)
    }

    fn update_insertion_quality(
        &mut self,
        trace_start: usize,
        cancellation: &dyn MeshingCancellationSignal,
    ) -> Result<(), VolumeStateError> {
        let edits = self.topology.active_edits();
        let mut removed = BTreeSet::new();
        let mut created = BTreeSet::new();
        for edit in edits.iter().skip(trace_start) {
            if let TopologyEdit::ReplaceCavity {
                removed: old,
                created: new,
            } = edit
            {
                for mut key in old.iter().copied() {
                    key.sort_unstable();
                    if !created.remove(&key) {
                        removed.insert(key);
                    }
                }
                for (mut key, _) in new.iter().cloned() {
                    key.sort_unstable();
                    if !removed.remove(&key) {
                        created.insert(key);
                    }
                }
            }
        }
        let created = created
            .into_iter()
            .map(|identities| {
                let vertices = identities.map(|identity| {
                    self.topology
                        .node_index(identity)
                        .expect("checked insertion corner")
                });
                self.topology.tetrahedron_index(vertices).ok_or_else(|| {
                    error(
                        VolumeStateErrorKind::InvalidTopology,
                        "checked insertion result cell is absent",
                    )
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        self.prepared
            .replace_quality(
                &mut self.quality,
                &removed.into_iter().collect::<Vec<_>>(),
                &created,
                &*self.topology,
                cancellation,
            )
            .map_err(quality_error)
    }
}
