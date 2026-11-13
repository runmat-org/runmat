//! One atomic point-and-connectivity change. Invalid intermediate stars never
//! escape this owner, and rollback retains the original spatial indexes.
use super::{
    point_update::PointMove, storage::error, TopologyEdit, TopologyMutation, TopologyMutationError,
};
use crate::cdt::{topology::VolumeTopologyRead, DelaunayTopologyOptions, DelaunayVolumeNode};
use runmat_geometry_core::PersistentEntityId;
use runmat_meshing_core::{MeshingCancellationSignal, StableDigest};
use std::collections::BTreeSet;

pub(in crate::cdt) struct CavityRelocation {
    pub(in crate::cdt) replacements: Vec<(u32, DelaunayVolumeNode)>,
    pub(in crate::cdt) removed: BTreeSet<u32>,
    pub(in crate::cdt) created: Vec<([u32; 4], Option<PersistentEntityId>)>,
}
impl TopologyMutation {
    pub(in crate::cdt) fn relocate_closed_cavity(
        &mut self,
        change: CavityRelocation,
        protected_nodes: &BTreeSet<StableDigest>,
        protected_faces: &BTreeSet<[StableDigest; 3]>,
        options: DelaunayTopologyOptions,
        cancellation: &dyn MeshingCancellationSignal,
    ) -> Result<Vec<u32>, TopologyMutationError> {
        let CavityRelocation {
            replacements,
            removed,
            created,
        } = change;
        let mut indices = BTreeSet::new();
        let mut identities = Vec::with_capacity(replacements.len());
        if replacements.is_empty() {
            return Err(error("joint relocation needs a nonempty point set"));
        }
        for (index, replacement) in &replacements {
            if !indices.insert(*index)
                || self.nodes.get(*index as usize).is_none()
                || self.vertex_is_boundary(*index)
            {
                return Err(error(
                    "joint relocation requires distinct existing interior points",
                ));
            }
            identities.push((self.nodes[*index as usize].identity, *replacement));
        }
        let original = removed
            .iter()
            .map(|cell| {
                self.tetrahedron(*cell)
                    .map(|cell| {
                        cell.vertex_indices
                            .map(|vertex| self.nodes[vertex as usize].identity)
                    })
                    .ok_or_else(|| error("joint relocation names an inactive cell"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let traced = self.trace_enabled && !self.trial_journals.is_empty();
        let trace_len = self.edit_trace.len();
        let mut trial = self.cavity_trial();
        for (index, replacement) in replacements {
            trial.move_point(
                PointMove {
                    index,
                    replacement,
                    replaced_cells: Some(&removed),
                },
                protected_nodes,
                options,
                cancellation,
            )?;
        }
        let result = trial.replace_closed_cavity(
            &removed,
            created,
            protected_faces,
            options,
            cancellation,
        )?;
        if traced {
            let created = result
                .iter()
                .map(|cell| {
                    let cell = trial.tetrahedron(*cell).expect("admitted joint cavity");
                    (
                        cell.vertex_indices
                            .map(|vertex| trial.nodes[vertex as usize].identity),
                        cell.region_id.clone(),
                    )
                })
                .collect();
            trial.edit_trace.truncate(trace_len);
            trial.edit_trace.push(TopologyEdit::RelocateCavity {
                replacements: identities,
                removed: original,
                created,
            });
        }
        trial.commit();
        Ok(result)
    }
}
