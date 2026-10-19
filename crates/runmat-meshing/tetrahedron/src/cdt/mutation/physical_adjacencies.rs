//! Every cavity records changed faces against a verified original physical mesh.
//! Unchanged adjacency is retained authority; dirty faces require fresh admission.
use super::TopologyMutation;
use crate::cdt::insertion::AdmittedPhysicalTopology;

impl TopologyMutation {
    pub(in crate::cdt) fn from_admitted(topology: AdmittedPhysicalTopology) -> Self {
        let mut result = Self::from_canonical(topology.into_topology());
        result.physical_dirty_faces = Some(Default::default());
        result
    }

    pub(in crate::cdt) fn adopt_admitted_with_spatial_index<E>(
        &mut self,
        topology: AdmittedPhysicalTopology,
        checkpoint: &mut impl FnMut() -> Result<(), E>,
    ) -> Result<(), E> {
        self.adopt_canonical_with_spatial_index(topology.into_topology(), checkpoint)?;
        self.physical_dirty_faces = Some(Default::default());
        Ok(())
    }

    pub(super) fn record_physical_cell_change(&mut self, vertices: [u32; 4]) {
        if let Some(dirty) = &mut self.physical_dirty_faces {
            for opposite in 0..4 {
                dirty.insert(Self::face(vertices, opposite));
            }
        }
    }

    pub(in crate::cdt) fn physical_dirty_faces(
        &self,
    ) -> Option<&std::collections::BTreeSet<[u32; 3]>> {
        self.physical_dirty_faces.as_ref()
    }

    /// Only the insertion owner's exact dirty-face verifier may discharge this work.
    pub(in crate::cdt) fn discharge_physical_dirty_faces(&mut self) {
        self.record_trial_dirty_discharge();
        self.physical_dirty_faces
            .as_mut()
            .expect("verified physical prefix")
            .clear();
    }
}
