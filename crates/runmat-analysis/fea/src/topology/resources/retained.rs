//! Count retained projection storage from the borrowed raw mesh without geometric construction.
use super::*;

impl AnalysisProjection {
    /// Conservative typed retained-heap bound for a projection constructed from this mesh.
    /// Raw directed neighbor occurrences bound the deduplicated adjacency. This does not
    /// validate connectivity, geometry or identity and is not mathematical certification or RSS.
    pub fn retained_resident_bytes_upper_bound(
        mesh: &SolverMeshArtifact,
    ) -> Result<u64, AnalysisProjectionError> {
        let topology = &mesh.topology;
        let mut total = 0;
        add(&mut total, vector::<ProjectedNode>(topology.nodes.len())?)?;
        add(
            &mut total,
            vector::<ProjectedElement>(topology.volume_elements.len())?,
        )?;
        add(
            &mut total,
            vector::<ProjectedBoundaryFace>(topology.boundary_faces.len())?,
        )?;
        add(&mut total, slots::<(u64, usize)>(topology.nodes.len(), 1)?)?;
        add(
            &mut total,
            slots::<(u64, usize)>(topology.volume_elements.len(), 1)?,
        )?;

        let neighbor_references = super::super::neighbors::occurrence_count(
            topology
                .volume_elements
                .iter()
                .map(|element| element.node_ids.len()),
        )?;
        for element in &topology.volume_elements {
            let count = element.node_ids.len();
            add(&mut total, vector::<usize>(count)?)?;
            add(&mut total, entity_payload_bytes(&element.region_id)?)?;
            for entity in &element.provenance {
                add(&mut total, entity_payload_bytes(entity)?)?;
            }
            add(
                &mut total,
                vector::<runmat_meshing_core::PersistentEntityId>(element.provenance.len())?,
            )?;
        }
        for face in &topology.boundary_faces {
            for entity in &face.provenance {
                add(&mut total, entity_payload_bytes(entity)?)?;
            }
            add(&mut total, vector::<usize>(face.node_ids.len())?)?;
            add(
                &mut total,
                vector::<usize>(face.adjacent_volume_element_ids.len())?,
            )?;
            add(
                &mut total,
                vector::<runmat_meshing_core::PersistentEntityId>(face.provenance.len())?,
            )?;
        }
        add(&mut total, slots::<Vec<usize>>(topology.nodes.len(), 1)?)?;
        add(&mut total, slots::<usize>(neighbor_references, 1)?)?;
        Ok(total)
    }
}
