//! Numerical storage added by an identical-topology solved-temperature consumer.
//!
//! This inventory is independent of source artifact encoding/metadata. Portable source
//! decoding must additionally use PreparedInputResourceFacts at the transport boundary.

use runmat_analysis_core::{ProcedureStep, StateKind};
use runmat_meshing_core::SolverMeshArtifact;

use super::ResourcePlanningError;

#[derive(Debug, Default, PartialEq)]
pub(super) struct ThermalStateStorage {
    /// Source values/IDs, aligned values, node lookup and equivalent load coexist during binding
    /// and assembly. Conservatively retain this reservation throughout numerical execution.
    pub resident_bytes: u64,
    /// Four temperatures plus four six-component thermal strain and stress tensors per element,
    /// one coupling residual scalar, and owned displacement/von-Mises snapshot copies.
    pub output_values: u64,
}

impl ThermalStateStorage {
    pub fn derive(
        step: &ProcedureStep,
        mesh: Option<&SolverMeshArtifact>,
    ) -> Result<Self, ResourcePlanningError> {
        if !step
            .state_inputs
            .iter()
            .any(|input| input.required_kind == StateKind::Thermal)
        {
            return Ok(Self::default());
        }
        let mesh = mesh.ok_or_else(|| {
            ResourcePlanningError::Invalid(
                "solved thermal resource planning requires the receiving solver mesh".into(),
            )
        })?;
        let node_count = u64::try_from(mesh.topology.nodes.len())
            .map_err(|_| ResourcePlanningError::Overflow("thermal source nodes"))?;
        let element_count = u64::try_from(mesh.topology.volume_elements.len())
            .map_err(|_| ResourcePlanningError::Overflow("thermal recovery elements"))?;
        // Per node: source ID + T (16), aligned T (8), load vector (24), and a conservative
        // 64-byte lookup-entry reservation. This is an allocation envelope, not wire size.
        let mut resident_bytes =
            node_count
                .checked_mul(112)
                .ok_or(ResourcePlanningError::Overflow(
                    "thermal source numerical storage",
                ))?;
        for element in &mesh.topology.volume_elements {
            // Local nodal temperatures; 4 T + 4 eigenstrain + alpha + Tref; and a 32-byte
            // local-vector header/allocation allowance. Geometry is owned by solid recovery.
            let nodes = u64::try_from(element.node_ids.len())
                .map_err(|_| ResourcePlanningError::Overflow("thermal element nodes"))?;
            let bytes = nodes
                .checked_add(10)
                .and_then(|n| n.checked_mul(8))
                .and_then(|n| n.checked_add(32))
                .ok_or(ResourcePlanningError::Overflow("thermal element storage"))?;
            resident_bytes = resident_bytes
                .checked_add(bytes)
                .ok_or(ResourcePlanningError::Overflow("thermal retained storage"))?;
        }
        // AnalysisField owns Vec<f64>: field_snapshot and the renamed published snapshot
        // coexist during construction. Account for both copies, not an assumed shared alias.
        let snapshot_values = node_count
            .checked_mul(3)
            .and_then(|values| values.checked_add(element_count))
            .ok_or(ResourcePlanningError::Overflow(
                "thermal structural snapshots",
            ))?;
        resident_bytes = snapshot_values
            .checked_mul(8)
            .and_then(|bytes| resident_bytes.checked_add(bytes))
            .ok_or(ResourcePlanningError::Overflow(
                "thermal snapshot construction",
            ))?;
        Ok(Self {
            resident_bytes,
            output_values: element_count
                .checked_mul(52)
                .and_then(|values| values.checked_add(snapshot_values))
                .and_then(|values| values.checked_add(1))
                .ok_or(ResourcePlanningError::Overflow(
                    "thermal recovered output values",
                ))?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_analysis_core::{InitialStateConflictPolicy, StateInput, StateSource, StateUse};
    use runmat_meshing_core::{fixtures::canonical_tetrahedron_solver_mesh, ElementOrder};

    #[test]
    fn thermal_inventory_tracks_actual_element_order_and_requires_topology() {
        let mut step = ProcedureStep::thermal_steady("consumer");
        assert_eq!(
            ThermalStateStorage::derive(&step, None).unwrap(),
            ThermalStateStorage::default()
        );
        step.state_inputs.push(StateInput {
            input_id: "temperature".into(),
            required_kind: StateKind::Thermal,
            source: StateSource::Upstream {
                producer_step: step.step_id.clone(),
                state_id: step.state_outputs[0].state_id.clone(),
            },
            use_as: StateUse::Prestress,
            conflict: InitialStateConflictPolicy::Reject,
        });
        assert!(ThermalStateStorage::derive(&step, None).is_err());
        for (order, nodes) in [(ElementOrder::Tet4, 4), (ElementOrder::Tet10, 10)] {
            let mesh = canonical_tetrahedron_solver_mesh(order);
            let facts = ThermalStateStorage::derive(&step, Some(&mesh)).unwrap();
            assert_eq!(
                facts.resident_bytes,
                nodes * 112 + (nodes + 10) * 8 + 32 + (3 * nodes + 1) * 8
            );
            assert_eq!(facts.output_values, 4 + 4 * 6 + 4 * 6 + 1 + 3 * nodes + 1);
        }
    }
}
