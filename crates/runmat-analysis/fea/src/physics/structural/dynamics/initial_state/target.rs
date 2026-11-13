use runmat_analysis_core::{InitialConditionTarget, InitialConditionValue, StructuralModel};
use runmat_meshing_core::SolverMeshArtifact;

use crate::physics::structural::topology::{
    explicit_target_node_indices, node_space::CanonicalStructuralNodeSpace,
    target::resolve_mixed_structural_target,
};

use super::StructuralInitialStateError;

pub(super) fn target_nodes<Q>(
    mesh: Option<&SolverMeshArtifact>,
    structural: &StructuralModel,
    topology: &CanonicalStructuralNodeSpace,
    condition: &InitialConditionValue<Q>,
) -> Result<Vec<usize>, StructuralInitialStateError> {
    match &condition.target {
        InitialConditionTarget::WholeModel => Ok((0..topology.nodes.len()).collect()),
        InitialConditionTarget::Region(target) => {
            let result = if let Some(mesh) = mesh {
                resolve_mixed_structural_target(target, mesh, structural, topology)
                    .map(|ownership| ownership.into_canonical_nodes())
                    .map_err(|error| error.to_string())
            } else {
                let mut explicit = explicit_target_node_indices(structural, &target.region_id)
                    .map_err(StructuralInitialStateError::Storage)?;
                if explicit.is_empty() {
                    Err("target does not resolve to explicit structural topology".into())
                } else {
                    for node in &mut explicit {
                        *node = topology.explicit_node_indices[*node];
                    }
                    Ok(explicit)
                }
            };
            result.map_err(|detail| StructuralInitialStateError::InvalidTarget {
                initial_condition_id: condition.initial_condition_id.clone(),
                selector: target.region_id.clone(),
                detail,
            })
        }
    }
}
