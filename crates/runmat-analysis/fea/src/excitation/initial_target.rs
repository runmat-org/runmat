//! Original volume-region node selection shared by physical initial conditions.
use crate::topology::AnalysisProjection;
use runmat_analysis_core::{EntityDimension, ExpectedTopology, InitialConditionTarget};
#[derive(Debug)]
pub(crate) enum InitialTargetError {
    InvalidTarget,
    UnmappedRegion,
    InvalidSource,
}
pub(crate) fn initial_target_nodes(
    projection: &AnalysisProjection,
    target: &InitialConditionTarget,
) -> Result<Vec<usize>, InitialTargetError> {
    match target {
        InitialConditionTarget::WholeModel => Ok((0..projection.nodes.len()).collect()),
        InitialConditionTarget::Region(target) => {
            if target.entity_dimension != EntityDimension::Volume
                || target.expected_topology != ExpectedTopology::MeshElement
            {
                return Err(InitialTargetError::InvalidTarget);
            }
            let mut selected = vec![false; projection.nodes.len()];
            let mut matched = false;
            for element in &projection.elements {
                if target.selects_volume_element(&element.region_id, &element.provenance) {
                    matched = true;
                    for &node in &element.node_indices {
                        *selected
                            .get_mut(node)
                            .ok_or(InitialTargetError::InvalidSource)? = true;
                    }
                }
            }
            if !matched {
                return Err(InitialTargetError::UnmappedRegion);
            }
            Ok(selected
                .iter()
                .enumerate()
                .filter_map(|(node, v)| v.then_some(node))
                .collect())
        }
    }
}
