//! Volume L2 discrepancy of original element-constant stress and nodal recovery.
use super::*;

pub(super) fn estimate(
    artifact: &SolverMeshArtifact,
    stress_topology_id: &str,
    topology: &runmat_meshing_core::FieldTopologyMap,
    values: &[f64],
    options: StructuralRecoveryEstimatorOptions,
) -> Result<StructuralRecoveryEstimate, StructuralRecoveryEstimatorError> {
    let stress_by_element = topology
        .ordered_entity_ids
        .iter()
        .enumerate()
        .map(|(index, element_id)| {
            let offset = index * STRESS_COMPONENT_COUNT;
            (
                *element_id,
                &values[offset..offset + STRESS_COMPONENT_COUNT],
            )
        })
        .collect::<BTreeMap<_, _>>();
    let coordinates = artifact
        .topology
        .nodes
        .iter()
        .map(|node| (node.node_id, node.coordinates_m.canonical_value()))
        .collect::<BTreeMap<_, _>>();
    let mut region_side_nodal_stress =
        BTreeMap::<(u64, &PersistentEntityId), ([f64; STRESS_COMPONENT_COUNT], f64)>::new();
    let mut work = 0_u64;
    for element in &artifact.topology.volume_elements {
        checkpoint(&mut work, options)?;
        let volume = tetrahedron_volume(element, &coordinates)?;
        let element_stress = stress_by_element
            .get(&element.element_id)
            .ok_or(StructuralRecoveryEstimatorError::InvalidStressField)?;
        for node_id in &element.node_ids {
            let (sum, weight) = region_side_nodal_stress
                .entry((*node_id, &element.region_id))
                .or_insert(([0.0; STRESS_COMPONENT_COUNT], 0.0));
            for (sum, value) in sum.iter_mut().zip(*element_stress) {
                *sum += volume * value;
            }
            *weight += volume;
        }
    }

    let mut reference_squared_total = 0.0;
    let mut indicators = Vec::with_capacity(artifact.topology.volume_elements.len());
    for element in &artifact.topology.volume_elements {
        checkpoint(&mut work, options)?;
        let node_count = element.node_ids.len();
        if !matches!(node_count, 4 | 10) {
            return Err(StructuralRecoveryEstimatorError::InvalidElementGeometry);
        }
        let mut nodal = [[0.0; STRESS_COMPONENT_COUNT]; 10];
        for (&node_id, value) in element.node_ids.iter().zip(&mut nodal) {
            let (sum, weight) = region_side_nodal_stress
                .get(&(node_id, &element.region_id))
                .ok_or(StructuralRecoveryEstimatorError::InvalidElementGeometry)?;
            if !weight.is_finite() || *weight <= 0.0 {
                return Err(StructuralRecoveryEstimatorError::InvalidElementGeometry);
            }
            *value = sum.map(|s| s / weight);
        }
        let mut nodes_m = [[0.0; 3]; 10];
        for (&node_id, position) in element.node_ids.iter().zip(&mut nodes_m) {
            *position = *coordinates
                .get(&node_id)
                .ok_or(StructuralRecoveryEstimatorError::InvalidElementGeometry)?;
        }
        let original = stress_by_element
            .get(&element.element_id)
            .ok_or(StructuralRecoveryEstimatorError::InvalidStressField)?;
        let original = std::array::from_fn(|i| original[i]);
        let (error_squared, reference_squared) = super::element_norm::integrate(
            element.order,
            &nodes_m[..node_count],
            &nodal[..node_count],
            original,
            &mut work,
            options,
        )?;
        reference_squared_total += reference_squared;
        if !reference_squared_total.is_finite() {
            return Err(StructuralRecoveryEstimatorError::InvalidStressField);
        }
        let noise_floor = RECOVERY_ROUNDOFF_RELATIVE_TOLERANCE.powi(2) * reference_squared;
        let error = if error_squared <= noise_floor {
            0.0
        } else {
            error_squared.sqrt()
        };

        indicators.push(StructuralRecoveryIndicator {
            element_stable_identity: element.stable_identity,
            error,
        });
    }
    finish_estimate(
        artifact.canonical_digest,
        stress_topology_id,
        indicators,
        reference_squared_total.sqrt(),
        options,
    )
}
