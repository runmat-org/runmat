use std::collections::{BTreeMap, BTreeSet, VecDeque};

use runmat_analysis_core::{StructuralElementKind, StructuralModel};
use runmat_meshing_core::SolverMeshArtifact;

use crate::physics::structural::topology::node_space::CanonicalStructuralNodeSpace;

use super::{
    canonical_frame::{self, CanonicalFrameTopology},
    solid_components::solid_face_connected_components,
    validate_operator, MatrixDefiniteness, MixedStructuralNullspaceInput,
    NullspaceEliminationEvidence, NullspaceEvidenceError, NullspaceKind, OperatorRole,
    RealSparseMatrix,
};

pub(super) fn prove(
    input: MixedStructuralNullspaceInput<'_>,
) -> Result<NullspaceEliminationEvidence, NullspaceEvidenceError> {
    let MixedStructuralNullspaceInput {
        mesh,
        structural,
        topology,
        layout,
        element_operator,
        connector_operator,
        combined_operator,
        constraints,
        constraint_plan,
        constrained_dofs,
    } = input;
    if topology.solid_node_count != mesh.topology.nodes.len()
        || topology.explicit_node_indices.len() != structural.nodes.len()
        || topology.nodes.len() != layout.node_count()
        || layout.reference_point_count() != constraint_plan.reference_points.len()
        || element_operator.rows() != layout.total_dof_count()
        || combined_operator.rows() != layout.total_dof_count()
        || element_operator.columns() != element_operator.rows()
        || combined_operator.columns() != combined_operator.rows()
        || constraints.full_dof_count != combined_operator.rows()
        || constrained_dofs.len() != combined_operator.rows()
    {
        return Err(NullspaceEvidenceError::DimensionMismatch);
    }
    validate_operator(
        element_operator,
        OperatorRole::Stiffness,
        MatrixDefiniteness::PositiveSemidefinite,
    )?;
    validate_operator(
        combined_operator,
        OperatorRole::Stiffness,
        MatrixDefiniteness::PositiveSemidefinite,
    )?;
    if let Some(connector) = connector_operator {
        validate_operator(
            connector,
            OperatorRole::Stiffness,
            MatrixDefiniteness::PositiveSemidefinite,
        )?;
        let expected = RealSparseMatrix::scaled_sum(
            &[(1.0, element_operator), (1.0, connector)],
            OperatorRole::Stiffness,
            combined_operator.properties(),
        )
        .map_err(|_| NullspaceEvidenceError::DimensionMismatch)?;
        if expected != *combined_operator {
            return Err(NullspaceEvidenceError::BindingMismatch);
        }
    } else if element_operator != combined_operator {
        return Err(NullspaceEvidenceError::BindingMismatch);
    }
    let components = mixed_components(mesh, structural, topology)?;
    let positions = topology
        .nodes
        .iter()
        .map(|node| node.coordinates_m.canonical_value())
        .collect::<Vec<_>>();
    let identities = topology
        .nodes
        .iter()
        .map(|node| node.identity)
        .collect::<Vec<_>>();
    let reference_points =
        canonical_frame::reference_points(constraint_plan, &identities, &components)?;
    canonical_frame::prove(
        CanonicalFrameTopology {
            positions_m: &positions,
            node_identities: &identities,
            components,
            reference_points: &reference_points,
            topology_identity: topology.identity.discretization_digest(),
            kind: NullspaceKind::MixedStructuralRigidBody3d,
        },
        layout,
        element_operator,
        connector_operator,
        combined_operator,
        Some(constraints),
        constrained_dofs,
    )
}

fn mixed_components(
    mesh: &SolverMeshArtifact,
    structural: &StructuralModel,
    topology: &CanonicalStructuralNodeSpace,
) -> Result<Vec<Vec<usize>>, NullspaceEvidenceError> {
    let node_by_id = mesh
        .topology
        .nodes
        .iter()
        .enumerate()
        .map(|(index, node)| (node.node_id, index))
        .collect::<BTreeMap<_, _>>();
    // Sheet and wire meshes carry no volume elements: their nodes connect only through the
    // shells and beams realized on them.
    let solid_components = if mesh.topology.volume_elements.is_empty() {
        Vec::new()
    } else {
        solid_face_connected_components(mesh, &node_by_id)?
    };
    let explicit_by_id = structural
        .nodes
        .iter()
        .enumerate()
        .map(|(index, node)| (node.node_id, index))
        .collect::<BTreeMap<_, _>>();
    if explicit_by_id.len() != structural.nodes.len() {
        return Err(NullspaceEvidenceError::DimensionMismatch);
    }
    let mut neighbors = vec![BTreeSet::new(); topology.nodes.len()];
    for component in solid_components {
        connect_component(&mut neighbors, &component);
    }
    for element in &structural.elements {
        let node_ids: &[u32] = match &element.kind {
            StructuralElementKind::Beam(beam) => &beam.node_ids,
            StructuralElementKind::Shell(shell) => &shell.node_ids,
        };
        let nodes = node_ids
            .iter()
            .map(|node_id| {
                explicit_by_id
                    .get(node_id)
                    .and_then(|explicit| topology.explicit_node_indices.get(*explicit))
                    .copied()
                    .ok_or(NullspaceEvidenceError::DimensionMismatch)
            })
            .collect::<Result<Vec<_>, _>>()?;
        connect_component(&mut neighbors, &nodes);
    }
    let mut seen = vec![false; neighbors.len()];
    let mut components = Vec::new();
    for root in 0..neighbors.len() {
        if seen[root] {
            continue;
        }
        seen[root] = true;
        let mut queue = VecDeque::from([root]);
        let mut component = Vec::new();
        while let Some(node) = queue.pop_front() {
            component.push(node);
            for neighbor in &neighbors[node] {
                if !seen[*neighbor] {
                    seen[*neighbor] = true;
                    queue.push_back(*neighbor);
                }
            }
        }
        component.sort_unstable();
        components.push(component);
    }
    Ok(components)
}

fn connect_component(neighbors: &mut [BTreeSet<usize>], nodes: &[usize]) {
    if let Some((&first, rest)) = nodes.split_first() {
        for &node in rest {
            neighbors[first].insert(node);
            neighbors[node].insert(first);
        }
    }
}
