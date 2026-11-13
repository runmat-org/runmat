use std::collections::{BTreeMap, BTreeSet};

use runmat_execution::Digest;
use runmat_meshing_core::SolverMeshArtifact;

use crate::{
    assembly::dofs::{StructuralDofKind, StructuralDofLayout},
    constraints::{CanonicalStructuralConstraintPlan, ResolvedConstraintMatrix},
};

use super::{
    canonical_frame::{self, CanonicalFrameTopology},
    digest, numerical_rank,
    rigid_body::RigidBodyFrame,
    solid_components::solid_face_connected_components,
    validate_operator, verify_null_mode, ConnectorAnchoredSolidInput, MatrixDefiniteness,
    NullspaceAnchoring, NullspaceComponentEvidence, NullspaceEliminationEvidence,
    NullspaceEvidenceError, NullspaceKind, OperatorRole, RealSparseMatrix, CONSTRAINT_DOMAIN,
};

pub(super) fn prove(
    mesh: &SolverMeshArtifact,
    layout: &StructuralDofLayout,
    plan: &CanonicalStructuralConstraintPlan,
    constraints: &ResolvedConstraintMatrix,
    operator: &RealSparseMatrix,
    constrained_dofs: &[bool],
) -> Result<NullspaceEliminationEvidence, NullspaceEvidenceError> {
    constraints
        .validate()
        .map_err(|_| NullspaceEvidenceError::InvalidEvidence)?;
    let expected_constraints = crate::assembly::resolve_solid_constraint_matrix(plan, mesh, layout)
        .map_err(|_| NullspaceEvidenceError::InvalidEvidence)?;
    if constraints != &expected_constraints {
        return Err(NullspaceEvidenceError::BindingMismatch);
    }
    let node_count = mesh.topology.nodes.len();
    if layout.node_count() != node_count
        || layout.total_dof_count() != operator.rows()
        || operator.rows() != operator.columns()
        || constraints.full_dof_count != operator.rows()
        || constrained_dofs.len() != operator.rows()
    {
        return Err(NullspaceEvidenceError::DimensionMismatch);
    }
    validate_operator(
        operator,
        OperatorRole::Stiffness,
        MatrixDefiniteness::PositiveSemidefinite,
    )?;
    let node_by_id = mesh
        .topology
        .nodes
        .iter()
        .enumerate()
        .map(|(index, node)| (node.node_id, index))
        .collect::<BTreeMap<_, _>>();
    let component_nodes = solid_face_connected_components(mesh, &node_by_id)?;
    let component_frames = component_nodes
        .iter()
        .map(|nodes| {
            let positions = nodes
                .iter()
                .map(|node| mesh.topology.nodes[*node].coordinates_m.canonical_value())
                .collect::<Vec<_>>();
            RigidBodyFrame::from_positions(&positions)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let component_by_node = component_nodes
        .iter()
        .enumerate()
        .flat_map(|(component, nodes)| nodes.iter().map(move |node| (*node, component)))
        .collect::<BTreeMap<_, _>>();
    let stable_node_index = mesh
        .topology
        .nodes
        .iter()
        .enumerate()
        .map(|(index, node)| (node.stable_identity, index))
        .collect::<BTreeMap<_, _>>();
    let mut reference_component = BTreeMap::new();
    for group in &plan.groups {
        let Some(coupling) = &group.coupling else {
            continue;
        };
        let associated = coupling
            .target_weights
            .iter()
            .map(|weight| {
                stable_node_index
                    .get(&weight.node_identity)
                    .and_then(|node| component_by_node.get(node))
                    .copied()
            })
            .collect::<Option<BTreeSet<_>>>()
            .ok_or(NullspaceEvidenceError::AmbiguousReferencePoint)?;
        let associated = associated.into_iter().collect::<Vec<_>>();
        let [component] = associated.as_slice() else {
            return Err(NullspaceEvidenceError::AmbiguousReferencePoint);
        };
        if reference_component
            .insert(coupling.reference_point_identity, *component)
            .is_some_and(|existing| existing != *component)
        {
            return Err(NullspaceEvidenceError::AmbiguousReferencePoint);
        }
    }
    if plan
        .reference_points
        .iter()
        .any(|point| !reference_component.contains_key(&point.identity))
    {
        return Err(NullspaceEvidenceError::AmbiguousReferencePoint);
    }

    let mode_count = component_nodes
        .len()
        .checked_mul(6)
        .ok_or(NullspaceEvidenceError::DimensionMismatch)?;
    let rigid_modes = RigidModeContext {
        layout,
        mesh,
        plan,
        component_by_node: &component_by_node,
        reference_component: &reference_component,
        component_frames: &component_frames,
        mode_count,
    };
    let mut modes = vec![vec![0.0; operator.columns()]; mode_count];
    for column in 0..operator.columns() {
        let response = rigid_mode_response(column, &rigid_modes)?;
        for (mode, value) in response.into_iter().enumerate() {
            modes[mode][column] = value;
        }
    }
    for mode in &modes {
        let applied =
            super::spmv(operator, mode).map_err(|_| NullspaceEvidenceError::DimensionMismatch)?;
        verify_null_mode(operator, mode, &applied)?;
    }
    let mut projected = vec![vec![0.0; mode_count]; constraints.row_count()];
    for (row, projected_row) in projected.iter_mut().enumerate() {
        for entry in constraints.row_offsets[row]..constraints.row_offsets[row + 1] {
            let column = constraints.column_indices[entry];
            let coefficient = constraints.coefficients[entry];
            let response = rigid_mode_response(column, &rigid_modes)?;
            for (value, mode_value) in projected_row.iter_mut().zip(response) {
                *value += coefficient * mode_value;
            }
        }
    }
    if numerical_rank(projected) != mode_count {
        return Err(NullspaceEvidenceError::UnderconstrainedComponent);
    }
    let components = component_nodes
        .into_iter()
        .enumerate()
        .map(|(component_index, nodes)| {
            let mut node_identities = nodes
                .iter()
                .map(|node| mesh.topology.nodes[*node].stable_identity)
                .collect::<Vec<_>>();
            node_identities.sort_unstable();
            NullspaceComponentEvidence {
                component_index,
                node_identities,
                null_mode_count: 6,
                projected_anchor_rank: 6,
                anchoring: NullspaceAnchoring::ConstraintRowSpace,
            }
        })
        .collect();
    let plan_identity = plan
        .content_digest()
        .map_err(|error| NullspaceEvidenceError::Encoding(error.to_string()))?;
    NullspaceEliminationEvidence::new(
        NullspaceKind::SolidRigidBody3d,
        Digest::from_bytes(*mesh.canonical_digest.bytes()),
        operator,
        digest(CONSTRAINT_DOMAIN, &(plan_identity, constraints))?,
        constrained_dofs,
        components,
    )
}

pub(super) fn prove_with_connector_anchors(
    input: ConnectorAnchoredSolidInput<'_>,
) -> Result<NullspaceEliminationEvidence, NullspaceEvidenceError> {
    let ConnectorAnchoredSolidInput {
        mesh,
        layout,
        element_operator,
        connector_operator,
        combined_operator,
        plan,
        constraints,
        constrained_dofs,
    } = input;
    constraints
        .validate()
        .map_err(|_| NullspaceEvidenceError::InvalidEvidence)?;
    let expected_constraints = crate::assembly::resolve_solid_constraint_matrix(plan, mesh, layout)
        .map_err(|_| NullspaceEvidenceError::InvalidEvidence)?;
    if constraints != &expected_constraints
        || layout.node_count() != mesh.topology.nodes.len()
        || layout.total_dof_count() != combined_operator.rows()
        || element_operator.rows() != combined_operator.rows()
        || element_operator.columns() != combined_operator.columns()
        || connector_operator.rows() != combined_operator.rows()
        || connector_operator.columns() != combined_operator.columns()
        || constraints.full_dof_count != combined_operator.rows()
        || constrained_dofs.len() != combined_operator.rows()
    {
        return Err(NullspaceEvidenceError::BindingMismatch);
    }
    for operator in [element_operator, connector_operator, combined_operator] {
        validate_operator(
            operator,
            OperatorRole::Stiffness,
            MatrixDefiniteness::PositiveSemidefinite,
        )?;
    }
    let expected = RealSparseMatrix::scaled_sum(
        &[(1.0, element_operator), (1.0, connector_operator)],
        OperatorRole::Stiffness,
        combined_operator.properties(),
    )
    .map_err(|_| NullspaceEvidenceError::DimensionMismatch)?;
    if expected != *combined_operator {
        return Err(NullspaceEvidenceError::BindingMismatch);
    }
    let node_by_id = mesh
        .topology
        .nodes
        .iter()
        .enumerate()
        .map(|(index, node)| (node.node_id, index))
        .collect::<BTreeMap<_, _>>();
    let components = solid_face_connected_components(mesh, &node_by_id)?;
    let positions = mesh
        .topology
        .nodes
        .iter()
        .map(|node| node.coordinates_m.canonical_value())
        .collect::<Vec<_>>();
    let identities = mesh
        .topology
        .nodes
        .iter()
        .map(|node| node.stable_identity)
        .collect::<Vec<_>>();
    let reference_points = canonical_frame::reference_points(plan, &identities, &components)?;
    canonical_frame::prove(
        CanonicalFrameTopology {
            positions_m: &positions,
            node_identities: &identities,
            components,
            reference_points: &reference_points,
            topology_identity: Digest::from_bytes(*mesh.canonical_digest.bytes()),
            kind: NullspaceKind::SolidRigidBody3d,
        },
        layout,
        element_operator,
        Some(connector_operator),
        combined_operator,
        Some(constraints),
        constrained_dofs,
    )
}

struct RigidModeContext<'a> {
    layout: &'a StructuralDofLayout,
    mesh: &'a SolverMeshArtifact,
    plan: &'a CanonicalStructuralConstraintPlan,
    component_by_node: &'a BTreeMap<usize, usize>,
    reference_component: &'a BTreeMap<Digest, usize>,
    component_frames: &'a [RigidBodyFrame],
    mode_count: usize,
}

fn rigid_mode_response(
    column: usize,
    context: &RigidModeContext<'_>,
) -> Result<Vec<f64>, NullspaceEvidenceError> {
    let mut response = vec![0.0; context.mode_count];
    let (component, position, kind) = if let Some(address) = context.layout.address(column) {
        let node = context
            .mesh
            .topology
            .nodes
            .get(address.node_index)
            .ok_or(NullspaceEvidenceError::DimensionMismatch)?;
        (
            *context
                .component_by_node
                .get(&address.node_index)
                .ok_or(NullspaceEvidenceError::DimensionMismatch)?,
            node.coordinates_m.canonical_value(),
            address.kind,
        )
    } else {
        let reference_rows = context
            .layout
            .reference_point_count()
            .checked_mul(6)
            .ok_or(NullspaceEvidenceError::DimensionMismatch)?;
        let base = context
            .layout
            .total_dof_count()
            .checked_sub(reference_rows)
            .ok_or(NullspaceEvidenceError::DimensionMismatch)?;
        let relative = column
            .checked_sub(base)
            .ok_or(NullspaceEvidenceError::DimensionMismatch)?;
        let reference_index = relative / 6;
        let component_index = relative % 6;
        let point = context
            .plan
            .reference_points
            .get(reference_index)
            .ok_or(NullspaceEvidenceError::DimensionMismatch)?;
        (
            *context
                .reference_component
                .get(&point.identity)
                .ok_or(NullspaceEvidenceError::AmbiguousReferencePoint)?,
            point.position_meters,
            StructuralDofKind::ORDER[component_index],
        )
    };
    let offset = component * 6;
    let frame = context
        .component_frames
        .get(component)
        .ok_or(NullspaceEvidenceError::DimensionMismatch)?;
    let values = frame.response(position, kind);
    response[offset..offset + 6].copy_from_slice(&values);
    Ok(response)
}
