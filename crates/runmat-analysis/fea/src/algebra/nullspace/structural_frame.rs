use std::collections::{BTreeMap, BTreeSet, VecDeque};

use crate::assembly::dofs::StructuralDofLayout;
use crate::constraints::{CanonicalStructuralConstraintPlan, ResolvedConstraintMatrix};
use crate::physics::structural::topology::node_space::CanonicalStructuralNodeSpace;
use runmat_analysis_core::{StructuralElementKind, StructuralModel};

use super::{
    validate_operator, ConnectorAnchoredStructuralFrameInput, MatrixDefiniteness,
    NullspaceEliminationEvidence, NullspaceEvidenceError, NullspaceKind, OperatorRole,
    RealSparseMatrix,
};

struct StructuralFrameProofInput<'a> {
    structural: &'a StructuralModel,
    layout: &'a StructuralDofLayout,
    element_operator: &'a RealSparseMatrix,
    connector_operator: Option<&'a RealSparseMatrix>,
    combined_operator: &'a RealSparseMatrix,
    plan: &'a CanonicalStructuralConstraintPlan,
    constraints: &'a ResolvedConstraintMatrix,
    constrained_dofs: &'a [bool],
}

pub(super) fn prove(
    structural: &StructuralModel,
    layout: &StructuralDofLayout,
    operator: &RealSparseMatrix,
    plan: &CanonicalStructuralConstraintPlan,
    constraints: &ResolvedConstraintMatrix,
    constrained_dofs: &[bool],
) -> Result<NullspaceEliminationEvidence, NullspaceEvidenceError> {
    prove_impl(StructuralFrameProofInput {
        structural,
        layout,
        element_operator: operator,
        connector_operator: None,
        combined_operator: operator,
        plan,
        constraints,
        constrained_dofs,
    })
}

pub(super) fn prove_with_connector_anchors(
    input: ConnectorAnchoredStructuralFrameInput<'_>,
) -> Result<NullspaceEliminationEvidence, NullspaceEvidenceError> {
    prove_impl(StructuralFrameProofInput {
        structural: input.structural,
        layout: input.layout,
        element_operator: input.element_operator,
        connector_operator: Some(input.connector_operator),
        combined_operator: input.combined_operator,
        plan: input.plan,
        constraints: input.constraints,
        constrained_dofs: input.constrained_dofs,
    })
}

fn prove_impl(
    input: StructuralFrameProofInput<'_>,
) -> Result<NullspaceEliminationEvidence, NullspaceEvidenceError> {
    let StructuralFrameProofInput {
        structural,
        layout,
        element_operator,
        connector_operator,
        combined_operator,
        plan,
        constraints,
        constrained_dofs,
    } = input;
    if structural.nodes.is_empty()
        || layout.node_count() != structural.nodes.len()
        || layout.total_dof_count() != combined_operator.rows()
        || combined_operator.rows() != combined_operator.columns()
        || element_operator.rows() != combined_operator.rows()
        || element_operator.columns() != combined_operator.columns()
        || connector_operator.is_some_and(|operator| {
            operator.rows() != combined_operator.rows()
                || operator.columns() != combined_operator.columns()
        })
        || constrained_dofs.len() != combined_operator.rows()
        || constraints.full_dof_count != combined_operator.rows()
        || layout.reference_point_count() != plan.reference_points.len()
        || structural.elements.iter().any(|element| {
            !matches!(
                element.kind,
                StructuralElementKind::Beam(_) | StructuralElementKind::Shell(_)
            )
        })
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
    if let Some(operator) = connector_operator {
        validate_operator(
            operator,
            OperatorRole::Stiffness,
            MatrixDefiniteness::PositiveSemidefinite,
        )?;
        let expected = RealSparseMatrix::scaled_sum(
            &[(1.0, element_operator), (1.0, operator)],
            OperatorRole::Stiffness,
            combined_operator.properties(),
        )
        .map_err(|_| NullspaceEvidenceError::DimensionMismatch)?;
        if expected != *combined_operator {
            return Err(NullspaceEvidenceError::BindingMismatch);
        }
    }

    let node_by_id = structural
        .nodes
        .iter()
        .enumerate()
        .map(|(index, node)| (node.node_id, index))
        .collect::<BTreeMap<_, _>>();
    if node_by_id.len() != structural.nodes.len() {
        return Err(NullspaceEvidenceError::DimensionMismatch);
    }
    let components = structural_frame_components(structural, &node_by_id)?;
    let topology = CanonicalStructuralNodeSpace::derive_explicit(structural)
        .map_err(|_| NullspaceEvidenceError::DimensionMismatch)?;
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
        super::canonical_frame::reference_points(plan, &identities, &components)?;
    super::canonical_frame::prove(
        super::canonical_frame::CanonicalFrameTopology {
            positions_m: &positions,
            node_identities: &identities,
            components,
            reference_points: &reference_points,
            topology_identity: topology.identity.discretization_digest(),
            kind: NullspaceKind::StructuralFrameRigidBody3d,
        },
        layout,
        element_operator,
        connector_operator,
        combined_operator,
        Some(constraints),
        constrained_dofs,
    )
}

fn structural_frame_components(
    structural: &StructuralModel,
    node_by_id: &BTreeMap<u32, usize>,
) -> Result<Vec<Vec<usize>>, NullspaceEvidenceError> {
    let mut neighbors = vec![BTreeSet::<usize>::new(); structural.nodes.len()];
    for element in &structural.elements {
        let node_ids: &[u32] = match &element.kind {
            StructuralElementKind::Beam(beam) => &beam.node_ids,
            StructuralElementKind::Shell(shell) => &shell.node_ids,
        };
        let nodes = node_ids
            .iter()
            .map(|node_id| {
                node_by_id
                    .get(node_id)
                    .copied()
                    .ok_or(NullspaceEvidenceError::DimensionMismatch)
            })
            .collect::<Result<Vec<_>, _>>()?;
        for left in 0..nodes.len() {
            for right in left + 1..nodes.len() {
                neighbors[nodes[left]].insert(nodes[right]);
                neighbors[nodes[right]].insert(nodes[left]);
            }
        }
    }
    let mut seen = vec![false; structural.nodes.len()];
    let mut components = Vec::new();
    for root in 0..structural.nodes.len() {
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
