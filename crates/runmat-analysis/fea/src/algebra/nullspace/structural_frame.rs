use std::collections::{BTreeMap, BTreeSet, VecDeque};

use runmat_analysis_core::{StructuralElementKind, StructuralModel};
use runmat_meshing_core::StableDigest;

use crate::assembly::dofs::StructuralDofLayout;

use super::{
    digest, numerical_rank, rigid_body::RigidBodyFrame, validate_operator, verify_null_mode,
    MatrixDefiniteness, NullspaceAnchoring, NullspaceComponentEvidence,
    NullspaceEliminationEvidence, NullspaceEvidenceError, NullspaceKind, OperatorRole,
    RealSparseMatrix,
};

const TOPOLOGY_DOMAIN: &str = "fea-explicit-structural-frame-topology";
const NODE_DOMAIN: &str = "fea-explicit-structural-frame-node";
const CONSTRAINT_DOMAIN: &str = "fea-explicit-structural-frame-constraint-state";
const RIGID_MODE_COUNT: usize = 6;

pub(super) fn prove(
    structural: &StructuralModel,
    layout: &StructuralDofLayout,
    operator: &RealSparseMatrix,
    constrained_dofs: &[bool],
) -> Result<NullspaceEliminationEvidence, NullspaceEvidenceError> {
    if structural.nodes.is_empty()
        || layout.node_count() != structural.nodes.len()
        || layout.total_dof_count() != operator.rows()
        || operator.rows() != operator.columns()
        || constrained_dofs.len() != operator.rows()
        || layout.reference_point_count() != 0
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
        operator,
        OperatorRole::Stiffness,
        MatrixDefiniteness::PositiveSemidefinite,
    )?;

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
    let component_by_node = components
        .iter()
        .enumerate()
        .flat_map(|(component, nodes)| nodes.iter().map(move |node| (*node, component)))
        .collect::<BTreeMap<_, _>>();
    let frames = components
        .iter()
        .map(|nodes| {
            let positions = nodes
                .iter()
                .map(|node| structural.nodes[*node].coordinates_m)
                .collect::<Vec<_>>();
            RigidBodyFrame::from_positions(&positions)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mode_count = components
        .len()
        .checked_mul(RIGID_MODE_COUNT)
        .ok_or(NullspaceEvidenceError::DimensionMismatch)?;
    let context = FrameModeContext {
        structural,
        layout,
        component_by_node: &component_by_node,
        frames: &frames,
        mode_count,
    };

    let mut modes = vec![vec![0.0; operator.columns()]; mode_count];
    for column in 0..operator.columns() {
        for (mode, value) in rigid_mode_response(column, &context)?
            .into_iter()
            .enumerate()
        {
            modes[mode][column] = value;
        }
    }
    for mode in &modes {
        let applied =
            super::spmv(operator, mode).map_err(|_| NullspaceEvidenceError::DimensionMismatch)?;
        verify_null_mode(operator, mode, &applied)?;
    }
    let projected = constrained_dofs
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, constrained)| *constrained)
        .map(|(column, _)| rigid_mode_response(column, &context))
        .collect::<Result<Vec<_>, _>>()?;
    if numerical_rank(projected) != mode_count {
        return Err(NullspaceEvidenceError::UnderconstrainedComponent);
    }

    let component_evidence = components
        .into_iter()
        .enumerate()
        .map(|(component_index, nodes)| {
            let mut node_identities = nodes
                .iter()
                .map(|node| explicit_node_identity(structural, *node))
                .collect::<Result<Vec<_>, _>>()?;
            node_identities.sort_unstable();
            Ok(NullspaceComponentEvidence {
                component_index,
                node_identities,
                null_mode_count: RIGID_MODE_COUNT,
                projected_constraint_rank: RIGID_MODE_COUNT,
                anchoring: NullspaceAnchoring::ConstraintRowSpace,
            })
        })
        .collect::<Result<Vec<_>, NullspaceEvidenceError>>()?;
    NullspaceEliminationEvidence::new(
        NullspaceKind::StructuralFrameRigidBody3d,
        digest(TOPOLOGY_DOMAIN, &(&structural.nodes, &structural.elements))?,
        operator,
        digest(CONSTRAINT_DOMAIN, &constrained_dofs)?,
        constrained_dofs,
        component_evidence,
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

struct FrameModeContext<'a> {
    structural: &'a StructuralModel,
    layout: &'a StructuralDofLayout,
    component_by_node: &'a BTreeMap<usize, usize>,
    frames: &'a [RigidBodyFrame],
    mode_count: usize,
}

fn rigid_mode_response(
    column: usize,
    context: &FrameModeContext<'_>,
) -> Result<Vec<f64>, NullspaceEvidenceError> {
    let address = context
        .layout
        .address(column)
        .ok_or(NullspaceEvidenceError::DimensionMismatch)?;
    let node = context
        .structural
        .nodes
        .get(address.node_index)
        .ok_or(NullspaceEvidenceError::DimensionMismatch)?;
    let component = context
        .component_by_node
        .get(&address.node_index)
        .copied()
        .ok_or(NullspaceEvidenceError::DimensionMismatch)?;
    let frame = context
        .frames
        .get(component)
        .ok_or(NullspaceEvidenceError::DimensionMismatch)?;
    let values = frame.response(node.coordinates_m, address.kind);
    let mut response = vec![0.0; context.mode_count];
    let offset = component * RIGID_MODE_COUNT;
    response[offset..offset + RIGID_MODE_COUNT].copy_from_slice(&values);
    Ok(response)
}

fn explicit_node_identity(
    structural: &StructuralModel,
    node: usize,
) -> Result<StableDigest, NullspaceEvidenceError> {
    let node = structural
        .nodes
        .get(node)
        .ok_or(NullspaceEvidenceError::DimensionMismatch)?;
    let identity = digest(NODE_DOMAIN, &(node.node_id, node.coordinates_m))?;
    Ok(StableDigest::from_bytes(*identity.bytes()))
}
