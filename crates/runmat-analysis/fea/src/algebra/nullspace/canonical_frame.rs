use std::collections::{BTreeMap, BTreeSet};

use runmat_execution::Digest;
use runmat_meshing_core::StableDigest;

use crate::assembly::dofs::{StructuralDofKind, StructuralDofLayout};
use crate::constraints::{CanonicalStructuralConstraintPlan, ResolvedConstraintMatrix};

use super::{
    digest, numerical_rank, rigid_body::RigidBodyFrame, verify_null_mode, NullspaceAnchoring,
    NullspaceComponentEvidence, NullspaceEliminationEvidence, NullspaceEvidenceError,
    NullspaceKind, RealSparseMatrix,
};

const RIGID_MODE_COUNT: usize = 6;
const CONSTRAINT_DOMAIN: &str = "fea-canonical-frame-constraint-state";

pub(super) struct CanonicalFrameTopology<'a> {
    pub positions_m: &'a [[f64; 3]],
    pub node_identities: &'a [StableDigest],
    pub components: Vec<Vec<usize>>,
    pub reference_points: &'a [CanonicalFrameReferencePoint],
    pub topology_identity: Digest,
    pub kind: NullspaceKind,
}

pub(super) struct CanonicalFrameReferencePoint {
    pub position_m: [f64; 3],
    pub component: usize,
}

pub(super) fn reference_points(
    plan: &CanonicalStructuralConstraintPlan,
    node_identities: &[StableDigest],
    components: &[Vec<usize>],
) -> Result<Vec<CanonicalFrameReferencePoint>, NullspaceEvidenceError> {
    let component_by_node = components
        .iter()
        .enumerate()
        .flat_map(|(component, nodes)| nodes.iter().map(move |node| (*node, component)))
        .collect::<BTreeMap<_, _>>();
    let node_by_identity = node_identities
        .iter()
        .copied()
        .enumerate()
        .map(|(index, identity)| (identity, index))
        .collect::<BTreeMap<_, _>>();
    let mut reference_components = BTreeMap::new();
    for group in &plan.groups {
        let Some(coupling) = &group.coupling else {
            continue;
        };
        let associated = coupling
            .target_weights
            .iter()
            .map(|weight| {
                node_by_identity
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
        if reference_components
            .insert(coupling.reference_point_identity, *component)
            .is_some_and(|existing| existing != *component)
        {
            return Err(NullspaceEvidenceError::AmbiguousReferencePoint);
        }
    }
    plan.reference_points
        .iter()
        .map(|point| {
            Ok(CanonicalFrameReferencePoint {
                position_m: point.position_meters,
                component: *reference_components
                    .get(&point.identity)
                    .ok_or(NullspaceEvidenceError::AmbiguousReferencePoint)?,
            })
        })
        .collect()
}

pub(super) fn prove(
    topology: CanonicalFrameTopology<'_>,
    layout: &StructuralDofLayout,
    element_operator: &RealSparseMatrix,
    connector_operator: Option<&RealSparseMatrix>,
    combined_operator: &RealSparseMatrix,
    constraints: Option<&ResolvedConstraintMatrix>,
    constrained_dofs: &[bool],
) -> Result<NullspaceEliminationEvidence, NullspaceEvidenceError> {
    if topology.positions_m.len() != topology.node_identities.len()
        || layout.node_count() != topology.positions_m.len()
        || layout.reference_point_count() != topology.reference_points.len()
        || layout.total_dof_count() != combined_operator.rows()
        || constraints.is_some_and(|matrix| matrix.full_dof_count != combined_operator.rows())
        || constrained_dofs.len() != combined_operator.rows()
    {
        return Err(NullspaceEvidenceError::DimensionMismatch);
    }
    let component_by_node = topology
        .components
        .iter()
        .enumerate()
        .flat_map(|(component, nodes)| nodes.iter().map(move |node| (*node, component)))
        .collect::<BTreeMap<_, _>>();
    if component_by_node.len() != topology.positions_m.len() {
        return Err(NullspaceEvidenceError::DimensionMismatch);
    }
    let frames = topology
        .components
        .iter()
        .map(|nodes| {
            let positions = nodes
                .iter()
                .map(|node| topology.positions_m[*node])
                .collect::<Vec<_>>();
            RigidBodyFrame::from_positions(&positions)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mode_count = topology
        .components
        .len()
        .checked_mul(RIGID_MODE_COUNT)
        .ok_or(NullspaceEvidenceError::DimensionMismatch)?;
    let context = FrameModeContext {
        positions_m: topology.positions_m,
        layout,
        component_by_node: &component_by_node,
        frames: &frames,
        reference_points: topology.reference_points,
        mode_count,
    };
    let mut modes = vec![vec![0.0; combined_operator.columns()]; mode_count];
    for column in 0..combined_operator.columns() {
        for (mode, value) in rigid_mode_response(column, &context)?
            .into_iter()
            .enumerate()
        {
            modes[mode][column] = value;
        }
    }
    for mode in &modes {
        let applied = super::spmv(element_operator, mode)
            .map_err(|_| NullspaceEvidenceError::DimensionMismatch)?;
        verify_null_mode(element_operator, mode, &applied)?;
    }
    let mut projected = if let Some(constraints) = constraints {
        constraints
            .validate()
            .map_err(|error| NullspaceEvidenceError::Encoding(error.to_string()))?;
        (0..constraints.row_count())
            .map(|row| {
                let start = constraints.row_offsets[row];
                let end = constraints.row_offsets[row + 1];
                let mut response = vec![0.0; mode_count];
                for entry in start..end {
                    let column = constraints.column_indices[entry];
                    let coefficient = constraints.coefficients[entry];
                    for (mode, value) in modes.iter().enumerate() {
                        response[mode] += coefficient * value[column];
                    }
                }
                response
            })
            .collect::<Vec<_>>()
    } else {
        constrained_dofs
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, constrained)| *constrained)
            .map(|(column, _)| rigid_mode_response(column, &context))
            .collect::<Result<Vec<_>, _>>()?
    };
    if let Some(operator) = connector_operator {
        let applied_modes = modes
            .iter()
            .map(|mode| {
                super::spmv(operator, mode).map_err(|_| NullspaceEvidenceError::DimensionMismatch)
            })
            .collect::<Result<Vec<_>, _>>()?;
        projected.extend((0..operator.rows()).map(|row| {
            applied_modes
                .iter()
                .map(|applied| applied[row])
                .collect::<Vec<_>>()
        }));
    }
    if numerical_rank(projected) != mode_count {
        return Err(NullspaceEvidenceError::UnderconstrainedComponent);
    }
    let components = if connector_operator.is_some() {
        let mut identities = topology.node_identities.to_vec();
        identities.sort_unstable();
        vec![NullspaceComponentEvidence {
            component_index: 0,
            node_identities: identities,
            null_mode_count: mode_count,
            projected_anchor_rank: mode_count,
            anchoring: NullspaceAnchoring::OperatorAndConstraintRowSpace,
        }]
    } else {
        topology
            .components
            .into_iter()
            .enumerate()
            .map(|(component_index, nodes)| {
                let mut identities = nodes
                    .into_iter()
                    .map(|node| topology.node_identities[node])
                    .collect::<Vec<_>>();
                identities.sort_unstable();
                NullspaceComponentEvidence {
                    component_index,
                    node_identities: identities,
                    null_mode_count: RIGID_MODE_COUNT,
                    projected_anchor_rank: RIGID_MODE_COUNT,
                    anchoring: NullspaceAnchoring::ConstraintRowSpace,
                }
            })
            .collect()
    };
    NullspaceEliminationEvidence::new(
        topology.kind,
        topology.topology_identity,
        combined_operator,
        match constraints {
            Some(matrix) => digest(CONSTRAINT_DOMAIN, matrix)?,
            None => digest(CONSTRAINT_DOMAIN, &constrained_dofs)?,
        },
        constrained_dofs,
        components,
    )
}

struct FrameModeContext<'a> {
    positions_m: &'a [[f64; 3]],
    layout: &'a StructuralDofLayout,
    component_by_node: &'a BTreeMap<usize, usize>,
    frames: &'a [RigidBodyFrame],
    reference_points: &'a [CanonicalFrameReferencePoint],
    mode_count: usize,
}

fn rigid_mode_response(
    column: usize,
    context: &FrameModeContext<'_>,
) -> Result<Vec<f64>, NullspaceEvidenceError> {
    let (position, component, kind) = if let Some(address) = context.layout.address(column) {
        (
            context
                .positions_m
                .get(address.node_index)
                .copied()
                .ok_or(NullspaceEvidenceError::DimensionMismatch)?,
            context
                .component_by_node
                .get(&address.node_index)
                .copied()
                .ok_or(NullspaceEvidenceError::DimensionMismatch)?,
            address.kind,
        )
    } else {
        let relative = column
            .checked_sub(context.layout.node_dof_count())
            .ok_or(NullspaceEvidenceError::DimensionMismatch)?;
        let point = context
            .reference_points
            .get(relative / StructuralDofKind::ORDER.len())
            .ok_or(NullspaceEvidenceError::DimensionMismatch)?;
        (
            point.position_m,
            point.component,
            StructuralDofKind::ORDER[relative % StructuralDofKind::ORDER.len()],
        )
    };
    let values = context
        .frames
        .get(component)
        .ok_or(NullspaceEvidenceError::DimensionMismatch)?
        .response(position, kind);
    let mut response = vec![0.0; context.mode_count];
    let offset = component * RIGID_MODE_COUNT;
    response[offset..offset + RIGID_MODE_COUNT].copy_from_slice(&values);
    Ok(response)
}
