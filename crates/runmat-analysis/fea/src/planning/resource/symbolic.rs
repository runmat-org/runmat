use std::collections::BTreeSet;

use runmat_analysis_core::{AnalysisModel, StructuralElementKind, StructuralModel};
use runmat_canonical_codec::{encode_contract, CanonicalLimits};
use runmat_execution::Digest;
use runmat_meshing_core::SolverMeshArtifact;

use crate::{
    algebra::SparsePattern,
    assembly::{dofs::StructuralDofKind, dofs::StructuralDofLayout, mixed_dof_layout},
    constraints::{CanonicalStructuralConstraintPlan, CanonicalStructuralTopologyIdentity},
    physics::structural::topology::{
        node_space::CanonicalStructuralNodeSpace, target::resolve_mixed_structural_target,
    },
    topology::AnalysisProjection,
};

use super::ResourcePlanningError;

const SYMBOLIC_PREFIX: &[u8] = b"runmat-fea-symbolic-operators/v1\0";
const SYMBOLIC_LIMITS: CanonicalLimits = CanonicalLimits::new(1024 * 1024, 64, 64, 8);
const TRANSLATIONAL_DOF_COUNT: usize = 3;
const REFERENCE_POINT_DOF_COUNT: usize = 6;

/// Material-independent sparse storage bounds for a structural solve.
///
/// The stiffness pattern contains every coupling that element assembly may emit, including
/// coefficients that happen to evaluate to zero for a particular material or geometry. Solid-only
/// mass follows its admitted diagonal storage contract. Connector-bearing solid damping reserves
/// the stiffness graph because axial dashpots can couple distinct nodes. Mixed systems conservatively
/// reserve the complete element-and-connector stiffness bound for all three operators because
/// explicit consistent mass and Rayleigh damping can populate those couplings. This plan is derived
/// before numerical assembly and therefore can be used for allocation admission without doing the
/// work it is intended to guard.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StructuralSymbolicOperators {
    stiffness_upper_bound: SparsePattern,
    mass: SparsePattern,
    damping: SparsePattern,
    identity: Digest,
}

impl StructuralSymbolicOperators {
    pub fn derive(
        projection: &AnalysisProjection,
        constraints: &CanonicalStructuralConstraintPlan,
        model: &AnalysisModel,
        mesh: &SolverMeshArtifact,
    ) -> Result<Self, ResourcePlanningError> {
        if constraints.topology.mesh_digest() != Some(projection.mesh_digest) {
            return Err(ResourcePlanningError::Invalid(
                "constraint plan and analysis projection name different topology".into(),
            ));
        }
        if matches!(
            constraints.topology,
            CanonicalStructuralTopologyIdentity::ConformalMixed { .. }
        ) {
            return Self::derive_mixed(projection, constraints, model, mesh);
        }
        let mesh_dofs = projection
            .nodes
            .len()
            .checked_mul(TRANSLATIONAL_DOF_COUNT)
            .ok_or(ResourcePlanningError::Overflow("structural mesh DOFs"))?;
        let reference_dofs = constraints
            .reference_points
            .len()
            .checked_mul(REFERENCE_POINT_DOF_COUNT)
            .ok_or(ResourcePlanningError::Overflow(
                "structural reference-point DOFs",
            ))?;
        let total_dofs = mesh_dofs
            .checked_add(reference_dofs)
            .ok_or(ResourcePlanningError::Overflow("structural DOFs"))?;
        if total_dofs == 0 {
            return Err(ResourcePlanningError::Invalid(
                "structural symbolic planning requires at least one DOF".into(),
            ));
        }

        let mut stiffness_rows = (0..total_dofs)
            .map(|_| BTreeSet::<usize>::new())
            .collect::<Vec<_>>();
        for element in &projection.elements {
            for &row_node in &element.node_indices {
                let row_offset = checked_dof_offset(row_node, projection.nodes.len())?;
                for &column_node in &element.node_indices {
                    let column_offset = checked_dof_offset(column_node, projection.nodes.len())?;
                    for row_axis in 0..TRANSLATIONAL_DOF_COUNT {
                        for column_axis in 0..TRANSLATIONAL_DOF_COUNT {
                            stiffness_rows[row_offset + row_axis]
                                .insert(column_offset + column_axis);
                        }
                    }
                }
            }
        }
        if !model.connectors.is_empty() {
            let topology = CanonicalStructuralNodeSpace::derive_solid(mesh)
                .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
            let layout = mixed_dof_layout(&topology, constraints.reference_points.len());
            if layout.total_dof_count() != total_dofs {
                return Err(ResourcePlanningError::Invalid(
                    "solid connector layout and symbolic DOF count differ".into(),
                ));
            }
            add_mesh_connector_pattern(
                &mut stiffness_rows,
                model,
                mesh,
                &StructuralModel::default(),
                &topology,
                &layout,
            )?;
        }
        let stiffness_upper_bound = pattern_from_rows(stiffness_rows, total_dofs)?;
        let diagonal = diagonal_pattern(total_dofs)?;
        let damping = if model.connectors.is_empty() {
            diagonal.clone()
        } else {
            stiffness_upper_bound.clone()
        };
        let pattern_identities = (
            stiffness_upper_bound
                .identity()
                .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?,
            diagonal
                .identity()
                .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?,
            damping
                .identity()
                .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?,
        );
        let encoded = encode_contract(
            SYMBOLIC_PREFIX,
            "structural-symbolic-operators",
            &pattern_identities,
            SYMBOLIC_LIMITS,
        )
        .map_err(ResourcePlanningError::from)?;
        Ok(Self {
            stiffness_upper_bound,
            mass: diagonal.clone(),
            damping,
            identity: Digest::sha256(encoded),
        })
    }

    pub fn derive_explicit(
        constraints: &CanonicalStructuralConstraintPlan,
        model: &AnalysisModel,
    ) -> Result<Self, ResourcePlanningError> {
        let structural = model.structural.as_ref().ok_or_else(|| {
            ResourcePlanningError::Invalid(
                "explicit symbolic planning requires structural topology".into(),
            )
        })?;
        let topology = CanonicalStructuralNodeSpace::derive_explicit(structural)
            .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
        if constraints.topology != topology.identity {
            return Err(ResourcePlanningError::Invalid(
                "constraint plan and explicit structural topology differ".into(),
            ));
        }
        let layout = mixed_dof_layout(&topology, constraints.reference_points.len());
        let total_dofs = layout.total_dof_count();
        let mut rows = (0..total_dofs)
            .map(|_| BTreeSet::<usize>::new())
            .collect::<Vec<_>>();
        for element in &structural.elements {
            let explicit_nodes = match &element.kind {
                StructuralElementKind::Beam(beam) => beam.node_ids.as_slice(),
                StructuralElementKind::Shell(shell) => shell.node_ids.as_slice(),
            };
            let mut element_dofs = Vec::new();
            for node_id in explicit_nodes {
                let explicit = structural
                    .nodes
                    .iter()
                    .position(|node| node.node_id == *node_id)
                    .ok_or_else(|| {
                        ResourcePlanningError::Invalid(format!(
                            "element `{}` references absent node {}",
                            element.element_id, node_id
                        ))
                    })?;
                let canonical = topology.explicit_node_indices[explicit];
                element_dofs.extend(
                    StructuralDofKind::ORDER
                        .into_iter()
                        .filter_map(|dof| layout.index(canonical, dof)),
                );
            }
            add_dense_element_pattern(&mut rows, &element_dofs);
        }
        for connector in &model.connectors {
            let mut connector_nodes = Vec::new();
            connector.kind.visit_targets(|target| {
                connector_nodes.extend(
                    crate::physics::structural::topology::explicit_target_node_indices(
                        structural,
                        &target.region_id,
                    ),
                );
            });
            connector_nodes.sort_unstable();
            connector_nodes.dedup();
            if connector_nodes.is_empty() {
                return Err(ResourcePlanningError::Invalid(format!(
                    "connector `{}` resolves to no explicit structural nodes",
                    connector.connector_id
                )));
            }
            let mut connector_dofs = Vec::new();
            for node in connector_nodes {
                connector_dofs.extend(
                    [
                        StructuralDofKind::Ux,
                        StructuralDofKind::Uy,
                        StructuralDofKind::Uz,
                    ]
                    .into_iter()
                    .filter_map(|dof| layout.index(node, dof)),
                );
            }
            add_dense_element_pattern(&mut rows, &connector_dofs);
        }
        for (row, columns) in rows.iter_mut().enumerate() {
            columns.insert(row);
        }
        let stiffness_upper_bound = pattern_from_rows(rows, total_dofs)?;
        Self::from_patterns(
            stiffness_upper_bound.clone(),
            stiffness_upper_bound.clone(),
            stiffness_upper_bound,
        )
    }

    fn derive_mixed(
        projection: &AnalysisProjection,
        constraints: &CanonicalStructuralConstraintPlan,
        model: &AnalysisModel,
        mesh: &SolverMeshArtifact,
    ) -> Result<Self, ResourcePlanningError> {
        let structural = model.structural.as_ref().ok_or_else(|| {
            ResourcePlanningError::Invalid(
                "mixed symbolic planning requires explicit structural topology".into(),
            )
        })?;
        let topology = CanonicalStructuralNodeSpace::derive(mesh, structural)
            .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
        if constraints.topology != topology.identity {
            return Err(ResourcePlanningError::Invalid(
                "constraint plan and mixed structural topology differ".into(),
            ));
        }
        let layout = mixed_dof_layout(&topology, constraints.reference_points.len());
        let total_dofs = layout.total_dof_count();
        let mut rows = (0..total_dofs)
            .map(|_| BTreeSet::<usize>::new())
            .collect::<Vec<_>>();
        for element in &projection.elements {
            let element_dofs = element
                .node_indices
                .iter()
                .flat_map(|node| {
                    [
                        StructuralDofKind::Ux,
                        StructuralDofKind::Uy,
                        StructuralDofKind::Uz,
                    ]
                    .into_iter()
                    .filter_map(|dof| layout.index(*node, dof))
                })
                .collect::<Vec<_>>();
            add_dense_element_pattern(&mut rows, &element_dofs);
        }
        for element in &structural.elements {
            let explicit_nodes = match &element.kind {
                StructuralElementKind::Beam(beam) => beam.node_ids.as_slice(),
                StructuralElementKind::Shell(shell) => shell.node_ids.as_slice(),
            };
            let mut element_dofs = Vec::new();
            for node_id in explicit_nodes {
                let explicit = structural
                    .nodes
                    .iter()
                    .position(|node| node.node_id == *node_id)
                    .ok_or_else(|| {
                        ResourcePlanningError::Invalid(format!(
                            "element `{}` references absent node {}",
                            element.element_id, node_id
                        ))
                    })?;
                let canonical = topology.explicit_node_indices[explicit];
                element_dofs.extend(
                    StructuralDofKind::ORDER
                        .into_iter()
                        .filter_map(|dof| layout.index(canonical, dof)),
                );
            }
            add_dense_element_pattern(&mut rows, &element_dofs);
        }
        add_mesh_connector_pattern(&mut rows, model, mesh, structural, &topology, &layout)?;
        for (row, columns) in rows.iter_mut().enumerate() {
            columns.insert(row);
        }
        let stiffness_upper_bound = pattern_from_rows(rows, total_dofs)?;
        let mass = stiffness_upper_bound.clone();
        let damping = stiffness_upper_bound.clone();
        Self::from_patterns(stiffness_upper_bound, mass, damping)
    }

    fn from_patterns(
        stiffness_upper_bound: SparsePattern,
        mass: SparsePattern,
        damping: SparsePattern,
    ) -> Result<Self, ResourcePlanningError> {
        let pattern_identities = (
            stiffness_upper_bound
                .identity()
                .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?,
            mass.identity()
                .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?,
            damping
                .identity()
                .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?,
        );
        let encoded = encode_contract(
            SYMBOLIC_PREFIX,
            "structural-symbolic-operators",
            &pattern_identities,
            SYMBOLIC_LIMITS,
        )
        .map_err(ResourcePlanningError::from)?;
        Ok(Self {
            stiffness_upper_bound,
            mass,
            damping,
            identity: Digest::sha256(encoded),
        })
    }

    pub const fn identity(&self) -> Digest {
        self.identity
    }

    pub fn stiffness_upper_bound(&self) -> &SparsePattern {
        &self.stiffness_upper_bound
    }

    pub fn mass_upper_bound(&self) -> &SparsePattern {
        &self.mass
    }

    pub fn damping_upper_bound(&self) -> &SparsePattern {
        &self.damping
    }

    pub fn patterns(&self) -> [&SparsePattern; 3] {
        [&self.stiffness_upper_bound, &self.mass, &self.damping]
    }
}

fn add_mesh_connector_pattern(
    rows: &mut [BTreeSet<usize>],
    model: &AnalysisModel,
    mesh: &SolverMeshArtifact,
    structural: &StructuralModel,
    topology: &CanonicalStructuralNodeSpace,
    layout: &StructuralDofLayout,
) -> Result<(), ResourcePlanningError> {
    for connector in &model.connectors {
        let mut connector_nodes = Vec::new();
        let mut resolution_error = None;
        connector.kind.visit_targets(|target| {
            match resolve_mixed_structural_target(target, mesh, structural, topology) {
                Ok(ownership) => connector_nodes.extend_from_slice(ownership.canonical_nodes()),
                Err(error) => resolution_error = Some(error),
            }
        });
        if let Some(error) = resolution_error {
            return Err(ResourcePlanningError::Invalid(error.to_string()));
        }
        connector_nodes.sort_unstable();
        connector_nodes.dedup();
        if connector_nodes.is_empty() {
            return Err(ResourcePlanningError::Invalid(format!(
                "connector `{}` resolves to no structural nodes",
                connector.connector_id
            )));
        }
        let mut connector_dofs = Vec::new();
        for node in connector_nodes {
            connector_dofs.extend(
                [
                    StructuralDofKind::Ux,
                    StructuralDofKind::Uy,
                    StructuralDofKind::Uz,
                ]
                .into_iter()
                .filter_map(|dof| layout.index(node, dof)),
            );
        }
        add_dense_element_pattern(rows, &connector_dofs);
    }
    Ok(())
}

fn add_dense_element_pattern(rows: &mut [BTreeSet<usize>], element_dofs: &[usize]) {
    for &row in element_dofs {
        rows[row].extend(element_dofs.iter().copied());
    }
}

fn checked_dof_offset(
    node_index: usize,
    node_count: usize,
) -> Result<usize, ResourcePlanningError> {
    if node_index >= node_count {
        return Err(ResourcePlanningError::Invalid(
            "projected element references an absent node".into(),
        ));
    }
    node_index
        .checked_mul(TRANSLATIONAL_DOF_COUNT)
        .ok_or(ResourcePlanningError::Overflow("structural DOF offset"))
}

fn diagonal_pattern(dofs: usize) -> Result<SparsePattern, ResourcePlanningError> {
    let rows = (0..dofs)
        .map(|row| BTreeSet::from([row]))
        .collect::<Vec<_>>();
    pattern_from_rows(rows, dofs)
}

fn pattern_from_rows(
    rows: Vec<BTreeSet<usize>>,
    columns: usize,
) -> Result<SparsePattern, ResourcePlanningError> {
    let mut offsets = Vec::with_capacity(rows.len() + 1);
    let mut column_indices = Vec::new();
    offsets.push(0_u64);
    for row in rows {
        for column in row {
            column_indices.push(
                u64::try_from(column)
                    .map_err(|_| ResourcePlanningError::Overflow("sparse column index"))?,
            );
        }
        offsets.push(
            u64::try_from(column_indices.len())
                .map_err(|_| ResourcePlanningError::Overflow("sparse row offset"))?,
        );
    }
    SparsePattern::try_new(
        u64::try_from(offsets.len() - 1)
            .map_err(|_| ResourcePlanningError::Overflow("sparse row count"))?,
        u64::try_from(columns)
            .map_err(|_| ResourcePlanningError::Overflow("sparse column count"))?,
        offsets,
        column_indices,
    )
    .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))
}
