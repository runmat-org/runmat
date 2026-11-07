#[cfg(test)]
use std::collections::BTreeSet;

use runmat_analysis_core::{AnalysisModel, StructuralElementKind};
use runmat_canonical_codec::{encode_contract, CanonicalLimits};
use runmat_execution::Digest;
use runmat_meshing_core::SolverMeshArtifact;

use crate::{
    algebra::{CanonicalRealOperators, SparsePattern},
    assembly::{
        dofs::StructuralDofKind, dofs::StructuralDofLayout,
        explicit_connectors::compile_model_connector_plan, mixed_dof_layout,
    },
    connectors::{CanonicalConnectorEntry, CanonicalStructuralConnectorPlan},
    constraints::{CanonicalStructuralConstraintPlan, CanonicalStructuralTopologyIdentity},
    physics::structural::topology::node_space::CanonicalStructuralNodeSpace,
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
/// mass follows the same-axis consistent-mass element graph. Solid damping reserves the stiffness
/// graph for Rayleigh mass/stiffness terms and axial dashpots. Mixed systems conservatively
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

        // Node coupling of the solid elements; every node carries the same translational block.
        let nodes = CliqueAdjacency::derive(
            projection.nodes.len(),
            projection
                .elements
                .iter()
                .map(|element| element.node_indices.as_slice()),
            "projected element references an absent node",
        )?;
        let mass = solid_mass_pattern(&nodes, mesh_dofs, total_dofs)?;
        let connectors = if model.connectors.is_empty() {
            None
        } else {
            let topology = CanonicalStructuralNodeSpace::derive_solid(mesh)
                .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
            let layout = mixed_dof_layout(&topology, constraints.reference_points.len());
            if layout.total_dof_count() != total_dofs {
                return Err(ResourcePlanningError::Invalid(
                    "solid connector layout and symbolic DOF count differ".into(),
                ));
            }
            let plan =
                compile_model_connector_plan(model, Some(mesh), constraints.reference_points.len())
                    .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
            let cliques = connector_dof_cliques(&plan, constraints.topology, &layout)?;
            Some(CliqueAdjacency::derive(
                total_dofs,
                cliques.iter().map(Vec::as_slice),
                "connector DOF lies outside the structural layout",
            )?)
        };
        let stiffness_upper_bound =
            solid_stiffness_pattern(&nodes, mesh_dofs, total_dofs, connectors.as_ref())?;
        Self::from_patterns(stiffness_upper_bound.clone(), mass, stiffness_upper_bound)
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
        let mut cliques = Vec::new();
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
            cliques.push(element_dofs);
        }
        if !model.connectors.is_empty() {
            let plan =
                compile_model_connector_plan(model, None, constraints.reference_points.len())
                    .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
            cliques.extend(connector_dof_cliques(&plan, constraints.topology, &layout)?);
        }
        let stiffness_upper_bound = dof_clique_pattern(total_dofs, &cliques)?;
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
        let mut cliques = Vec::new();
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
            cliques.push(element_dofs);
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
            cliques.push(element_dofs);
        }
        if !model.connectors.is_empty() {
            let plan =
                compile_model_connector_plan(model, Some(mesh), constraints.reference_points.len())
                    .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
            cliques.extend(connector_dof_cliques(&plan, constraints.topology, &layout)?);
        }
        let stiffness_upper_bound = dof_clique_pattern(total_dofs, &cliques)?;
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

    /// Reconcile the assembled graphs against the assembly-free storage claim before solve.
    /// This checks exact K/M/C entries, not only total nonzero counts.
    pub(crate) fn verify_assembled(
        &self,
        operators: &CanonicalRealOperators,
    ) -> Result<(), ResourcePlanningError> {
        let actual = [
            operators.stiffness().pattern(),
            operators.mass().pattern(),
            operators.damping().pattern(),
        ];
        for ((role, planned), assembled) in ["stiffness", "mass", "damping"]
            .into_iter()
            .zip(self.patterns())
            .zip(actual)
        {
            match planned.first_uncovered_entry(assembled) {
                Ok(None) => {}
                Ok(Some((row, column))) => {
                    return Err(ResourcePlanningError::Invalid(format!(
                        "assembled {role} entry ({row}, {column}) exceeds symbolic storage bound"
                    )));
                }
                Err(error) => {
                    return Err(ResourcePlanningError::Invalid(format!(
                        "assembled {role} shape differs from symbolic storage bound: {error}"
                    )));
                }
            }
        }
        Ok(())
    }
}

/// The DOFs each connector couples densely.
fn connector_dof_cliques(
    plan: &CanonicalStructuralConnectorPlan,
    topology: CanonicalStructuralTopologyIdentity,
    layout: &StructuralDofLayout,
) -> Result<Vec<Vec<usize>>, ResourcePlanningError> {
    if plan.topology != topology || plan.node_count != layout.node_count() {
        return Err(ResourcePlanningError::Invalid(
            "connector plan and symbolic topology differ".into(),
        ));
    }
    let mut cliques = Vec::with_capacity(plan.entries.len());
    for connector in &plan.entries {
        let mut connector_nodes = Vec::new();
        match connector {
            CanonicalConnectorEntry::AxialSpring {
                primary, secondary, ..
            }
            | CanonicalConnectorEntry::AxialDashpot {
                primary, secondary, ..
            } => connector_nodes.extend([primary.node_index, secondary.node_index]),
            CanonicalConnectorEntry::IsotropicGroundSpring { target, .. } => {
                connector_nodes.push(target.node_index);
            }
            CanonicalConnectorEntry::SurfaceNormalFoundation { quadrature, .. } => {
                connector_nodes.extend(
                    quadrature
                        .iter()
                        .flat_map(|point| point.nodes.iter().map(|node| node.node_index)),
                );
            }
        }
        connector_nodes.sort_unstable();
        connector_nodes.dedup();
        if connector_nodes.is_empty() {
            return Err(ResourcePlanningError::Invalid(format!(
                "connector `{}` resolves to no structural nodes",
                connector.connector_id()
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
                .map(|dof| {
                    layout.index(node, dof).ok_or_else(|| {
                        ResourcePlanningError::Invalid(format!(
                            "connector `{}` lacks a translational DOF",
                            connector.connector_id()
                        ))
                    })
                })
                .collect::<Result<Vec<_>, _>>()?,
            );
        }
        cliques.push(connector_dofs);
    }
    Ok(cliques)
}

/// Consistent-mass pattern of solid elements: each translational DOF couples the same axis of
/// every node it shares an element with; every DOF keeps its diagonal.
fn solid_mass_pattern(
    nodes: &CliqueAdjacency,
    mesh_dofs: usize,
    total_dofs: usize,
) -> Result<SparsePattern, ResourcePlanningError> {
    pattern_from_row_builder(total_dofs, |row, columns| {
        columns.push(row);
        if row < mesh_dofs {
            let axis = row % TRANSLATIONAL_DOF_COUNT;
            columns.extend(
                nodes
                    .row(row / TRANSLATIONAL_DOF_COUNT)
                    .iter()
                    .map(|node| node * TRANSLATIONAL_DOF_COUNT + axis),
            );
        }
    })
}

/// Stiffness pattern of solid elements (full translational blocks between nodes that share an
/// element) plus the DOF cliques of connectors.
fn solid_stiffness_pattern(
    nodes: &CliqueAdjacency,
    mesh_dofs: usize,
    total_dofs: usize,
    connectors: Option<&CliqueAdjacency>,
) -> Result<SparsePattern, ResourcePlanningError> {
    pattern_from_row_builder(total_dofs, |row, columns| {
        if row < mesh_dofs {
            for node in nodes.row(row / TRANSLATIONAL_DOF_COUNT) {
                let offset = node * TRANSLATIONAL_DOF_COUNT;
                columns.extend(offset..offset + TRANSLATIONAL_DOF_COUNT);
            }
        }
        if let Some(connectors) = connectors {
            columns.extend_from_slice(connectors.row(row));
        }
    })
}

/// Pattern of densely coupled DOF cliques with a full diagonal.
fn dof_clique_pattern(
    dimension: usize,
    cliques: &[Vec<usize>],
) -> Result<SparsePattern, ResourcePlanningError> {
    let adjacency = CliqueAdjacency::derive(
        dimension,
        cliques.iter().map(Vec::as_slice),
        "element DOF lies outside the structural layout",
    )?;
    pattern_from_row_builder(dimension, |row, columns| {
        columns.push(row);
        columns.extend_from_slice(adjacency.row(row));
    })
}

/// Sorted, duplicate-free neighbor lists of indices coupled by cliques: an index neighbors
/// every index it shares a clique with, itself included; an index in no clique has none.
///
/// Flat storage: an ordered set per row costs one small allocation per few entries, which on
/// a large mesh churns gigabytes through the allocator during planning and leaves the
/// planning process holding that freed memory while its solve worker runs.
pub(super) struct CliqueAdjacency {
    offsets: Vec<usize>,
    neighbors: Vec<usize>,
}

impl CliqueAdjacency {
    pub(super) fn derive<'a>(
        dimension: usize,
        cliques: impl Iterator<Item = &'a [usize]>,
        out_of_range: &'static str,
    ) -> Result<Self, ResourcePlanningError> {
        let mut clique_offsets = vec![0_usize];
        let mut members = Vec::new();
        let mut incidence_offsets = vec![0_usize; dimension + 1];
        for clique in cliques {
            for &index in clique {
                if index >= dimension {
                    return Err(ResourcePlanningError::Invalid(out_of_range.into()));
                }
                incidence_offsets[index + 1] += 1;
                members.push(index);
            }
            clique_offsets.push(members.len());
        }
        for index in 0..dimension {
            incidence_offsets[index + 1] += incidence_offsets[index];
        }
        let mut cursor = incidence_offsets[..dimension].to_vec();
        let mut incidence = vec![0_usize; members.len()];
        for clique in 0..clique_offsets.len() - 1 {
            for &index in &members[clique_offsets[clique]..clique_offsets[clique + 1]] {
                incidence[cursor[index]] = clique;
                cursor[index] += 1;
            }
        }
        drop(cursor);
        let gather = |index: usize, scratch: &mut Vec<usize>| {
            scratch.clear();
            for &clique in &incidence[incidence_offsets[index]..incidence_offsets[index + 1]] {
                scratch.extend_from_slice(
                    &members[clique_offsets[clique]..clique_offsets[clique + 1]],
                );
            }
            scratch.sort_unstable();
            scratch.dedup();
        };
        let mut scratch = Vec::new();
        let mut total = 0_usize;
        for index in 0..dimension {
            gather(index, &mut scratch);
            total += scratch.len();
        }
        let mut offsets = Vec::with_capacity(dimension + 1);
        let mut neighbors = Vec::with_capacity(total);
        offsets.push(0);
        for index in 0..dimension {
            gather(index, &mut scratch);
            neighbors.extend_from_slice(&scratch);
            offsets.push(neighbors.len());
        }
        Ok(Self { offsets, neighbors })
    }

    pub(super) fn row(&self, index: usize) -> &[usize] {
        &self.neighbors[self.offsets[index]..self.offsets[index + 1]]
    }
}

/// Square pattern whose row `r` holds the columns `row(r, columns)` writes (any order,
/// duplicates allowed). Rows are produced twice so the column storage is allocated once at its
/// exact size.
fn pattern_from_row_builder(
    dimension: usize,
    row: impl Fn(usize, &mut Vec<usize>),
) -> Result<SparsePattern, ResourcePlanningError> {
    let build = |index: usize, scratch: &mut Vec<usize>| {
        scratch.clear();
        row(index, scratch);
        scratch.sort_unstable();
        scratch.dedup();
    };
    let mut scratch = Vec::new();
    let mut total = 0_usize;
    for index in 0..dimension {
        build(index, &mut scratch);
        total += scratch.len();
    }
    let mut offsets = Vec::with_capacity(dimension + 1);
    let mut column_indices = Vec::with_capacity(total);
    offsets.push(0_u64);
    for index in 0..dimension {
        build(index, &mut scratch);
        for &column in &scratch {
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
    let size = u64::try_from(dimension)
        .map_err(|_| ResourcePlanningError::Overflow("sparse row count"))?;
    SparsePattern::try_new(size, size, offsets, column_indices)
        .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))
}

#[cfg(test)]
pub(super) fn pattern_from_rows(
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::{MatrixProperties, OperatorRole, RealSparseMatrix};

    /// Deterministic pseudo-random cliques over `dimension` indices.
    fn cliques(dimension: usize, count: usize, size: usize, seed: u64) -> Vec<Vec<usize>> {
        let mut state = seed;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as usize % dimension
        };
        (0..count)
            .map(|_| (0..size).map(|_| next()).collect())
            .collect()
    }

    fn set_rows_pattern(rows: Vec<BTreeSet<usize>>, dimension: usize) -> SparsePattern {
        pattern_from_rows(rows, dimension).unwrap()
    }

    #[test]
    fn flat_solid_patterns_equal_the_per_row_set_construction() {
        let node_count = 40;
        let reference_dofs = 6;
        let mesh_dofs = node_count * TRANSLATIONAL_DOF_COUNT;
        let total_dofs = mesh_dofs + reference_dofs;
        let elements = cliques(node_count, 25, 10, 7);
        let connector_cliques = cliques(total_dofs, 3, 9, 11);
        let nodes = CliqueAdjacency::derive(
            node_count,
            elements.iter().map(Vec::as_slice),
            "absent node",
        )
        .unwrap();
        let connectors = CliqueAdjacency::derive(
            total_dofs,
            connector_cliques.iter().map(Vec::as_slice),
            "absent dof",
        )
        .unwrap();

        let mut stiffness_rows = vec![BTreeSet::new(); total_dofs];
        let mut mass_rows = (0..total_dofs)
            .map(|row| BTreeSet::from([row]))
            .collect::<Vec<_>>();
        for element in &elements {
            for &row_node in element {
                for &column_node in element {
                    for row_axis in 0..TRANSLATIONAL_DOF_COUNT {
                        for column_axis in 0..TRANSLATIONAL_DOF_COUNT {
                            stiffness_rows[row_node * 3 + row_axis]
                                .insert(column_node * 3 + column_axis);
                        }
                        mass_rows[row_node * 3 + row_axis].insert(column_node * 3 + row_axis);
                    }
                }
            }
        }
        for clique in &connector_cliques {
            for &row in clique {
                stiffness_rows[row].extend(clique.iter().copied());
            }
        }
        assert_eq!(
            solid_mass_pattern(&nodes, mesh_dofs, total_dofs).unwrap(),
            set_rows_pattern(mass_rows, total_dofs)
        );
        assert_eq!(
            solid_stiffness_pattern(&nodes, mesh_dofs, total_dofs, Some(&connectors)).unwrap(),
            set_rows_pattern(stiffness_rows, total_dofs)
        );
    }

    #[test]
    fn flat_dof_clique_pattern_equals_the_per_row_set_construction() {
        let dimension = 90;
        let element_cliques = cliques(dimension, 30, 12, 3);
        let mut rows = vec![BTreeSet::new(); dimension];
        for clique in &element_cliques {
            for &row in clique {
                rows[row].extend(clique.iter().copied());
            }
        }
        for (row, columns) in rows.iter_mut().enumerate() {
            columns.insert(row);
        }
        assert_eq!(
            dof_clique_pattern(dimension, &element_cliques).unwrap(),
            set_rows_pattern(rows, dimension)
        );
        assert!(matches!(
            dof_clique_pattern(4, &[vec![1, 4]]),
            Err(ResourcePlanningError::Invalid(_))
        ));
    }

    #[test]
    fn assembled_operator_reconciliation_rejects_each_unplanned_role() {
        let diagonal = SparsePattern::try_new(2, 2, vec![0, 1, 2], vec![0, 1]).unwrap();
        let planned = StructuralSymbolicOperators::from_patterns(
            diagonal.clone(),
            diagonal.clone(),
            diagonal,
        )
        .unwrap();
        let matrix = |role, coupled| {
            RealSparseMatrix::from_dense(
                &[2.0, coupled, coupled, 2.0],
                2,
                role,
                MatrixProperties::SYMMETRIC_POSITIVE_DEFINITE,
            )
            .unwrap()
        };
        for unplanned in 0..3 {
            let operators = CanonicalRealOperators::try_new(
                matrix(
                    OperatorRole::Stiffness,
                    if unplanned == 0 { 1.0 } else { 0.0 },
                ),
                matrix(OperatorRole::Mass, if unplanned == 1 { 1.0 } else { 0.0 }),
                matrix(
                    OperatorRole::Damping,
                    if unplanned == 2 { 1.0 } else { 0.0 },
                ),
            )
            .unwrap();
            let role = ["stiffness", "mass", "damping"][unplanned];
            assert!(matches!(
                planned.verify_assembled(&operators),
                Err(ResourcePlanningError::Invalid(message))
                    if message.contains(role) && message.contains("(0, 1)")
            ));
        }
        let covered = CanonicalRealOperators::try_new(
            matrix(OperatorRole::Stiffness, 0.0),
            matrix(OperatorRole::Mass, 0.0),
            matrix(OperatorRole::Damping, 0.0),
        )
        .unwrap();
        planned.verify_assembled(&covered).unwrap();
    }
}
