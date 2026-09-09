use std::collections::{BTreeMap, BTreeSet, VecDeque};

use runmat_canonical_codec::{encode_contract, CanonicalLimits};
use runmat_execution::Digest;
use runmat_meshing_core::{SolverMeshArtifact, StableDigest};
use serde::Serialize;
use thiserror::Error;

use crate::{
    assembly::dofs::{StructuralDofKind, StructuralDofLayout},
    constraints::{CanonicalStructuralConstraintPlan, ResolvedConstraintMatrix},
    topology::AnalysisProjection,
};

use super::{spmv, MatrixDefiniteness, MatrixSymmetry, OperatorRole, RealSparseMatrix};

mod solid_components;

use solid_components::solid_face_connected_components;

const EVIDENCE_PREFIX: &[u8] = b"runmat-fea-nullspace-evidence/v1\0";
const EVIDENCE_DOMAIN: &str = "fea-nullspace-evidence";
const BINDING_DOMAIN: &str = "fea-prepared-nullspace-binding";
const CONSTRAINT_DOMAIN: &str = "fea-nullspace-constraint-state";
const RHS_DOMAIN: &str = "fea-nullspace-prepared-rhs";
const LIMITS: CanonicalLimits = CanonicalLimits::new(64 * 1024 * 1024, 1_000_000, 65_536, 64);
const RANK_TOLERANCE: f64 = 1.0e-10;
const NULL_MODE_RELATIVE_TOLERANCE: f64 = 1.0e-9;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum NullspaceKind {
    ScalarConstantPerComponent,
    SolidRigidBody3d,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct NullspaceComponentEvidence {
    component_index: usize,
    node_identities: Vec<StableDigest>,
    null_mode_count: usize,
    projected_constraint_rank: usize,
    anchoring: NullspaceAnchoring,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum NullspaceAnchoring {
    PrescribedDof {
        count: usize,
    },
    PositiveRobin {
        count: usize,
    },
    PrescribedAndPositiveRobin {
        prescribed_count: usize,
        robin_count: usize,
    },
    ConstraintRowSpace,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct NullspaceEliminationEvidence {
    schema_version: u16,
    kind: NullspaceKind,
    topology_identity: Digest,
    operator_pattern_identity: Digest,
    operator_values_identity: Digest,
    constraint_identity: Digest,
    constrained_dofs_identity: Digest,
    components: Vec<NullspaceComponentEvidence>,
    proof_identity: Digest,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct PreparedNullspaceEvidence {
    source: NullspaceEliminationEvidence,
    prepared_pattern_identity: Digest,
    prepared_values_identity: Digest,
    prepared_rhs_identity: Digest,
    binding_identity: Digest,
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum NullspaceEvidenceError {
    #[error("nullspace evidence dimensions do not match the physical topology")]
    DimensionMismatch,
    #[error("a connected component retains an uneliminated null mode")]
    UnderconstrainedComponent,
    #[error("a reference point cannot be assigned to one structural component")]
    AmbiguousReferencePoint,
    #[error(
        "solid elements touch only through a node or edge; an explicit contact or tie is required"
    )]
    AmbiguousStructuralConnectivity,
    #[error("nullspace evidence is malformed or has been altered")]
    InvalidEvidence,
    #[error("nullspace evidence does not match the operator or constraints")]
    BindingMismatch,
    #[error("nullspace evidence was requested for an incompatible operator")]
    IncompatibleOperator,
    #[error("the physical operator does not contain the declared null mode")]
    NullModeViolation,
    #[error("a declared positive boundary anchor is absent from the operator")]
    MissingPositiveAnchor,
    #[error("nullspace evidence identity encoding failed: {0}")]
    Encoding(String),
}

impl NullspaceEliminationEvidence {
    pub fn prove_thermal(
        projection: &AnalysisProjection,
        operator: &RealSparseMatrix,
        prescribed: &[bool],
        positive_robin_weight: &[f64],
    ) -> Result<Self, NullspaceEvidenceError> {
        let count = projection.nodes.len();
        if operator.rows() != count
            || operator.columns() != count
            || prescribed.len() != count
            || positive_robin_weight.len() != count
            || positive_robin_weight
                .iter()
                .any(|weight| !weight.is_finite() || *weight < 0.0)
        {
            return Err(NullspaceEvidenceError::DimensionMismatch);
        }
        validate_operator(
            operator,
            OperatorRole::Conductivity,
            MatrixDefiniteness::PositiveSemidefinite,
        )?;
        let component_nodes = connected_components(&projection.node_neighbors)?;
        let mut components = Vec::with_capacity(component_nodes.len());
        for (component_index, nodes) in component_nodes.into_iter().enumerate() {
            let prescribed_count = nodes.iter().filter(|node| prescribed[**node]).count();
            let robin_count = nodes
                .iter()
                .filter(|node| positive_robin_weight[**node] > 0.0)
                .count();
            let anchoring = match (prescribed_count, robin_count) {
                (0, 0) => return Err(NullspaceEvidenceError::UnderconstrainedComponent),
                (count, 0) => NullspaceAnchoring::PrescribedDof { count },
                (0, count) => NullspaceAnchoring::PositiveRobin { count },
                (prescribed_count, robin_count) => NullspaceAnchoring::PrescribedAndPositiveRobin {
                    prescribed_count,
                    robin_count,
                },
            };
            let mut constant_mode = vec![0.0; count];
            for &node in &nodes {
                constant_mode[node] = 1.0;
            }
            let applied = spmv(operator, &constant_mode)
                .map_err(|_| NullspaceEvidenceError::DimensionMismatch)?;
            if robin_count == 0 {
                verify_null_mode(operator, &constant_mode, &applied)?;
            } else {
                let energy = constant_mode
                    .iter()
                    .zip(&applied)
                    .map(|(left, right)| left * right)
                    .sum::<f64>();
                let diagonal_anchor = nodes
                    .iter()
                    .map(|node| positive_robin_weight[*node])
                    .sum::<f64>();
                let tolerance = operator_scale(operator).max(1.0) * f64::EPSILON * 64.0;
                if !energy.is_finite()
                    || energy <= tolerance
                    || energy + tolerance < diagonal_anchor
                {
                    return Err(NullspaceEvidenceError::MissingPositiveAnchor);
                }
            }
            let mut node_identities = nodes
                .iter()
                .map(|node| projection.nodes[*node].stable_identity)
                .collect::<Vec<_>>();
            node_identities.sort_unstable();
            components.push(NullspaceComponentEvidence {
                component_index,
                node_identities,
                null_mode_count: 1,
                projected_constraint_rank: 1,
                anchoring,
            });
        }
        let constraint_identity = digest(CONSTRAINT_DOMAIN, &(prescribed, positive_robin_weight))?;
        Self::new(
            NullspaceKind::ScalarConstantPerComponent,
            Digest::from_bytes(*projection.mesh_digest.bytes()),
            operator,
            constraint_identity,
            prescribed,
            components,
        )
    }

    pub fn prove_solid(
        mesh: &SolverMeshArtifact,
        layout: &StructuralDofLayout,
        plan: &CanonicalStructuralConstraintPlan,
        constraints: &ResolvedConstraintMatrix,
        operator: &RealSparseMatrix,
        constrained_dofs: &[bool],
    ) -> Result<Self, NullspaceEvidenceError> {
        constraints
            .validate()
            .map_err(|_| NullspaceEvidenceError::InvalidEvidence)?;
        let expected_constraints =
            crate::assembly::resolve_solid_constraint_matrix(plan, mesh, layout)
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
            .map(|nodes| component_frame(mesh, nodes))
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
                spmv(operator, mode).map_err(|_| NullspaceEvidenceError::DimensionMismatch)?;
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
        let rank = numerical_rank(projected);
        if rank != mode_count {
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
                    projected_constraint_rank: 6,
                    anchoring: NullspaceAnchoring::ConstraintRowSpace,
                }
            })
            .collect();
        let plan_identity = plan
            .content_digest()
            .map_err(|error| NullspaceEvidenceError::Encoding(error.to_string()))?;
        let constraint_identity = digest(CONSTRAINT_DOMAIN, &(plan_identity, constraints))?;
        Self::new(
            NullspaceKind::SolidRigidBody3d,
            Digest::from_bytes(*mesh.canonical_digest.bytes()),
            operator,
            constraint_identity,
            constrained_dofs,
            components,
        )
    }

    fn new(
        kind: NullspaceKind,
        topology_identity: Digest,
        operator: &RealSparseMatrix,
        constraint_identity: Digest,
        constrained_dofs: &[bool],
        components: Vec<NullspaceComponentEvidence>,
    ) -> Result<Self, NullspaceEvidenceError> {
        let mut evidence = Self {
            schema_version: 1,
            kind,
            topology_identity,
            operator_pattern_identity: operator
                .pattern_identity()
                .map_err(|error| NullspaceEvidenceError::Encoding(error.to_string()))?,
            operator_values_identity: operator
                .values_identity()
                .map_err(|error| NullspaceEvidenceError::Encoding(error.to_string()))?,
            constraint_identity,
            constrained_dofs_identity: digest(CONSTRAINT_DOMAIN, &constrained_dofs)?,
            components,
            proof_identity: Digest::sha256([]),
        };
        evidence.proof_identity = evidence.expected_identity()?;
        evidence.validate()?;
        Ok(evidence)
    }

    pub(super) fn validate_for_system(
        &self,
        operator: &RealSparseMatrix,
        constrained_dofs: &[bool],
    ) -> Result<(), NullspaceEvidenceError> {
        self.validate()?;
        let pattern = operator
            .pattern_identity()
            .map_err(|error| NullspaceEvidenceError::Encoding(error.to_string()))?;
        let values = operator
            .values_identity()
            .map_err(|error| NullspaceEvidenceError::Encoding(error.to_string()))?;
        if self.operator_pattern_identity != pattern
            || self.operator_values_identity != values
            || self.constrained_dofs_identity != digest(CONSTRAINT_DOMAIN, &constrained_dofs)?
        {
            return Err(NullspaceEvidenceError::BindingMismatch);
        }
        Ok(())
    }

    pub(super) fn bind_prepared(
        &self,
        matrix: &RealSparseMatrix,
        rhs: &[f64],
    ) -> Result<PreparedNullspaceEvidence, NullspaceEvidenceError> {
        self.validate()?;
        let mut binding = PreparedNullspaceEvidence {
            source: self.clone(),
            prepared_pattern_identity: matrix
                .pattern_identity()
                .map_err(|error| NullspaceEvidenceError::Encoding(error.to_string()))?,
            prepared_values_identity: matrix
                .values_identity()
                .map_err(|error| NullspaceEvidenceError::Encoding(error.to_string()))?,
            prepared_rhs_identity: digest(RHS_DOMAIN, &rhs)?,
            binding_identity: Digest::sha256([]),
        };
        binding.binding_identity = binding.expected_identity()?;
        binding.validate()?;
        Ok(binding)
    }

    fn validate(&self) -> Result<(), NullspaceEvidenceError> {
        if self.schema_version != 1
            || self.components.is_empty()
            || self
                .components
                .iter()
                .enumerate()
                .any(|(index, component)| {
                    component.component_index != index
                        || component.node_identities.is_empty()
                        || !component
                            .node_identities
                            .windows(2)
                            .all(|pair| pair[0] < pair[1])
                        || component.null_mode_count == 0
                        || component.projected_constraint_rank != component.null_mode_count
                        || !anchoring_matches(component, self.kind)
                })
            || self.expected_identity()? != self.proof_identity
        {
            return Err(NullspaceEvidenceError::InvalidEvidence);
        }
        let unique_nodes = self
            .components
            .iter()
            .flat_map(|component| component.node_identities.iter().copied())
            .collect::<BTreeSet<_>>();
        if unique_nodes.len()
            != self
                .components
                .iter()
                .map(|component| component.node_identities.len())
                .sum::<usize>()
        {
            return Err(NullspaceEvidenceError::InvalidEvidence);
        }
        Ok(())
    }

    fn expected_identity(&self) -> Result<Digest, NullspaceEvidenceError> {
        digest(
            EVIDENCE_DOMAIN,
            &(
                self.schema_version,
                self.kind,
                self.topology_identity,
                self.operator_pattern_identity,
                self.operator_values_identity,
                self.constraint_identity,
                self.constrained_dofs_identity,
                &self.components,
            ),
        )
    }
}

impl PreparedNullspaceEvidence {
    pub(super) const fn identity(&self) -> Digest {
        self.binding_identity
    }

    fn validate(&self) -> Result<(), NullspaceEvidenceError> {
        self.source.validate()?;
        if self.binding_identity != self.expected_identity()? {
            return Err(NullspaceEvidenceError::InvalidEvidence);
        }
        Ok(())
    }

    fn expected_identity(&self) -> Result<Digest, NullspaceEvidenceError> {
        digest(
            BINDING_DOMAIN,
            &(
                self.source.proof_identity,
                self.prepared_pattern_identity,
                self.prepared_values_identity,
                self.prepared_rhs_identity,
            ),
        )
    }
}

fn anchoring_matches(component: &NullspaceComponentEvidence, kind: NullspaceKind) -> bool {
    match (&component.anchoring, kind) {
        (
            NullspaceAnchoring::PrescribedDof { count },
            NullspaceKind::ScalarConstantPerComponent,
        )
        | (
            NullspaceAnchoring::PositiveRobin { count },
            NullspaceKind::ScalarConstantPerComponent,
        ) => *count > 0,
        (
            NullspaceAnchoring::PrescribedAndPositiveRobin {
                prescribed_count,
                robin_count,
            },
            NullspaceKind::ScalarConstantPerComponent,
        ) => *prescribed_count > 0 && *robin_count > 0,
        (NullspaceAnchoring::ConstraintRowSpace, NullspaceKind::SolidRigidBody3d) => true,
        _ => false,
    }
}

fn connected_components(
    neighbors: &[Vec<usize>],
) -> Result<Vec<Vec<usize>>, NullspaceEvidenceError> {
    if neighbors
        .iter()
        .any(|adjacent| adjacent.iter().any(|node| *node >= neighbors.len()))
    {
        return Err(NullspaceEvidenceError::DimensionMismatch);
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
            for &neighbor in &neighbors[node] {
                if !seen[neighbor] {
                    seen[neighbor] = true;
                    queue.push_back(neighbor);
                }
            }
        }
        component.sort_unstable();
        components.push(component);
    }
    Ok(components)
}

struct RigidModeContext<'a> {
    layout: &'a StructuralDofLayout,
    mesh: &'a SolverMeshArtifact,
    plan: &'a CanonicalStructuralConstraintPlan,
    component_by_node: &'a BTreeMap<usize, usize>,
    reference_component: &'a BTreeMap<Digest, usize>,
    component_frames: &'a [([f64; 3], f64)],
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
            node.coordinates_m,
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
        let kind = StructuralDofKind::ORDER[component_index];
        (
            *context
                .reference_component
                .get(&point.identity)
                .ok_or(NullspaceEvidenceError::AmbiguousReferencePoint)?,
            point.position_meters,
            kind,
        )
    };
    let offset = component * 6;
    let (centroid, characteristic_length) = context
        .component_frames
        .get(component)
        .ok_or(NullspaceEvidenceError::DimensionMismatch)?;
    let [x, y, z] = [
        (position[0] - centroid[0]) / characteristic_length,
        (position[1] - centroid[1]) / characteristic_length,
        (position[2] - centroid[2]) / characteristic_length,
    ];
    let inverse_length = characteristic_length.recip();
    let values = match kind {
        StructuralDofKind::Ux => [1.0, 0.0, 0.0, 0.0, z, -y],
        StructuralDofKind::Uy => [0.0, 1.0, 0.0, -z, 0.0, x],
        StructuralDofKind::Uz => [0.0, 0.0, 1.0, y, -x, 0.0],
        StructuralDofKind::Rx => [0.0, 0.0, 0.0, inverse_length, 0.0, 0.0],
        StructuralDofKind::Ry => [0.0, 0.0, 0.0, 0.0, inverse_length, 0.0],
        StructuralDofKind::Rz => [0.0, 0.0, 0.0, 0.0, 0.0, inverse_length],
    };
    response[offset..offset + 6].copy_from_slice(&values);
    Ok(response)
}

fn component_frame(
    mesh: &SolverMeshArtifact,
    nodes: &[usize],
) -> Result<([f64; 3], f64), NullspaceEvidenceError> {
    if nodes.is_empty() {
        return Err(NullspaceEvidenceError::DimensionMismatch);
    }
    let mut centroid = [0.0; 3];
    for &node in nodes {
        let position = mesh
            .topology
            .nodes
            .get(node)
            .ok_or(NullspaceEvidenceError::DimensionMismatch)?
            .coordinates_m;
        for axis in 0..3 {
            centroid[axis] += position[axis];
        }
    }
    for coordinate in &mut centroid {
        *coordinate /= nodes.len() as f64;
    }
    let characteristic_length = nodes
        .iter()
        .map(|node| {
            let position = mesh.topology.nodes[*node].coordinates_m;
            position
                .iter()
                .zip(centroid)
                .map(|(coordinate, center)| (coordinate - center).powi(2))
                .sum::<f64>()
                .sqrt()
        })
        .fold(0.0_f64, f64::max);
    if !characteristic_length.is_finite() || characteristic_length <= f64::MIN_POSITIVE {
        return Err(NullspaceEvidenceError::DimensionMismatch);
    }
    Ok((centroid, characteristic_length))
}

fn numerical_rank(mut rows: Vec<Vec<f64>>) -> usize {
    for row in &mut rows {
        let scale = row.iter().map(|value| value.abs()).fold(0.0_f64, f64::max);
        if scale > 0.0 && scale.is_finite() {
            for value in row {
                *value /= scale;
            }
        }
    }
    let columns = rows.first().map(Vec::len).unwrap_or(0);
    let scale = rows
        .iter()
        .flatten()
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max)
        .max(1.0);
    let tolerance = scale * RANK_TOLERANCE;
    let mut rank = 0;
    for column in 0..columns {
        let pivot = (rank..rows.len()).max_by(|left, right| {
            rows[*left][column]
                .abs()
                .total_cmp(&rows[*right][column].abs())
        });
        let Some(pivot) = pivot.filter(|pivot| rows[*pivot][column].abs() > tolerance) else {
            continue;
        };
        rows.swap(rank, pivot);
        let divisor = rows[rank][column];
        for value in &mut rows[rank][column..] {
            *value /= divisor;
        }
        for row in 0..rows.len() {
            if row == rank {
                continue;
            }
            let factor = rows[row][column];
            if factor.abs() <= tolerance {
                continue;
            }
            for entry in column..columns {
                rows[row][entry] -= factor * rows[rank][entry];
            }
        }
        rank += 1;
        if rank == rows.len() {
            break;
        }
    }
    rank
}

fn validate_operator(
    operator: &RealSparseMatrix,
    role: OperatorRole,
    definiteness: MatrixDefiniteness,
) -> Result<(), NullspaceEvidenceError> {
    if operator.role() != role
        || operator.properties().symmetry != MatrixSymmetry::Symmetric
        || operator.properties().definiteness != definiteness
    {
        return Err(NullspaceEvidenceError::IncompatibleOperator);
    }
    Ok(())
}

fn verify_null_mode(
    operator: &RealSparseMatrix,
    mode: &[f64],
    applied: &[f64],
) -> Result<(), NullspaceEvidenceError> {
    let mode_scale = mode.iter().map(|value| value.abs()).fold(0.0_f64, f64::max);
    let residual = applied
        .iter()
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max);
    let reference = (operator_scale(operator) * mode_scale).max(1.0);
    if mode_scale == 0.0
        || !residual.is_finite()
        || residual > reference * NULL_MODE_RELATIVE_TOLERANCE
    {
        return Err(NullspaceEvidenceError::NullModeViolation);
    }
    Ok(())
}

fn operator_scale(operator: &RealSparseMatrix) -> f64 {
    operator
        .pattern()
        .row_offsets()
        .windows(2)
        .map(|row| {
            operator.values()[row[0] as usize..row[1] as usize]
                .iter()
                .map(|value| value.abs())
                .sum::<f64>()
        })
        .fold(0.0_f64, f64::max)
}

fn digest<T: Serialize>(domain: &'static str, value: &T) -> Result<Digest, NullspaceEvidenceError> {
    encode_contract(EVIDENCE_PREFIX, domain, value, LIMITS)
        .map(Digest::sha256)
        .map_err(|error| NullspaceEvidenceError::Encoding(error.to_string()))
}
