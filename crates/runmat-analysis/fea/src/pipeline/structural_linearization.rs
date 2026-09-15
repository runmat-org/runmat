use runmat_canonical_codec::{encode_contract, CanonicalLimits};
use runmat_execution::Digest;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use thiserror::Error;

use crate::{
    algebra::{
        ContributionDomain, ContributionId, ContributionProvenance, MatrixProperties, OperatorRole,
        RealSparseAssembler, RealSparseMatrix,
    },
    assembly::{
        dofs::StructuralDofKind,
        elements::solid::{
            elasticity_matrix, strain_displacement_matrix,
            tetrahedron10_geometric_stiffness_matrix, tetrahedron10_quadrature_stress,
            tetrahedron4_geometric_stiffness_matrix, Tetrahedron10ElementGeometry,
            Tetrahedron4ElementGeometry,
        },
        AssemblySummary, SolidRecoveryGeometry,
    },
    AnalysisRunIdentity,
};

const LINEARIZATION_PREFIX: &[u8] = b"runmat-fea-structural-linearization/v1\0";
const LINEARIZATION_LIMITS: CanonicalLimits =
    CanonicalLimits::new(1024 * 1024 * 1024, 64_000_000, 65_536, 32);

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct StructuralLinearizationIdentity(Digest);

impl StructuralLinearizationIdentity {
    pub const fn digest(self) -> Digest {
        self.0
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuralInitialStressState {
    pub element_id: String,
    pub region_id: String,
    /// Full signed, tension-positive stress tensors at the geometric-stiffness quadrature points.
    pub quadrature_stress_pa: Vec<[[f64; 3]; 3]>,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuralLinearizationEvidence {
    pub reference_load_factor: f64,
    pub solid_element_count: usize,
    pub stress_quadrature_point_count: usize,
    pub geometric_stiffness_nonzero_count: usize,
    pub maximum_symmetry_residual: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "StructuralLinearizationArtifactData")]
pub struct StructuralLinearizationArtifact {
    schema_version: u16,
    identity: StructuralLinearizationIdentity,
    source_run: AnalysisRunIdentity,
    topology_digest: Digest,
    elastic_stiffness_digest: Digest,
    base_displacement: Vec<f64>,
    reference_load: Vec<f64>,
    initial_stress: Vec<StructuralInitialStressState>,
    geometric_stiffness: RealSparseMatrix,
    evidence: StructuralLinearizationEvidence,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct StructuralLinearizationArtifactData {
    schema_version: u16,
    identity: StructuralLinearizationIdentity,
    source_run: AnalysisRunIdentity,
    topology_digest: Digest,
    elastic_stiffness_digest: Digest,
    base_displacement: Vec<f64>,
    reference_load: Vec<f64>,
    initial_stress: Vec<StructuralInitialStressState>,
    geometric_stiffness: RealSparseMatrix,
    evidence: StructuralLinearizationEvidence,
}

#[derive(Debug, Error)]
pub enum StructuralLinearizationError {
    #[error("structural linearization requires an exact finite full-DOF solution")]
    InvalidSolution,
    #[error("structural linearization currently requires a solid-only structural operator")]
    UnsupportedStructuralFormulation,
    #[error("structural linearization element `{element_id}` is invalid: {message}")]
    InvalidElement { element_id: String, message: String },
    #[error("structural linearization sparse assembly failed: {0}")]
    Sparse(String),
    #[error("structural linearization identity encoding failed: {0}")]
    Identity(String),
    #[error("structural linearization artifact is inconsistent: {0}")]
    InvalidArtifact(&'static str),
}

impl TryFrom<StructuralLinearizationArtifactData> for StructuralLinearizationArtifact {
    type Error = StructuralLinearizationError;

    fn try_from(data: StructuralLinearizationArtifactData) -> Result<Self, Self::Error> {
        let artifact = Self {
            schema_version: data.schema_version,
            identity: data.identity,
            source_run: data.source_run,
            topology_digest: data.topology_digest,
            elastic_stiffness_digest: data.elastic_stiffness_digest,
            base_displacement: data.base_displacement,
            reference_load: data.reference_load,
            initial_stress: data.initial_stress,
            geometric_stiffness: data.geometric_stiffness,
            evidence: data.evidence,
        };
        artifact.validate()?;
        Ok(artifact)
    }
}

impl StructuralLinearizationArtifact {
    pub fn from_solution(
        summary: &AssemblySummary,
        source_run: AnalysisRunIdentity,
        topology_digest: Digest,
        solution: &[f64],
    ) -> Result<Self, StructuralLinearizationError> {
        if solution.len() != summary.dof_count || solution.iter().any(|value| !value.is_finite()) {
            return Err(StructuralLinearizationError::InvalidSolution);
        }
        if summary.structural_beam_element_count != 0 || summary.structural_shell_element_count != 0
        {
            return Err(StructuralLinearizationError::UnsupportedStructuralFormulation);
        }
        if summary.structural_solid_recovery.is_empty()
            || summary.structural_solid_recovery.len() != summary.structural_solid_element_count
        {
            return Err(StructuralLinearizationError::UnsupportedStructuralFormulation);
        }

        let mut assembler = RealSparseAssembler::square(
            summary.dof_count,
            OperatorRole::GeometricStiffness,
            MatrixProperties::SYMMETRIC_INDEFINITE,
        );
        assembler.add_contribution(ContributionProvenance {
            domain: ContributionDomain::Structural,
            contribution_id: ContributionId("solid_initial_stress".into()),
        });
        let mut initial_stress = Vec::with_capacity(summary.structural_solid_recovery.len());
        let mut quadrature_count = 0usize;
        for element in &summary.structural_solid_recovery {
            let stresses = match &element.geometry {
                SolidRecoveryGeometry::Tetrahedron4 {
                    node_indices,
                    coordinates_m,
                } => {
                    let geometry = Tetrahedron4ElementGeometry {
                        nodes_m: *coordinates_m,
                    };
                    let displacement = gather_element_displacement::<4, 12>(
                        &summary.structural_dof_layout,
                        node_indices,
                        solution,
                    )?;
                    let b = strain_displacement_matrix(geometry)
                        .map_err(|error| element_error(&element.element_id, error))?;
                    let elasticity = elasticity_matrix(element.material)
                        .map_err(|error| element_error(&element.element_id, error))?;
                    let strain = std::array::from_fn::<_, 6, _>(|row| {
                        b[row]
                            .iter()
                            .zip(displacement)
                            .map(|(left, right)| left * right)
                            .sum::<f64>()
                    });
                    let stress = std::array::from_fn::<_, 6, _>(|row| {
                        elasticity[row]
                            .iter()
                            .zip(strain)
                            .map(|(left, right)| left * right)
                            .sum::<f64>()
                    });
                    let tensor = voigt_stress_tensor(stress);
                    let matrix = tetrahedron4_geometric_stiffness_matrix(geometry, tensor)
                        .map_err(|error| element_error(&element.element_id, error))?;
                    assemble_element_matrix(
                        &mut assembler,
                        &summary.structural_dof_layout,
                        node_indices,
                        &matrix,
                    )?;
                    vec![tensor]
                }
                SolidRecoveryGeometry::Tetrahedron10 {
                    node_indices,
                    coordinates_m,
                } => {
                    let geometry = Tetrahedron10ElementGeometry {
                        nodes_m: *coordinates_m,
                    };
                    let displacement = gather_element_displacement::<10, 30>(
                        &summary.structural_dof_layout,
                        node_indices,
                        solution,
                    )?;
                    let stresses =
                        tetrahedron10_quadrature_stress(element.material, geometry, displacement)
                            .map_err(|error| element_error(&element.element_id, error))?;
                    let matrix = tetrahedron10_geometric_stiffness_matrix(geometry, stresses)
                        .map_err(|error| element_error(&element.element_id, error))?;
                    assemble_element_matrix(
                        &mut assembler,
                        &summary.structural_dof_layout,
                        node_indices,
                        &matrix,
                    )?;
                    stresses.to_vec()
                }
            };
            quadrature_count += stresses.len();
            initial_stress.push(StructuralInitialStressState {
                element_id: element.element_id.clone(),
                region_id: element.region_id.clone(),
                quadrature_stress_pa: stresses,
            });
        }
        let geometric_stiffness = assembler
            .finish()
            .map_err(|error| StructuralLinearizationError::Sparse(error.to_string()))?;
        let evidence = StructuralLinearizationEvidence {
            reference_load_factor: 1.0,
            solid_element_count: initial_stress.len(),
            stress_quadrature_point_count: quadrature_count,
            geometric_stiffness_nonzero_count: geometric_stiffness.values().len(),
            maximum_symmetry_residual: symmetry_residual(&geometric_stiffness),
        };
        let elastic_stiffness_digest = summary
            .operator
            .operators()
            .stiffness()
            .numerical_identity()
            .map_err(|error| StructuralLinearizationError::Identity(error.to_string()))?;
        let mut artifact = Self {
            schema_version: 1,
            identity: StructuralLinearizationIdentity(Digest::sha256([])),
            source_run,
            topology_digest,
            elastic_stiffness_digest,
            base_displacement: solution.to_vec(),
            reference_load: summary.physical_load.clone(),
            initial_stress,
            geometric_stiffness,
            evidence,
        };
        artifact.identity = artifact.derive_identity()?;
        artifact.validate()?;
        Ok(artifact)
    }

    pub const fn identity(&self) -> StructuralLinearizationIdentity {
        self.identity
    }
    pub const fn source_run(&self) -> &AnalysisRunIdentity {
        &self.source_run
    }
    pub const fn topology_digest(&self) -> Digest {
        self.topology_digest
    }
    pub const fn elastic_stiffness_digest(&self) -> Digest {
        self.elastic_stiffness_digest
    }
    pub fn base_displacement(&self) -> &[f64] {
        &self.base_displacement
    }
    pub fn reference_load(&self) -> &[f64] {
        &self.reference_load
    }
    pub fn initial_stress(&self) -> &[StructuralInitialStressState] {
        &self.initial_stress
    }
    pub fn geometric_stiffness(&self) -> &RealSparseMatrix {
        &self.geometric_stiffness
    }
    pub const fn evidence(&self) -> StructuralLinearizationEvidence {
        self.evidence
    }

    pub fn validate(&self) -> Result<(), StructuralLinearizationError> {
        let dimension = self.base_displacement.len();
        if self.schema_version != 1
            || dimension == 0
            || self.reference_load.len() != dimension
            || self
                .base_displacement
                .iter()
                .any(|value| !value.is_finite())
            || self.reference_load.iter().any(|value| !value.is_finite())
            || self.geometric_stiffness.rows() != dimension
            || self.geometric_stiffness.columns() != dimension
            || self.geometric_stiffness.role() != OperatorRole::GeometricStiffness
            || self.geometric_stiffness.properties() != MatrixProperties::SYMMETRIC_INDEFINITE
        {
            return Err(StructuralLinearizationError::InvalidArtifact(
                "operator dimensions or typed metadata are invalid",
            ));
        }
        let mut element_ids = BTreeSet::new();
        if self.initial_stress.is_empty()
            || self.initial_stress.iter().any(|state| {
                state.element_id.is_empty()
                    || state.region_id.is_empty()
                    || !element_ids.insert(state.element_id.as_str())
                    || !matches!(state.quadrature_stress_pa.len(), 1 | 4)
                    || state
                        .quadrature_stress_pa
                        .iter()
                        .flatten()
                        .flatten()
                        .any(|value| !value.is_finite())
                    || state
                        .quadrature_stress_pa
                        .iter()
                        .any(|stress| stress_is_asymmetric(*stress))
            })
        {
            return Err(StructuralLinearizationError::InvalidArtifact(
                "initial stress inventory is invalid",
            ));
        }
        let expected_evidence = StructuralLinearizationEvidence {
            reference_load_factor: 1.0,
            solid_element_count: self.initial_stress.len(),
            stress_quadrature_point_count: self
                .initial_stress
                .iter()
                .map(|state| state.quadrature_stress_pa.len())
                .sum(),
            geometric_stiffness_nonzero_count: self.geometric_stiffness.values().len(),
            maximum_symmetry_residual: symmetry_residual(&self.geometric_stiffness),
        };
        if self.evidence != expected_evidence
            || self.evidence.maximum_symmetry_residual > 1.0e-12
            || self.identity != self.derive_identity()?
        {
            return Err(StructuralLinearizationError::InvalidArtifact(
                "evidence or identity does not converge",
            ));
        }
        Ok(())
    }

    fn derive_identity(
        &self,
    ) -> Result<StructuralLinearizationIdentity, StructuralLinearizationError> {
        encode_contract(
            LINEARIZATION_PREFIX,
            "analysis.fea.structural-linearization",
            &(
                &self.source_run,
                self.topology_digest,
                self.elastic_stiffness_digest,
                &self.base_displacement,
                &self.reference_load,
                &self.initial_stress,
                &self.geometric_stiffness,
                self.evidence,
            ),
            LINEARIZATION_LIMITS,
        )
        .map(Digest::sha256)
        .map(StructuralLinearizationIdentity)
        .map_err(|error| StructuralLinearizationError::Identity(error.to_string()))
    }
}

fn gather_element_displacement<const N: usize, const D: usize>(
    layout: &crate::assembly::dofs::StructuralDofLayout,
    node_indices: &[usize; N],
    solution: &[f64],
) -> Result<[f64; D], StructuralLinearizationError> {
    debug_assert_eq!(D, N * 3);
    let kinds = [
        StructuralDofKind::Ux,
        StructuralDofKind::Uy,
        StructuralDofKind::Uz,
    ];
    let mut gathered = [0.0; D];
    for (local_node, node_index) in node_indices.iter().copied().enumerate() {
        for (axis, kind) in kinds.into_iter().enumerate() {
            let row = layout
                .index(node_index, kind)
                .ok_or(StructuralLinearizationError::InvalidSolution)?;
            gathered[local_node * 3 + axis] = *solution
                .get(row)
                .ok_or(StructuralLinearizationError::InvalidSolution)?;
        }
    }
    Ok(gathered)
}

fn assemble_element_matrix<const D: usize>(
    assembler: &mut RealSparseAssembler,
    layout: &crate::assembly::dofs::StructuralDofLayout,
    node_indices: &[usize],
    local: &[[f64; D]; D],
) -> Result<(), StructuralLinearizationError> {
    let kinds = [
        StructuralDofKind::Ux,
        StructuralDofKind::Uy,
        StructuralDofKind::Uz,
    ];
    let rows = node_indices
        .iter()
        .flat_map(|node| kinds.into_iter().map(|kind| layout.index(*node, kind)))
        .collect::<Option<Vec<_>>>()
        .ok_or(StructuralLinearizationError::InvalidSolution)?;
    if D != rows.len() {
        return Err(StructuralLinearizationError::InvalidSolution);
    }
    for (local_row, global_row) in rows.iter().copied().enumerate() {
        for (local_column, global_column) in rows.iter().copied().enumerate() {
            assembler.add(global_row, global_column, local[local_row][local_column]);
        }
    }
    Ok(())
}

fn voigt_stress_tensor(stress: [f64; 6]) -> [[f64; 3]; 3] {
    [
        [stress[0], stress[5], stress[4]],
        [stress[5], stress[1], stress[3]],
        [stress[4], stress[3], stress[2]],
    ]
}

fn symmetry_residual(matrix: &RealSparseMatrix) -> f64 {
    (0..matrix.rows())
        .flat_map(|row| {
            (row..matrix.columns())
                .map(move |column| (matrix.get(row, column) - matrix.get(column, row)).abs())
        })
        .fold(0.0, f64::max)
}

fn stress_is_asymmetric(stress: [[f64; 3]; 3]) -> bool {
    (0..3).any(|row| {
        (row + 1..3).any(|column| {
            let scale = stress[row][column]
                .abs()
                .max(stress[column][row].abs())
                .max(1.0);
            (stress[row][column] - stress[column][row]).abs() > 1.0e-12 * scale
        })
    })
}

fn element_error(element_id: &str, error: impl std::fmt::Display) -> StructuralLinearizationError {
    StructuralLinearizationError::InvalidElement {
        element_id: element_id.into(),
        message: error.to_string(),
    }
}
