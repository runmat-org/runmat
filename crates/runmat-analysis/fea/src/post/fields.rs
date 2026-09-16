use runmat_analysis_core::{
    AnalysisField, FieldLocation, ObservationQuantity, ObservationValueShape,
};

use crate::contracts::{
    FEA_FIELD_STRUCTURAL_BEAM_AXIAL_FORCE, FEA_FIELD_STRUCTURAL_BEAM_BENDING_MOMENT,
    FEA_FIELD_STRUCTURAL_BEAM_BENDING_STRESS, FEA_FIELD_STRUCTURAL_BEAM_SHEAR_FORCE,
    FEA_FIELD_STRUCTURAL_BEAM_TORSION_MOMENT, FEA_FIELD_STRUCTURAL_BEAM_TORSION_STRESS,
    FEA_FIELD_STRUCTURAL_CONSTRAINT_REACTION_FORCE,
    FEA_FIELD_STRUCTURAL_CONSTRAINT_REACTION_MOMENT_ABOUT_GLOBAL_ORIGIN,
    FEA_FIELD_STRUCTURAL_DISPLACEMENT, FEA_FIELD_STRUCTURAL_EQUATION_SCALE,
    FEA_FIELD_STRUCTURAL_NODAL_VON_MISES, FEA_FIELD_STRUCTURAL_REACTION_FORCE,
    FEA_FIELD_STRUCTURAL_REACTION_MOMENT, FEA_FIELD_STRUCTURAL_RESIDUAL_NORM,
    FEA_FIELD_STRUCTURAL_ROTATION, FEA_FIELD_STRUCTURAL_SHELL_BENDING_MOMENT,
    FEA_FIELD_STRUCTURAL_SHELL_MEMBRANE_FORCE, FEA_FIELD_STRUCTURAL_SHELL_TRANSVERSE_SHEAR,
    FEA_FIELD_STRUCTURAL_SHELL_VON_MISES, FEA_FIELD_STRUCTURAL_STRAIN,
    FEA_FIELD_STRUCTURAL_STRAIN_ENERGY_DENSITY, FEA_FIELD_STRUCTURAL_STRESS,
    FEA_FIELD_STRUCTURAL_TOTAL_STRAIN_ENERGY, FEA_FIELD_STRUCTURAL_VON_MISES,
};
use crate::{
    algebra::apply_k,
    assembly::{
        dofs::StructuralDofKind,
        elements::{
            beam::{local_stiffness_matrix, BEAM_ELEMENT_DOF_COUNT},
            shell::{recover_resultants as recover_shell_resultants, SHELL_ELEMENT_DOF_COUNT},
        },
        AssemblySummary, BeamRecoveryElementSummary, ShellRecoveryElementSummary,
        StructuralMaterialSummary,
    },
    post::{error::StructuralFieldRecoveryError, solid::recover_solid_fields},
    results::field::{
        RecoveredField, RecoveredFieldDefinition, RecoveredFieldEntityDomain, RecoveredFieldError,
        RecoveredFieldNumericRepresentation, RecoveredFieldSample,
    },
    solve::linear::LinearSolveResult,
};

const VECTOR_COMPONENT_COUNT: usize = 3;
const TENSOR_COMPONENT_COUNT: usize = 6;

/// Structural quantities whose recovery is linear in the displacement state.
///
/// Statistical modal procedures may combine these component transfers before applying any
/// nonlinear derived transform. Equivalent/principal stress, energy, and beam peak-end selectors
/// are intentionally absent.
#[derive(
    Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, serde::Serialize, serde::Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum StructuralLinearFieldQuantity {
    Displacement,
    Rotation,
    Strain,
    Stress,
    ShellMembraneForce,
    ShellBendingMoment,
    ShellTransverseShear,
}

impl StructuralLinearFieldQuantity {
    pub const ALL: [Self; 7] = [
        Self::Displacement,
        Self::Rotation,
        Self::Strain,
        Self::Stress,
        Self::ShellMembraneForce,
        Self::ShellBendingMoment,
        Self::ShellTransverseShear,
    ];

    pub const fn source_field_id(self) -> &'static str {
        match self {
            Self::Displacement => FEA_FIELD_STRUCTURAL_DISPLACEMENT,
            Self::Rotation => FEA_FIELD_STRUCTURAL_ROTATION,
            Self::Strain => FEA_FIELD_STRUCTURAL_STRAIN,
            Self::Stress => FEA_FIELD_STRUCTURAL_STRESS,
            Self::ShellMembraneForce => FEA_FIELD_STRUCTURAL_SHELL_MEMBRANE_FORCE,
            Self::ShellBendingMoment => FEA_FIELD_STRUCTURAL_SHELL_BENDING_MOMENT,
            Self::ShellTransverseShear => FEA_FIELD_STRUCTURAL_SHELL_TRANSVERSE_SHEAR,
        }
    }
}

pub fn recover_result_fields(
    summary: &AssemblySummary,
    solve_result: &LinearSolveResult,
) -> Result<Vec<AnalysisField>, StructuralFieldRecoveryError> {
    if solve_result.solution.is_empty()
        || solve_result.solution.iter().any(|value| !value.is_finite())
    {
        return Ok(empty_structural_fields());
    }

    let node_count = structural_displacement_node_count(summary, summary.dof_count.max(3));
    let internal_force = apply_k(&summary.operator, &solve_result.solution);
    let reaction_values = recover_reaction_force(summary, &internal_force, node_count);
    let reaction_moment_values = recover_reaction_moment(summary, &internal_force);
    let strain_energy = recover_total_strain_energy(&solve_result.solution, &internal_force);
    let residual_metrics = recover_residual_metrics(summary, &solve_result.solution);
    let mut fields = recover_deformation_result_fields(summary, &solve_result.solution)?;
    fields.extend([
        AnalysisField::host_f64(
            FEA_FIELD_STRUCTURAL_REACTION_FORCE,
            vec![node_count, VECTOR_COMPONENT_COUNT],
            reaction_values,
        ),
        AnalysisField::host_f64(
            FEA_FIELD_STRUCTURAL_REACTION_MOMENT,
            rotation_shape(summary),
            reaction_moment_values,
        ),
        AnalysisField::host_f64(
            FEA_FIELD_STRUCTURAL_TOTAL_STRAIN_ENERGY,
            vec![1],
            vec![strain_energy],
        ),
        AnalysisField::host_f64(
            FEA_FIELD_STRUCTURAL_RESIDUAL_NORM,
            vec![1],
            vec![residual_metrics.normalized_residual_norm],
        ),
        AnalysisField::host_f64(
            FEA_FIELD_STRUCTURAL_EQUATION_SCALE,
            vec![1],
            vec![residual_metrics.equation_scale],
        ),
    ]);
    Ok(fields)
}

/// Recovers only deformation-dependent physical fields from a structural state vector.
///
/// Modal shapes have no static load equilibrium, so reaction and residual fields would be
/// misleading. This shared path keeps their strain, stress, energy-density, beam, and shell
/// recovery identical to linear-static recovery without manufacturing equilibrium quantities.
pub fn recover_deformation_result_fields(
    summary: &AssemblySummary,
    solution: &[f64],
) -> Result<Vec<AnalysisField>, StructuralFieldRecoveryError> {
    if solution.is_empty() || solution.iter().any(|value| !value.is_finite()) {
        return Ok(Vec::new());
    }
    if solution.len() != summary.dof_count {
        return Err(StructuralFieldRecoveryError::DisplacementShapeMismatch {
            expected: summary.dof_count,
            actual: solution.len(),
        });
    }
    let node_count = structural_displacement_node_count(summary, summary.dof_count.max(3));
    let displacement_values = recover_displacement(summary, solution, node_count);
    let continuum_fields = if publishes_continuum_fields(summary) {
        let strain_recovery = recover_structural_strain(summary, &displacement_values)?;
        let element_count = strain_recovery.element_count;
        let strain_values = strain_recovery.values;
        let stress_values = strain_recovery
            .stress
            .unwrap_or_else(|| recover_stress(&strain_values, summary.structural_material));
        let von_mises_values = recover_von_mises(&stress_values);
        let nodal_von_mises_values =
            recover_nodal_averaged_scalar(summary, &von_mises_values, node_count);
        let strain_energy_density_values = strain_recovery
            .strain_energy_density
            .unwrap_or_else(|| recover_strain_energy_density(&strain_values, &stress_values));
        Some((
            element_count,
            strain_values,
            stress_values,
            von_mises_values,
            nodal_von_mises_values,
            strain_energy_density_values,
        ))
    } else {
        None
    };

    let mut fields = vec![
        AnalysisField::host_f64(
            FEA_FIELD_STRUCTURAL_DISPLACEMENT,
            vec![node_count, VECTOR_COMPONENT_COUNT],
            displacement_values,
        ),
        AnalysisField::host_f64(
            FEA_FIELD_STRUCTURAL_ROTATION,
            rotation_shape(summary),
            recover_rotation(summary, solution),
        ),
    ];
    if let Some((
        element_count,
        strain_values,
        stress_values,
        von_mises_values,
        nodal_von_mises_values,
        strain_energy_density_values,
    )) = continuum_fields
    {
        fields.extend([
            AnalysisField::host_f64(
                FEA_FIELD_STRUCTURAL_VON_MISES,
                vec![element_count],
                von_mises_values,
            ),
            AnalysisField::host_f64(
                FEA_FIELD_STRUCTURAL_STRAIN,
                vec![element_count, TENSOR_COMPONENT_COUNT],
                strain_values,
            ),
            AnalysisField::host_f64(
                FEA_FIELD_STRUCTURAL_STRESS,
                vec![element_count, TENSOR_COMPONENT_COUNT],
                stress_values,
            ),
            AnalysisField::host_f64(
                FEA_FIELD_STRUCTURAL_STRAIN_ENERGY_DENSITY,
                vec![element_count],
                strain_energy_density_values,
            ),
            AnalysisField::host_f64(
                FEA_FIELD_STRUCTURAL_NODAL_VON_MISES,
                vec![node_count],
                nodal_von_mises_values,
            ),
        ]);
    }
    if let Some(beam_fields) = recover_beam_result_fields(summary, solution) {
        fields.extend(beam_fields);
    }
    if let Some(shell_fields) = recover_shell_result_fields(summary, solution)? {
        fields.extend(shell_fields);
    }
    Ok(fields)
}

fn publishes_continuum_fields(summary: &AssemblySummary) -> bool {
    !summary.structural_solid_recovery.is_empty()
}

/// Moves spatial fields with stable physical semantics into the observation domain.
///
/// Diagnostics, global summaries, derived display conveniences, and unsupported beam/shell
/// resultants remain in `fields`; they are not mislabeled merely to make them observable.
pub fn take_structural_observable_fields(
    fields: &mut Vec<AnalysisField>,
) -> Result<Vec<RecoveredField>, RecoveredFieldError> {
    let declarations = structural_observable_field_definitions()?;
    let mut recovered = Vec::with_capacity(declarations.len());
    for definition in declarations {
        let Some(index) = fields.iter().position(|field| {
            field.field_id == definition.field_id()
                && field
                    .shape
                    .first()
                    .is_some_and(|entity_count| *entity_count > 0)
        }) else {
            continue;
        };
        let storage = fields.remove(index);
        recovered.push(RecoveredField::new(
            definition,
            RecoveredFieldSample::Final,
            storage,
        )?);
    }
    Ok(recovered)
}

/// Semantic field inventory produced by the structural solid recovery path.
///
/// Resource and observation planning use the same declarations as postprocessing so output
/// selection cannot drift from the fields that are actually recoverable.
pub fn structural_observable_field_definitions(
) -> Result<Vec<RecoveredFieldDefinition>, RecoveredFieldError> {
    let declarations = [
        (
            FEA_FIELD_STRUCTURAL_DISPLACEMENT,
            ObservationQuantity::Displacement,
            FieldLocation::Node,
            RecoveredFieldEntityDomain::StructuralNodes,
            ObservationValueShape::Vector3,
        ),
        (
            FEA_FIELD_STRUCTURAL_ROTATION,
            ObservationQuantity::Rotation,
            FieldLocation::Node,
            RecoveredFieldEntityDomain::StructuralNodes,
            ObservationValueShape::Vector3,
        ),
        (
            FEA_FIELD_STRUCTURAL_STRAIN,
            ObservationQuantity::Strain,
            FieldLocation::Element,
            RecoveredFieldEntityDomain::SolidElements,
            ObservationValueShape::SymmetricTensor3,
        ),
        (
            FEA_FIELD_STRUCTURAL_STRESS,
            ObservationQuantity::Stress,
            FieldLocation::Element,
            RecoveredFieldEntityDomain::SolidElements,
            ObservationValueShape::SymmetricTensor3,
        ),
        (
            FEA_FIELD_STRUCTURAL_VON_MISES,
            ObservationQuantity::VonMisesStress,
            FieldLocation::Element,
            RecoveredFieldEntityDomain::SolidElements,
            ObservationValueShape::Scalar,
        ),
        (
            FEA_FIELD_STRUCTURAL_NODAL_VON_MISES,
            ObservationQuantity::VonMisesStress,
            FieldLocation::Node,
            RecoveredFieldEntityDomain::StructuralNodes,
            ObservationValueShape::Scalar,
        ),
        (
            FEA_FIELD_STRUCTURAL_STRAIN_ENERGY_DENSITY,
            ObservationQuantity::StrainEnergy,
            FieldLocation::Element,
            RecoveredFieldEntityDomain::SolidElements,
            ObservationValueShape::Scalar,
        ),
        (
            FEA_FIELD_STRUCTURAL_REACTION_FORCE,
            ObservationQuantity::ReactionForce,
            FieldLocation::Node,
            RecoveredFieldEntityDomain::StructuralNodes,
            ObservationValueShape::Vector3,
        ),
        (
            FEA_FIELD_STRUCTURAL_CONSTRAINT_REACTION_FORCE,
            ObservationQuantity::ReactionForce,
            FieldLocation::Constraint,
            RecoveredFieldEntityDomain::ConstraintGroups,
            ObservationValueShape::Vector3,
        ),
        (
            FEA_FIELD_STRUCTURAL_CONSTRAINT_REACTION_MOMENT_ABOUT_GLOBAL_ORIGIN,
            ObservationQuantity::ReactionMoment,
            FieldLocation::Constraint,
            RecoveredFieldEntityDomain::ConstraintGroups,
            ObservationValueShape::Vector3,
        ),
        (
            FEA_FIELD_STRUCTURAL_BEAM_AXIAL_FORCE,
            ObservationQuantity::BeamAxialForce,
            FieldLocation::Element,
            RecoveredFieldEntityDomain::BeamElements,
            ObservationValueShape::Scalar,
        ),
        (
            FEA_FIELD_STRUCTURAL_BEAM_SHEAR_FORCE,
            ObservationQuantity::BeamShearForce,
            FieldLocation::Element,
            RecoveredFieldEntityDomain::BeamElements,
            ObservationValueShape::Tensor { extents: vec![2] },
        ),
        (
            FEA_FIELD_STRUCTURAL_BEAM_TORSION_MOMENT,
            ObservationQuantity::BeamTorsionMoment,
            FieldLocation::Element,
            RecoveredFieldEntityDomain::BeamElements,
            ObservationValueShape::Scalar,
        ),
        (
            FEA_FIELD_STRUCTURAL_BEAM_BENDING_MOMENT,
            ObservationQuantity::BeamBendingMoment,
            FieldLocation::Element,
            RecoveredFieldEntityDomain::BeamElements,
            ObservationValueShape::Tensor { extents: vec![2] },
        ),
        (
            FEA_FIELD_STRUCTURAL_BEAM_BENDING_STRESS,
            ObservationQuantity::BeamBendingStress,
            FieldLocation::Element,
            RecoveredFieldEntityDomain::BeamElements,
            ObservationValueShape::Tensor { extents: vec![2] },
        ),
        (
            FEA_FIELD_STRUCTURAL_BEAM_TORSION_STRESS,
            ObservationQuantity::BeamTorsionStress,
            FieldLocation::Element,
            RecoveredFieldEntityDomain::BeamElements,
            ObservationValueShape::Scalar,
        ),
        (
            FEA_FIELD_STRUCTURAL_SHELL_MEMBRANE_FORCE,
            ObservationQuantity::ShellMembraneForce,
            FieldLocation::Element,
            RecoveredFieldEntityDomain::ShellElements,
            ObservationValueShape::Tensor { extents: vec![3] },
        ),
        (
            FEA_FIELD_STRUCTURAL_SHELL_BENDING_MOMENT,
            ObservationQuantity::ShellBendingMoment,
            FieldLocation::Element,
            RecoveredFieldEntityDomain::ShellElements,
            ObservationValueShape::Tensor { extents: vec![3] },
        ),
        (
            FEA_FIELD_STRUCTURAL_SHELL_TRANSVERSE_SHEAR,
            ObservationQuantity::ShellTransverseShear,
            FieldLocation::Element,
            RecoveredFieldEntityDomain::ShellElements,
            ObservationValueShape::Tensor { extents: vec![2] },
        ),
        (
            FEA_FIELD_STRUCTURAL_SHELL_VON_MISES,
            ObservationQuantity::ShellVonMises,
            FieldLocation::Element,
            RecoveredFieldEntityDomain::ShellElements,
            ObservationValueShape::Scalar,
        ),
    ];
    declarations
        .into_iter()
        .map(
            |(field_id, quantity, location, entity_domain, value_shape)| {
                RecoveredFieldDefinition::new(
                    field_id,
                    quantity,
                    location,
                    entity_domain,
                    value_shape,
                    RecoveredFieldNumericRepresentation::Real,
                )
            },
        )
        .collect()
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StructuralFieldRecoveryMetrics {
    pub active_stiffness_edge_count: usize,
    pub constrained_edge_count: usize,
    pub recovery_element_count: usize,
    pub solver_mesh_node_count: usize,
    pub solver_mesh_element_count: usize,
    pub max_edge_displacement_jump: f64,
    pub max_edge_strain_norm: f64,
    pub mean_edge_stiffness_ratio: f64,
    pub mean_edge_length_m: f64,
    pub strain_component_coverage_ratio: f64,
    pub element_geometry_node_count: usize,
    pub element_geometry_edge_count: usize,
    pub element_geometry_coverage_ratio: f64,
    pub basis: &'static str,
}

pub fn structural_field_recovery_metrics(
    summary: &AssemblySummary,
    displacement: &[f64],
) -> Result<StructuralFieldRecoveryMetrics, StructuralFieldRecoveryError> {
    let recovery_edges = structural_recovery_edges(summary);
    let strain_recovery = recover_structural_strain(summary, displacement)?;
    let active_stiffness_edge_count = recovery_edges.len();
    let constrained_edge_count = constrained_recovery_edge_count(summary);
    let mut max_edge_displacement_jump = 0.0_f64;
    let mut max_edge_strain_norm = 0.0_f64;
    let mut stiffness_ratio_sum = 0.0_f64;
    let mut edge_length_sum = 0.0_f64;
    for edge in &recovery_edges {
        let jump = displacement
            .get(edge.to_dof)
            .zip(displacement.get(edge.from_dof))
            .map(|(right, left)| (right - left).abs())
            .unwrap_or(0.0);
        max_edge_displacement_jump = max_edge_displacement_jump.max(jump);
        stiffness_ratio_sum += edge.stiffness_ratio;
        edge_length_sum += edge.edge_length_m;
    }
    for strain_tensor in strain_recovery.values.chunks_exact(TENSOR_COMPONENT_COUNT) {
        let strain_norm = strain_tensor
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt();
        max_edge_strain_norm = max_edge_strain_norm.max(strain_norm);
    }
    let recovery_edge_count = recovery_edges.len();
    let mean_edge_stiffness_ratio = if recovery_edge_count == 0 {
        0.0
    } else {
        stiffness_ratio_sum / recovery_edge_count as f64
    };
    let mean_edge_length_m = if recovery_edge_count == 0 {
        0.0
    } else {
        edge_length_sum / recovery_edge_count as f64
    };
    let strain_component_coverage_ratio =
        strain_component_coverage_ratio(displacement, &recovery_edges);
    let solver_mesh_element_count = summary.structural_solid_recovery.len();
    let solver_mesh_node_count = if solver_mesh_element_count == 0 {
        0
    } else {
        let mut nodes = summary
            .structural_solid_recovery
            .iter()
            .flat_map(|element| element.geometry.node_indices().iter().copied())
            .collect::<Vec<_>>();
        nodes.sort_unstable();
        nodes.dedup();
        nodes.len()
    };

    Ok(StructuralFieldRecoveryMetrics {
        active_stiffness_edge_count,
        constrained_edge_count,
        recovery_element_count: strain_recovery.element_count,
        solver_mesh_node_count,
        solver_mesh_element_count,
        max_edge_displacement_jump,
        max_edge_strain_norm,
        mean_edge_stiffness_ratio,
        mean_edge_length_m,
        strain_component_coverage_ratio,
        element_geometry_node_count: solver_mesh_node_count,
        element_geometry_edge_count: solver_mesh_element_count.saturating_mul(6),
        element_geometry_coverage_ratio: f64::from(solver_mesh_element_count > 0),
        basis: strain_recovery.basis,
    })
}

struct StructuralStrainRecovery {
    values: Vec<f64>,
    stress: Option<Vec<f64>>,
    strain_energy_density: Option<Vec<f64>>,
    element_count: usize,
    basis: &'static str,
}

fn empty_structural_fields() -> Vec<AnalysisField> {
    vec![
        AnalysisField::host_f64(FEA_FIELD_STRUCTURAL_DISPLACEMENT, vec![0, 3], Vec::new()),
        AnalysisField::host_f64(FEA_FIELD_STRUCTURAL_ROTATION, vec![0, 3], Vec::new()),
        AnalysisField::host_f64(FEA_FIELD_STRUCTURAL_VON_MISES, vec![0], Vec::new()),
        AnalysisField::host_f64(FEA_FIELD_STRUCTURAL_STRAIN, vec![0, 6], Vec::new()),
        AnalysisField::host_f64(FEA_FIELD_STRUCTURAL_STRESS, vec![0, 6], Vec::new()),
        AnalysisField::host_f64(FEA_FIELD_STRUCTURAL_NODAL_VON_MISES, vec![0], Vec::new()),
        AnalysisField::host_f64(
            FEA_FIELD_STRUCTURAL_STRAIN_ENERGY_DENSITY,
            vec![0],
            Vec::new(),
        ),
        AnalysisField::host_f64(FEA_FIELD_STRUCTURAL_REACTION_FORCE, vec![0, 3], Vec::new()),
        AnalysisField::host_f64(FEA_FIELD_STRUCTURAL_REACTION_MOMENT, vec![0, 3], Vec::new()),
        AnalysisField::host_f64(
            FEA_FIELD_STRUCTURAL_TOTAL_STRAIN_ENERGY,
            vec![0],
            Vec::new(),
        ),
        AnalysisField::host_f64(FEA_FIELD_STRUCTURAL_RESIDUAL_NORM, vec![0], Vec::new()),
        AnalysisField::host_f64(FEA_FIELD_STRUCTURAL_EQUATION_SCALE, vec![0], Vec::new()),
    ]
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct StructuralRecoveryEdge {
    from_dof: usize,
    to_dof: usize,
    component: usize,
    hop: usize,
    edge_length_m: f64,
    stiffness_ratio: f64,
    basis: StructuralRecoveryBasis,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StructuralRecoveryBasis {
    OperatorConnectivity,
}

impl StructuralRecoveryBasis {
    const fn as_str(self) -> &'static str {
        match self {
            Self::OperatorConnectivity => "operator_connectivity",
        }
    }
}

fn structural_recovery_edges(summary: &AssemblySummary) -> Vec<StructuralRecoveryEdge> {
    let matrix = summary.operator.operators().stiffness();
    let constrained = summary.operator.constrained_dofs();
    let mut edges = Vec::new();
    for from_dof in 0..matrix.rows() {
        for entry in matrix.pattern().row_range(from_dof) {
            let to_dof = matrix.pattern().column(entry);
            let stiffness = matrix.values()[entry].abs();
            if to_dof <= from_dof
                || stiffness == 0.0
                || constrained[from_dof]
                || constrained[to_dof]
            {
                continue;
            }
            let left_diag = matrix.get(from_dof, from_dof);
            let right_diag = matrix.get(to_dof, to_dof);
            let diag_scale = (0.5 * (left_diag + right_diag)).abs().max(1.0);
            edges.push(StructuralRecoveryEdge {
                from_dof,
                to_dof,
                component: from_dof % VECTOR_COMPONENT_COUNT,
                hop: to_dof - from_dof,
                edge_length_m: 1.0,
                stiffness_ratio: stiffness / diag_scale,
                basis: StructuralRecoveryBasis::OperatorConnectivity,
            });
        }
    }
    edges
}

fn constrained_recovery_edge_count(summary: &AssemblySummary) -> usize {
    structural_recovery_edges_including_constraints(summary)
        .into_iter()
        .filter(|(left, right)| {
            summary.operator.constrained_dofs()[*left]
                || summary.operator.constrained_dofs()[*right]
        })
        .count()
}

fn structural_recovery_edges_including_constraints(
    summary: &AssemblySummary,
) -> Vec<(usize, usize)> {
    let matrix = summary.operator.operators().stiffness();
    let mut edges = Vec::new();
    for row in 0..matrix.rows() {
        for entry in matrix.pattern().row_range(row) {
            let column = matrix.pattern().column(entry);
            if column > row && matrix.values()[entry] != 0.0 {
                edges.push((row, column));
            }
        }
    }
    edges
}

fn recover_structural_strain(
    summary: &AssemblySummary,
    displacement: &[f64],
) -> Result<StructuralStrainRecovery, StructuralFieldRecoveryError> {
    if !summary.structural_solid_recovery.is_empty() {
        let recovered = recover_solid_fields(&summary.structural_solid_recovery, displacement)?;
        let basis = if summary.structural_solid_recovery.iter().all(|element| {
            matches!(
                element.geometry,
                crate::assembly::SolidRecoveryGeometry::Tetrahedron4 { .. }
            )
        }) {
            "solid_tetrahedron4_constant_strain"
        } else {
            "solid_tetrahedron10_volume_average_strain"
        };
        return Ok(StructuralStrainRecovery {
            values: recovered.strain,
            stress: Some(recovered.stress),
            strain_energy_density: Some(recovered.strain_energy_density),
            element_count: summary.structural_solid_recovery.len().max(1),
            basis,
        });
    }

    let recovery_edges = structural_recovery_edges(summary);
    Ok(StructuralStrainRecovery {
        values: recover_edge_strain(displacement, &recovery_edges),
        stress: None,
        strain_energy_density: None,
        element_count: recovery_edges.len().max(1),
        basis: recovery_edges
            .first()
            .map(|edge| edge.basis.as_str())
            .unwrap_or(StructuralRecoveryBasis::OperatorConnectivity.as_str()),
    })
}

fn recover_nodal_averaged_scalar(
    summary: &AssemblySummary,
    element_values: &[f64],
    node_count: usize,
) -> Vec<f64> {
    let mut nodal = vec![0.0_f64; node_count];
    let mut counts = vec![0_usize; node_count];
    for (element_index, element) in summary.structural_solid_recovery.iter().enumerate() {
        let Some(value) = element_values.get(element_index).copied() else {
            continue;
        };
        for &node_index in element.geometry.node_indices() {
            if let Some(accumulator) = nodal.get_mut(node_index) {
                *accumulator += value;
                counts[node_index] += 1;
            }
        }
    }
    for (value, count) in nodal.iter_mut().zip(counts) {
        if count > 0 {
            *value /= count as f64;
        }
    }
    nodal
}

fn recover_edge_strain(
    displacement: &[f64],
    recovery_edges: &[StructuralRecoveryEdge],
) -> Vec<f64> {
    let element_count = recovery_edges.len().max(1);
    let mut strain = vec![0.0; element_count * TENSOR_COMPONENT_COUNT];
    for (element_index, edge) in recovery_edges.iter().enumerate() {
        let edge_strain = edge_strain_tensor(displacement, edge);
        let base = element_index * TENSOR_COMPONENT_COUNT;
        strain[base..base + TENSOR_COMPONENT_COUNT].copy_from_slice(&edge_strain);
    }
    strain
}

fn edge_strain_tensor(displacement: &[f64], edge: &StructuralRecoveryEdge) -> [f64; 6] {
    let length = if edge.edge_length_m.is_finite() && edge.edge_length_m > 0.0 {
        edge.edge_length_m
    } else {
        edge.hop.max(1) as f64
    };
    let from_node = edge.from_dof / VECTOR_COMPONENT_COUNT;
    let to_node = edge.to_dof / VECTOR_COMPONENT_COUNT;
    if from_node == to_node {
        let jump = displacement.get(edge.to_dof).copied().unwrap_or(0.0)
            - displacement.get(edge.from_dof).copied().unwrap_or(0.0);
        let mut tensor = [0.0; TENSOR_COMPONENT_COUNT];
        tensor[edge.component] = jump / length;
        return tensor;
    }

    let du = nodal_displacement(displacement, to_node);
    let u0 = nodal_displacement(displacement, from_node);
    let gradient = [
        (du[0] - u0[0]) / length,
        (du[1] - u0[1]) / length,
        (du[2] - u0[2]) / length,
    ];
    [
        gradient[0],
        gradient[1],
        gradient[2],
        0.5 * (gradient[0] + gradient[1]),
        0.5 * (gradient[1] + gradient[2]),
        0.5 * (gradient[0] + gradient[2]),
    ]
}

fn nodal_displacement(displacement: &[f64], node: usize) -> [f64; VECTOR_COMPONENT_COUNT] {
    let base = node * VECTOR_COMPONENT_COUNT;
    [
        displacement.get(base).copied().unwrap_or(0.0),
        displacement.get(base + 1).copied().unwrap_or(0.0),
        displacement.get(base + 2).copied().unwrap_or(0.0),
    ]
}

fn strain_component_coverage_ratio(
    displacement: &[f64],
    recovery_edges: &[StructuralRecoveryEdge],
) -> f64 {
    let expected_component_count = recovery_edges.len() * TENSOR_COMPONENT_COUNT;
    if expected_component_count == 0 {
        return 0.0;
    }
    let active_component_count = recovery_edges
        .iter()
        .map(|edge| {
            edge_strain_tensor(displacement, edge)
                .iter()
                .filter(|value| value.abs() > 0.0)
                .count()
        })
        .sum::<usize>();
    active_component_count as f64 / expected_component_count as f64
}

fn recover_stress(strain: &[f64], material: StructuralMaterialSummary) -> Vec<f64> {
    let lambda = material.lame_lambda_pa.max(0.0);
    let mu = material.shear_modulus_pa.max(0.0);
    let mut stress = vec![0.0; strain.len()];
    for (element_index, strain_tensor) in strain.chunks_exact(TENSOR_COMPONENT_COUNT).enumerate() {
        let trace = strain_tensor[0] + strain_tensor[1] + strain_tensor[2];
        let base = element_index * TENSOR_COMPONENT_COUNT;
        stress[base] = lambda * trace + 2.0 * mu * strain_tensor[0];
        stress[base + 1] = lambda * trace + 2.0 * mu * strain_tensor[1];
        stress[base + 2] = lambda * trace + 2.0 * mu * strain_tensor[2];
        stress[base + 3] = mu * strain_tensor[3];
        stress[base + 4] = mu * strain_tensor[4];
        stress[base + 5] = mu * strain_tensor[5];
    }
    stress
}

fn recover_von_mises(stress: &[f64]) -> Vec<f64> {
    stress
        .chunks_exact(TENSOR_COMPONENT_COUNT)
        .map(|tensor| {
            let sxx = tensor[0];
            let syy = tensor[1];
            let szz = tensor[2];
            let txy = tensor[3];
            let tyz = tensor[4];
            let txz = tensor[5];
            (0.5 * ((sxx - syy).powi(2) + (syy - szz).powi(2) + (szz - sxx).powi(2))
                + 3.0 * (txy.powi(2) + tyz.powi(2) + txz.powi(2)))
            .sqrt()
        })
        .collect()
}

fn recover_strain_energy_density(strain: &[f64], stress: &[f64]) -> Vec<f64> {
    strain
        .chunks_exact(TENSOR_COMPONENT_COUNT)
        .zip(stress.chunks_exact(TENSOR_COMPONENT_COUNT))
        .map(|(strain_tensor, stress_tensor)| {
            0.5 * strain_tensor
                .iter()
                .zip(stress_tensor.iter())
                .map(|(strain, stress)| strain * stress)
                .sum::<f64>()
        })
        .collect()
}

struct BeamResultValues {
    axial_force: Vec<f64>,
    shear_force: Vec<f64>,
    torsion_moment: Vec<f64>,
    bending_moment: Vec<f64>,
    bending_stress: Vec<f64>,
    torsion_stress: Vec<f64>,
}

fn recover_beam_result_fields(
    summary: &AssemblySummary,
    solution: &[f64],
) -> Option<Vec<AnalysisField>> {
    if summary.structural_beam_recovery.is_empty() {
        return None;
    }
    let values = recover_beam_result_values(summary, solution);
    let element_count = summary.structural_beam_recovery.len();
    Some(vec![
        AnalysisField::host_f64(
            FEA_FIELD_STRUCTURAL_BEAM_AXIAL_FORCE,
            vec![element_count],
            values.axial_force,
        ),
        AnalysisField::host_f64(
            FEA_FIELD_STRUCTURAL_BEAM_SHEAR_FORCE,
            vec![element_count, 2],
            values.shear_force,
        ),
        AnalysisField::host_f64(
            FEA_FIELD_STRUCTURAL_BEAM_TORSION_MOMENT,
            vec![element_count],
            values.torsion_moment,
        ),
        AnalysisField::host_f64(
            FEA_FIELD_STRUCTURAL_BEAM_BENDING_MOMENT,
            vec![element_count, 2],
            values.bending_moment,
        ),
        AnalysisField::host_f64(
            FEA_FIELD_STRUCTURAL_BEAM_BENDING_STRESS,
            vec![element_count, 2],
            values.bending_stress,
        ),
        AnalysisField::host_f64(
            FEA_FIELD_STRUCTURAL_BEAM_TORSION_STRESS,
            vec![element_count],
            values.torsion_stress,
        ),
    ])
}

fn recover_beam_result_values(summary: &AssemblySummary, solution: &[f64]) -> BeamResultValues {
    let mut axial_force = Vec::with_capacity(summary.structural_beam_recovery.len());
    let mut shear_force = Vec::with_capacity(summary.structural_beam_recovery.len() * 2);
    let mut torsion_moment = Vec::with_capacity(summary.structural_beam_recovery.len());
    let mut bending_moment = Vec::with_capacity(summary.structural_beam_recovery.len() * 2);
    let mut bending_stress = Vec::with_capacity(summary.structural_beam_recovery.len() * 2);
    let mut torsion_stress = Vec::with_capacity(summary.structural_beam_recovery.len());

    for beam in &summary.structural_beam_recovery {
        let q_global = beam_global_displacement(summary, solution, beam);
        let q_local = multiply_beam_matrix_vector(&beam.transform_global_to_local, &q_global);
        let k_local = local_stiffness_matrix(beam.section, beam.material, beam.length_m)
            .unwrap_or([[0.0; BEAM_ELEMENT_DOF_COUNT]; BEAM_ELEMENT_DOF_COUNT]);
        let f_local = multiply_beam_matrix_vector(&k_local, &q_local);

        let axial = signed_peak_pair(f_local[0], f_local[6]);
        let shear_y = signed_peak_pair(f_local[1], f_local[7]);
        let shear_z = signed_peak_pair(f_local[2], f_local[8]);
        let torsion = signed_peak_pair(f_local[3], f_local[9]);
        let bending_y = signed_peak_pair(f_local[4], f_local[10]);
        let bending_z = signed_peak_pair(f_local[5], f_local[11]);

        axial_force.push(axial);
        shear_force.extend([shear_y, shear_z]);
        torsion_moment.push(torsion);
        bending_moment.extend([bending_y, bending_z]);
        bending_stress.extend([
            bending_stress_from_moment(bending_y, beam.section.outer_fiber_z_m, beam.section.iy_m4),
            bending_stress_from_moment(bending_z, beam.section.outer_fiber_y_m, beam.section.iz_m4),
        ]);
        torsion_stress.push(torsion_stress_from_moment(
            torsion,
            beam.section.torsion_outer_radius_m,
            beam.section.torsion_j_m4,
        ));
    }

    BeamResultValues {
        axial_force,
        shear_force,
        torsion_moment,
        bending_moment,
        bending_stress,
        torsion_stress,
    }
}

fn beam_global_displacement(
    summary: &AssemblySummary,
    solution: &[f64],
    beam: &BeamRecoveryElementSummary,
) -> [f64; BEAM_ELEMENT_DOF_COUNT] {
    let mut q = [0.0; BEAM_ELEMENT_DOF_COUNT];
    for (node_offset, node_index) in [beam.node_i_index, beam.node_j_index].iter().enumerate() {
        for (component, kind) in StructuralDofKind::ORDER.iter().enumerate() {
            let target = node_offset * StructuralDofKind::ORDER.len() + component;
            q[target] = summary
                .structural_dof_layout
                .index(*node_index, *kind)
                .and_then(|row| solution.get(row).copied())
                .unwrap_or(0.0);
        }
    }
    q
}

fn multiply_beam_matrix_vector(
    matrix: &[[f64; BEAM_ELEMENT_DOF_COUNT]; BEAM_ELEMENT_DOF_COUNT],
    vector: &[f64; BEAM_ELEMENT_DOF_COUNT],
) -> [f64; BEAM_ELEMENT_DOF_COUNT] {
    let mut out = [0.0; BEAM_ELEMENT_DOF_COUNT];
    for (row, out_value) in out.iter_mut().enumerate() {
        *out_value = matrix[row]
            .iter()
            .zip(vector.iter())
            .map(|(entry, value)| entry * value)
            .sum();
    }
    out
}

fn signed_peak_pair(a: f64, b: f64) -> f64 {
    if a.abs() >= b.abs() {
        a
    } else {
        b
    }
}

fn bending_stress_from_moment(moment_n_m: f64, outer_fiber_m: f64, inertia_m4: f64) -> f64 {
    if outer_fiber_m > 0.0 && inertia_m4 > 0.0 {
        moment_n_m * outer_fiber_m / inertia_m4
    } else {
        0.0
    }
}

fn torsion_stress_from_moment(moment_n_m: f64, outer_radius_m: f64, torsion_j_m4: f64) -> f64 {
    if outer_radius_m > 0.0 && torsion_j_m4 > 0.0 {
        moment_n_m * outer_radius_m / torsion_j_m4
    } else {
        0.0
    }
}

struct ShellResultValues {
    membrane_force: Vec<f64>,
    bending_moment: Vec<f64>,
    transverse_shear: Vec<f64>,
    von_mises: Vec<f64>,
}

fn recover_shell_result_fields(
    summary: &AssemblySummary,
    solution: &[f64],
) -> Result<Option<Vec<AnalysisField>>, StructuralFieldRecoveryError> {
    if summary.structural_shell_recovery.is_empty() {
        return Ok(None);
    }
    let values = recover_shell_result_values(summary, solution)?;
    let element_count = summary.structural_shell_recovery.len();
    Ok(Some(vec![
        AnalysisField::host_f64(
            FEA_FIELD_STRUCTURAL_SHELL_MEMBRANE_FORCE,
            vec![element_count, 3],
            values.membrane_force,
        ),
        AnalysisField::host_f64(
            FEA_FIELD_STRUCTURAL_SHELL_BENDING_MOMENT,
            vec![element_count, 3],
            values.bending_moment,
        ),
        AnalysisField::host_f64(
            FEA_FIELD_STRUCTURAL_SHELL_TRANSVERSE_SHEAR,
            vec![element_count, 2],
            values.transverse_shear,
        ),
        AnalysisField::host_f64(
            FEA_FIELD_STRUCTURAL_SHELL_VON_MISES,
            vec![element_count],
            values.von_mises,
        ),
    ]))
}

fn recover_shell_result_values(
    summary: &AssemblySummary,
    solution: &[f64],
) -> Result<ShellResultValues, StructuralFieldRecoveryError> {
    let mut membrane_force = Vec::with_capacity(summary.structural_shell_recovery.len() * 3);
    let mut bending_moment = Vec::with_capacity(summary.structural_shell_recovery.len() * 3);
    let mut transverse_shear = Vec::with_capacity(summary.structural_shell_recovery.len() * 2);
    let mut von_mises = Vec::with_capacity(summary.structural_shell_recovery.len());

    for shell in &summary.structural_shell_recovery {
        let q = shell_node_values(summary, solution, shell);
        let global_displacement =
            std::array::from_fn::<_, SHELL_ELEMENT_DOF_COUNT, _>(|index| q[index / 6][index % 6]);
        let result = recover_shell_resultants(
            shell.section,
            shell.material,
            shell.geometry,
            global_displacement,
        )
        .map_err(|source| StructuralFieldRecoveryError::Shell {
            element_id: shell.element_id.clone(),
            source,
        })?;
        membrane_force.extend(result.membrane_force_n_per_m);
        bending_moment.extend(result.bending_moment_n);
        transverse_shear.extend(result.transverse_shear_n_per_m);
        von_mises.push(result.surface_von_mises_pa);
    }

    Ok(ShellResultValues {
        membrane_force,
        bending_moment,
        transverse_shear,
        von_mises,
    })
}

fn shell_node_values(
    summary: &AssemblySummary,
    solution: &[f64],
    shell: &ShellRecoveryElementSummary,
) -> [[f64; 6]; 3] {
    let mut out = [[0.0; 6]; 3];
    for (node_offset, node_index) in shell.node_indices.iter().enumerate() {
        for (component, kind) in StructuralDofKind::ORDER.iter().enumerate() {
            out[node_offset][component] = summary
                .structural_dof_layout
                .index(*node_index, *kind)
                .and_then(|row| solution.get(row).copied())
                .unwrap_or(0.0);
        }
    }
    out
}

fn recover_reaction_force(
    summary: &AssemblySummary,
    internal_force: &[f64],
    node_count: usize,
) -> Vec<f64> {
    let mut reactions = vec![0.0; node_count * VECTOR_COMPONENT_COUNT];
    for (dof, is_constrained) in summary.operator.constrained_dofs().iter().enumerate() {
        if !*is_constrained && !is_affine_constraint_dof(summary, dof) {
            continue;
        }
        let Some(address) = summary.structural_dof_layout.address(dof) else {
            continue;
        };
        let Some(component) = translational_component(address.kind) else {
            continue;
        };
        let target = address.node_index * VECTOR_COMPONENT_COUNT + component;
        if target >= reactions.len() {
            continue;
        }
        let rhs = summary.operator.load().get(dof).copied().unwrap_or(0.0);
        let internal = internal_force.get(dof).copied().unwrap_or(0.0);
        reactions[target] = internal - rhs;
    }
    reactions
}

fn recover_rotation(summary: &AssemblySummary, solution: &[f64]) -> Vec<f64> {
    let mut rotation = vec![0.0; rotation_shape(summary).iter().product()];
    if rotation.is_empty() {
        return rotation;
    }
    for row in 0..summary.structural_dof_layout.total_dof_count() {
        let Some(address) = summary.structural_dof_layout.address(row) else {
            continue;
        };
        let Some(component) = rotational_component(address.kind) else {
            continue;
        };
        let target = address.node_index * VECTOR_COMPONENT_COUNT + component;
        if target < rotation.len() {
            rotation[target] = solution.get(row).copied().unwrap_or(0.0);
        }
    }
    rotation
}

fn recover_displacement(
    summary: &AssemblySummary,
    solution: &[f64],
    node_count: usize,
) -> Vec<f64> {
    if summary.structural_rotational_dof_count == 0 {
        let mut displacement = solution.to_vec();
        displacement.resize(node_count * VECTOR_COMPONENT_COUNT, 0.0);
        return displacement;
    }

    let mut displacement = vec![0.0; node_count * VECTOR_COMPONENT_COUNT];
    for row in 0..summary.structural_dof_layout.total_dof_count() {
        let Some(address) = summary.structural_dof_layout.address(row) else {
            continue;
        };
        let Some(component) = translational_component(address.kind) else {
            continue;
        };
        let target = address.node_index * VECTOR_COMPONENT_COUNT + component;
        if target < displacement.len() {
            displacement[target] = solution.get(row).copied().unwrap_or(0.0);
        }
    }
    displacement
}

fn translational_component(kind: StructuralDofKind) -> Option<usize> {
    match kind {
        StructuralDofKind::Ux => Some(0),
        StructuralDofKind::Uy => Some(1),
        StructuralDofKind::Uz => Some(2),
        _ => None,
    }
}

fn recover_reaction_moment(summary: &AssemblySummary, internal_force: &[f64]) -> Vec<f64> {
    let mut reactions = vec![0.0; rotation_shape(summary).iter().product()];
    if reactions.is_empty() {
        return reactions;
    }
    for (dof, is_constrained) in summary.operator.constrained_dofs().iter().enumerate() {
        if !*is_constrained && !is_affine_constraint_dof(summary, dof) {
            continue;
        }
        let Some(address) = summary.structural_dof_layout.address(dof) else {
            continue;
        };
        let Some(component) = rotational_component(address.kind) else {
            continue;
        };
        let target = address.node_index * VECTOR_COMPONENT_COUNT + component;
        if target < reactions.len() {
            let rhs = summary.operator.load().get(dof).copied().unwrap_or(0.0);
            let internal = internal_force.get(dof).copied().unwrap_or(0.0);
            reactions[target] = internal - rhs;
        }
    }
    reactions
}

fn rotational_component(kind: StructuralDofKind) -> Option<usize> {
    match kind {
        StructuralDofKind::Rx => Some(0),
        StructuralDofKind::Ry => Some(1),
        StructuralDofKind::Rz => Some(2),
        _ => None,
    }
}

fn rotation_shape(summary: &AssemblySummary) -> Vec<usize> {
    if summary.structural_rotational_dof_count == 0 {
        return vec![0, VECTOR_COMPONENT_COUNT];
    }
    vec![
        summary.structural_dof_layout.node_count(),
        VECTOR_COMPONENT_COUNT,
    ]
}

fn structural_displacement_node_count(summary: &AssemblySummary, dof_count: usize) -> usize {
    if summary.structural_rotational_dof_count > 0 {
        summary.structural_node_count.max(1)
    } else {
        dof_count.div_ceil(VECTOR_COMPONENT_COUNT).max(1)
    }
}

fn recover_total_strain_energy(displacement: &[f64], internal_force: &[f64]) -> f64 {
    0.5 * displacement
        .iter()
        .zip(internal_force.iter())
        .map(|(u, force)| u * force)
        .sum::<f64>()
}

struct StructuralResidualMetrics {
    normalized_residual_norm: f64,
    equation_scale: f64,
}

fn recover_residual_metrics(
    summary: &AssemblySummary,
    displacement: &[f64],
) -> StructuralResidualMetrics {
    let mut solution = displacement.to_vec();
    solution.resize(summary.dof_count, 0.0);
    let applied = apply_k(&summary.operator, &solution);
    let mut full_residual = applied
        .iter()
        .zip(summary.operator.load().iter())
        .map(|(lhs, rhs)| lhs - rhs)
        .collect::<Vec<_>>();
    if summary.constraint_matrix.is_none() && summary.affine_constraint_reduction.is_none() {
        for (value, constrained) in full_residual
            .iter_mut()
            .zip(summary.operator.constrained_dofs())
        {
            if *constrained {
                *value = 0.0;
            }
        }
    }
    let residual = summary
        .affine_constraint_reduction
        .as_ref()
        .and_then(|reduction| reduction.transpose_apply(&full_residual).ok())
        .unwrap_or(full_residual);
    let scaled_rhs = summary
        .affine_constraint_reduction
        .as_ref()
        .and_then(|reduction| reduction.transpose_apply(summary.operator.load()).ok())
        .unwrap_or_else(|| summary.operator.load().to_vec());
    let residual_norm = residual
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    let equation_scale = scaled_rhs
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt()
        .max(1.0);
    StructuralResidualMetrics {
        normalized_residual_norm: residual_norm / equation_scale,
        equation_scale,
    }
}

fn is_affine_constraint_dof(summary: &AssemblySummary, dof: usize) -> bool {
    summary
        .constraint_matrix
        .as_ref()
        .is_some_and(|matrix| matrix.column_indices.contains(&dof))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        assembly::{assemble_structural_system, StructuralFormulationRequirement},
        excitation::{realize_step_excitations, ExcitationSamplePoint, RealizedExcitations},
        fixtures::{fixture_model, FixtureId},
    };
    use runmat_analysis_core::{
        AnalysisConstraint, CoordinateFrame, Displacement, PrescribedStructuralDofs,
        PrescribedValue, RegionTarget, StructuralConstraint, StructuralConstraintKind,
        StructuralTranslationDof, TranslationDofValue,
    };
    use runmat_meshing_core::{ElementOrder, SolverMeshArtifact};

    fn static_excitations(model: &runmat_analysis_core::AnalysisModel) -> RealizedExcitations {
        realize_step_excitations(
            model,
            &model.sequence.steps[0],
            ExcitationSamplePoint::Static,
        )
        .expect("field-recovery fixture excitations should realize")
    }

    #[test]
    fn synthetic_operator_recovery_does_not_publish_continuum_fields() {
        let model = fixture_model(FixtureId::CantileverLinearStatic);
        let summary = assemble_structural_system(
            &model,
            &static_excitations(&model),
            None,
            StructuralFormulationRequirement::ExperimentalSyntheticOperator,
        )
        .expect("the synthetic fixture operator should assemble");

        let fields = recover_deformation_result_fields(&summary, &vec![0.0; summary.dof_count])
            .expect("kinematic recovery should remain available");

        assert!(fields
            .iter()
            .any(|field| field.field_id == FEA_FIELD_STRUCTURAL_DISPLACEMENT));
        for field_id in [
            FEA_FIELD_STRUCTURAL_STRAIN,
            FEA_FIELD_STRUCTURAL_STRESS,
            FEA_FIELD_STRUCTURAL_VON_MISES,
            FEA_FIELD_STRUCTURAL_NODAL_VON_MISES,
            FEA_FIELD_STRUCTURAL_STRAIN_ENERGY_DENSITY,
        ] {
            assert!(
                fields.iter().all(|field| field.field_id != field_id),
                "synthetic topology cannot support physical field `{field_id}`"
            );
        }
    }

    #[test]
    fn solid_tetrahedron4_recovery_uses_solver_mesh_field_shapes() {
        let mut model = fixture_model(FixtureId::CantileverLinearStatic);
        model.excitations.clear();
        model.constraints.clear();
        let summary = assemble_structural_system(
            &model,
            &static_excitations(&model),
            Some(tetrahedron4_mesh()),
            StructuralFormulationRequirement::PhysicalDiscretization,
        )
        .expect("valid Tet4 system should assemble");
        let solve = LinearSolveResult {
            iterations: 1,
            residual_norm: 0.0,
            converged: true,
            host_sync_count: 0,
            solver_backend: crate::solve::backend::kind::LinearAlgebraBackendKind::CpuReference,
            device_apply_k_count: 0,
            device_apply_k_attempt_count: 0,
            accelerator_evidence: None,
            solution: vec![
                0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.02, 0.0, 0.0, 0.0, 0.03,
            ],
            solver_method: crate::contracts::SolverMethod::MatrixFreePcg,
            preconditioner: crate::contracts::SolverPreconditioner::None,
            diagnostics: Vec::new(),
        };

        let fields = recover_result_fields(&summary, &solve).unwrap();
        assert_eq!(
            field_shape(&fields, FEA_FIELD_STRUCTURAL_DISPLACEMENT),
            &[4, 3]
        );
        assert_eq!(field_shape(&fields, FEA_FIELD_STRUCTURAL_STRAIN), &[1, 6]);
        assert_eq!(field_shape(&fields, FEA_FIELD_STRUCTURAL_STRESS), &[1, 6]);
        assert_eq!(field_shape(&fields, FEA_FIELD_STRUCTURAL_VON_MISES), &[1]);
        assert_eq!(
            field_shape(&fields, FEA_FIELD_STRUCTURAL_STRAIN_ENERGY_DENSITY),
            &[1]
        );
        assert_eq!(
            field_shape(&fields, FEA_FIELD_STRUCTURAL_NODAL_VON_MISES),
            &[4]
        );
        assert!(host_field(&fields, FEA_FIELD_STRUCTURAL_STRAIN_ENERGY_DENSITY)[0] >= 0.0);
        let element_von_mises = host_field(&fields, FEA_FIELD_STRUCTURAL_VON_MISES)[0];
        assert!(host_field(&fields, FEA_FIELD_STRUCTURAL_NODAL_VON_MISES)
            .iter()
            .all(|value| (*value - element_von_mises).abs() <= 1.0e-12));
        let metrics = structural_field_recovery_metrics(&summary, &solve.solution).unwrap();
        assert_eq!(metrics.basis, "solid_tetrahedron4_constant_strain");
        assert_eq!(metrics.solver_mesh_node_count, 4);
        assert_eq!(metrics.solver_mesh_element_count, 1);
    }

    #[test]
    fn solid_tetrahedron10_recovery_uses_midside_displacements() {
        let mut model = fixture_model(FixtureId::CantileverLinearStatic);
        model.excitations.clear();
        model.constraints.clear();
        let mesh = crate::assembly::solver_solid::tests::artifact(ElementOrder::Tet10);
        let displacement = mesh
            .topology
            .nodes
            .iter()
            .flat_map(|node| [node.coordinates_m[0].powi(2), 0.0, 0.0])
            .collect::<Vec<_>>();
        let summary = assemble_structural_system(
            &model,
            &static_excitations(&model),
            Some(mesh),
            StructuralFormulationRequirement::PhysicalDiscretization,
        )
        .expect("valid Tet10 system should assemble");
        let solve = LinearSolveResult {
            iterations: 1,
            residual_norm: 0.0,
            converged: true,
            host_sync_count: 0,
            solver_backend: crate::solve::backend::kind::LinearAlgebraBackendKind::CpuReference,
            device_apply_k_count: 0,
            device_apply_k_attempt_count: 0,
            accelerator_evidence: None,
            solution: displacement,
            solver_method: crate::contracts::SolverMethod::MatrixFreePcg,
            preconditioner: crate::contracts::SolverPreconditioner::None,
            diagnostics: Vec::new(),
        };

        let fields = recover_result_fields(&summary, &solve).unwrap();
        let strain = host_field(&fields, FEA_FIELD_STRUCTURAL_STRAIN);
        assert_eq!(field_shape(&fields, FEA_FIELD_STRUCTURAL_STRAIN), &[1, 6]);
        assert!((strain[0] - 0.5).abs() < 1.0e-13);
        assert!(strain[1..].iter().all(|value| value.abs() < 1.0e-13));
        assert_eq!(
            summary.structural_formulations[0].recovery,
            Some(runmat_analysis_core::StructuralRecoveryImplementation::Tet10VolumeAverageStrain)
        );
        let metrics = structural_field_recovery_metrics(&summary, &solve.solution).unwrap();
        assert_eq!(metrics.basis, "solid_tetrahedron10_volume_average_strain");
        assert_eq!(metrics.solver_mesh_node_count, 10);
    }

    #[test]
    fn reaction_force_retains_the_constrained_node_and_component() {
        let mut model = fixture_model(FixtureId::CantileverLinearStatic);
        model.excitations.clear();
        model.constraints = vec![AnalysisConstraint::Structural(StructuralConstraint {
            constraint_id: "node_y".into(),
            frame: CoordinateFrame::Global,
            kind: StructuralConstraintKind::Prescribed(PrescribedStructuralDofs {
                target: RegionTarget::point("node:2"),
                translations: vec![TranslationDofValue {
                    dof: StructuralTranslationDof::Uy,
                    value: PrescribedValue::Constant(Displacement::new(0.0)),
                }],
                rotations: Vec::new(),
            }),
        })];
        let mesh = tetrahedron4_mesh();
        let node_index = mesh
            .topology
            .nodes
            .iter()
            .position(|node| node.node_id == 2)
            .expect("fixture contains node 2");
        let summary = assemble_structural_system(
            &model,
            &static_excitations(&model),
            Some(mesh),
            StructuralFormulationRequirement::PhysicalDiscretization,
        )
        .expect("component constraint should assemble");
        let constrained_dof = summary
            .structural_dof_layout
            .index(node_index, StructuralDofKind::Uy)
            .expect("node y DOF should exist");
        let mut internal_force = vec![0.0; summary.dof_count];
        internal_force[constrained_dof] = 42.0;

        let reactions = recover_reaction_force(
            &summary,
            &internal_force,
            summary.structural_dof_layout.node_count(),
        );

        assert_eq!(reactions.len(), 4 * VECTOR_COMPONENT_COUNT);
        assert_eq!(reactions[node_index * VECTOR_COMPONENT_COUNT + 1], 42.0);
        assert_eq!(
            reactions
                .iter()
                .enumerate()
                .filter(|(_, value)| **value != 0.0)
                .map(|(index, _)| index)
                .collect::<Vec<_>>(),
            vec![node_index * VECTOR_COMPONENT_COUNT + 1]
        );
    }

    #[test]
    fn finite_nonconverged_solve_still_recovers_solver_mesh_fields() {
        let mut model = fixture_model(FixtureId::CantileverLinearStatic);
        model.excitations.clear();
        model.constraints.clear();
        let summary = assemble_structural_system(
            &model,
            &static_excitations(&model),
            Some(tetrahedron4_mesh()),
            StructuralFormulationRequirement::PhysicalDiscretization,
        )
        .expect("valid Tet4 system should assemble");
        let solve = LinearSolveResult {
            iterations: 10,
            residual_norm: 1.0e-4,
            converged: false,
            host_sync_count: 0,
            solver_backend: crate::solve::backend::kind::LinearAlgebraBackendKind::CpuReference,
            device_apply_k_count: 0,
            device_apply_k_attempt_count: 0,
            accelerator_evidence: None,
            solution: vec![
                0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.02, 0.0, 0.0, 0.0, 0.03,
            ],
            solver_method: crate::contracts::SolverMethod::MatrixFreePcg,
            preconditioner: crate::contracts::SolverPreconditioner::None,
            diagnostics: Vec::new(),
        };

        let fields = recover_result_fields(&summary, &solve).unwrap();

        assert_eq!(field_shape(&fields, FEA_FIELD_STRUCTURAL_VON_MISES), &[1]);
        assert_eq!(field_shape(&fields, FEA_FIELD_STRUCTURAL_STRAIN), &[1, 6]);
        assert_eq!(
            field_shape(&fields, FEA_FIELD_STRUCTURAL_DISPLACEMENT),
            &[4, 3]
        );
    }

    #[test]
    fn observable_catalog_types_every_structural_deformation_family() {
        let definitions = structural_observable_field_definitions().unwrap();
        let expected = [
            (FEA_FIELD_STRUCTURAL_ROTATION, ObservationQuantity::Rotation),
            (
                FEA_FIELD_STRUCTURAL_BEAM_AXIAL_FORCE,
                ObservationQuantity::BeamAxialForce,
            ),
            (
                FEA_FIELD_STRUCTURAL_BEAM_SHEAR_FORCE,
                ObservationQuantity::BeamShearForce,
            ),
            (
                FEA_FIELD_STRUCTURAL_SHELL_MEMBRANE_FORCE,
                ObservationQuantity::ShellMembraneForce,
            ),
            (
                FEA_FIELD_STRUCTURAL_SHELL_BENDING_MOMENT,
                ObservationQuantity::ShellBendingMoment,
            ),
        ];
        for (field_id, quantity) in expected {
            assert!(definitions.iter().any(|definition| {
                definition.field_id() == field_id && definition.quantity() == quantity
            }));
        }
        let mut identities = definitions
            .iter()
            .map(|definition| definition.identity())
            .collect::<Vec<_>>();
        identities.sort();
        identities.dedup();
        assert_eq!(identities.len(), definitions.len());
    }

    fn field_shape<'a>(fields: &'a [AnalysisField], field_id: &str) -> &'a [usize] {
        fields
            .iter()
            .find(|field| field.field_id == field_id)
            .map(|field| field.shape.as_slice())
            .expect("field should be present")
    }

    fn host_field<'a>(fields: &'a [AnalysisField], field_id: &str) -> &'a [f64] {
        fields
            .iter()
            .find(|field| field.field_id == field_id)
            .and_then(AnalysisField::as_host_f64)
            .expect("field should be present")
    }

    fn tetrahedron4_mesh() -> SolverMeshArtifact {
        crate::assembly::solver_solid::tests::artifact(ElementOrder::Tet4)
    }
}
