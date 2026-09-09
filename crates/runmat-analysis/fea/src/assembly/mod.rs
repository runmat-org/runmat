pub mod dofs;
pub mod elements;
mod moment_projection;
mod reduced_rhs;
mod solid_matrix;
pub mod solver_solid;
mod solver_solid_boundary;
mod solver_solid_mass;
mod structural_inputs;

use std::{collections::BTreeMap, fmt};

use runmat_analysis_core::{
    AnalysisModel, BeamSectionModel, BoundaryTransferImplementation, ShellSectionModel,
    SourceImplementation, StructuralConstraintKind, StructuralElementKind,
    StructuralFormulationRealization, StructuralModel, StructuralOperatorImplementation,
    StructuralRecoveryImplementation, StructuralRotationDof, StructuralTranslationDof,
};
use runmat_meshing_core::{ElementOrder, SolverMeshArtifact};
use serde::{Deserialize, Serialize};

use self::elements::solid::SolidMaterial;
use self::{
    dofs::{StructuralDofKind, StructuralDofLayout, StructuralNodeDofSet},
    elements::beam::{
        global_stiffness_matrix as beam_global_stiffness_matrix, transformation_matrix,
        BeamElementError, BeamElementGeometry, BeamMaterial, BeamSection, BeamTransform12,
        BEAM_ELEMENT_DOF_COUNT,
    },
    elements::shell::{
        global_stiffness_matrix as shell_global_stiffness_matrix, ShellElementError,
        ShellElementGeometry, ShellMaterial, ShellSection, SHELL_ELEMENT_DOF_COUNT,
        SHELL_NODE_DOF_COUNT,
    },
    solver_solid::{
        assemble_solver_solid_stiffness_csr, solver_solid_topology, SolverSolidAssemblyError,
    },
    solver_solid_boundary::apply_solver_mesh_structural_loads,
    solver_solid_mass::assemble_solver_solid_mass,
    structural_inputs::{
        apply_synthetic_structural_rhs, explicit_load_kind, requested_moment_norm,
        structural_boundary_implementations, structural_constraint_kind_name,
        structural_source_implementations, validate_structural_inputs, StructuralProjectionBasis,
    },
};
use moment_projection::project_moment;
use reduced_rhs::apply_reduced_multiphysics_rhs;

use crate::constraints::{
    AffineConstraintReduction, CanonicalStructuralConstraintPlan, CanonicalStructuralDof,
    CanonicalStructuralDofOwner, ConstraintAlgebraError, ConstraintPlanError, ReactionRecovery,
    ResolvedConstraintMatrix,
};
use crate::excitation::{RealizedExcitationKind, RealizedExcitations};
use crate::operator::{CsrMatrix, OperatorSystem};
use crate::{FeaElectroThermalContext, FeaThermoMechanicalContext};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AssemblySummary {
    pub dof_count: usize,
    #[serde(default)]
    pub structural_node_count: usize,
    #[serde(default)]
    pub structural_translational_dof_count: usize,
    #[serde(default)]
    pub structural_rotational_dof_count: usize,
    #[serde(default)]
    pub structural_rotation_node_count: usize,
    #[serde(default)]
    pub structural_moment_load_count: usize,
    #[serde(default)]
    pub structural_direct_rotational_moment_load_count: usize,
    #[serde(default)]
    pub structural_requested_moment_norm_n_m: f64,
    #[serde(default)]
    pub structural_moment_projection: Vec<MomentProjectionSummary>,
    #[serde(default)]
    pub spatial_projection_evidence: Vec<crate::excitation::SpatialProjectionEvidence>,
    #[serde(default)]
    pub structural_rotational_constraint_count: usize,
    #[serde(default)]
    pub structural_beam_element_count: usize,
    #[serde(default)]
    pub structural_shell_element_count: usize,
    #[serde(default)]
    pub structural_solid_element_count: usize,
    #[serde(default)]
    pub structural_solid_total_mass_kg: f64,
    #[serde(default)]
    pub structural_solid_recovery: Vec<SolidRecoveryElementSummary>,
    #[serde(default)]
    pub structural_dof_layout: StructuralDofLayout,
    #[serde(default)]
    pub structural_beam_recovery: Vec<BeamRecoveryElementSummary>,
    #[serde(default)]
    pub structural_shell_recovery: Vec<ShellRecoveryElementSummary>,
    #[serde(default)]
    pub structural_formulation: Option<StructuralFormulationRealization>,
    #[serde(default)]
    pub structural_sources: Vec<SourceImplementation>,
    #[serde(default)]
    pub structural_boundary_transfers: Vec<BoundaryTransferImplementation>,
    pub constrained_dof_count: usize,
    #[serde(default)]
    pub constraint_matrix: Option<ResolvedConstraintMatrix>,
    #[serde(default)]
    pub affine_constraint_reduction: Option<AffineConstraintReduction>,
    pub load_count: usize,
    pub structural_material: StructuralMaterialSummary,
    pub thermo_mechanical: Option<ThermoMechanicalAssemblySummary>,
    pub electro_thermal: Option<ElectroThermalAssemblySummary>,
    pub operator: OperatorSystem,
}

#[derive(Debug, Clone, PartialEq)]
pub enum LinearAssemblyError {
    ConstraintPlan(ConstraintPlanError),
    ConstraintAlgebra(ConstraintAlgebraError),
    Structural(StructuralAssemblyError),
    SolidStiffness(SolverSolidAssemblyError),
    SolverMeshRegionMapping(SolverMeshRegionMappingError),
}

impl fmt::Display for LinearAssemblyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LinearAssemblyError::ConstraintPlan(err) => {
                write!(f, "constraint planning failed: {err}")
            }
            LinearAssemblyError::ConstraintAlgebra(err) => {
                write!(f, "constraint algebra preparation failed: {err}")
            }
            LinearAssemblyError::Structural(err) => write!(f, "{err}"),
            LinearAssemblyError::SolidStiffness(err) => {
                write!(f, "solid stiffness assembly failed: {err:?}")
            }
            LinearAssemblyError::SolverMeshRegionMapping(err) => {
                write!(f, "{err}")
            }
        }
    }
}

impl std::error::Error for LinearAssemblyError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StructuralLoadKind {
    Force,
    Moment,
    Traction,
    Pressure,
    BodyForce,
    BaseDisplacement,
    BaseVelocity,
    BaseAcceleration,
    HeatGeneration,
    HeatFlux,
    TemperatureField,
    AmbientTemperature,
    TerminalVoltage,
    ElectricCurrent,
    ElectricCharge,
    CurrentDensity,
    MagneticVectorPotential,
    AcousticPressure,
    NormalVelocity,
    InletVelocity,
    MassFlow,
    WallMotion,
}

impl StructuralLoadKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Force => "force",
            Self::Moment => "moment",
            Self::Traction => "traction",
            Self::Pressure => "pressure",
            Self::BodyForce => "body_force",
            Self::BaseDisplacement => "base_displacement",
            Self::BaseVelocity => "base_velocity",
            Self::BaseAcceleration => "base_acceleration",
            Self::HeatGeneration => "heat_generation",
            Self::HeatFlux => "heat_flux",
            Self::TemperatureField => "temperature_field",
            Self::AmbientTemperature => "ambient_temperature",
            Self::TerminalVoltage => "terminal_voltage",
            Self::ElectricCurrent => "electric_current",
            Self::ElectricCharge => "electric_charge",
            Self::CurrentDensity => "current_density",
            Self::MagneticVectorPotential => "magnetic_vector_potential",
            Self::AcousticPressure => "acoustic_pressure",
            Self::NormalVelocity => "normal_velocity",
            Self::InletVelocity => "inlet_velocity",
            Self::MassFlow => "mass_flow",
            Self::WallMotion => "wall_motion",
        }
    }
}

impl fmt::Display for StructuralLoadKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StructuralAssemblyError {
    MissingDiscretization,
    MixedSolidAndExplicit {
        explicit_element_count: usize,
    },
    MixedExplicitDimensions {
        beam_element_count: usize,
        shell_element_count: usize,
    },
    MissingNodes,
    DuplicateNodeId {
        node_id: u32,
    },
    DuplicateElementId {
        element_id: String,
    },
    UnknownNode {
        element_id: String,
        node_id: u32,
    },
    UnknownBeamSection {
        element_id: String,
        section_id: String,
    },
    UnknownShellSection {
        element_id: String,
        section_id: String,
    },
    BeamElement {
        element_id: String,
        error: BeamElementError,
    },
    ShellElement {
        element_id: String,
        error: ShellElementError,
    },
    UnsupportedStructuralLoad {
        load_id: String,
        load_kind: StructuralLoadKind,
    },
    UnmappedStructuralLoad {
        load_id: String,
        region_id: String,
        load_kind: StructuralLoadKind,
    },
    UnsupportedStructuralConstraint {
        constraint_id: String,
        constraint_kind: &'static str,
    },
    UnmappedExplicitConstraint {
        constraint_id: String,
        region_id: String,
    },
    UnsupportedScheduledConstraint {
        constraint_id: String,
    },
    UnsupportedConstraintFrame {
        constraint_id: String,
        frame: String,
    },
    ConstraintPlanNodeAbsent,
}

impl fmt::Display for StructuralAssemblyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingDiscretization => write!(
                f,
                "structural analysis requires a canonical solid mesh or explicit beam/shell connectivity"
            ),
            Self::MixedSolidAndExplicit {
                explicit_element_count,
            } => write!(
                f,
                "solid and explicit beam/shell discretizations cannot be assembled together until typed coupling is implemented: explicit_element_count={explicit_element_count}"
            ),
            Self::MixedExplicitDimensions {
                beam_element_count,
                shell_element_count,
            } => write!(
                f,
                "mixed beam and shell connectivity requires an explicit coupling formulation: beam_element_count={beam_element_count} shell_element_count={shell_element_count}"
            ),
            Self::MissingNodes => write!(
                f,
                "explicit structural connectivity requires at least one structural node"
            ),
            Self::DuplicateNodeId { node_id } => {
                write!(f, "explicit structural node id is duplicated: node_id={node_id}")
            }
            Self::DuplicateElementId { element_id } => write!(
                f,
                "explicit structural element id is duplicated: element_id={element_id}"
            ),
            Self::UnknownNode {
                element_id,
                node_id,
            } => write!(
                f,
                "explicit structural element references an unknown node: element_id={element_id} node_id={node_id}"
            ),
            Self::UnknownBeamSection {
                element_id,
                section_id,
            } => write!(
                f,
                "beam element references an unknown section: element_id={element_id} section_id={section_id}"
            ),
            Self::UnknownShellSection {
                element_id,
                section_id,
            } => write!(
                f,
                "shell element references an unknown section: element_id={element_id} section_id={section_id}"
            ),
            Self::BeamElement { element_id, error } => {
                write!(f, "beam element assembly failed: element_id={element_id} error={error}")
            }
            Self::ShellElement { element_id, error } => {
                write!(f, "shell element assembly failed: element_id={element_id} error={error}")
            }
            Self::UnsupportedStructuralLoad { load_id, load_kind } => {
                if *load_kind == StructuralLoadKind::Moment {
                    write!(
                        f,
                        "moment loads require rotational-DOF structural elements; use a wrench load for equivalent force-couple moments on solid-only face regions: load_id={load_id}"
                    )
                } else {
                    write!(
                        f,
                        "the selected structural formulation does not implement this load: load_id={load_id} load_kind={load_kind}"
                    )
                }
            }
            Self::UnmappedStructuralLoad {
                load_id,
                region_id,
                load_kind,
            } => write!(
                f,
                "structural load target does not resolve to the selected discretization: load_id={load_id} region_id={region_id} load_kind={load_kind}"
            ),
            Self::UnsupportedStructuralConstraint {
                constraint_id,
                constraint_kind,
            } => write!(
                f,
                "the selected structural formulation does not implement this constraint: constraint_id={constraint_id} constraint_kind={constraint_kind}"
            ),
            Self::UnmappedExplicitConstraint {
                constraint_id,
                region_id,
            } => write!(
                f,
                "explicit structural constraint target does not resolve to connectivity: constraint_id={constraint_id} region_id={region_id}"
            ),
            Self::UnsupportedScheduledConstraint { constraint_id } => write!(
                f,
                "structural assembly requires study-constant prescribed constraints: constraint_id={constraint_id}"
            ),
            Self::UnsupportedConstraintFrame {
                constraint_id,
                frame,
            } => write!(
                f,
                "structural constraint frame is not implemented by the selected formulation: constraint_id={constraint_id} frame={frame}"
            ),
            Self::ConstraintPlanNodeAbsent => write!(
                f,
                "canonical constraint plan references a node absent from the solver mesh"
            ),
        }
    }
}

impl std::error::Error for StructuralAssemblyError {}

/// The structural formulation that an assembly caller is willing to admit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StructuralFormulationRequirement {
    PhysicalDiscretization,
    ExperimentalSyntheticOperator,
}

#[derive(Debug, Clone, PartialEq)]
pub enum SolverMeshRegionMappingError {
    InvalidAnalysisProjection(String),
    UnmappedLoadRegion {
        load_id: String,
        region_id: String,
        load_kind: StructuralLoadKind,
    },
    UncoupledReferencePoint {
        load_id: String,
        reference_point_id: String,
    },
    ReferencePointPositionMismatch {
        load_id: String,
        reference_point_id: String,
    },
    UnsupportedReferencePointLoad {
        load_id: String,
        load_kind: StructuralLoadKind,
    },
    ProjectionConservation {
        load_id: String,
        resultant_relative_error: f64,
        moment_relative_error: f64,
        relative_tolerance: f64,
    },
}

impl fmt::Display for SolverMeshRegionMappingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SolverMeshRegionMappingError::InvalidAnalysisProjection(reason) => {
                write!(f, "solver mesh analysis projection failed: {reason}")
            }
            SolverMeshRegionMappingError::UnmappedLoadRegion {
                load_id,
                region_id,
                load_kind,
            } => write!(
                f,
                "solver mesh load region did not resolve to persistent boundary provenance: load_id={load_id} region_id={region_id} load_kind={load_kind}"
            ),
            SolverMeshRegionMappingError::UncoupledReferencePoint {
                load_id,
                reference_point_id,
            } => write!(
                f,
                "reference-point load has no rigid or distributing coupling: load_id={load_id} reference_point_id={reference_point_id}"
            ),
            SolverMeshRegionMappingError::ReferencePointPositionMismatch {
                load_id,
                reference_point_id,
            } => write!(
                f,
                "reference-point load position differs from its coupling: load_id={load_id} reference_point_id={reference_point_id}"
            ),
            SolverMeshRegionMappingError::UnsupportedReferencePointLoad { load_id, load_kind } => write!(
                f,
                "load kind is not supported on a structural reference point: load_id={load_id} load_kind={load_kind}"
            ),
            SolverMeshRegionMappingError::ProjectionConservation {
                load_id,
                resultant_relative_error,
                moment_relative_error,
                relative_tolerance,
            } => write!(
                f,
                "spatial load projection violates its conservation requirement: load_id={load_id} resultant_relative_error={resultant_relative_error:e} moment_relative_error={moment_relative_error:e} relative_tolerance={relative_tolerance:e}"
            ),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct StructuralMaterialSummary {
    pub youngs_modulus_pa: f64,
    pub poisson_ratio: f64,
    #[serde(default)]
    pub density_kg_per_m3: f64,
    pub lame_lambda_pa: f64,
    pub shear_modulus_pa: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BeamRecoveryElementSummary {
    pub element_id: String,
    pub region_id: String,
    pub node_i_index: usize,
    pub node_j_index: usize,
    pub length_m: f64,
    pub section: BeamSection,
    pub material: BeamMaterial,
    pub transform_global_to_local: BeamTransform12,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ShellRecoveryElementSummary {
    pub element_id: String,
    pub region_id: String,
    pub node_indices: [usize; 3],
    pub area_m2: f64,
    pub section: ShellSection,
    pub material: ShellMaterial,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SolidRecoveryElementSummary {
    pub element_id: String,
    pub region_id: String,
    pub node_indices: [usize; 4],
    pub coordinates_m: [[f64; 3]; 4],
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MomentProjectionSummary {
    pub load_id: String,
    pub region_id: String,
    pub target_node_count: usize,
    pub projection_point_m: [f64; 3],
    pub applied_force: [f64; 3],
    pub applied_moment: [f64; 3],
    pub force_residual: [f64; 3],
    pub moment_residual: [f64; 3],
    pub projection_applied: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ThermoMechanicalAssemblySummary {
    pub enabled: bool,
    pub reference_temperature_k: f64,
    pub applied_temperature_delta_k: f64,
    pub thermal_expansion_coefficient: f64,
    pub thermal_strain_scale: f64,
    pub thermal_load_scale: f64,
    pub constitutive_temperature_factor: f64,
    pub constitutive_poisson_coupling: f64,
    pub effective_modulus_scale: f64,
    pub constitutive_material_spread_ratio: f64,
    pub assignment_heterogeneity_index: f64,
    pub spatial_gradient_index: f64,
    pub spatial_coverage_ratio: f64,
    pub region_delta_count: usize,
    pub coupling_fingerprint: u64,
    pub profile_sources: Vec<SourceImplementation>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ElectroThermalAssemblySummary {
    pub enabled: bool,
    pub reference_temperature_k: f64,
    pub applied_voltage_v: f64,
    pub base_electrical_conductivity_s_per_m: f64,
    pub resistive_heating_coefficient: f64,
    pub joule_heating_scale: f64,
    pub conductivity_spread_ratio: f64,
    pub region_scale_count: usize,
    pub coupling_fingerprint: u64,
}

/// Internal reduced operator used by non-structural experimental pipelines.
///
/// Structural callers must use [`assemble_structural_system`] so synthetic admission is explicit.
pub(crate) fn assemble_linear_system(
    model: &AnalysisModel,
    excitations: &RealizedExcitations,
    solver_mesh: Option<SolverMeshArtifact>,
    thermo_mechanical_context: Option<FeaThermoMechanicalContext>,
    electro_thermal_context: Option<FeaElectroThermalContext>,
) -> Result<AssemblySummary, LinearAssemblyError> {
    assemble_linear_system_impl(
        model,
        excitations,
        solver_mesh,
        thermo_mechanical_context,
        electro_thermal_context,
        None,
        None,
    )
}

pub(crate) fn assemble_structural_system(
    model: &AnalysisModel,
    excitations: &RealizedExcitations,
    solver_mesh: Option<SolverMeshArtifact>,
    thermo_mechanical_context: Option<FeaThermoMechanicalContext>,
    electro_thermal_context: Option<FeaElectroThermalContext>,
    requirement: StructuralFormulationRequirement,
) -> Result<AssemblySummary, LinearAssemblyError> {
    assemble_linear_system_impl(
        model,
        excitations,
        solver_mesh,
        thermo_mechanical_context,
        electro_thermal_context,
        Some(requirement),
        None,
    )
}

pub(crate) fn assemble_structural_system_with_constraint_plan(
    model: &AnalysisModel,
    excitations: &RealizedExcitations,
    solver_mesh: SolverMeshArtifact,
    thermo_mechanical_context: Option<FeaThermoMechanicalContext>,
    electro_thermal_context: Option<FeaElectroThermalContext>,
    requirement: StructuralFormulationRequirement,
    constraint_plan: &CanonicalStructuralConstraintPlan,
) -> Result<AssemblySummary, LinearAssemblyError> {
    assemble_linear_system_impl(
        model,
        excitations,
        Some(solver_mesh),
        thermo_mechanical_context,
        electro_thermal_context,
        Some(requirement),
        Some(constraint_plan),
    )
}

fn assemble_linear_system_impl(
    model: &AnalysisModel,
    excitations: &RealizedExcitations,
    solver_mesh: Option<SolverMeshArtifact>,
    thermo_mechanical_context: Option<FeaThermoMechanicalContext>,
    electro_thermal_context: Option<FeaElectroThermalContext>,
    structural_requirement: Option<StructuralFormulationRequirement>,
    supplied_constraint_plan: Option<&CanonicalStructuralConstraintPlan>,
) -> Result<AssemblySummary, LinearAssemblyError> {
    let explicit_element_count = model
        .structural
        .as_ref()
        .map(|structural| structural.elements.len())
        .unwrap_or(0);
    if structural_requirement.is_some() && solver_mesh.is_some() && explicit_element_count > 0 {
        return Err(LinearAssemblyError::Structural(
            StructuralAssemblyError::MixedSolidAndExplicit {
                explicit_element_count,
            },
        ));
    }
    if structural_requirement.is_some() && solver_mesh.is_none() && explicit_element_count > 0 {
        if supplied_constraint_plan.is_some() {
            return Err(LinearAssemblyError::ConstraintPlan(
                ConstraintPlanError::Invalid(
                    "a solid constraint plan cannot be applied to explicit connectivity".into(),
                ),
            ));
        }
        return assemble_explicit_structural_system(model, excitations)
            .map_err(LinearAssemblyError::Structural);
    }
    if solver_mesh.is_none()
        && structural_requirement == Some(StructuralFormulationRequirement::PhysicalDiscretization)
    {
        return Err(LinearAssemblyError::Structural(
            StructuralAssemblyError::MissingDiscretization,
        ));
    }

    let base_dof_count = (excitations.len() * 3).max(3);
    let solid_topology = solver_mesh
        .as_ref()
        .map(|mesh| solver_solid_topology(mesh, base_dof_count))
        .transpose()
        .map_err(LinearAssemblyError::SolidStiffness)?;
    if structural_requirement.is_some() {
        validate_structural_inputs(
            model,
            excitations,
            if solid_topology.is_some() {
                StructuralProjectionBasis::CanonicalSolid
            } else {
                StructuralProjectionBasis::Synthetic
            },
        )
        .map_err(LinearAssemblyError::Structural)?;
    }
    let compiled_constraint_plan = match (
        structural_requirement,
        supplied_constraint_plan,
        solver_mesh.as_ref(),
    ) {
        (Some(_), None, Some(mesh)) => Some(
            CanonicalStructuralConstraintPlan::compile(model, mesh)
                .map_err(LinearAssemblyError::ConstraintPlan)?,
        ),
        _ => None,
    };
    let constraint_plan = supplied_constraint_plan.or(compiled_constraint_plan.as_ref());
    if let (Some(plan), Some(mesh)) = (constraint_plan, solver_mesh.as_ref()) {
        plan.validate_against(model, mesh)
            .map_err(LinearAssemblyError::ConstraintPlan)?;
    }
    let mesh_dof_count = solid_topology
        .as_ref()
        .map(|topology| topology.dof_count)
        .unwrap_or(base_dof_count);
    let reference_point_count = constraint_plan
        .map(|plan| plan.reference_points.len())
        .unwrap_or(0);
    let dof_count = mesh_dof_count + reference_point_count * 6;
    let structural_solid_recovery = solver_mesh
        .as_ref()
        .map(solid_recovery_from_solver_mesh)
        .unwrap_or_default();

    let avg_youngs_modulus = if model.materials.is_empty() {
        1.0e9
    } else {
        model
            .materials
            .iter()
            .map(|material| material.mechanical.youngs_modulus_pa.max(1.0))
            .sum::<f64>()
            / model.materials.len() as f64
    };
    let avg_poisson_ratio = if model.materials.is_empty() {
        0.3
    } else {
        model
            .materials
            .iter()
            .map(|material| material.mechanical.poisson_ratio.clamp(0.0, 0.49))
            .sum::<f64>()
            / model.materials.len() as f64
    };
    let avg_reference_temperature_k = if model.materials.is_empty() {
        293.15
    } else {
        model
            .materials
            .iter()
            .map(|material| material.thermal.reference_temperature_k)
            .sum::<f64>()
            / model.materials.len() as f64
    };
    let shear_modulus_pa = avg_youngs_modulus / (2.0 * (1.0 + avg_poisson_ratio)).max(1.0e-9);
    let avg_density_kg_per_m3 = if model.materials.is_empty() {
        7850.0
    } else {
        model
            .materials
            .iter()
            .map(|material| material.mechanical.density_kg_per_m3.max(1.0))
            .sum::<f64>()
            / model.materials.len() as f64
    };
    let lame_lambda_pa = avg_youngs_modulus * avg_poisson_ratio
        / ((1.0 + avg_poisson_ratio) * (1.0 - 2.0 * avg_poisson_ratio)).max(1.0e-9);
    let structural_material = StructuralMaterialSummary {
        youngs_modulus_pa: avg_youngs_modulus,
        poisson_ratio: avg_poisson_ratio,
        density_kg_per_m3: avg_density_kg_per_m3,
        lame_lambda_pa,
        shear_modulus_pa,
    };
    let stiffness_base = (avg_youngs_modulus / 2.0e3).max(1.0e5);

    let mut stiffness_diag = vec![0.0; dof_count];
    let mut stiffness_upper = vec![0.0; dof_count.saturating_sub(1)];
    let mut mass_diag = vec![0.0; dof_count];
    let mut damping_diag = vec![0.0; dof_count];
    if solid_topology.is_none() {
        for i in 0..mesh_dof_count {
            let factor = 1.0 + (i as f64) * 0.05;
            stiffness_diag[i] = stiffness_base * factor;
            mass_diag[i] = 1.0 + (i as f64) * 0.01;
            damping_diag[i] = 0.05 * factor;
        }
    }

    let mut rhs = vec![0.0; dof_count];
    if structural_requirement
        == Some(StructuralFormulationRequirement::ExperimentalSyntheticOperator)
    {
        apply_synthetic_structural_rhs(excitations, &mut rhs);
    } else if structural_requirement.is_none() {
        apply_reduced_multiphysics_rhs(excitations, &mut rhs);
    }

    let structural_dof_layout = StructuralDofLayout::reduced_translational_rows(mesh_dof_count)
        .with_reference_points(reference_point_count);
    let structural_moment_load_count = excitations
        .iter()
        .filter(|load| matches!(load.kind, RealizedExcitationKind::Moment(_)))
        .count();
    let structural_requested_moment_norm_n_m = requested_moment_norm(excitations);
    let structural_direct_rotational_moment_load_count =
        if structural_dof_layout.has_rotational_dofs() {
            structural_moment_load_count
        } else {
            0
        };

    let reduced_constraint_count = if structural_requirement.is_some() {
        model
            .constraints
            .iter()
            .filter(|constraint| constraint.structural().is_some())
            .count()
    } else {
        model.constraints.len()
    };
    let reduced_constrained_dof_count = reduced_constraint_count.min(dof_count);
    let mut constrained = vec![false; dof_count];
    let constraint_offset = 0;
    for idx in 0..reduced_constrained_dof_count {
        let dof = (constraint_offset + idx * 2) % dof_count.max(1);
        constrained[dof] = true;
        rhs[dof] = 0.0;
    }
    let mut structural_moment_projection = Vec::new();
    let mut spatial_projection_evidence = Vec::new();
    let mut constraint_matrix = None;
    let mut affine_constraint_reduction = None;
    if structural_requirement.is_some() {
        if let (Some(mesh), Some(_)) = (solver_mesh.as_ref(), solid_topology.as_ref()) {
            let constraint_plan = constraint_plan.ok_or_else(|| {
                LinearAssemblyError::ConstraintPlan(ConstraintPlanError::Invalid(
                    "canonical solid assembly has no compiled constraint plan".into(),
                ))
            })?;
            let source_projection = apply_solver_mesh_structural_loads(
                excitations,
                mesh,
                constraint_plan,
                &structural_dof_layout,
                &mut rhs,
                true,
            )
            .map_err(LinearAssemblyError::SolverMeshRegionMapping)?;
            structural_moment_projection = source_projection.moment_projection;
            spatial_projection_evidence = source_projection.evidence;
            constrained.fill(false);
            if constraint_plan.rows.iter().any(|row| row.terms.len() > 1) {
                let matrix =
                    resolve_solid_constraint_matrix(constraint_plan, mesh, &structural_dof_layout)
                        .map_err(LinearAssemblyError::ConstraintAlgebra)?;
                if constraint_plan.enforcement
                    != crate::constraints::ConstraintEnforcement::LagrangeMultiplier
                {
                    affine_constraint_reduction = Some(
                        matrix
                            .affine_reduction()
                            .map_err(LinearAssemblyError::ConstraintAlgebra)?,
                    );
                }
                constraint_matrix = Some(matrix);
            } else {
                apply_solid_constraint_plan(
                    constraint_plan,
                    mesh,
                    &structural_dof_layout,
                    &mut constrained,
                    &mut rhs,
                )
                .map_err(LinearAssemblyError::Structural)?;
            }
        }
    }

    let mut thermo_mechanical = None;
    let mut electro_thermal = None;
    for i in 0..stiffness_upper.len() {
        let coupling = 0.05 * stiffness_diag[i].min(stiffness_diag[i + 1]);
        stiffness_upper[i] = if constrained[i] || constrained[i + 1] {
            0.0
        } else {
            coupling
        };
    }

    if let Some(context) = thermo_mechanical_context {
        let thermal_strain_scale = (context.thermal_expansion_coefficient
            * context.applied_temperature_delta_k.abs())
        .clamp(0.0, 0.05);
        let thermal_load_scale = (context.applied_temperature_delta_k / 50.0).clamp(-2.0, 2.0);
        let constitutive_temperature_factor = if model.materials.is_empty() {
            (-(2.5e-4) * context.applied_temperature_delta_k).clamp(-0.25, 0.25)
        } else {
            let response = model
                .materials
                .iter()
                .map(|material| {
                    let adjusted_delta = context.applied_temperature_delta_k
                        + (context.reference_temperature_k
                            - material.thermal.reference_temperature_k)
                        + (avg_reference_temperature_k - material.thermal.reference_temperature_k)
                            * 0.1;
                    material.thermal.modulus_temp_coeff_per_k * adjusted_delta
                })
                .sum::<f64>()
                / model.materials.len() as f64;
            response.clamp(-0.25, 0.25)
        };
        let constitutive_poisson_coupling =
            (0.6 + avg_poisson_ratio.clamp(0.0, 0.49)).clamp(0.6, 1.2);
        let modulus_temperature_scale = (1.0
            + constitutive_temperature_factor * constitutive_poisson_coupling)
            .clamp(0.72, 1.15);
        let thermal_stiffening_scale = (1.0 + 0.35 * thermal_strain_scale).clamp(1.0, 1.06);
        let effective_modulus_scale =
            (modulus_temperature_scale * thermal_stiffening_scale).clamp(0.75, 1.2);
        let mut dof_adjustments = vec![0.0_f64; dof_count];
        let assignment_heterogeneity_index = apply_thermo_material_heterogeneity(
            model,
            dof_count,
            constitutive_temperature_factor,
            context.reference_temperature_k,
            context.applied_temperature_delta_k,
            &mut dof_adjustments,
        );
        let spatial_field = apply_thermo_spatial_field(&context, dof_count, &mut dof_adjustments);
        let mut local_modulus_scales = vec![effective_modulus_scale; dof_count];
        for i in 0..mesh_dof_count {
            let thermal_bias = 1.0 + thermal_strain_scale * (1.0 + (i % 3) as f64 * 0.1);
            let local_scale =
                (effective_modulus_scale * (1.0 + dof_adjustments[i])).clamp(0.75, 1.2);
            local_modulus_scales[i] = local_scale;
            stiffness_diag[i] *= thermal_bias * local_scale;
            if !constrained[i] {
                rhs[i] += thermal_load_scale * (1.0 + (i % 5) as f64 * 0.05);
            }
        }
        for i in 0..stiffness_upper.len() {
            let edge_scale = 0.5 * (local_modulus_scales[i] + local_modulus_scales[i + 1]);
            stiffness_upper[i] *= edge_scale;
        }
        let min_modulus_scale = local_modulus_scales
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
        let max_modulus_scale = local_modulus_scales.iter().copied().fold(0.0_f64, f64::max);
        let constitutive_material_spread_ratio =
            if min_modulus_scale.is_finite() && min_modulus_scale > 0.0 {
                max_modulus_scale / min_modulus_scale
            } else {
                1.0
            };
        thermo_mechanical = Some(ThermoMechanicalAssemblySummary {
            enabled: true,
            reference_temperature_k: context.reference_temperature_k,
            applied_temperature_delta_k: context.applied_temperature_delta_k,
            thermal_expansion_coefficient: context.thermal_expansion_coefficient,
            thermal_strain_scale,
            thermal_load_scale,
            constitutive_temperature_factor,
            constitutive_poisson_coupling,
            effective_modulus_scale,
            constitutive_material_spread_ratio,
            assignment_heterogeneity_index,
            spatial_gradient_index: spatial_field.gradient_index,
            spatial_coverage_ratio: spatial_field.coverage_ratio,
            region_delta_count: context.region_temperature_deltas.len(),
            coupling_fingerprint: thermo_mechanical_fingerprint(
                &context,
                ThermoMechanicalFingerprintInputs {
                    dof_count,
                    constitutive_temperature_factor,
                    constitutive_poisson_coupling,
                    effective_modulus_scale,
                    constitutive_material_spread_ratio,
                    assignment_heterogeneity_index,
                    spatial_gradient_index: spatial_field.gradient_index,
                },
            ),
            profile_sources: thermomechanical_profile_sources(&context),
        });
    }

    if let Some(context) = electro_thermal_context {
        let conductivity_scales = vec![1.0_f64; dof_count];
        let min_scale = conductivity_scales
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min)
            .max(1.0e-6);
        let max_scale = conductivity_scales.iter().copied().fold(0.0_f64, f64::max);
        let conductivity_spread_ratio = (max_scale / min_scale).clamp(1.0, 8.0);
        let joule_heating_scale = (context.applied_voltage_v.powi(2)
            * context.base_electrical_conductivity_s_per_m.max(1.0e-9)
            * context.resistive_heating_coefficient.max(0.0)
            / 1.0e6)
            .clamp(0.0, 10.0);

        for i in 0..mesh_dof_count {
            let local = conductivity_scales[i];
            damping_diag[i] *= (1.0 + 0.02 * local).clamp(1.0, 1.1);
            if !constrained[i] {
                rhs[i] += joule_heating_scale * local * (1.0 + (i % 7) as f64 * 0.01);
            }
        }

        electro_thermal = Some(ElectroThermalAssemblySummary {
            enabled: true,
            reference_temperature_k: context.reference_temperature_k,
            applied_voltage_v: context.applied_voltage_v,
            base_electrical_conductivity_s_per_m: context.base_electrical_conductivity_s_per_m,
            resistive_heating_coefficient: context.resistive_heating_coefficient,
            joule_heating_scale,
            conductivity_spread_ratio,
            region_scale_count: 0,
            coupling_fingerprint: electro_thermal_fingerprint(
                &context,
                dof_count,
                joule_heating_scale,
                conductivity_spread_ratio,
            ),
        });
    }

    let default_solid_material = (model.materials.len() == 1).then_some(SolidMaterial {
        youngs_modulus_pa: structural_material.youngs_modulus_pa,
        poisson_ratio: structural_material.poisson_ratio,
        density_kg_per_m3: structural_material.density_kg_per_m3,
    });
    let solid_materials_by_region = solid_materials_by_region(model);
    let mut stiffness_csr = match solver_mesh.as_ref() {
        Some(mesh) => Some(
            assemble_solver_solid_stiffness_csr(
                mesh,
                default_solid_material,
                &solid_materials_by_region,
                base_dof_count,
            )
            .map_err(LinearAssemblyError::SolidStiffness)?,
        ),
        None => None,
    };
    let mut structural_solid_total_mass_kg = 0.0;
    let mut mass_csr = match solver_mesh.as_ref() {
        Some(mesh) => {
            let assembled = assemble_solver_solid_mass(
                mesh,
                default_solid_material,
                &solid_materials_by_region,
                base_dof_count,
            )
            .map_err(LinearAssemblyError::SolidStiffness)?;
            structural_solid_total_mass_kg = assembled.total_mass_kg;
            mass_diag[..mesh_dof_count].copy_from_slice(&assembled.lumped_diag[..mesh_dof_count]);
            Some(assembled.consistent)
        }
        None => None,
    };
    if let Some(csr) = stiffness_csr.as_mut() {
        let terminal_offset = csr.row_offsets.last().copied().unwrap_or(0);
        csr.row_offsets.resize(dof_count + 1, terminal_offset);
    }
    if let Some(csr) = mass_csr.as_mut() {
        let terminal_offset = csr.row_offsets.last().copied().unwrap_or(0);
        csr.row_offsets.resize(dof_count + 1, terminal_offset);
    }
    if let Some(csr) = stiffness_csr.as_ref() {
        apply_csr_constraints(csr, &constrained, &mut rhs, dof_count);
        for (i, diagonal) in stiffness_diag.iter_mut().enumerate().take(dof_count) {
            if i >= mesh_dof_count {
                *diagonal = 0.0;
                continue;
            }
            let start = csr.row_offsets[i];
            let end = csr.row_offsets[i + 1];
            *diagonal = csr.column_indices[start..end]
                .iter()
                .zip(csr.values[start..end].iter())
                .find_map(|(&column, &value)| (column == i).then_some(value.abs()))
                .unwrap_or(1.0e-12)
                .max(1.0e-12);
        }
        stiffness_upper.fill(0.0);
    }

    let constrained_dof_count = affine_constraint_reduction
        .as_ref()
        .map(|reduction| reduction.pivot_dof_indices.len())
        .or_else(|| {
            constraint_matrix
                .as_ref()
                .map(ResolvedConstraintMatrix::row_count)
        })
        .unwrap_or_else(|| constrained.iter().filter(|value| **value).count());

    Ok(AssemblySummary {
        dof_count,
        structural_node_count: structural_dof_layout.node_count(),
        structural_translational_dof_count: structural_dof_layout.translational_dof_count(),
        structural_rotational_dof_count: structural_dof_layout.rotational_dof_count(),
        structural_rotation_node_count: structural_dof_layout.rotation_node_count(),
        structural_moment_load_count,
        structural_direct_rotational_moment_load_count,
        structural_requested_moment_norm_n_m,
        structural_moment_projection,
        spatial_projection_evidence,
        structural_rotational_constraint_count: 0,
        structural_beam_element_count: 0,
        structural_shell_element_count: 0,
        structural_solid_element_count: solid_topology
            .as_ref()
            .map(|topology| topology.volume_element_count)
            .unwrap_or(0),
        structural_solid_total_mass_kg,
        structural_solid_recovery,
        structural_dof_layout,
        structural_beam_recovery: Vec::new(),
        structural_shell_recovery: Vec::new(),
        structural_formulation: structural_requirement.and_then(|requirement| {
            solid_topology
                .as_ref()
                .map(|topology| {
                    let (operator, recovery) = match topology.order {
                        ElementOrder::Tet4 => (
                            StructuralOperatorImplementation::Tet4LinearElasticity,
                            Some(StructuralRecoveryImplementation::Tet4ConstantStrain),
                        ),
                        ElementOrder::Tet10 => (
                            StructuralOperatorImplementation::Tet10LinearElasticity,
                            None,
                        ),
                    };
                    StructuralFormulationRealization::new(operator, recovery)
                })
                .or_else(|| {
                    (requirement == StructuralFormulationRequirement::ExperimentalSyntheticOperator)
                        .then(|| {
                            StructuralFormulationRealization::new(
                                StructuralOperatorImplementation::SyntheticDiagonalChain,
                                None,
                            )
                        })
                })
        }),
        structural_sources: structural_requirement.map_or_else(
            || vec![SourceImplementation::NotApplicable],
            |_| {
                structural_source_implementations(
                    excitations,
                    if solid_topology.is_some() {
                        StructuralProjectionBasis::CanonicalSolid
                    } else {
                        StructuralProjectionBasis::Synthetic
                    },
                )
            },
        ),
        structural_boundary_transfers: structural_requirement.map_or_else(
            || vec![BoundaryTransferImplementation::NotApplicable],
            |_| {
                structural_boundary_implementations(
                    model,
                    if solid_topology.is_some() {
                        StructuralProjectionBasis::CanonicalSolid
                    } else {
                        StructuralProjectionBasis::Synthetic
                    },
                )
            },
        ),
        constrained_dof_count,
        constraint_matrix,
        affine_constraint_reduction,
        load_count: excitations.len(),
        structural_material,
        thermo_mechanical,
        electro_thermal,
        operator: OperatorSystem {
            dof_count,
            constrained,
            stiffness_dense: None,
            stiffness_csr,
            stiffness_diag,
            stiffness_upper,
            mass_csr,
            mass_diag,
            damping_diag,
            rhs,
        },
    })
}

fn solid_recovery_from_solver_mesh(mesh: &SolverMeshArtifact) -> Vec<SolidRecoveryElementSummary> {
    let node_indices = mesh
        .topology
        .nodes
        .iter()
        .enumerate()
        .map(|(index, node)| (node.node_id, index))
        .collect::<BTreeMap<_, _>>();

    mesh.topology
        .volume_elements
        .iter()
        .filter(|element| element.node_ids.len() == 4)
        .filter_map(|element| {
            let mut indices = [0_usize; 4];
            let mut coordinates_m = [[0.0_f64; 3]; 4];
            for (local, node_id) in element.node_ids.iter().copied().enumerate() {
                let node_index = *node_indices.get(&node_id)?;
                indices[local] = node_index;
                coordinates_m[local] = mesh.topology.nodes.get(node_index)?.coordinates_m;
            }
            Some(SolidRecoveryElementSummary {
                element_id: element.element_id.to_string(),
                region_id: element.region_id.source_topology_id.clone(),
                node_indices: indices,
                coordinates_m,
            })
        })
        .collect()
}

fn electro_thermal_fingerprint(
    context: &FeaElectroThermalContext,
    dof_count: usize,
    joule_heating_scale: f64,
    conductivity_spread_ratio: f64,
) -> u64 {
    let mut hash = 1469598103934665603_u64;
    for value in [
        context.reference_temperature_k.to_bits(),
        context.applied_voltage_v.to_bits(),
        context.base_electrical_conductivity_s_per_m.to_bits(),
        context.resistive_heating_coefficient.to_bits(),
        joule_heating_scale.to_bits(),
        conductivity_spread_ratio.to_bits(),
        dof_count as u64,
    ] {
        hash ^= value;
        hash = hash.wrapping_mul(1099511628211);
    }
    hash
}

fn assemble_explicit_structural_system(
    model: &AnalysisModel,
    excitations: &RealizedExcitations,
) -> Result<AssemblySummary, StructuralAssemblyError> {
    let structural = model
        .structural
        .as_ref()
        .ok_or(StructuralAssemblyError::MissingDiscretization)?;
    validate_structural_inputs(
        model,
        excitations,
        StructuralProjectionBasis::ExplicitConnectivity,
    )?;
    let beam_elements = structural
        .elements
        .iter()
        .filter(|element| matches!(element.kind, StructuralElementKind::Beam(_)))
        .collect::<Vec<_>>();
    let shell_elements = structural
        .elements
        .iter()
        .filter(|element| matches!(element.kind, StructuralElementKind::Shell(_)))
        .collect::<Vec<_>>();
    if !beam_elements.is_empty() && !shell_elements.is_empty() {
        return Err(StructuralAssemblyError::MixedExplicitDimensions {
            beam_element_count: beam_elements.len(),
            shell_element_count: shell_elements.len(),
        });
    }
    if beam_elements.is_empty() && shell_elements.is_empty() {
        return Err(StructuralAssemblyError::MissingDiscretization);
    }
    if structural.nodes.is_empty() {
        return Err(StructuralAssemblyError::MissingNodes);
    }
    let mut node_ids = std::collections::BTreeSet::new();
    for node in &structural.nodes {
        if !node_ids.insert(node.node_id) {
            return Err(StructuralAssemblyError::DuplicateNodeId {
                node_id: node.node_id,
            });
        }
    }
    let mut element_ids = std::collections::BTreeSet::new();
    for element in &structural.elements {
        if !element_ids.insert(element.element_id.as_str()) {
            return Err(StructuralAssemblyError::DuplicateElementId {
                element_id: element.element_id.clone(),
            });
        }
    }

    let structural_material = structural_material_summary(model);
    let beam_material = BeamMaterial {
        youngs_modulus_pa: structural_material.youngs_modulus_pa,
        shear_modulus_pa: structural_material.shear_modulus_pa,
    };
    let shell_material = ShellMaterial {
        youngs_modulus_pa: structural_material.youngs_modulus_pa,
        poisson_ratio: structural_material.poisson_ratio,
        shear_modulus_pa: structural_material.shear_modulus_pa,
    };
    let node_count = structural.nodes.len();
    let structural_dof_layout = StructuralDofLayout::from_node_sets(vec![
        StructuralNodeDofSet::translational_rotational();
        node_count
    ]);
    let dof_count = structural_dof_layout.total_dof_count();
    let mut dense = vec![0.0_f64; dof_count * dof_count];
    let mut rhs = vec![0.0_f64; dof_count];
    let mut mass_diag = vec![0.0_f64; dof_count];
    let mut damping_diag = vec![0.0_f64; dof_count];
    let mut structural_beam_recovery = Vec::new();
    let mut structural_shell_recovery = Vec::new();

    for element in &beam_elements {
        let StructuralElementKind::Beam(beam) = &element.kind else {
            continue;
        };
        let node_i_index =
            structural_node_index(structural, beam.node_ids[0]).ok_or_else(|| {
                StructuralAssemblyError::UnknownNode {
                    element_id: element.element_id.clone(),
                    node_id: beam.node_ids[0],
                }
            })?;
        let node_j_index =
            structural_node_index(structural, beam.node_ids[1]).ok_or_else(|| {
                StructuralAssemblyError::UnknownNode {
                    element_id: element.element_id.clone(),
                    node_id: beam.node_ids[1],
                }
            })?;
        let section = structural_beam_section(structural, &beam.section_id).ok_or_else(|| {
            StructuralAssemblyError::UnknownBeamSection {
                element_id: element.element_id.clone(),
                section_id: beam.section_id.clone(),
            }
        })?;
        let geometry = BeamElementGeometry {
            node_i_m: structural.nodes[node_i_index].coordinates_m,
            node_j_m: structural.nodes[node_j_index].coordinates_m,
            reference_axis: beam.reference_axis,
        };
        let frame =
            geometry
                .local_frame()
                .map_err(|error| StructuralAssemblyError::BeamElement {
                    element_id: element.element_id.clone(),
                    error,
                })?;
        let transform_global_to_local = transformation_matrix(frame);
        let stiffness =
            beam_global_stiffness_matrix(section, beam_material, geometry).map_err(|error| {
                StructuralAssemblyError::BeamElement {
                    element_id: element.element_id.clone(),
                    error,
                }
            })?;
        let element_dofs =
            beam_element_dof_indices(&structural_dof_layout, node_i_index, node_j_index)
                .expect("six-degree-of-freedom structural layout must address every beam node");
        for (local_row, &global_row) in element_dofs.iter().enumerate() {
            for (local_col, &global_col) in element_dofs.iter().enumerate() {
                dense[global_row * dof_count + global_col] += stiffness[local_row][local_col];
            }
        }
        structural_beam_recovery.push(BeamRecoveryElementSummary {
            element_id: element.element_id.clone(),
            region_id: element.region_id.clone(),
            node_i_index,
            node_j_index,
            length_m: frame.length_m,
            section,
            material: beam_material,
            transform_global_to_local,
        });

        add_lumped_beam_mass_and_damping(
            &mut mass_diag,
            &mut damping_diag,
            &element_dofs,
            section,
            structural_material.density_kg_per_m3,
            frame.length_m,
            &transform_global_to_local,
        );
    }

    for element in &shell_elements {
        let StructuralElementKind::Shell(shell) = &element.kind else {
            continue;
        };
        let mut node_indices = [0_usize; 3];
        for (local, node_id) in shell.node_ids.iter().copied().enumerate() {
            node_indices[local] = structural_node_index(structural, node_id).ok_or_else(|| {
                StructuralAssemblyError::UnknownNode {
                    element_id: element.element_id.clone(),
                    node_id,
                }
            })?;
        }
        let section = structural_shell_section(structural, &shell.section_id).ok_or_else(|| {
            StructuralAssemblyError::UnknownShellSection {
                element_id: element.element_id.clone(),
                section_id: shell.section_id.clone(),
            }
        })?;
        let geometry = ShellElementGeometry {
            nodes_m: [
                structural.nodes[node_indices[0]].coordinates_m,
                structural.nodes[node_indices[1]].coordinates_m,
                structural.nodes[node_indices[2]].coordinates_m,
            ],
            reference_axis: shell.reference_axis,
        };
        let frame =
            geometry
                .local_frame()
                .map_err(|error| StructuralAssemblyError::ShellElement {
                    element_id: element.element_id.clone(),
                    error,
                })?;
        let stiffness =
            shell_global_stiffness_matrix(section, shell_material, geometry).map_err(|error| {
                StructuralAssemblyError::ShellElement {
                    element_id: element.element_id.clone(),
                    error,
                }
            })?;
        let element_dofs = shell_element_dof_indices(&structural_dof_layout, node_indices)
            .expect("six-degree-of-freedom structural layout must address every shell node");
        for (local_row, &global_row) in element_dofs.iter().enumerate() {
            for (local_col, &global_col) in element_dofs.iter().enumerate() {
                dense[global_row * dof_count + global_col] += stiffness[local_row][local_col];
            }
        }
        add_lumped_shell_mass_and_damping(
            &mut mass_diag,
            &mut damping_diag,
            &element_dofs,
            section,
            structural_material.density_kg_per_m3,
            frame.area_m2,
        );
        structural_shell_recovery.push(ShellRecoveryElementSummary {
            element_id: element.element_id.clone(),
            region_id: element.region_id.clone(),
            node_indices,
            area_m2: frame.area_m2,
            section,
            material: shell_material,
        });
    }

    let mut direct_rotational_moment_load_count = 0usize;
    let mut structural_moment_projection = Vec::new();
    let spatial_projection_evidence = Vec::new();
    let structural_node_coordinates_m = structural
        .nodes
        .iter()
        .map(|node| node.coordinates_m)
        .collect::<Vec<_>>();
    for load in excitations.iter() {
        if matches!(
            load.kind,
            RealizedExcitationKind::TemperatureField(_)
                | RealizedExcitationKind::TerminalVoltage(_)
        ) {
            continue;
        }
        let region_id = load.target.solver_region_id().ok_or_else(|| {
            StructuralAssemblyError::UnmappedStructuralLoad {
                load_id: load.excitation_id.0.clone(),
                region_id: "<non-region target>".to_string(),
                load_kind: explicit_load_kind(&load.kind),
            }
        })?;
        let target_nodes = structural_target_nodes(structural, region_id);
        if target_nodes.is_empty() {
            return Err(StructuralAssemblyError::UnmappedStructuralLoad {
                load_id: load.excitation_id.0.clone(),
                region_id: region_id.to_string(),
                load_kind: explicit_load_kind(&load.kind),
            });
        }
        let scale = 1.0 / target_nodes.len() as f64;
        match load.kind {
            RealizedExcitationKind::Force(value) => {
                let [fx, fy, fz] = value.real;
                for node_index in target_nodes {
                    add_rhs(
                        &structural_dof_layout,
                        &mut rhs,
                        node_index,
                        StructuralDofKind::Ux,
                        fx * scale,
                    );
                    add_rhs(
                        &structural_dof_layout,
                        &mut rhs,
                        node_index,
                        StructuralDofKind::Uy,
                        fy * scale,
                    );
                    add_rhs(
                        &structural_dof_layout,
                        &mut rhs,
                        node_index,
                        StructuralDofKind::Uz,
                        fz * scale,
                    );
                }
            }
            RealizedExcitationKind::Moment(value) => {
                if target_nodes.iter().all(|node_index| {
                    structural_dof_layout
                        .index(*node_index, StructuralDofKind::Rx)
                        .is_some()
                }) {
                    let [mx, my, mz] = value.real;
                    direct_rotational_moment_load_count += 1;
                    for node_index in target_nodes {
                        add_rhs(
                            &structural_dof_layout,
                            &mut rhs,
                            node_index,
                            StructuralDofKind::Rx,
                            mx * scale,
                        );
                        add_rhs(
                            &structural_dof_layout,
                            &mut rhs,
                            node_index,
                            StructuralDofKind::Ry,
                            my * scale,
                        );
                        add_rhs(
                            &structural_dof_layout,
                            &mut rhs,
                            node_index,
                            StructuralDofKind::Rz,
                            mz * scale,
                        );
                    }
                } else {
                    let summary = add_moment_projection_rhs(
                        &structural_node_coordinates_m,
                        &structural_dof_layout,
                        &mut rhs,
                        &target_nodes,
                        value.real,
                    );
                    structural_moment_projection.push(MomentProjectionSummary {
                        load_id: load.excitation_id.0.clone(),
                        region_id: region_id.to_string(),
                        ..summary
                    });
                }
            }
            _ => {}
        }
    }

    let mut constrained = vec![false; dof_count];
    for constraint in model
        .constraints
        .iter()
        .filter_map(|value| value.structural())
    {
        if !matches!(
            constraint.kind,
            StructuralConstraintKind::Fixed { .. } | StructuralConstraintKind::Prescribed(_)
        ) {
            return Err(StructuralAssemblyError::UnsupportedStructuralConstraint {
                constraint_id: constraint.constraint_id.clone(),
                constraint_kind: structural_constraint_kind_name(&constraint.kind),
            });
        }
        let target = constraint.kind.target().ok_or_else(|| {
            StructuralAssemblyError::UnsupportedStructuralConstraint {
                constraint_id: constraint.constraint_id.clone(),
                constraint_kind: structural_constraint_kind_name(&constraint.kind),
            }
        })?;
        let target_nodes = structural_target_nodes(structural, &target.region_id);
        if target_nodes.is_empty() {
            return Err(StructuralAssemblyError::UnmappedExplicitConstraint {
                constraint_id: constraint.constraint_id.clone(),
                region_id: target.region_id.clone(),
            });
        }
        for node_index in target_nodes {
            match &constraint.kind {
                StructuralConstraintKind::Fixed { .. } => {
                    for kind in StructuralDofKind::ORDER {
                        constrain_dof(
                            &structural_dof_layout,
                            &mut constrained,
                            &mut rhs,
                            node_index,
                            kind,
                            0.0,
                        );
                    }
                }
                StructuralConstraintKind::Prescribed(values) => {
                    for prescribed in &values.translations {
                        let Some(value) = prescribed.value.study_constant(&model.signals) else {
                            continue;
                        };
                        let kind = match prescribed.dof {
                            StructuralTranslationDof::Ux => StructuralDofKind::Ux,
                            StructuralTranslationDof::Uy => StructuralDofKind::Uy,
                            StructuralTranslationDof::Uz => StructuralDofKind::Uz,
                        };
                        constrain_dof(
                            &structural_dof_layout,
                            &mut constrained,
                            &mut rhs,
                            node_index,
                            kind,
                            value.meters,
                        );
                    }
                    for prescribed in &values.rotations {
                        let Some(value) = prescribed.value.study_constant(&model.signals) else {
                            continue;
                        };
                        let kind = match prescribed.dof {
                            StructuralRotationDof::Rx => StructuralDofKind::Rx,
                            StructuralRotationDof::Ry => StructuralDofKind::Ry,
                            StructuralRotationDof::Rz => StructuralDofKind::Rz,
                        };
                        constrain_dof(
                            &structural_dof_layout,
                            &mut constrained,
                            &mut rhs,
                            node_index,
                            kind,
                            value.radians,
                        );
                    }
                }
                _ => {}
            }
        }
    }
    apply_dense_constraints(&dense, &constrained, &mut rhs, dof_count);

    let stiffness_diag = (0..dof_count)
        .map(|idx| {
            if constrained[idx] {
                1.0
            } else {
                dense[idx * dof_count + idx].abs().max(1.0e-12)
            }
        })
        .collect::<Vec<_>>();
    let stiffness_upper = vec![0.0; dof_count.saturating_sub(1)];
    let constrained_dof_count = constrained.iter().filter(|value| **value).count();
    let rotational_constraint_count = constrained
        .iter()
        .enumerate()
        .filter(|(_, is_constrained)| **is_constrained)
        .filter(|(row, _)| {
            structural_dof_layout
                .address(*row)
                .is_some_and(|address| address.kind.is_rotational())
        })
        .count();

    let structural_formulation = if !beam_elements.is_empty() {
        StructuralFormulationRealization::new(
            StructuralOperatorImplementation::EulerBernoulliBeam3d,
            Some(StructuralRecoveryImplementation::BeamSectionResultants),
        )
    } else {
        StructuralFormulationRealization::new(
            StructuralOperatorImplementation::PairSpringShellProxy,
            Some(StructuralRecoveryImplementation::PairSpringShellDerivedResultants),
        )
    };

    Ok(AssemblySummary {
        dof_count,
        structural_node_count: structural_dof_layout.node_count(),
        structural_translational_dof_count: structural_dof_layout.translational_dof_count(),
        structural_rotational_dof_count: structural_dof_layout.rotational_dof_count(),
        structural_rotation_node_count: structural_dof_layout.rotation_node_count(),
        structural_moment_load_count: excitations
            .iter()
            .filter(|load| matches!(load.kind, RealizedExcitationKind::Moment(_)))
            .count(),
        structural_direct_rotational_moment_load_count: direct_rotational_moment_load_count,
        structural_requested_moment_norm_n_m: requested_moment_norm(excitations),
        structural_moment_projection,
        spatial_projection_evidence,
        structural_rotational_constraint_count: rotational_constraint_count,
        structural_beam_element_count: beam_elements.len(),
        structural_shell_element_count: shell_elements.len(),
        structural_solid_element_count: 0,
        structural_solid_total_mass_kg: 0.0,
        structural_solid_recovery: Vec::new(),
        structural_dof_layout,
        structural_beam_recovery,
        structural_shell_recovery,
        structural_formulation: Some(structural_formulation),
        structural_sources: structural_source_implementations(
            excitations,
            StructuralProjectionBasis::ExplicitConnectivity,
        ),
        structural_boundary_transfers: structural_boundary_implementations(
            model,
            StructuralProjectionBasis::ExplicitConnectivity,
        ),
        constrained_dof_count,
        constraint_matrix: None,
        affine_constraint_reduction: None,
        load_count: excitations.len(),
        structural_material,
        thermo_mechanical: None,
        electro_thermal: None,
        operator: OperatorSystem {
            dof_count,
            constrained,
            stiffness_dense: Some(dense),
            stiffness_csr: None,
            stiffness_diag,
            stiffness_upper,
            mass_csr: None,
            mass_diag,
            damping_diag,
            rhs,
        },
    })
}

fn structural_material_summary(model: &AnalysisModel) -> StructuralMaterialSummary {
    let avg_youngs_modulus = if model.materials.is_empty() {
        1.0e9
    } else {
        model
            .materials
            .iter()
            .map(|material| material.mechanical.youngs_modulus_pa.max(1.0))
            .sum::<f64>()
            / model.materials.len() as f64
    };
    let avg_poisson_ratio = if model.materials.is_empty() {
        0.3
    } else {
        model
            .materials
            .iter()
            .map(|material| material.mechanical.poisson_ratio.clamp(0.0, 0.49))
            .sum::<f64>()
            / model.materials.len() as f64
    };
    let shear_modulus_pa = avg_youngs_modulus / (2.0 * (1.0 + avg_poisson_ratio)).max(1.0e-9);
    let avg_density_kg_per_m3 = if model.materials.is_empty() {
        7850.0
    } else {
        model
            .materials
            .iter()
            .map(|material| material.mechanical.density_kg_per_m3.max(1.0))
            .sum::<f64>()
            / model.materials.len() as f64
    };
    let lame_lambda_pa = avg_youngs_modulus * avg_poisson_ratio
        / ((1.0 + avg_poisson_ratio) * (1.0 - 2.0 * avg_poisson_ratio)).max(1.0e-9);
    StructuralMaterialSummary {
        youngs_modulus_pa: avg_youngs_modulus,
        poisson_ratio: avg_poisson_ratio,
        density_kg_per_m3: avg_density_kg_per_m3,
        lame_lambda_pa,
        shear_modulus_pa,
    }
}

fn solid_materials_by_region(model: &AnalysisModel) -> BTreeMap<String, SolidMaterial> {
    let materials = model
        .materials
        .iter()
        .map(|material| {
            (
                material.material_id.clone(),
                SolidMaterial {
                    youngs_modulus_pa: material.mechanical.youngs_modulus_pa.max(1.0),
                    poisson_ratio: material.mechanical.poisson_ratio.clamp(0.0, 0.49),
                    density_kg_per_m3: material.mechanical.density_kg_per_m3.max(1.0e-12),
                },
            )
        })
        .collect::<BTreeMap<_, _>>();
    model
        .material_assignments
        .iter()
        .filter_map(|assignment| {
            materials
                .get(&assignment.assigned_material_id)
                .copied()
                .map(|material| (assignment.region_id.clone(), material))
        })
        .collect()
}

fn structural_node_index(structural: &StructuralModel, node_id: u32) -> Option<usize> {
    structural
        .nodes
        .iter()
        .position(|node| node.node_id == node_id)
}

fn structural_beam_section(structural: &StructuralModel, section_id: &str) -> Option<BeamSection> {
    structural
        .beam_sections
        .iter()
        .find(|section| section.section_id == section_id)
        .map(section_from_model)
}

fn structural_shell_section(
    structural: &StructuralModel,
    section_id: &str,
) -> Option<ShellSection> {
    structural
        .shell_sections
        .iter()
        .find(|section| section.section_id == section_id)
        .map(shell_section_from_model)
}

fn section_from_model(section: &BeamSectionModel) -> BeamSection {
    BeamSection {
        area_m2: section.area_m2,
        iy_m4: section.iy_m4,
        iz_m4: section.iz_m4,
        torsion_j_m4: section.torsion_j_m4,
        outer_fiber_y_m: section.outer_fiber_y_m,
        outer_fiber_z_m: section.outer_fiber_z_m,
        torsion_outer_radius_m: section.torsion_outer_radius_m,
    }
}

fn shell_section_from_model(section: &ShellSectionModel) -> ShellSection {
    ShellSection {
        thickness_m: section.thickness_m,
        shear_correction: section.shear_correction,
        drilling_stiffness_scale: section.drilling_stiffness_scale,
    }
}

fn beam_element_dof_indices(
    layout: &StructuralDofLayout,
    node_i_index: usize,
    node_j_index: usize,
) -> Option<[usize; BEAM_ELEMENT_DOF_COUNT]> {
    let mut indices = [0usize; BEAM_ELEMENT_DOF_COUNT];
    for (local, kind) in StructuralDofKind::ORDER.iter().copied().enumerate() {
        indices[local] = layout.index(node_i_index, kind)?;
        indices[local + 6] = layout.index(node_j_index, kind)?;
    }
    Some(indices)
}

fn shell_element_dof_indices(
    layout: &StructuralDofLayout,
    node_indices: [usize; 3],
) -> Option<[usize; SHELL_ELEMENT_DOF_COUNT]> {
    let mut indices = [0usize; SHELL_ELEMENT_DOF_COUNT];
    for (node_offset, node_index) in node_indices.iter().enumerate() {
        for (component, kind) in StructuralDofKind::ORDER.iter().enumerate() {
            indices[node_offset * SHELL_NODE_DOF_COUNT + component] =
                layout.index(*node_index, *kind)?;
        }
    }
    Some(indices)
}

fn add_lumped_beam_mass_and_damping(
    mass_diag: &mut [f64],
    damping_diag: &mut [f64],
    element_dofs: &[usize; BEAM_ELEMENT_DOF_COUNT],
    section: BeamSection,
    density_kg_per_m3: f64,
    length_m: f64,
    transform_global_to_local: &BeamTransform12,
) {
    let density = density_kg_per_m3.max(1.0);
    let length = length_m.max(1.0e-12);
    let nodal_mass = density * section.area_m2.max(1.0e-18) * length / 2.0;
    let local_rotational_inertia = [
        density * section.torsion_j_m4.max(1.0e-24) * length / 2.0,
        density * section.iy_m4.max(1.0e-24) * length / 2.0,
        density * section.iz_m4.max(1.0e-24) * length / 2.0,
    ];

    for node_offset in [0usize, 6] {
        for component in 0..3 {
            let dof = element_dofs[node_offset + component];
            mass_diag[dof] += nodal_mass;
            damping_diag[dof] += 0.01;
        }
        for global_component in 0..3 {
            let local_col = node_offset + 3 + global_component;
            let inertia = local_rotational_inertia
                .iter()
                .enumerate()
                .map(|(local_component, local_inertia)| {
                    let row = node_offset + 3 + local_component;
                    local_inertia * transform_global_to_local[row][local_col].powi(2)
                })
                .sum::<f64>();
            let dof = element_dofs[node_offset + 3 + global_component];
            mass_diag[dof] += inertia.max(1.0e-18);
            damping_diag[dof] += 0.01;
        }
    }
}

fn add_lumped_shell_mass_and_damping(
    mass_diag: &mut [f64],
    damping_diag: &mut [f64],
    element_dofs: &[usize; SHELL_ELEMENT_DOF_COUNT],
    section: ShellSection,
    density_kg_per_m3: f64,
    area_m2: f64,
) {
    let density = density_kg_per_m3.max(1.0);
    let thickness = section.thickness_m.max(1.0e-12);
    let area = area_m2.max(1.0e-18);
    let nodal_mass = density * thickness * area / 3.0;
    let nodal_rotary_inertia = nodal_mass * thickness.powi(2) / 12.0;
    for node_offset in [0usize, 6, 12] {
        for component in 0..3 {
            let dof = element_dofs[node_offset + component];
            mass_diag[dof] += nodal_mass;
            damping_diag[dof] += 0.01;
        }
        for component in 3..6 {
            let dof = element_dofs[node_offset + component];
            mass_diag[dof] += nodal_rotary_inertia.max(1.0e-18);
            damping_diag[dof] += 0.01;
        }
    }
}

fn structural_target_nodes(structural: &StructuralModel, region_id: &str) -> Vec<usize> {
    if let Some(node_id) = structural_node_selector(region_id) {
        return structural_node_index(structural, node_id)
            .into_iter()
            .collect();
    }
    let mut nodes = Vec::new();
    for element in &structural.elements {
        if element.region_id != region_id {
            continue;
        }
        match &element.kind {
            StructuralElementKind::Beam(beam) => {
                for node_id in beam.node_ids {
                    if let Some(index) = structural_node_index(structural, node_id) {
                        if !nodes.contains(&index) {
                            nodes.push(index);
                        }
                    }
                }
            }
            StructuralElementKind::Shell(shell) => {
                for node_id in shell.node_ids {
                    if let Some(index) = structural_node_index(structural, node_id) {
                        if !nodes.contains(&index) {
                            nodes.push(index);
                        }
                    }
                }
            }
        }
    }
    nodes
}

fn structural_node_selector(region_id: &str) -> Option<u32> {
    region_id
        .strip_prefix("node:")
        .unwrap_or(region_id)
        .parse::<u32>()
        .ok()
}

fn add_rhs(
    layout: &StructuralDofLayout,
    rhs: &mut [f64],
    node_index: usize,
    kind: StructuralDofKind,
    value: f64,
) {
    if let Some(dof) = layout.index(node_index, kind) {
        rhs[dof] += value;
    }
}

fn add_moment_projection_rhs(
    node_coordinates_m: &[[f64; 3]],
    layout: &StructuralDofLayout,
    rhs: &mut [f64],
    target_nodes: &[usize],
    requested_moment: [f64; 3],
) -> MomentProjectionSummary {
    let projection = project_moment(node_coordinates_m, target_nodes, requested_moment);
    for (&node_index, &force) in target_nodes.iter().zip(&projection.nodal_forces) {
        add_translational_rhs(layout, rhs, node_index, force);
    }
    MomentProjectionSummary {
        load_id: String::new(),
        region_id: String::new(),
        target_node_count: target_nodes.len(),
        projection_point_m: projection.point_m,
        applied_force: projection.applied_force,
        applied_moment: projection.applied_moment,
        force_residual: projection.applied_force.map(|value| -value),
        moment_residual: [
            requested_moment[0] - projection.applied_moment[0],
            requested_moment[1] - projection.applied_moment[1],
            requested_moment[2] - projection.applied_moment[2],
        ],
        projection_applied: projection.projection_applied,
    }
}

fn add_translational_rhs(
    layout: &StructuralDofLayout,
    rhs: &mut [f64],
    node_index: usize,
    force: [f64; 3],
) {
    add_rhs(layout, rhs, node_index, StructuralDofKind::Ux, force[0]);
    add_rhs(layout, rhs, node_index, StructuralDofKind::Uy, force[1]);
    add_rhs(layout, rhs, node_index, StructuralDofKind::Uz, force[2]);
}

fn constrain_dof(
    layout: &StructuralDofLayout,
    constrained: &mut [bool],
    rhs: &mut [f64],
    node_index: usize,
    kind: StructuralDofKind,
    value: f64,
) {
    if let Some(dof) = layout.index(node_index, kind) {
        constrained[dof] = true;
        rhs[dof] = value;
    }
}

fn apply_solid_constraint_plan(
    plan: &CanonicalStructuralConstraintPlan,
    mesh: &SolverMeshArtifact,
    layout: &StructuralDofLayout,
    constrained: &mut [bool],
    rhs: &mut [f64],
) -> Result<(), StructuralAssemblyError> {
    let node_indices = mesh
        .topology
        .nodes
        .iter()
        .enumerate()
        .map(|(index, node)| (node.stable_identity, index))
        .collect::<BTreeMap<_, _>>();
    for row in &plan.rows {
        if row.reaction_recovery == ReactionRecovery::RedundantEquation {
            continue;
        }
        let [term] = row.terms.as_slice() else {
            return Err(StructuralAssemblyError::UnsupportedStructuralConstraint {
                constraint_id: row.constraint_id.clone(),
                constraint_kind: "general_constraint_equation",
            });
        };
        let CanonicalStructuralDofOwner::MeshNode(node_identity) = term.owner else {
            return Err(StructuralAssemblyError::ConstraintPlanNodeAbsent);
        };
        let node_index = node_indices
            .get(&node_identity)
            .copied()
            .ok_or(StructuralAssemblyError::ConstraintPlanNodeAbsent)?;
        let kind = match term.dof {
            CanonicalStructuralDof::Translation(StructuralTranslationDof::Ux) => {
                StructuralDofKind::Ux
            }
            CanonicalStructuralDof::Translation(StructuralTranslationDof::Uy) => {
                StructuralDofKind::Uy
            }
            CanonicalStructuralDof::Translation(StructuralTranslationDof::Uz) => {
                StructuralDofKind::Uz
            }
            CanonicalStructuralDof::Rotation(_) => {
                return Err(StructuralAssemblyError::UnsupportedStructuralConstraint {
                    constraint_id: row.constraint_id.clone(),
                    constraint_kind: "rotational_dof_on_solid",
                });
            }
        };
        constrain_dof(
            layout,
            constrained,
            rhs,
            node_index,
            kind,
            row.right_hand_side / term.coefficient,
        );
    }
    Ok(())
}

fn resolve_solid_constraint_matrix(
    plan: &CanonicalStructuralConstraintPlan,
    mesh: &SolverMeshArtifact,
    layout: &StructuralDofLayout,
) -> Result<ResolvedConstraintMatrix, ConstraintAlgebraError> {
    let node_indices = mesh
        .topology
        .nodes
        .iter()
        .enumerate()
        .map(|(index, node)| (node.stable_identity, index))
        .collect::<BTreeMap<_, _>>();
    let reference_point_indices = plan
        .reference_points
        .iter()
        .enumerate()
        .map(|(index, point)| (point.identity, index))
        .collect::<BTreeMap<_, _>>();
    ResolvedConstraintMatrix::from_plan(plan, layout.total_dof_count(), |owner, dof| {
        let kind = match dof {
            CanonicalStructuralDof::Translation(StructuralTranslationDof::Ux) => {
                StructuralDofKind::Ux
            }
            CanonicalStructuralDof::Translation(StructuralTranslationDof::Uy) => {
                StructuralDofKind::Uy
            }
            CanonicalStructuralDof::Translation(StructuralTranslationDof::Uz) => {
                StructuralDofKind::Uz
            }
            CanonicalStructuralDof::Rotation(StructuralRotationDof::Rx) => StructuralDofKind::Rx,
            CanonicalStructuralDof::Rotation(StructuralRotationDof::Ry) => StructuralDofKind::Ry,
            CanonicalStructuralDof::Rotation(StructuralRotationDof::Rz) => StructuralDofKind::Rz,
        };
        match owner {
            CanonicalStructuralDofOwner::MeshNode(node) => {
                layout.index(node_indices.get(&node).copied()?, kind)
            }
            CanonicalStructuralDofOwner::ReferencePoint(identity) => {
                layout.reference_point_index(reference_point_indices.get(&identity).copied()?, kind)
            }
        }
    })
}

fn apply_dense_constraints(dense: &[f64], constrained: &[bool], rhs: &mut [f64], dof_count: usize) {
    for dof in 0..dof_count {
        if !constrained[dof] {
            continue;
        }
        for row in 0..dof_count {
            if !constrained[row] {
                rhs[row] -= dense[row * dof_count + dof] * rhs[dof];
            }
        }
    }
}

fn apply_csr_constraints(csr: &CsrMatrix, constrained: &[bool], rhs: &mut [f64], dof_count: usize) {
    for row in 0..dof_count {
        if constrained[row] {
            continue;
        }
        let start = csr.row_offsets[row];
        let end = csr.row_offsets[row + 1];
        for entry in start..end {
            let column = csr.column_indices[entry];
            if constrained[column] {
                rhs[row] -= csr.values[entry] * rhs[column];
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct ThermoMechanicalFingerprintInputs {
    dof_count: usize,
    constitutive_temperature_factor: f64,
    constitutive_poisson_coupling: f64,
    effective_modulus_scale: f64,
    constitutive_material_spread_ratio: f64,
    assignment_heterogeneity_index: f64,
    spatial_gradient_index: f64,
}

fn thermo_mechanical_fingerprint(
    context: &FeaThermoMechanicalContext,
    inputs: ThermoMechanicalFingerprintInputs,
) -> u64 {
    let mut hash = 1469598103934665603_u64;
    for value in [
        inputs.dof_count as u64,
        context.reference_temperature_k.to_bits(),
        context.applied_temperature_delta_k.to_bits(),
        context.thermal_expansion_coefficient.to_bits(),
        inputs.constitutive_temperature_factor.to_bits(),
        inputs.constitutive_poisson_coupling.to_bits(),
        inputs.effective_modulus_scale.to_bits(),
        inputs.constitutive_material_spread_ratio.to_bits(),
        inputs.assignment_heterogeneity_index.to_bits(),
        inputs.spatial_gradient_index.to_bits(),
    ] {
        hash ^= value;
        hash = hash.wrapping_mul(1099511628211_u64);
    }
    hash
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct ThermoSpatialFieldSummary {
    gradient_index: f64,
    coverage_ratio: f64,
}

fn apply_thermo_spatial_field(
    context: &FeaThermoMechanicalContext,
    dof_count: usize,
    dof_adjustments: &mut [f64],
) -> ThermoSpatialFieldSummary {
    if dof_count == 0 || context.region_temperature_deltas.is_empty() {
        return ThermoSpatialFieldSummary {
            gradient_index: 0.0,
            coverage_ratio: 0.0,
        };
    }
    let mut touched = vec![false; dof_count];
    let mut min_delta = f64::INFINITY;
    let mut max_delta = -f64::INFINITY;
    for (idx, region_delta) in context.region_temperature_deltas.iter().enumerate() {
        min_delta = min_delta.min(region_delta.temperature_delta_k);
        max_delta = max_delta.max(region_delta.temperature_delta_k);
        let normalized = ((region_delta.temperature_delta_k - context.applied_temperature_delta_k)
            / 240.0)
            .clamp(-0.45, 0.45);
        let start =
            ((region_hash(&region_delta.region_id) as usize).wrapping_add(idx * 5)) % dof_count;
        let stride = context
            .region_temperature_deltas
            .len()
            .saturating_add(3)
            .max(2);
        let mut cursor = start;
        for hop in 0..dof_count {
            if hop > 0 && cursor == start {
                break;
            }
            let wave = 1.0 + ((hop + idx) % 7) as f64 * 0.02;
            dof_adjustments[cursor] += normalized * wave;
            touched[cursor] = true;
            cursor = (cursor + stride) % dof_count;
        }
    }
    if !min_delta.is_finite() || !max_delta.is_finite() {
        return ThermoSpatialFieldSummary {
            gradient_index: 0.0,
            coverage_ratio: 0.0,
        };
    }
    let touched_count = touched.iter().filter(|entry| **entry).count() as f64;
    ThermoSpatialFieldSummary {
        gradient_index: ((max_delta - min_delta).abs() / 240.0).clamp(0.0, 1.0),
        coverage_ratio: (touched_count / dof_count as f64).clamp(0.0, 1.0),
    }
}

fn thermomechanical_profile_sources(
    context: &FeaThermoMechanicalContext,
) -> Vec<SourceImplementation> {
    let mut sources = vec![SourceImplementation::PrescribedTemperatureDeltaProfile];
    if !context.region_temperature_deltas.is_empty() {
        sources.push(SourceImplementation::OrdinalRegionTemperatureDistribution);
    }
    sources
}

fn apply_thermo_material_heterogeneity(
    model: &AnalysisModel,
    dof_count: usize,
    constitutive_temperature_factor: f64,
    reference_temperature_k: f64,
    applied_temperature_delta_k: f64,
    dof_adjustments: &mut [f64],
) -> f64 {
    if dof_count == 0 || model.material_assignments.is_empty() {
        return 0.0;
    }
    let base_amplitude = (constitutive_temperature_factor.abs() * 0.8).clamp(0.0, 0.15);
    if base_amplitude <= 0.0 {
        return 0.0;
    }
    let mut weighted_activity = 0.0_f64;
    let mut weight_sum = 0.0_f64;
    for (idx, assignment) in model.material_assignments.iter().enumerate() {
        let confidence_weight = match assignment.confidence {
            runmat_analysis_core::EvidenceConfidence::Verified => 1.0,
            runmat_analysis_core::EvidenceConfidence::Probable => 0.65,
            runmat_analysis_core::EvidenceConfidence::Inferred => 0.4,
        };
        let expected_modulus = model
            .materials
            .iter()
            .find(|material| material.material_id == assignment.expected_material_id)
            .map(|material| material.mechanical.youngs_modulus_pa)
            .unwrap_or(1.0e9)
            .max(1.0);
        let assigned_modulus = model
            .materials
            .iter()
            .find(|material| material.material_id == assignment.assigned_material_id)
            .map(|material| material.mechanical.youngs_modulus_pa)
            .unwrap_or(expected_modulus)
            .max(1.0);
        let modulus_delta_ratio =
            ((assigned_modulus - expected_modulus) / expected_modulus).clamp(-0.6, 0.6);
        let expected_temp_response = model
            .materials
            .iter()
            .find(|material| material.material_id == assignment.expected_material_id)
            .map(|material| {
                material.thermal.modulus_temp_coeff_per_k
                    * (applied_temperature_delta_k
                        + (reference_temperature_k - material.thermal.reference_temperature_k))
            })
            .unwrap_or(constitutive_temperature_factor)
            .clamp(-0.4, 0.2);
        let assigned_temp_response = model
            .materials
            .iter()
            .find(|material| material.material_id == assignment.assigned_material_id)
            .map(|material| {
                material.thermal.modulus_temp_coeff_per_k
                    * (applied_temperature_delta_k
                        + (reference_temperature_k - material.thermal.reference_temperature_k))
            })
            .unwrap_or(expected_temp_response)
            .clamp(-0.4, 0.2);
        let response_delta = (assigned_temp_response - expected_temp_response).clamp(-0.35, 0.35);
        let region_phase = ((region_hash(&assignment.region_id) % 11) as f64) / 10.0;
        let activity =
            (0.7 * modulus_delta_ratio.abs() + 0.3 * response_delta.abs()).clamp(0.0, 1.0);
        let signed_bias = base_amplitude
            * confidence_weight
            * (0.45 * modulus_delta_ratio
                + 0.35 * response_delta
                + 0.2 * modulus_delta_ratio.signum() * region_phase);
        let stride = model.material_assignments.len().saturating_add(1).max(2);
        let start = ((region_hash(&assignment.region_id) as usize).wrapping_add(idx * 3))
            % dof_count.max(1);
        let mut cursor = start;
        for hop in 0..dof_count {
            if hop > 0 && cursor == start {
                break;
            }
            let wave = 1.0 + ((hop + idx) % 5) as f64 * 0.03;
            dof_adjustments[cursor] += signed_bias * wave;
            cursor = (cursor + stride) % dof_count.max(1);
        }
        weighted_activity += activity * confidence_weight;
        weight_sum += confidence_weight;
    }
    for value in dof_adjustments.iter_mut() {
        *value = value.clamp(-0.18, 0.18);
    }
    if weight_sum > 0.0 {
        (weighted_activity / weight_sum).clamp(0.0, 1.0)
    } else {
        0.0
    }
}

fn region_hash(region_id: &str) -> u64 {
    let mut hash = 1469598103934665603_u64;
    for byte in region_id.as_bytes() {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(1099511628211_u64);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixtures::{
        fixture_model, replace_fixture_excitations, replace_fixture_with_electric_current,
        FixtureExcitation, FixtureExcitationKind, FixtureId,
    };
    use runmat_meshing_core::{
        ElementOrder, PersistentEntityId, PersistentEntityKind, SolverMeshArtifact, StableDigest,
    };

    fn fixed_constraint(region_id: &str) -> runmat_analysis_core::AnalysisConstraint {
        runmat_analysis_core::AnalysisConstraint::Structural(
            runmat_analysis_core::StructuralConstraint {
                constraint_id: "fixed_root".to_string(),
                frame: runmat_analysis_core::CoordinateFrame::Global,
                kind: runmat_analysis_core::StructuralConstraintKind::Fixed {
                    target: runmat_analysis_core::RegionTarget::authored(region_id),
                },
            },
        )
    }

    fn static_excitations(model: &AnalysisModel) -> RealizedExcitations {
        crate::excitation::realize_step_excitations(
            model,
            &model.sequence.steps[0],
            crate::excitation::ExcitationSamplePoint::Static,
        )
        .expect("test model excitations should realize at the static sample")
    }

    fn scaled_force_excitations(
        excitations: &RealizedExcitations,
        scale: f64,
    ) -> RealizedExcitations {
        RealizedExcitations::new(
            excitations
                .iter()
                .cloned()
                .map(|mut excitation| {
                    if let RealizedExcitationKind::Force(force) = &mut excitation.kind {
                        force.real.iter_mut().for_each(|value| *value *= scale);
                        force.imaginary.iter_mut().for_each(|value| *value *= scale);
                    }
                    excitation
                })
                .collect(),
        )
    }

    fn assemble_physical_structural_system(
        model: &AnalysisModel,
        solver_mesh: Option<SolverMeshArtifact>,
    ) -> Result<AssemblySummary, LinearAssemblyError> {
        let excitations = static_excitations(model);
        assemble_structural_system(
            model,
            &excitations,
            solver_mesh,
            None,
            None,
            StructuralFormulationRequirement::PhysicalDiscretization,
        )
    }

    #[test]
    fn solver_mesh_populates_sparse_solid_stiffness_operator() {
        let mut model = fixture_model(FixtureId::CantileverLinearStatic);
        model.excitations.clear();
        model.constraints.clear();
        let summary = assemble_physical_structural_system(&model, Some(tetrahedron4_mesh()))
            .expect("valid Tet4 system should assemble");

        assert_eq!(summary.dof_count, 12);
        assert_eq!(summary.structural_solid_element_count, 1);
        assert_eq!(summary.structural_solid_recovery.len(), 1);
        assert_eq!(
            summary.structural_solid_recovery[0].node_indices,
            [0, 1, 2, 3]
        );
        assert_eq!(
            summary.structural_solid_recovery[0].coordinates_m,
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        );
        assert!(summary.operator.stiffness_dense.is_none());
        let expected_mass_kg = model.materials[0].mechanical.density_kg_per_m3 / 6.0;
        assert!(
            (summary.structural_solid_total_mass_kg - expected_mass_kg).abs()
                <= expected_mass_kg * 1.0e-12
        );
        assert!(summary.operator.mass_csr.is_some());
        assert!(summary.operator.mass_diag.iter().all(|mass| *mass > 0.0));
        assert!(summary
            .operator
            .damping_diag
            .iter()
            .all(|damping| *damping == 0.0));
        let csr = summary
            .operator
            .stiffness_csr
            .as_ref()
            .expect("solver mesh should assemble a sparse solid stiffness matrix");
        assert_eq!(csr.row_offsets.len(), summary.dof_count + 1);
        assert_eq!(csr.row_offsets.last().copied(), Some(csr.values.len()));
        assert_eq!(csr.column_indices.len(), csr.values.len());
        assert!(summary
            .operator
            .stiffness_diag
            .iter()
            .all(|value| *value > 0.0));
        for row in 0..summary.dof_count {
            let start = csr.row_offsets[row];
            let end = csr.row_offsets[row + 1];
            let diagonal = csr.column_indices[start..end]
                .iter()
                .zip(csr.values[start..end].iter())
                .find_map(|(&column, &value)| (column == row).then_some(value.abs()))
                .expect("csr row should include diagonal");
            assert!((summary.operator.stiffness_diag[row] - diagonal) <= 1.0e-8);
        }
    }

    #[test]
    fn solver_mesh_rejects_uncoupled_explicit_beam_topology() {
        let mut model = fixture_model(FixtureId::CantileverLinearStatic);
        model.structural = Some(runmat_analysis_core::StructuralModel {
            nodes: vec![
                runmat_analysis_core::StructuralNode {
                    node_id: 1,
                    coordinates_m: [0.0, 0.0, 0.0],
                },
                runmat_analysis_core::StructuralNode {
                    node_id: 2,
                    coordinates_m: [1.0, 0.0, 0.0],
                },
            ],
            elements: vec![runmat_analysis_core::StructuralElement {
                element_id: "beam_1".to_string(),
                region_id: "span".to_string(),
                kind: runmat_analysis_core::StructuralElementKind::Beam(
                    runmat_analysis_core::BeamElementModel {
                        node_ids: [1, 2],
                        section_id: "rect".to_string(),
                        reference_axis: [0.0, 1.0, 0.0],
                    },
                ),
            }],
            beam_sections: vec![runmat_analysis_core::BeamSectionModel {
                section_id: "rect".to_string(),
                area_m2: 1.0e-4,
                iy_m4: 1.0e-9,
                iz_m4: 1.0e-9,
                torsion_j_m4: 1.0e-9,
                outer_fiber_y_m: 0.01,
                outer_fiber_z_m: 0.01,
                torsion_outer_radius_m: 0.01,
            }],
            shell_sections: Vec::new(),
        });

        let error = assemble_structural_system(
            &model,
            &static_excitations(&model),
            Some(tetrahedron4_mesh()),
            None,
            None,
            StructuralFormulationRequirement::PhysicalDiscretization,
        )
        .expect_err("an uncoupled solid-plus-beam model must be rejected");

        assert_eq!(
            error,
            LinearAssemblyError::Structural(StructuralAssemblyError::MixedSolidAndExplicit {
                explicit_element_count: 1,
            })
        );
    }

    #[test]
    fn solver_mesh_material_regions_select_assigned_solid_materials() {
        let mut soft_model = fixture_model(FixtureId::CantileverLinearStatic);
        soft_model.excitations.clear();
        soft_model.constraints.clear();
        let mut hard = soft_model.materials[0].clone();
        hard.material_id = "mat_hard".to_string();
        hard.mechanical.youngs_modulus_pa = 200.0e9;
        hard.mechanical.poisson_ratio = 0.3;
        let mut soft = hard.clone();
        soft.material_id = "mat_soft".to_string();
        soft.mechanical.youngs_modulus_pa = 20.0e9;
        soft_model.materials = vec![hard.clone(), soft.clone()];
        soft_model.material_assignments = vec![runmat_analysis_core::MaterialAssignment {
            region_id: "soft_region".to_string(),
            expected_material_id: "mat_hard".to_string(),
            assigned_material_id: "mat_soft".to_string(),
            confidence: runmat_analysis_core::EvidenceConfidence::Verified,
        }];
        let mut soft_mesh = tetrahedron4_mesh();
        set_solver_mesh_region(&mut soft_mesh, "soft_region");

        let mut hard_model = soft_model.clone();
        hard_model.material_assignments = vec![runmat_analysis_core::MaterialAssignment {
            region_id: "soft_region".to_string(),
            expected_material_id: "mat_hard".to_string(),
            assigned_material_id: "mat_hard".to_string(),
            confidence: runmat_analysis_core::EvidenceConfidence::Verified,
        }];
        let hard_mesh = soft_mesh.clone();
        assert_eq!(soft_mesh.canonical_digest, hard_mesh.canonical_digest);

        let soft_summary = assemble_physical_structural_system(&soft_model, Some(soft_mesh))
            .expect("soft-material system should assemble");
        let hard_summary = assemble_physical_structural_system(&hard_model, Some(hard_mesh))
            .expect("hard-material system should assemble");

        assert!(
            first_csr_diagonal(&soft_summary) < first_csr_diagonal(&hard_summary) * 0.2,
            "soft material assignment should lower solid element stiffness"
        );
        assert_eq!(
            soft_summary.structural_solid_recovery[0].region_id,
            "soft_region"
        );
    }

    #[test]
    fn strict_solver_mesh_assembly_rejects_inverted_projection_geometry() {
        let model = fixture_model(FixtureId::CantileverLinearStatic);
        let mut mesh = tetrahedron4_mesh();
        mesh.topology.volume_elements[0].node_ids = vec![1, 3, 2, 4];
        set_boundary_regions(&mut mesh, 0, &["root"]);
        set_boundary_regions(&mut mesh, 1, &["tip"]);
        reseal(&mut mesh);

        let err = assemble_physical_structural_system(&model, Some(mesh)).expect_err(
            "strict solver mesh assembly should reject inverted Tetrahedron4 stiffness",
        );

        assert!(matches!(
            err,
            LinearAssemblyError::SolverMeshRegionMapping(
                SolverMeshRegionMappingError::InvalidAnalysisProjection(_)
            )
        ));
    }

    #[test]
    fn malformed_solid_connectivity_returns_a_typed_error() {
        let mut model = fixture_model(FixtureId::CantileverLinearStatic);
        model.excitations.clear();
        model.constraints.clear();

        for node_ids in [vec![1, 2, 3], vec![1, 2, 3, 4, 1]] {
            let mut mesh = tetrahedron4_mesh();
            mesh.topology.volume_elements[0].node_ids = node_ids;

            let error = assemble_structural_system(
                &model,
                &static_excitations(&model),
                Some(mesh),
                None,
                None,
                StructuralFormulationRequirement::PhysicalDiscretization,
            )
            .expect_err("Tet4 connectivity must contain exactly four nodes");
            assert!(matches!(error, LinearAssemblyError::SolidStiffness(_)));
        }
    }

    #[test]
    fn solver_mesh_boundary_regions_drive_solid_loads_and_constraints() {
        let mut model = fixture_model(FixtureId::CantileverLinearStatic);
        model.constraints = vec![fixed_constraint("root")];
        replace_fixture_excitations(
            &mut model,
            vec![FixtureExcitation {
                id: "load_tip".into(),
                region_id: "tip".into(),
                kind: FixtureExcitationKind::Force([0.0, -12.0, 0.0]),
            }],
        );
        let mut mesh = tetrahedron4_mesh();
        set_boundary_regions(&mut mesh, 0, &["root"]);
        set_boundary_regions(&mut mesh, 1, &["tip"]);
        reseal(&mut mesh);

        let summary = assemble_physical_structural_system(&model, Some(mesh))
            .expect("valid boundary regions should assemble");

        assert_eq!(summary.constrained_dof_count, 9);
        assert!(summary.operator.constrained[0]);
        assert!(summary.operator.constrained[1]);
        assert!(summary.operator.constrained[2]);
        assert_eq!(summary.operator.rhs[10], -4.0);
    }

    #[test]
    fn solver_mesh_projects_a_solid_boundary_moment_with_conserved_resultants() {
        let mut model = fixture_model(FixtureId::CantileverLinearStatic);
        model.constraints = Vec::new();
        replace_fixture_excitations(
            &mut model,
            vec![FixtureExcitation {
                id: "moment_tip".into(),
                region_id: "tip".into(),
                kind: FixtureExcitationKind::Moment([2.0, -3.0, 5.0]),
            }],
        );
        let mut mesh = tetrahedron4_mesh();
        set_boundary_regions(&mut mesh, 1, &["tip"]);
        reseal(&mut mesh);

        let summary = assemble_physical_structural_system(&model, Some(mesh))
            .expect("solid boundary moment should project to a nodal force couple");

        assert_eq!(summary.structural_moment_projection.len(), 1);
        let projection = &summary.structural_moment_projection[0];
        assert_eq!(projection.load_id, "moment_tip");
        assert_eq!(projection.region_id, "tip");
        assert_eq!(projection.target_node_count, 3);
        assert!(projection.projection_applied);
        for residual in projection
            .force_residual
            .iter()
            .chain(projection.moment_residual.iter())
        {
            assert!(residual.abs() <= 1.0e-12, "residual={residual}");
        }
        assert_eq!(
            summary.structural_sources,
            vec![SourceImplementation::BoundaryResultantMomentProjection]
        );
    }

    #[test]
    fn strict_solver_mesh_assembly_rejects_unmapped_load_region() {
        let mut model = fixture_model(FixtureId::CantileverLinearStatic);
        model.constraints = vec![fixed_constraint("root")];
        replace_fixture_excitations(
            &mut model,
            vec![FixtureExcitation {
                id: "load_tip".into(),
                region_id: "missing_tip".into(),
                kind: FixtureExcitationKind::Force([0.0, -12.0, 0.0]),
            }],
        );
        let mut mesh = tetrahedron4_mesh();
        clear_boundary_regions(&mut mesh);
        set_boundary_regions(&mut mesh, 0, &["root"]);
        reseal(&mut mesh);

        let err = assemble_physical_structural_system(&model, Some(mesh))
            .expect_err("strict solver mesh assembly should reject unmapped loads");

        assert!(matches!(
            err,
            LinearAssemblyError::SolverMeshRegionMapping(
                SolverMeshRegionMappingError::UnmappedLoadRegion { .. }
            )
        ));
        let message = err.to_string();
        assert!(message.contains("load_id=load_tip"));
        assert!(message.contains("region_id=missing_tip"));
    }

    #[test]
    fn strict_solver_mesh_assembly_rejects_unmapped_constraint_region() {
        let mut model = fixture_model(FixtureId::CantileverLinearStatic);
        model.constraints = vec![fixed_constraint("missing_root")];
        replace_fixture_excitations(
            &mut model,
            vec![FixtureExcitation {
                id: "load_tip".into(),
                region_id: "tip".into(),
                kind: FixtureExcitationKind::Force([0.0, -12.0, 0.0]),
            }],
        );
        let mut mesh = tetrahedron4_mesh();
        clear_boundary_regions(&mut mesh);
        set_boundary_regions(&mut mesh, 0, &["tip"]);
        reseal(&mut mesh);

        let err = assemble_physical_structural_system(&model, Some(mesh))
            .expect_err("strict solver mesh assembly should reject unmapped constraints");

        assert!(matches!(
            err,
            LinearAssemblyError::ConstraintPlan(ConstraintPlanError::EmptyTarget { .. })
        ));
        let message = err.to_string();
        assert!(message.contains("`fixed_root`"));
        assert!(message.contains("`missing_root`"));
    }

    #[test]
    fn solver_mesh_boundary_regions_integrate_pressure_loads() {
        let mut model = fixture_model(FixtureId::CantileverLinearStatic);
        model.constraints = Vec::new();
        replace_fixture_excitations(
            &mut model,
            vec![FixtureExcitation {
                id: "pressure_tip".into(),
                region_id: "tip".into(),
                kind: FixtureExcitationKind::Pressure(12.0),
            }],
        );
        let mut mesh = tetrahedron4_mesh();
        set_boundary_regions(&mut mesh, 1, &["tip"]);
        reseal(&mut mesh);

        let summary = assemble_structural_system(
            &model,
            &static_excitations(&model),
            Some(mesh),
            None,
            None,
            StructuralFormulationRequirement::PhysicalDiscretization,
        )
        .expect("valid pressure boundary should assemble");

        assert_close(summary.operator.rhs[1], -2.0);
        assert_close(summary.operator.rhs[4], -2.0);
        assert_close(summary.operator.rhs[10], -2.0);
        assert_eq!(
            summary.structural_sources,
            vec![SourceImplementation::CanonicalBoundaryPressureIntegration]
        );
    }

    #[test]
    fn tet10_pressure_quadrature_loads_midside_nodes_and_preserves_resultants() {
        let mut model = fixture_model(FixtureId::CantileverLinearStatic);
        model.constraints = Vec::new();
        let mesh = crate::assembly::solver_solid::tests::artifact(ElementOrder::Tet10);
        let region_id = mesh.topology.boundary_faces[1].provenance[0]
            .analysis_selection_id()
            .unwrap();
        replace_fixture_excitations(
            &mut model,
            vec![FixtureExcitation {
                id: "pressure_tip".into(),
                region_id: region_id.clone(),
                kind: FixtureExcitationKind::Pressure(12.0),
            }],
        );
        let projection = crate::topology::AnalysisProjection::derive(&mesh).unwrap();
        let face = projection
            .boundary_faces_matching(&region_id)
            .next()
            .unwrap()
            .clone();

        let summary = assemble_physical_structural_system(&model, Some(mesh))
            .expect("valid quadratic pressure boundary should assemble");
        let mut force = [0.0; 3];
        let mut moment = [0.0; 3];
        for &node_index in &face.node_indices {
            let nodal = [
                summary.operator.rhs[summary
                    .structural_dof_layout
                    .index(node_index, StructuralDofKind::Ux)
                    .unwrap()],
                summary.operator.rhs[summary
                    .structural_dof_layout
                    .index(node_index, StructuralDofKind::Uy)
                    .unwrap()],
                summary.operator.rhs[summary
                    .structural_dof_layout
                    .index(node_index, StructuralDofKind::Uz)
                    .unwrap()],
            ];
            for axis in 0..3 {
                force[axis] += nodal[axis];
            }
            let arm = std::array::from_fn::<_, 3, _>(|axis| {
                projection.nodes[node_index].coordinates_m[axis] - face.centroid_m[axis]
            });
            let nodal_moment = [
                arm[1] * nodal[2] - arm[2] * nodal[1],
                arm[2] * nodal[0] - arm[0] * nodal[2],
                arm[0] * nodal[1] - arm[1] * nodal[0],
            ];
            for axis in 0..3 {
                moment[axis] += nodal_moment[axis];
            }
        }
        let expected_force = face.oriented_area_vector_m2.map(|value| 12.0 * value);
        for axis in 0..3 {
            assert_close(force[axis], expected_force[axis]);
            assert_close(moment[axis], 0.0);
        }
        assert!(face.node_indices[3..].iter().any(|node_index| {
            summary.operator.rhs[summary
                .structural_dof_layout
                .index(*node_index, StructuralDofKind::Uy)
                .unwrap()]
            .abs()
                > 0.0
        }));
        let evidence = summary
            .spatial_projection_evidence
            .first()
            .expect("pressure projection must publish evidence");
        assert_eq!(
            evidence.integration_rule,
            crate::excitation::SpatialIntegrationRule::BoundaryDunavantSeven
        );
        assert_eq!(evidence.resolved_entity_count, 1);
        assert_eq!(evidence.resolved_node_count, 6);
        assert!(evidence.accepted);
        assert!(evidence.resultant_relative_error() <= evidence.relative_tolerance);
        assert!(evidence.moment_relative_error().unwrap() <= evidence.relative_tolerance);
    }

    #[test]
    fn tet10_traction_quadrature_preserves_force_and_moment() {
        let mut model = fixture_model(FixtureId::CantileverLinearStatic);
        model.constraints = Vec::new();
        let mesh = crate::assembly::solver_solid::tests::artifact(ElementOrder::Tet10);
        let region_id = mesh.topology.boundary_faces[1].provenance[0]
            .analysis_selection_id()
            .unwrap();
        let traction = [5.0, -7.0, 11.0];
        replace_fixture_excitations(
            &mut model,
            vec![FixtureExcitation {
                id: "traction_tip".into(),
                region_id: region_id.clone(),
                kind: FixtureExcitationKind::Traction(traction),
            }],
        );
        let projection = crate::topology::AnalysisProjection::derive(&mesh).unwrap();
        let face = projection
            .boundary_faces_matching(&region_id)
            .next()
            .unwrap();

        let summary = assemble_physical_structural_system(&model, Some(mesh))
            .expect("valid quadratic traction boundary should assemble");
        let evidence = summary
            .spatial_projection_evidence
            .first()
            .expect("traction projection must publish evidence");
        assert_eq!(
            evidence.integration_rule,
            crate::excitation::SpatialIntegrationRule::BoundaryDunavantSeven
        );
        let crate::excitation::SpatialProjectionBalance::VectorResultant {
            requested_resultant,
            projected_resultant,
            ..
        } = evidence.balance
        else {
            panic!("traction projection must report a vector resultant")
        };
        for (axis, component) in traction.into_iter().enumerate() {
            assert_close(requested_resultant.real[axis], component * face.area_m2);
            assert_close(projected_resultant.real[axis], component * face.area_m2);
        }
        assert_eq!(evidence.resolved_node_count, 6);
        assert!(evidence.accepted);
        assert!(evidence.resultant_relative_error() <= evidence.relative_tolerance);
        assert!(evidence.moment_relative_error().unwrap() <= evidence.relative_tolerance);
    }

    #[test]
    fn tet10_body_force_quadrature_preserves_integrated_volume_resultant() {
        let mut model = fixture_model(FixtureId::CantileverLinearStatic);
        model.constraints = Vec::new();
        let mesh = crate::assembly::solver_solid::tests::artifact(ElementOrder::Tet10);
        let region_id = mesh.topology.volume_elements[0]
            .region_id
            .analysis_selection_id()
            .unwrap();
        replace_fixture_excitations(
            &mut model,
            vec![FixtureExcitation {
                id: "body_force".into(),
                region_id,
                kind: FixtureExcitationKind::BodyForce([3.0, -6.0, 9.0]),
            }],
        );
        let projection = crate::topology::AnalysisProjection::derive(&mesh).unwrap();
        let volume = projection.elements[0].signed_volume_m3;

        let summary = assemble_physical_structural_system(&model, Some(mesh))
            .expect("valid quadratic body force should assemble");
        let mut force = [0.0; 3];
        for node_index in 0..projection.nodes.len() {
            for (axis, kind) in [
                StructuralDofKind::Ux,
                StructuralDofKind::Uy,
                StructuralDofKind::Uz,
            ]
            .into_iter()
            .enumerate()
            {
                force[axis] += summary.operator.rhs[summary
                    .structural_dof_layout
                    .index(node_index, kind)
                    .unwrap()];
            }
        }
        for (actual, expected) in force.into_iter().zip([3.0, -6.0, 9.0]) {
            assert_close(actual, expected * volume);
        }
        let evidence = summary
            .spatial_projection_evidence
            .first()
            .expect("body-force projection must publish evidence");
        assert_eq!(
            evidence.integration_rule,
            crate::excitation::SpatialIntegrationRule::VolumeKeastFifteen
        );
        assert_eq!(evidence.resolved_entity_count, 1);
        assert_eq!(evidence.resolved_node_count, 10);
        assert!(evidence.accepted);
        assert!(evidence.resultant_relative_error() <= evidence.relative_tolerance);
        assert!(evidence.moment_relative_error().unwrap() <= evidence.relative_tolerance);
    }

    #[test]
    fn solver_mesh_reference_point_coupling_receives_wrench_moments() {
        let mut model = fixture_model(FixtureId::CantileverLinearStatic);
        model.constraints = vec![runmat_analysis_core::AnalysisConstraint::Structural(
            runmat_analysis_core::StructuralConstraint {
                constraint_id: "tip_coupling".into(),
                frame: runmat_analysis_core::CoordinateFrame::Global,
                kind: runmat_analysis_core::StructuralConstraintKind::DistributingCoupling(
                    runmat_analysis_core::ReferencePointCoupling {
                        target: runmat_analysis_core::RegionTarget::boundary("tip"),
                        reference_point: runmat_analysis_core::ReferencePoint {
                            reference_point_id: "tip".into(),
                            position: runmat_analysis_core::Position::new([0.0, 0.0, 0.0]),
                        },
                    },
                ),
            },
        )];
        replace_fixture_excitations(
            &mut model,
            vec![FixtureExcitation {
                id: "wrench_tip".into(),
                region_id: "tip".into(),
                kind: FixtureExcitationKind::Wrench {
                    force: [0.0, 0.0, 0.0],
                    moment: [0.0, 6.0, 0.0],
                    point_m: [0.0, 0.0, 0.0],
                },
            }],
        );
        let mut mesh = tetrahedron4_mesh();
        set_boundary_regions(&mut mesh, 1, &["tip"]);
        reseal(&mut mesh);

        let summary = assemble_structural_system(
            &model,
            &static_excitations(&model),
            Some(mesh),
            None,
            None,
            StructuralFormulationRequirement::PhysicalDiscretization,
        )
        .expect("valid wrench boundary should assemble");

        assert!(summary.structural_moment_projection.is_empty());
        assert_eq!(summary.structural_dof_layout.reference_point_count(), 1);
        let reference_ry = summary
            .structural_dof_layout
            .reference_point_index(0, StructuralDofKind::Ry)
            .unwrap();
        assert_close(summary.operator.rhs[reference_ry], 6.0);
        assert_eq!(
            summary.structural_sources,
            vec![SourceImplementation::ReferencePointGeneralizedLoad]
        );
        assert_eq!(summary.spatial_projection_evidence.len(), 2);
        assert!(summary.spatial_projection_evidence.iter().all(|evidence| {
            evidence.integration_rule
                == crate::excitation::SpatialIntegrationRule::ReferencePointGeneralized
                && evidence.accepted
                && evidence.resolved_entity_count == 1
                && evidence.resolved_node_count == 0
        }));
    }

    #[test]
    fn malformed_explicit_elements_return_typed_errors_without_synthetic_fallback() {
        let base = fixture_model(FixtureId::StructuralBeamCantileverEndMomentReference);

        let mut unknown_node = base.clone();
        let StructuralElementKind::Beam(beam) = &mut unknown_node
            .structural
            .as_mut()
            .expect("fixture has structural topology")
            .elements[0]
            .kind
        else {
            panic!("fixture must contain a beam")
        };
        beam.node_ids[1] = u32::MAX;

        let mut unknown_section = base.clone();
        let StructuralElementKind::Beam(beam) = &mut unknown_section
            .structural
            .as_mut()
            .expect("fixture has structural topology")
            .elements[0]
            .kind
        else {
            panic!("fixture must contain a beam")
        };
        beam.section_id = "missing_section".to_string();

        let mut degenerate = base.clone();
        let structural = degenerate
            .structural
            .as_mut()
            .expect("fixture has structural topology");
        structural.nodes[1].coordinates_m = structural.nodes[0].coordinates_m;

        let shell_base = fixture_model(FixtureId::StructuralShellPlateMomentReference);
        let mut shell_unknown_node = shell_base.clone();
        let StructuralElementKind::Shell(shell) = &mut shell_unknown_node
            .structural
            .as_mut()
            .expect("fixture has structural topology")
            .elements[0]
            .kind
        else {
            panic!("fixture must contain a shell")
        };
        shell.node_ids[2] = u32::MAX - 1;

        let mut shell_unknown_section = shell_base.clone();
        let StructuralElementKind::Shell(shell) = &mut shell_unknown_section
            .structural
            .as_mut()
            .expect("fixture has structural topology")
            .elements[0]
            .kind
        else {
            panic!("fixture must contain a shell")
        };
        shell.section_id = "missing_shell_section".to_string();

        let mut shell_degenerate = shell_base.clone();
        let structural = shell_degenerate
            .structural
            .as_mut()
            .expect("fixture has structural topology");
        structural.nodes[1].coordinates_m = structural.nodes[0].coordinates_m;
        structural.nodes[2].coordinates_m = structural.nodes[0].coordinates_m;

        let element_id = base
            .structural
            .as_ref()
            .expect("fixture has structural topology")
            .elements[0]
            .element_id
            .clone();
        let shell_element_id = shell_base
            .structural
            .as_ref()
            .expect("fixture has structural topology")
            .elements[0]
            .element_id
            .clone();
        let cases = [
            (
                "unknown node",
                unknown_node,
                StructuralAssemblyError::UnknownNode {
                    element_id: element_id.clone(),
                    node_id: u32::MAX,
                },
            ),
            (
                "unknown section",
                unknown_section,
                StructuralAssemblyError::UnknownBeamSection {
                    element_id: element_id.clone(),
                    section_id: "missing_section".to_string(),
                },
            ),
            (
                "degenerate geometry",
                degenerate,
                StructuralAssemblyError::BeamElement {
                    element_id: element_id.clone(),
                    error: BeamElementError::DegenerateLength,
                },
            ),
            (
                "unknown shell node",
                shell_unknown_node,
                StructuralAssemblyError::UnknownNode {
                    element_id: shell_element_id.clone(),
                    node_id: u32::MAX - 1,
                },
            ),
            (
                "unknown shell section",
                shell_unknown_section,
                StructuralAssemblyError::UnknownShellSection {
                    element_id: shell_element_id.clone(),
                    section_id: "missing_shell_section".to_string(),
                },
            ),
            (
                "degenerate shell geometry",
                shell_degenerate,
                StructuralAssemblyError::ShellElement {
                    element_id: shell_element_id,
                    error: ShellElementError::DegenerateArea,
                },
            ),
        ];

        for (name, model, expected) in cases {
            let error = assemble_structural_system(
                &model,
                &static_excitations(&model),
                None,
                None,
                None,
                StructuralFormulationRequirement::ExperimentalSyntheticOperator,
            )
            .expect_err(name);
            assert_eq!(error, LinearAssemblyError::Structural(expected), "{name}");
        }
    }

    #[test]
    fn mixed_explicit_dimensions_are_rejected_without_partial_assembly() {
        let mut model = fixture_model(FixtureId::StructuralBeamCantileverEndMomentReference);
        let shell_model = fixture_model(FixtureId::StructuralShellPlateMomentReference);
        let structural = model
            .structural
            .as_mut()
            .expect("beam fixture has structural topology");
        let shell = shell_model
            .structural
            .expect("shell fixture has structural topology");
        structural.elements.push(shell.elements[0].clone());
        structural.shell_sections = shell.shell_sections;

        let error = assemble_structural_system(
            &model,
            &static_excitations(&model),
            None,
            None,
            None,
            StructuralFormulationRequirement::ExperimentalSyntheticOperator,
        )
        .expect_err("mixed beam-shell topology must require a coupling formulation");

        assert_eq!(
            error,
            LinearAssemblyError::Structural(StructuralAssemblyError::MixedExplicitDimensions {
                beam_element_count: 1,
                shell_element_count: 1,
            })
        );
    }

    #[test]
    fn physical_requirement_rejects_meshless_structural_assembly() {
        let model = fixture_model(FixtureId::CantileverLinearStatic);
        let error = assemble_structural_system(
            &model,
            &static_excitations(&model),
            None,
            None,
            None,
            StructuralFormulationRequirement::PhysicalDiscretization,
        )
        .expect_err("physical structural assembly must not synthesize an operator");

        assert_eq!(
            error,
            LinearAssemblyError::Structural(StructuralAssemblyError::MissingDiscretization)
        );
    }

    #[test]
    fn explicit_connectivity_rejects_unimplemented_sources_without_fallback() {
        let cases = [
            (
                FixtureExcitationKind::Pressure(12.0),
                StructuralLoadKind::Pressure,
            ),
            (
                FixtureExcitationKind::BodyForce([0.0, -9.81, 0.0]),
                StructuralLoadKind::BodyForce,
            ),
        ];
        for (kind, expected_kind) in cases {
            let mut model = fixture_model(FixtureId::StructuralBeamCantileverEndMomentReference);
            replace_fixture_excitations(
                &mut model,
                vec![FixtureExcitation {
                    id: format!("unsupported_{expected_kind}"),
                    region_id: "tip".to_string(),
                    kind,
                }],
            );
            let error = assemble_structural_system(
                &model,
                &static_excitations(&model),
                None,
                None,
                None,
                StructuralFormulationRequirement::PhysicalDiscretization,
            )
            .expect_err("unsupported explicit sources must not silently disappear");
            assert!(matches!(
                error,
                LinearAssemblyError::Structural(
                    StructuralAssemblyError::UnsupportedStructuralLoad { load_kind, .. }
                ) if load_kind == expected_kind
            ));
        }
    }

    #[test]
    fn structural_validation_rejects_cross_physics_sources() {
        let model = fixture_model(FixtureId::CantileverLinearStatic);
        let cases = [
            (
                RealizedExcitationKind::ElectricCurrent(crate::excitation::ComplexScalar::real(
                    4.0,
                )),
                StructuralLoadKind::ElectricCurrent,
                runmat_analysis_core::PhysicalQuantityKind::ElectricCurrent,
            ),
            (
                RealizedExcitationKind::CurrentDensity(crate::excitation::ComplexVector3::real([
                    1.0, 2.0, 3.0,
                ])),
                StructuralLoadKind::CurrentDensity,
                runmat_analysis_core::PhysicalQuantityKind::ElectricCurrentDensity,
            ),
        ];
        for (kind, expected_kind, quantity) in cases {
            let excitations =
                RealizedExcitations::new(vec![crate::excitation::RealizedExcitation {
                    excitation_id: runmat_analysis_core::ExcitationId(format!(
                        "unsupported_{expected_kind}"
                    )),
                    case_id: runmat_analysis_core::ExcitationCaseId("case".to_string()),
                    signal_id: runmat_analysis_core::SignalId("signal".to_string()),
                    quantity,
                    target: crate::excitation::RealizedExcitationTarget::WholeModel,
                    kind,
                    policy: crate::excitation::RealizedExcitationPolicy {
                        frame: runmat_analysis_core::CoordinateFrame::Global,
                        distribution: runmat_analysis_core::ExcitationDistribution::Concentrated,
                        kinematics: runmat_analysis_core::ExcitationKinematics::FixedFrame,
                        projection: runmat_analysis_core::ProjectionRequirement {
                            preserve_resultant: false,
                            preserve_moment: false,
                            relative_tolerance: 1.0e-12,
                        },
                    },
                }]);
            let error = validate_structural_inputs(
                &model,
                &excitations,
                StructuralProjectionBasis::Synthetic,
            )
            .expect_err("electrical sources must not become a mechanical right-hand side");
            assert!(matches!(
                error,
                StructuralAssemblyError::UnsupportedStructuralLoad { load_kind, .. }
                    if load_kind == expected_kind
            ));
        }
    }

    #[test]
    fn assembly_consumes_the_caller_selected_excitation_sample() {
        let model = fixture_model(FixtureId::CantileverLinearStatic);
        let first_sample = static_excitations(&model);
        let second_sample = scaled_force_excitations(&first_sample, 2.0);

        let first = assemble_linear_system(&model, &first_sample, None, None, None)
            .expect("first sampled load should assemble");
        let second = assemble_linear_system(&model, &second_sample, None, None, None)
            .expect("second sampled load should assemble");

        assert_eq!(first.operator.stiffness_csr, second.operator.stiffness_csr);
        assert_eq!(first.operator.rhs.len(), second.operator.rhs.len());
        assert!(first.operator.rhs.iter().any(|value| value.abs() > 0.0));
        for (first_value, second_value) in first.operator.rhs.iter().zip(&second.operator.rhs) {
            assert!((second_value - 2.0 * first_value).abs() <= 1.0e-12);
        }
    }

    #[test]
    fn nonstructural_scaffolding_ignores_explicit_structural_connectivity() {
        let mut model = fixture_model(FixtureId::StructuralBeamCantileverEndMomentReference);
        replace_fixture_with_electric_current(&mut model, "coil_current", "beam_span", 4.0);
        model.constraints.clear();

        let excitations = static_excitations(&model);
        let summary = assemble_linear_system(&model, &excitations, None, None, None)
            .expect("nonstructural scaffolding must not select explicit structural assembly");

        assert_eq!(summary.load_count, 1);
        assert_eq!(summary.structural_beam_element_count, 0);
        assert!(summary.structural_formulation.is_none());
        assert!(summary.operator.rhs.iter().any(|value| value.abs() > 0.0));

        let mesh_summary =
            assemble_linear_system(&model, &excitations, Some(tetrahedron4_mesh()), None, None)
                .expect("a solver mesh must not route a nonstructural source through mechanics");
        assert!(mesh_summary.structural_formulation.is_none());
        assert_eq!(
            mesh_summary.structural_sources,
            vec![SourceImplementation::NotApplicable]
        );
        assert_eq!(
            mesh_summary.structural_boundary_transfers,
            vec![BoundaryTransferImplementation::NotApplicable]
        );
        assert!(mesh_summary
            .operator
            .rhs
            .iter()
            .any(|value| value.abs() > 0.0));
    }

    #[test]
    fn explicit_connectivity_rejects_unmapped_load_targets() {
        let mut model = fixture_model(FixtureId::StructuralBeamCantileverEndMomentReference);
        replace_fixture_excitations(
            &mut model,
            vec![FixtureExcitation {
                id: "missing_tip".to_string(),
                region_id: "not_a_connectivity_region".to_string(),
                kind: FixtureExcitationKind::Force([0.0, -12.0, 0.0]),
            }],
        );
        let error = assemble_structural_system(
            &model,
            &static_excitations(&model),
            None,
            None,
            None,
            StructuralFormulationRequirement::PhysicalDiscretization,
        )
        .expect_err("unmapped explicit source must not be dropped");
        assert!(matches!(
            error,
            LinearAssemblyError::Structural(StructuralAssemblyError::UnmappedStructuralLoad {
                load_kind: StructuralLoadKind::Force,
                ..
            })
        ));
    }

    #[test]
    fn structural_assembly_rejects_scheduled_prescribed_constraints() {
        let mut model = fixture_model(FixtureId::StructuralBeamCantileverEndMomentReference);
        model.constraints = vec![runmat_analysis_core::AnalysisConstraint::Structural(
            runmat_analysis_core::StructuralConstraint {
                constraint_id: "scheduled_root".to_string(),
                frame: runmat_analysis_core::CoordinateFrame::Global,
                kind: runmat_analysis_core::StructuralConstraintKind::Prescribed(
                    runmat_analysis_core::PrescribedStructuralDofs {
                        target: runmat_analysis_core::RegionTarget::authored("root"),
                        translations: vec![runmat_analysis_core::TranslationDofValue {
                            dof: runmat_analysis_core::StructuralTranslationDof::Ux,
                            value: runmat_analysis_core::PrescribedValue::Signal(
                                runmat_analysis_core::SignalId("time_dependent".to_string()),
                            ),
                        }],
                        rotations: Vec::new(),
                    },
                ),
            },
        )];
        let error = assemble_structural_system(
            &model,
            &static_excitations(&model),
            None,
            None,
            None,
            StructuralFormulationRequirement::PhysicalDiscretization,
        )
        .expect_err("unsupported scheduled constraints must not vanish during assembly");
        assert_eq!(
            error,
            LinearAssemblyError::Structural(
                StructuralAssemblyError::UnsupportedScheduledConstraint {
                    constraint_id: "scheduled_root".to_string(),
                }
            )
        );
    }

    #[test]
    fn canonical_solid_records_executed_source_and_constraint_projection() {
        let mut model = fixture_model(FixtureId::CantileverLinearStatic);
        model.constraints = vec![fixed_constraint("root")];
        replace_fixture_excitations(
            &mut model,
            vec![FixtureExcitation {
                id: "load_tip".to_string(),
                region_id: "tip".to_string(),
                kind: FixtureExcitationKind::Force([0.0, -12.0, 0.0]),
            }],
        );
        let mut mesh = tetrahedron4_mesh();
        set_boundary_regions(&mut mesh, 0, &["root"]);
        set_boundary_regions(&mut mesh, 1, &["tip"]);
        reseal(&mut mesh);
        let summary = assemble_structural_system(
            &model,
            &static_excitations(&model),
            Some(mesh),
            None,
            None,
            StructuralFormulationRequirement::PhysicalDiscretization,
        )
        .expect("supported solid projections should assemble");
        assert_eq!(
            summary.structural_sources,
            vec![SourceImplementation::CanonicalBoundaryProjection]
        );
        assert_eq!(
            summary.structural_boundary_transfers,
            vec![BoundaryTransferImplementation::CanonicalBoundaryProjection]
        );
    }

    #[test]
    fn structural_formulation_evidence_identifies_every_implemented_operator_exactly() {
        let mut solid_model = fixture_model(FixtureId::CantileverLinearStatic);
        solid_model.excitations.clear();
        solid_model.constraints.clear();
        let tet4 = assemble_structural_system(
            &solid_model,
            &static_excitations(&solid_model),
            Some(crate::assembly::solver_solid::tests::artifact(
                ElementOrder::Tet4,
            )),
            None,
            None,
            StructuralFormulationRequirement::PhysicalDiscretization,
        )
        .expect("Tet4 assembly should succeed");
        let tet10 = assemble_structural_system(
            &solid_model,
            &static_excitations(&solid_model),
            Some(crate::assembly::solver_solid::tests::artifact(
                ElementOrder::Tet10,
            )),
            None,
            None,
            StructuralFormulationRequirement::PhysicalDiscretization,
        )
        .expect("Tet10 assembly should succeed");
        let beam_model = fixture_model(FixtureId::StructuralBeamCantileverEndMomentReference);
        let beam = assemble_structural_system(
            &beam_model,
            &static_excitations(&beam_model),
            None,
            None,
            None,
            StructuralFormulationRequirement::PhysicalDiscretization,
        )
        .expect("beam assembly should succeed");
        let shell_model = fixture_model(FixtureId::StructuralShellPlateMomentReference);
        let shell = assemble_structural_system(
            &shell_model,
            &static_excitations(&shell_model),
            None,
            None,
            None,
            StructuralFormulationRequirement::PhysicalDiscretization,
        )
        .expect("shell assembly should succeed");
        let synthetic_model = fixture_model(FixtureId::CantileverLinearStatic);
        let synthetic = assemble_structural_system(
            &synthetic_model,
            &static_excitations(&synthetic_model),
            None,
            None,
            None,
            StructuralFormulationRequirement::ExperimentalSyntheticOperator,
        )
        .expect("typed experimental admission should permit synthetic assembly");

        let cases = [
            (
                tet4,
                StructuralOperatorImplementation::Tet4LinearElasticity,
                Some(StructuralRecoveryImplementation::Tet4ConstantStrain),
                SourceImplementation::NotApplicable,
                BoundaryTransferImplementation::NotApplicable,
            ),
            (
                tet10,
                StructuralOperatorImplementation::Tet10LinearElasticity,
                None,
                SourceImplementation::NotApplicable,
                BoundaryTransferImplementation::NotApplicable,
            ),
            (
                beam,
                StructuralOperatorImplementation::EulerBernoulliBeam3d,
                Some(StructuralRecoveryImplementation::BeamSectionResultants),
                SourceImplementation::ExplicitRegionNodalDistribution,
                BoundaryTransferImplementation::ExplicitRegionConnectivity,
            ),
            (
                shell,
                StructuralOperatorImplementation::PairSpringShellProxy,
                Some(StructuralRecoveryImplementation::PairSpringShellDerivedResultants),
                SourceImplementation::ExplicitRegionNodalDistribution,
                BoundaryTransferImplementation::ExplicitRegionConnectivity,
            ),
            (
                synthetic,
                StructuralOperatorImplementation::SyntheticDiagonalChain,
                None,
                SourceImplementation::OrdinalDistribution,
                BoundaryTransferImplementation::GlobalCountProfile,
            ),
        ];

        for (summary, operator, recovery, source, boundary) in cases {
            let realization = summary
                .structural_formulation
                .expect("structural assembly must record its exact realization");
            assert_eq!(realization.operator, operator);
            assert_eq!(realization.recovery, recovery);
            assert_eq!(summary.structural_sources, vec![source]);
            assert_eq!(summary.structural_boundary_transfers, vec![boundary]);
        }
    }

    fn tetrahedron4_mesh() -> SolverMeshArtifact {
        crate::assembly::solver_solid::tests::artifact(ElementOrder::Tet4)
    }

    fn set_boundary_regions(mesh: &mut SolverMeshArtifact, face_index: usize, regions: &[&str]) {
        mesh.topology.boundary_faces[face_index].provenance = regions
            .iter()
            .map(|region| PersistentEntityId {
                kind: PersistentEntityKind::Face,
                source_topology_id: (*region).to_owned(),
                assembly_path: vec!["root".to_owned()],
            })
            .collect();
        mesh.topology.boundary_faces[face_index].provenance.sort();
    }

    fn clear_boundary_regions(mesh: &mut SolverMeshArtifact) {
        for (index, face) in mesh.topology.boundary_faces.iter_mut().enumerate() {
            face.provenance = vec![PersistentEntityId {
                kind: PersistentEntityKind::Face,
                source_topology_id: format!("unselected:{index}"),
                assembly_path: vec!["root".to_owned()],
            }];
        }
    }

    fn set_solver_mesh_region(mesh: &mut SolverMeshArtifact, region_id: &str) {
        let region = PersistentEntityId {
            kind: PersistentEntityKind::Region,
            source_topology_id: region_id.to_owned(),
            assembly_path: vec!["root".to_owned()],
        };
        mesh.topology.volume_elements[0].region_id = region.clone();
        mesh.topology.regions[0].region_id = region;
        reseal(mesh);
    }

    fn reseal(mesh: &mut SolverMeshArtifact) {
        mesh.canonical_digest = StableDigest::ZERO;
        mesh.seal_canonical_digest()
            .expect("test solver mesh should remain canonical");
    }

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() <= 1.0e-8,
            "expected {expected}, got {actual}"
        );
    }

    fn first_csr_diagonal(summary: &AssemblySummary) -> f64 {
        let csr = summary
            .operator
            .stiffness_csr
            .as_ref()
            .expect("solver mesh should assemble CSR stiffness");
        csr.column_indices[csr.row_offsets[0]..csr.row_offsets[1]]
            .iter()
            .zip(csr.values[csr.row_offsets[0]..csr.row_offsets[1]].iter())
            .find_map(|(&column, &value)| (column == 0).then_some(value.abs()))
            .expect("first row should contain diagonal")
    }
}
