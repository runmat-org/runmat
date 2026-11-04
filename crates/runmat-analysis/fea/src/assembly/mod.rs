pub mod complex_structural;
mod constraint_projection;
mod coupling_summary;
mod direct_constraints;
pub mod dofs;
pub mod elements;
mod entry_points;
pub(crate) mod explicit_connectors;
mod explicit_elements;
pub(crate) use explicit_elements::ExplicitCapacityPhase;
mod explicit_loads;
mod explicit_system;
pub mod identity;
mod mixed_structural;
mod moment_projection;
pub(crate) mod nonlinear_solid;
pub mod physical_fields;
mod reduced_material_summary;
mod solid_matrix;
mod solid_properties;
pub(crate) mod solid_recovery;
mod solid_recovery_types;
pub mod solver_solid;
mod solver_solid_boundary;
mod structural_inputs;
pub use structural_inputs::StructuralLoadKind;
mod structural_loads;
mod transient_basis;
pub(crate) use transient_basis::PreparedStructuralTransientBasis;
pub(crate) mod thermoelastic_load;

use std::{collections::BTreeMap, fmt};

#[cfg(test)]
use runmat_analysis_core::StructuralElementKind;
use runmat_analysis_core::{
    AnalysisModel, BoundaryTransferImplementation, SourceImplementation,
    StructuralFormulationRealization, StructuralOperatorImplementation,
    StructuralRecoveryImplementation, StructuralTranslationDof,
};
use runmat_meshing_core::{ElementOrder, SolverMeshArtifact};
use serde::{Deserialize, Serialize};
pub(crate) use {entry_points::*, mixed_structural::mixed_dof_layout};

use self::{
    constraint_projection::{
        apply_direct_constraints, resolve_constraint_matrix, rotational_row_count,
    },
    direct_constraints::DirectConstraintAssignments,
    dofs::{StructuralDofKind, StructuralDofLayout, StructuralNodeDofSet},
    elements::{
        beam::{BeamElementError, BeamMaterial, BeamSection, BeamTransform12},
        shell::{ShellConstitutive, ShellElementError, ShellElementGeometry},
    },
    explicit_connectors::{
        assemble_connectors_with_resolver, assemble_explicit_connectors, resolve_projected_surface,
        ConnectorAssemblyContext, ConnectorTargetRole, ResolvedConnectorTarget,
    },
    explicit_elements::{
        assemble_explicit_elements_with_owner, validate_explicit_topology, ExplicitElementAssembly,
    },
    solver_solid::{solver_solid_topology, SolverSolidAssemblyError},
    structural_inputs::{
        apply_synthetic_structural_rhs, explicit_load_kind, requested_moment_norm,
        structural_boundary_implementations, structural_source_implementations,
        validate_structural_inputs, StructuralProjectionBasis,
    },
    structural_loads::{
        apply as apply_structural_load_plan, compile_explicit as compile_explicit_load_plan,
        compile_solid as compile_solid_load_plan,
    },
};
use crate::physics::structural::topology::explicit_target_node_indices as structural_target_nodes;
use moment_projection::project_moment;
use reduced_material_summary::structural_material_summary;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum StructuralCapacityPhase {
    Solid(solver_solid::SolidCapacityPhase),
    Explicit(explicit_elements::ExplicitCapacityPhase),
    ThermoelasticLoadElement { element: usize },
    SolidRecoveryElement { element: usize },
}

struct StructuralCouplingContexts {
    thermo_mechanical: Option<FeaThermoMechanicalContext>,
    electro_thermal: Option<FeaElectroThermalContext>,
}

struct StructuralAssemblyBatch {
    maximum_block_elements: usize,
    awaited_elements: bool,
}

fn thermoelastic_capacity_error<A: From<LinearAssemblyError>>(
    error: thermoelastic_load::ThermoelasticLoadCapacityError<A>,
) -> A {
    match error {
        thermoelastic_load::ThermoelasticLoadCapacityError::Admission(error) => error,
        thermoelastic_load::ThermoelasticLoadCapacityError::Numerical(
            thermoelastic_load::ThermoelasticLoadError::Accounting(cause),
        ) => LinearAssemblyError::Structural(StructuralAssemblyError::Accounting(cause)).into(),
        thermoelastic_load::ThermoelasticLoadCapacityError::Numerical(
            thermoelastic_load::ThermoelasticLoadError::Cancelled,
        ) => LinearAssemblyError::Structural(StructuralAssemblyError::Cancelled).into(),
        thermoelastic_load::ThermoelasticLoadCapacityError::Numerical(error) => {
            LinearAssemblyError::CouplingRealization(error.to_string()).into()
        }
    }
}

use crate::algebra::{
    AssembledRealSystem, CanonicalRealOperators, ContributionDomain, ContributionId,
    ContributionProvenance, MatrixProperties, NullspaceEliminationEvidence, OperatorRole,
    RealSparseAssembler, RealSparseMatrix,
};
use crate::constraints::{
    AffineConstraintReduction, CanonicalStructuralConstraintPlan, CanonicalStructuralDof,
    CanonicalStructuralDofOwner, ConstraintAlgebraError, ConstraintEnforcement,
    ConstraintPlanError, ReactionRecovery, ResolvedConstraintMatrix,
};
use crate::excitation::{RealizedExcitationKind, RealizedExcitations};
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
    /// Canonical node-space index for each authored explicit structural node, in authored-node
    /// order. Conformal mixed nodes may point into the solid prefix; nonconformal nodes point into
    /// the appended explicit suffix.
    #[serde(default)]
    pub structural_explicit_node_indices: Vec<usize>,
    #[serde(default)]
    pub structural_load_plan_identity: Option<runmat_execution::Digest>,
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
    pub structural_connector_stiffness_count: usize,
    #[serde(default)]
    pub structural_connector_damping_count: usize,
    #[serde(default)]
    pub structural_connector_plan: Option<crate::connectors::CanonicalStructuralConnectorPlan>,
    #[serde(default)]
    pub structural_solid_element_count: usize,
    #[serde(default)]
    pub structural_solid_total_mass_kg: f64,
    #[serde(default)]
    pub structural_solid_recovery: Vec<SolidRecoveryElementSummary>,
    #[serde(default)]
    pub structural_dof_layout: StructuralDofLayout,
    /// Actual basis when the current operator has already been reduced.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub structural_reduced_basis: Option<physical_fields::StructuralReducedBasis>,
    #[serde(default)]
    pub structural_beam_recovery: Vec<BeamRecoveryElementSummary>,
    #[serde(default)]
    pub structural_shell_recovery: Vec<ShellRecoveryElementSummary>,
    #[serde(default)]
    pub structural_formulations: Vec<StructuralFormulationRealization>,
    #[serde(default)]
    pub structural_property_map_identity: Option<crate::properties::CompiledPropertyMapIdentity>,
    #[serde(default)]
    pub structural_sources: Vec<SourceImplementation>,
    #[serde(default)]
    pub structural_boundary_transfers: Vec<BoundaryTransferImplementation>,
    pub constrained_dof_count: usize,
    #[serde(default)]
    pub constraint_matrix: Option<ResolvedConstraintMatrix>,
    #[serde(default)]
    pub affine_constraint_reduction: Option<AffineConstraintReduction>,
    /// Physical generalized forcing before essential-constraint elimination. Prescribed values
    /// are carried by the constraint channel and never overwrite this load authority.
    pub physical_load: Vec<f64>,
    pub load_count: usize,
    pub structural_material: StructuralMaterialSummary,
    pub thermo_mechanical: Option<ThermoMechanicalAssemblySummary>,
    pub electro_thermal: Option<ElectroThermalAssemblySummary>,
    pub operator: AssembledRealSystem,
}

#[derive(Debug, Clone, PartialEq)]
pub enum LinearAssemblyError {
    CouplingRealization(String),
    SparseOperator(String),
    ConstraintPlan(ConstraintPlanError),
    ConstraintAlgebra(ConstraintAlgebraError),
    Structural(StructuralAssemblyError),
    SolidOperator(SolverSolidAssemblyError),
    SolverMeshRegionMapping(SolverMeshRegionMappingError),
}

impl fmt::Display for LinearAssemblyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LinearAssemblyError::CouplingRealization(message) => {
                write!(f, "coupling realization failed: {message}")
            }
            LinearAssemblyError::SparseOperator(message) => {
                write!(f, "sparse operator construction failed: {message}")
            }
            LinearAssemblyError::ConstraintPlan(err) => {
                write!(f, "constraint planning failed: {err}")
            }
            LinearAssemblyError::ConstraintAlgebra(err) => {
                write!(f, "constraint algebra preparation failed: {err}")
            }
            LinearAssemblyError::Structural(err) => write!(f, "{err}"),
            LinearAssemblyError::SolidOperator(err) => {
                write!(f, "solid operator assembly failed: {err:?}")
            }
            LinearAssemblyError::SolverMeshRegionMapping(err) => {
                write!(f, "{err}")
            }
        }
    }
}

impl std::error::Error for LinearAssemblyError {}

impl From<SolverSolidAssemblyError> for LinearAssemblyError {
    fn from(error: SolverSolidAssemblyError) -> Self {
        Self::SolidOperator(error)
    }
}

impl From<StructuralAssemblyError> for LinearAssemblyError {
    fn from(error: StructuralAssemblyError) -> Self {
        Self::Structural(error)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum StructuralAssemblyError {
    SparseOperator(String),
    LocalExecution(String),
    Accounting(runmat_simulation_usage::LedgerError),
    Cancelled,
    MissingDiscretization,
    MixedTopology(String),
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
    MissingRotationalDof {
        element_id: String,
    },
    InvalidDirectConstraintValue {
        constraint_id: String,
    },
    ConflictingDirectConstraints {
        first_constraint_id: String,
        second_constraint_id: String,
        node_index: usize,
        dof: StructuralDofKind,
    },
    InvalidDirectConstraintState {
        expected: usize,
        actual: usize,
    },
    InvalidCanonicalNodeMap {
        expected: usize,
        actual: usize,
    },
    UnknownBeamSection {
        element_id: String,
        section_id: String,
    },
    UnknownShellSection {
        element_id: String,
        section_id: String,
    },
    UnassignedExplicitRegion {
        element_id: String,
        region_id: String,
    },
    InvalidMaterialCoverage(String),
    ExplicitTopologyIdentity(String),
    InvalidConstraintPlan(String),
    InvalidLoadPlan(String),
    InvalidSpatialProjectionEvidence {
        load_id: String,
        reason: String,
    },
    ProjectionConservation {
        load_id: String,
        resultant_relative_error: f64,
        moment_relative_error: f64,
        relative_tolerance: f64,
    },
    UnmappedExplicitConnector {
        connector_id: String,
        target: &'static str,
        region_id: String,
    },
    InvalidExplicitConnector {
        connector_id: String,
        reason: &'static str,
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
    UnmappedMixedTarget {
        target_id: String,
        selector: String,
    },
    AmbiguousMixedTarget {
        target_id: String,
        selector: String,
        solid_node_count: usize,
        explicit_node_count: usize,
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
            Self::SparseOperator(message) => {
                write!(f, "structural sparse operator construction failed: {message}")
            }
            Self::LocalExecution(message) => {
                write!(f, "structural local execution failed: {message}")
            }
            Self::Cancelled => write!(f, "structural assembly cancelled"),
            Self::MissingDiscretization => write!(
                f,
                "structural analysis requires a canonical solid mesh or explicit beam/shell connectivity"
            ),
            Self::MixedTopology(message) => {
                write!(f, "mixed structural topology is invalid: {message}")
            }
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
            Self::MissingRotationalDof { element_id } => write!(
                f,
                "explicit structural element requires rotational DOFs on every attached canonical node: element_id={element_id}"
            ),
            Self::InvalidDirectConstraintValue { constraint_id } => write!(
                f,
                "structural constraint `{constraint_id}` resolves to a non-finite direct value"
            ),
            Self::ConflictingDirectConstraints {
                first_constraint_id,
                second_constraint_id,
                node_index,
                dof,
            } => write!(
                f,
                "structural constraints `{first_constraint_id}` and `{second_constraint_id}` prescribe conflicting values on canonical node {node_index} DOF {dof:?}"
            ),
            Self::InvalidDirectConstraintState { expected, actual } => write!(
                f,
                "direct structural constraint state has {actual} solver rows; expected {expected}"
            ),
            Self::InvalidCanonicalNodeMap { expected, actual } => write!(
                f,
                "explicit structural node projection has the wrong size: expected={expected} actual={actual}"
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
            Self::UnassignedExplicitRegion {
                element_id,
                region_id,
            } => write!(
                f,
                "explicit structural element region has no material assignment: element_id={element_id} region_id={region_id}"
            ),
            Self::InvalidMaterialCoverage(message) => {
                write!(f, "explicit structural material coverage is invalid: {message}")
            }
            Self::ExplicitTopologyIdentity(message) => write!(
                f,
                "explicit structural topology identity could not be derived: {message}"
            ),
            Self::InvalidConstraintPlan(message) => write!(
                f,
                "explicit structural constraint plan is invalid: {message}"
            ),
            Self::Accounting(error) => write!(f, "structural numerical accounting failed: {error}"),
            Self::InvalidLoadPlan(message) => {
                write!(f, "canonical structural load plan is invalid: {message}")
            }
            Self::InvalidSpatialProjectionEvidence { load_id, reason } => write!(
                f,
                "explicit structural load projection evidence is invalid: load_id={load_id} reason={reason}"
            ),
            Self::ProjectionConservation {
                load_id,
                resultant_relative_error,
                moment_relative_error,
                relative_tolerance,
            } => write!(
                f,
                "explicit structural load projection violates its conservation requirement: load_id={load_id} resultant_relative_error={resultant_relative_error} moment_relative_error={moment_relative_error} relative_tolerance={relative_tolerance}"
            ),
            Self::UnmappedExplicitConnector {
                connector_id,
                target,
                region_id,
            } => write!(
                f,
                "explicit structural connector target does not resolve to connectivity: connector_id={connector_id} target={target} region_id={region_id}"
            ),
            Self::InvalidExplicitConnector {
                connector_id,
                reason,
            } => write!(
                f,
                "explicit structural connector is invalid: connector_id={connector_id} reason={reason}"
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
            Self::UnmappedMixedTarget {
                target_id,
                selector,
            } => write!(
                f,
                "mixed structural target does not resolve to solid or explicit topology: target_id={target_id} selector={selector}"
            ),
            Self::AmbiguousMixedTarget {
                target_id,
                selector,
                solid_node_count,
                explicit_node_count,
            } => write!(
                f,
                "mixed structural target resolves to different solid and explicit node sets: target_id={target_id} selector={selector} solid_node_count={solid_node_count} explicit_node_count={explicit_node_count}"
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
    SyntheticOperator,
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
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    pub youngs_modulus_pa: runmat_types::physical::YoungsModulus,
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    pub poisson_ratio: runmat_types::physical::PoissonRatio,
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    #[serde(default)]
    pub density_kg_per_m3: runmat_types::physical::MassDensity,
    pub lame_lambda_pa: f64,
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    pub shear_modulus_pa: runmat_types::physical::ShearModulus,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BeamRecoveryElementSummary {
    pub element_id: String,
    pub region_id: String,
    pub node_i_index: usize,
    pub node_j_index: usize,
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    pub length_m: runmat_types::physical::Length,
    pub section: BeamSection,
    pub material: BeamMaterial,
    pub transform_global_to_local: BeamTransform12,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ShellRecoveryElementSummary {
    pub element_id: String,
    pub region_id: String,
    pub node_indices: [usize; 3],
    pub geometry: ShellElementGeometry,
    pub constitutive: ShellConstitutive,
}

pub use solid_recovery_types::{
    SolidRecoveryElementSummary, SolidRecoveryGeometry, SolidRecoveryGeometryError,
    SolvedThermalRecoveryQuadrature,
};

pub use moment_projection::MomentProjectionSummary;

pub use coupling_summary::{ElectroThermalAssemblySummary, ThermoMechanicalAssemblySummary};

pub(super) fn assemble_linear_system_impl(
    model: &AnalysisModel,
    excitations: &RealizedExcitations,
    solver_mesh: Option<&SolverMeshArtifact>,
    thermo_mechanical_context: Option<FeaThermoMechanicalContext>,
    electro_thermal_context: Option<FeaElectroThermalContext>,
    execution: AssemblyExecution<'_>,
) -> Result<AssemblySummary, LinearAssemblyError> {
    struct ReadyWake;
    impl std::task::Wake for ReadyWake {
        fn wake(self: std::sync::Arc<Self>) {}
    }
    let future = assemble_linear_system_impl_with_owner(
        model,
        excitations,
        solver_mesh,
        StructuralCouplingContexts {
            thermo_mechanical: thermo_mechanical_context,
            electro_thermal: electro_thermal_context,
        },
        execution,
        StructuralAssemblyBatch {
            maximum_block_elements: usize::MAX,
            awaited_elements: false,
        },
        |_, _| std::future::ready(Ok(())),
    );
    let mut future = std::pin::pin!(future);
    let waker = std::task::Waker::from(std::sync::Arc::new(ReadyWake));
    let mut context = std::task::Context::from_waker(&waker);
    match std::future::Future::poll(future.as_mut(), &mut context) {
        std::task::Poll::Ready(result) => result,
        std::task::Poll::Pending => unreachable!("immediate structural owner cannot suspend"),
    }
}

async fn assemble_linear_system_impl_with_owner<O, F, A>(
    model: &AnalysisModel,
    excitations: &RealizedExcitations,
    solver_mesh: Option<&SolverMeshArtifact>,
    contexts: StructuralCouplingContexts,
    execution: AssemblyExecution<'_>,
    batch: StructuralAssemblyBatch,
    mut before_block: O,
) -> Result<AssemblySummary, A>
where
    O: FnMut(StructuralCapacityPhase, runmat_simulation_usage::OperationCount) -> F,
    F: std::future::Future<Output = Result<(), A>>,
    A: From<LinearAssemblyError>
        + From<SolverSolidAssemblyError>
        + From<StructuralAssemblyError>
        + Send,
{
    let StructuralCouplingContexts {
        thermo_mechanical: thermo_mechanical_context,
        electro_thermal: electro_thermal_context,
    } = contexts;
    let StructuralAssemblyBatch {
        maximum_block_elements,
        awaited_elements,
    } = batch;
    let AssemblyExecution {
        structural_requirement,
        supplied_constraint_plan,
        local_execution,
        solved_nodal_temperature_k,
        solid_energy,
    } = execution;
    if structural_requirement.is_none() {
        return Err(LinearAssemblyError::CouplingRealization(
            "assembly requires an explicit structural formulation; other physics must use their physical assembly operators, not a synthetic shared RHS".into(),
        ).into());
    }
    if electro_thermal_context.is_some()
        && structural_requirement != Some(StructuralFormulationRequirement::SyntheticOperator)
    {
        return Err(LinearAssemblyError::CouplingRealization(
            "electro-thermal structural coupling requires a solved physical electrical/thermal transfer; the legacy synthetic Joule-heating RHS is not a physical formulation".into(),
        ).into());
    }
    // A prescribed temperature profile is applied only by the physical solid temperature-load
    // assembly. Explicit beam/shell connectivity and mesh-free assembly have no such load path,
    // so a thermomechanical context there would be dropped silently.
    if thermo_mechanical_context.is_some()
        && structural_requirement != Some(StructuralFormulationRequirement::SyntheticOperator)
        && (solver_mesh.is_none()
            || model
                .structural
                .as_ref()
                .is_some_and(|domain| !domain.elements.is_empty()))
    {
        return Err(LinearAssemblyError::CouplingRealization(
            "prescribed thermomechanical coupling requires the physical solid temperature-load assembly; other formulations require a solved physical thermal transfer".into(),
        ).into());
    }
    // Temperature fields are coupling inputs, never structural loads: with neither a
    // thermomechanical context nor a solved thermal state to consume them, assembling would
    // silently drop the authored temperature load.
    if thermo_mechanical_context.is_none() && solved_nodal_temperature_k.is_none() {
        let unrealized = excitations
            .iter()
            .filter(|excitation| {
                matches!(excitation.kind, RealizedExcitationKind::TemperatureField(_))
            })
            .map(|excitation| excitation.excitation_id.0.as_str())
            .collect::<std::collections::BTreeSet<_>>();
        if !unrealized.is_empty() {
            return Err(LinearAssemblyError::CouplingRealization(format!(
                "assembly requests Structural, but the active temperature-field coupling ({}) realizes Coupled(Thermomechanical); a prescribed temperature profile must be applied by a thermomechanical step",
                unrealized.into_iter().collect::<Vec<_>>().join(", ")
            )).into());
        }
    }
    // Terminal voltages are electro-thermal coupling inputs, never structural loads: with
    // neither an electro-thermal context nor a solved thermal state to consume them, assembling
    // would silently drop the authored voltage.
    if electro_thermal_context.is_none() && solved_nodal_temperature_k.is_none() {
        let unrealized = excitations
            .iter()
            .filter(|excitation| {
                matches!(excitation.kind, RealizedExcitationKind::TerminalVoltage(_))
            })
            .map(|excitation| excitation.excitation_id.0.as_str())
            .collect::<std::collections::BTreeSet<_>>();
        if !unrealized.is_empty() {
            return Err(LinearAssemblyError::CouplingRealization(format!(
                "assembly requests Structural, but the active terminal-voltage coupling ({}) realizes Coupled(Electrothermal); a terminal voltage must be applied by an electro-thermal step",
                unrealized.into_iter().collect::<Vec<_>>().join(", ")
            )).into());
        }
    }
    if solved_nodal_temperature_k.is_some()
        && (solver_mesh.is_none()
            || structural_requirement
                != Some(StructuralFormulationRequirement::PhysicalDiscretization)
            || thermo_mechanical_context.is_some()
            || electro_thermal_context.is_some()
            || model.thermo_mechanical.is_none()
            || model
                .structural
                .as_ref()
                .is_some_and(|domain| !domain.elements.is_empty()))
    {
        return Err(LinearAssemblyError::CouplingRealization(
            "solved thermal input requires exclusively physical solid thermomechanical assembly"
                .into(),
        )
        .into());
    }
    let explicit_element_count = model
        .structural
        .as_ref()
        .map(|structural| structural.elements.len())
        .unwrap_or(0);
    if structural_requirement.is_some() && explicit_element_count > 0 {
        if let Some(solver_mesh) = solver_mesh {
            return mixed_structural::assemble_mixed_structural_system_with_owner(
                model,
                excitations,
                solver_mesh,
                supplied_constraint_plan,
                local_execution,
                awaited_elements,
                |phase, required| {
                    let phase = match phase {
                        mixed_structural::MixedCapacityPhase::Solid(phase) => {
                            StructuralCapacityPhase::Solid(phase)
                        }
                        mixed_structural::MixedCapacityPhase::Explicit(phase) => {
                            StructuralCapacityPhase::Explicit(phase)
                        }
                    };
                    before_block(phase, required)
                },
            )
            .await;
        }
        return explicit_system::assemble_explicit_structural_system_with_owner(
            model,
            excitations,
            supplied_constraint_plan,
            local_execution,
            |phase, required| before_block(StructuralCapacityPhase::Explicit(phase), required),
            awaited_elements,
        )
        .await;
    }
    if solver_mesh.is_none()
        && structural_requirement == Some(StructuralFormulationRequirement::PhysicalDiscretization)
    {
        return Err(LinearAssemblyError::Structural(
            StructuralAssemblyError::MissingDiscretization,
        )
        .into());
    }

    let solid_properties = solid_properties::compile_for_assembly(
        model,
        solver_mesh,
        structural_requirement,
        thermo_mechanical_context.is_some(),
    )?;

    let base_dof_count = (excitations.len() * 3).max(3);
    let solid_topology = solver_mesh
        .map(solver_solid_topology)
        .transpose()
        .map_err(LinearAssemblyError::SolidOperator)?;
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
        solver_mesh,
    ) {
        (Some(_), None, Some(mesh)) => Some(
            CanonicalStructuralConstraintPlan::compile(model, mesh)
                .map_err(LinearAssemblyError::ConstraintPlan)?,
        ),
        _ => None,
    };
    let constraint_plan = supplied_constraint_plan.or(compiled_constraint_plan.as_ref());
    if let (Some(plan), Some(mesh)) = (constraint_plan, solver_mesh) {
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
    let avg_youngs_modulus = mean(model.materials.iter().filter_map(|material| {
        material
            .require_mechanical()
            .ok()
            .map(|mechanical| mechanical.youngs_modulus_pa.canonical_value())
    }))
    .unwrap_or(1.0e9);
    let avg_poisson_ratio = mean(model.materials.iter().filter_map(|material| {
        material
            .require_mechanical()
            .ok()
            .map(|mechanical| mechanical.poisson_ratio.canonical_value())
    }))
    .unwrap_or(0.3);
    let avg_reference_temperature_k = mean(model.materials.iter().filter_map(|material| {
        material
            .thermal
            .as_ref()
            .map(|thermal| thermal.reference_temperature_k.canonical_value())
    }))
    .unwrap_or(293.15);
    let shear_modulus_pa = avg_youngs_modulus / (2.0 * (1.0 + avg_poisson_ratio));
    let avg_density_kg_per_m3 = mean(model.materials.iter().filter_map(|material| {
        material
            .mass_density_kg_per_m3
            .map(|quantity| quantity.canonical_value())
    }))
    .unwrap_or(7850.0);
    let lame_lambda_pa = avg_youngs_modulus * avg_poisson_ratio
        / ((1.0 + avg_poisson_ratio) * (1.0 - 2.0 * avg_poisson_ratio));
    let structural_material = StructuralMaterialSummary {
        youngs_modulus_pa: runmat_types::physical::YoungsModulus::new(avg_youngs_modulus),
        poisson_ratio: runmat_types::physical::PoissonRatio::new(avg_poisson_ratio),
        density_kg_per_m3: runmat_types::physical::MassDensity::new(avg_density_kg_per_m3),
        lame_lambda_pa,
        shear_modulus_pa: runmat_types::physical::ShearModulus::new(shear_modulus_pa),
    };
    let stiffness_base = (avg_youngs_modulus / 2.0e3).max(1.0e5);

    let mut stiffness_diag = vec![0.0; dof_count];
    let mut stiffness_upper = vec![0.0; dof_count.saturating_sub(1)];
    let mut mass_diag = vec![0.0; dof_count];
    if solid_topology.is_none() {
        for i in 0..mesh_dof_count {
            let factor = 1.0 + (i as f64) * 0.05;
            stiffness_diag[i] = stiffness_base * factor;
            mass_diag[i] = 1.0 + (i as f64) * 0.01;
        }
    }

    let mut rhs = vec![0.0; dof_count];
    if solid_topology.is_none()
        && structural_requirement == Some(StructuralFormulationRequirement::SyntheticOperator)
    {
        apply_synthetic_structural_rhs(excitations, &mut rhs);
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
    let mut structural_load_plan_identity = None;
    let mut physical_load = rhs.clone();
    let mut constraint_matrix = None;
    let mut affine_constraint_reduction = None;
    if structural_requirement.is_some() {
        if let (Some(mesh), Some(_)) = (solver_mesh, solid_topology.as_ref()) {
            let constraint_plan = constraint_plan.ok_or_else(|| {
                LinearAssemblyError::ConstraintPlan(ConstraintPlanError::Invalid(
                    "canonical solid assembly has no compiled constraint plan".into(),
                ))
            })?;
            let forcing = transient_basis::forcing::lower(
                excitations,
                mesh,
                constraint_plan,
                &structural_dof_layout,
                &transient_basis::forcing::SolidLoadDomain::derive(mesh)?,
            )?;
            rhs = forcing.rhs;
            physical_load = forcing.physical_load;
            constrained = forcing.constrained;
            constraint_matrix = forcing.constraint_matrix;
            affine_constraint_reduction = forcing.affine_constraint_reduction;
            structural_moment_projection = forcing.moment_projection;
            spatial_projection_evidence = forcing.projection_evidence;
            structural_load_plan_identity = forcing.plan_identity;
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

    if let Some(temperatures) = solved_nodal_temperature_k {
        let (mesh, properties) = solver_mesh.zip(solid_properties.as_ref()).ok_or_else(|| {
            LinearAssemblyError::CouplingRealization(
                "solved thermal input lacks solid material coverage".into(),
            )
        })?;
        let thermal_load = thermoelastic_load::assemble_solved_temperature_load_with_owner(
            mesh,
            properties,
            temperatures,
            &mut |element, count| {
                before_block(
                    StructuralCapacityPhase::ThermoelasticLoadElement { element },
                    count,
                )
            },
        )
        .await
        .map_err(thermoelastic_capacity_error)?;
        for (index, value) in thermal_load.into_iter().enumerate() {
            if !constrained[index] {
                rhs[index] += value;
            }
            physical_load[index] += value;
        }
    }

    if let Some(context) = thermo_mechanical_context.as_ref() {
        let physical_thermal_load = if let Some((mesh, properties)) = solver_mesh
            .filter(|_| {
                structural_requirement
                    == Some(StructuralFormulationRequirement::PhysicalDiscretization)
            })
            .zip(solid_properties.as_ref())
        {
            Some(
                thermoelastic_load::assemble_prescribed_temperature_load_with_owner(
                    mesh,
                    properties,
                    context,
                    &mut |element, count| {
                        before_block(
                            StructuralCapacityPhase::ThermoelasticLoadElement { element },
                            count,
                        )
                    },
                )
                .await
                .map_err(thermoelastic_capacity_error)?,
            )
        } else {
            None
        };
        let thermal_strain_scale = (context.thermal_expansion_coefficient.canonical_value()
            * context.applied_temperature_delta_k.canonical_value().abs())
        .clamp(0.0, 0.05);
        let thermal_load_scale =
            (context.applied_temperature_delta_k.canonical_value() / 50.0).clamp(-2.0, 2.0);
        let constitutive_temperature_factor = if model.materials.is_empty() {
            (-(2.5e-4) * context.applied_temperature_delta_k.canonical_value()).clamp(-0.25, 0.25)
        } else {
            let response = mean(model.materials.iter().filter_map(|material| {
                material.thermal.as_ref().map(|thermal| {
                    let adjusted_delta = context.applied_temperature_delta_k.canonical_value()
                        + (context.reference_temperature_k.canonical_value()
                            - thermal.reference_temperature_k.canonical_value())
                        + (avg_reference_temperature_k
                            - thermal.reference_temperature_k.canonical_value())
                            * 0.1;
                    thermal.modulus_temp_coeff_per_k.canonical_value() * adjusted_delta
                })
            }))
            .unwrap_or(-(2.5e-4) * context.applied_temperature_delta_k.canonical_value());
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
            context.reference_temperature_k.canonical_value(),
            context.applied_temperature_delta_k.canonical_value(),
            &mut dof_adjustments,
        );
        let spatial_field = apply_thermo_spatial_field(context, dof_count, &mut dof_adjustments);
        let mut local_modulus_scales = vec![effective_modulus_scale; dof_count];
        for i in 0..mesh_dof_count {
            let thermal_bias = 1.0 + thermal_strain_scale * (1.0 + (i % 3) as f64 * 0.1);
            let local_scale =
                (effective_modulus_scale * (1.0 + dof_adjustments[i])).clamp(0.75, 1.2);
            local_modulus_scales[i] = local_scale;
            if physical_thermal_load.is_none() {
                stiffness_diag[i] *= thermal_bias * local_scale;
            }
            if physical_thermal_load.is_none() && !constrained[i] {
                rhs[i] += thermal_load_scale * (1.0 + (i % 5) as f64 * 0.05);
            }
        }
        for i in 0..stiffness_upper.len() {
            let edge_scale = 0.5 * (local_modulus_scales[i] + local_modulus_scales[i + 1]);
            stiffness_upper[i] *= edge_scale;
        }
        if let Some(thermal_load) = physical_thermal_load {
            for (index, value) in thermal_load.iter().copied().enumerate() {
                if !constrained[index] {
                    rhs[index] += value;
                }
                physical_load[index] += value;
            }
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
            reference_temperature_k: runmat_types::physical::Temperature::new(
                context.reference_temperature_k.canonical_value(),
            ),
            applied_temperature_delta_k: runmat_types::physical::TemperatureDifference::new(
                context.applied_temperature_delta_k.canonical_value(),
            ),
            thermal_expansion_coefficient: runmat_types::physical::ThermalExpansion::new(
                context.thermal_expansion_coefficient.canonical_value(),
            ),
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
                context,
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
            profile_sources: thermomechanical_profile_sources(context),
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
            * context
                .resistive_heating_coefficient
                .canonical_value()
                .max(0.0)
            / 1.0e6)
            .clamp(0.0, 10.0);

        for i in 0..mesh_dof_count {
            let local = conductivity_scales[i];
            if !constrained[i] {
                rhs[i] += joule_heating_scale * local * (1.0 + (i % 7) as f64 * 0.01);
            }
        }

        electro_thermal = Some(ElectroThermalAssemblySummary {
            enabled: true,
            reference_temperature_k: runmat_types::physical::Temperature::new(
                context.reference_temperature_k.canonical_value(),
            ),
            applied_voltage_v: context.applied_voltage_v,
            base_electrical_conductivity_s_per_m: context.base_electrical_conductivity_s_per_m,
            resistive_heating_coefficient: runmat_types::physical::InverseTemperature::new(
                context.resistive_heating_coefficient.canonical_value(),
            ),
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

    let structural_solid_recovery = match solver_mesh
        .filter(|_| structural_requirement.is_some())
        .zip(solid_properties.as_ref())
    {
        Some((mesh, properties)) => {
            let thermal_context = thermo_mechanical_context.as_ref().filter(|_| {
                structural_requirement
                    == Some(StructuralFormulationRequirement::PhysicalDiscretization)
            });
            if awaited_elements {
                solid_recovery::build_with_owner(
                    mesh,
                    properties,
                    thermal_context,
                    solved_nodal_temperature_k,
                    &mut |element, required| {
                        before_block(
                            StructuralCapacityPhase::SolidRecoveryElement { element },
                            required,
                        )
                    },
                )
                .await
                .map_err(|error| match error {
                    crate::algebra::coercivity::NumericalCapacityError::Numerical(error) => {
                        A::from(error)
                    }
                    crate::algebra::coercivity::NumericalCapacityError::Admission(error) => error,
                })?
            } else {
                solid_recovery::build(
                    mesh,
                    properties,
                    thermal_context,
                    solved_nodal_temperature_k,
                )?
            }
        }
        None => Vec::new(),
    };
    let solid_operators = match solver_mesh
        .filter(|_| structural_requirement.is_some())
        .zip(solid_properties.as_ref())
    {
        Some((mesh, properties)) => Some(
            solver_solid::assemble_preparing_with_owner(
                mesh,
                properties,
                solver_solid::SolidAssemblyExecution {
                    local_execution,
                    capture: solid_energy,
                    maximum_block_elements,
                    awaited_elements,
                },
                (&constrained, affine_constraint_reduction.as_ref()),
                |phase, required| before_block(StructuralCapacityPhase::Solid(phase), required),
            )
            .await?,
        ),
        None => None,
    };
    let structural_solid_total_mass_kg = solid_operators
        .as_ref()
        .map_or(0.0, |operators| operators.total_mass_kg);
    if let Some(operators) = solid_operators.as_ref() {
        mass_diag[..mesh_dof_count].copy_from_slice(&operators.lumped_mass_diag[..mesh_dof_count]);
    }
    // The solid operators move into the structural DOF space; no copy of them stays alive.
    let expanded = |matrix: RealSparseMatrix| {
        matrix
            .into_expanded_square(dof_count)
            .expect("structural DOF expansion is valid")
    };
    let (mut stiffness_csr, mass_csr, mut damping_csr) = match solid_operators {
        Some(operators) => (
            Some(expanded(operators.stiffness)),
            Some(expanded(operators.consistent_mass)),
            Some(expanded(operators.damping)),
        ),
        None => (None, None, None),
    };
    // The element-only stiffness is retained only when connectors join the solid operator: the
    // connector-anchored nullspace proof needs both parts.
    let mut solid_element_stiffness_csr = None;
    let mut structural_connector_stiffness_count = 0;
    let mut structural_connector_damping_count = 0;
    let mut structural_connector_plan = None;
    let mut structural_connector_stiffness_csr = None;
    if structural_requirement.is_some() && !model.connectors.is_empty() {
        let mesh = solver_mesh.ok_or({
            LinearAssemblyError::Structural(StructuralAssemblyError::MissingDiscretization)
        })?;
        let topology = crate::physics::structural::topology::node_space::CanonicalStructuralNodeSpace::derive_solid(mesh)
            .map_err(|error| {
                LinearAssemblyError::Structural(StructuralAssemblyError::MixedTopology(
                    error.to_string(),
                ))
            })?;
        let projection = crate::topology::AnalysisProjection::derive(mesh).map_err(|error| {
            LinearAssemblyError::Structural(StructuralAssemblyError::MixedTopology(
                error.to_string(),
            ))
        })?;
        let coordinates_m = topology
            .nodes
            .iter()
            .map(|node| node.coordinates_m.canonical_value())
            .collect::<Vec<_>>();
        let node_identities = topology
            .nodes
            .iter()
            .map(|node| node.identity)
            .collect::<Vec<_>>();
        let empty_explicit = runmat_analysis_core::StructuralModel::default();
        let mut connector_stiffness = RealSparseAssembler::square(
            dof_count,
            OperatorRole::Stiffness,
            MatrixProperties::SYMMETRIC_POSITIVE_SEMIDEFINITE,
        );
        let mut connector_damping = RealSparseAssembler::square(
            dof_count,
            OperatorRole::Damping,
            MatrixProperties::SYMMETRIC_POSITIVE_SEMIDEFINITE,
        );
        let summary = assemble_connectors_with_resolver(
            &model.connectors,
            ConnectorAssemblyContext {
                topology_identity: topology.identity,
                layout: &structural_dof_layout,
                coordinates_m: runmat_types::physical::Quantity::<runmat_types::physical::kind::Position,_>::new(&coordinates_m),
                node_identities: &node_identities,
            },
            local_execution,
            &mut connector_stiffness,
            &mut connector_damping,
            |connector, target, role| match role {
                ConnectorTargetRole::Point => {
                    let ownership = crate::physics::structural::topology::target::resolve_mixed_structural_target(
                        target,
                        mesh,
                        &empty_explicit,
                        &topology,
                    )
                    .map_err(|_| StructuralAssemblyError::UnmappedExplicitConnector {
                        connector_id: connector.connector_id.clone(),
                        target: "point",
                        region_id: target.region_id.clone(),
                    })?;
                    Ok(ResolvedConnectorTarget {
                        node_indices: ownership.canonical_nodes().to_vec(),
                        surface_samples: Vec::new(),
                    })
                }
                ConnectorTargetRole::Surface => {
                    resolve_projected_surface(connector, target, &projection)
                }
            },
        )
        .map_err(LinearAssemblyError::Structural)?;
        structural_connector_stiffness_count = summary.stiffness_count;
        structural_connector_damping_count = summary.damping_count;
        structural_connector_plan = Some(summary.plan.clone());
        if summary.stiffness_count > 0 {
            let connector = connector_stiffness
                .finish()
                .map_err(|error| LinearAssemblyError::SparseOperator(format!("{error:?}")))?;
            let solid = stiffness_csr.take().ok_or({
                LinearAssemblyError::Structural(StructuralAssemblyError::MissingDiscretization)
            })?;
            let combined = RealSparseMatrix::scaled_sum(
                &[(1.0, &solid), (1.0, &connector)],
                OperatorRole::Stiffness,
                MatrixProperties::SYMMETRIC_POSITIVE_SEMIDEFINITE,
            )
            .map_err(|error| LinearAssemblyError::SparseOperator(format!("{error:?}")))?;
            structural_connector_stiffness_csr = Some(connector);
            solid_element_stiffness_csr = Some(solid);
            stiffness_csr = Some(combined);
        }
        if summary.damping_count > 0 {
            let connector = connector_damping
                .finish()
                .map_err(|error| LinearAssemblyError::SparseOperator(format!("{error:?}")))?;
            let solid = damping_csr.as_ref().ok_or({
                LinearAssemblyError::Structural(StructuralAssemblyError::MissingDiscretization)
            })?;
            damping_csr = Some(
                RealSparseMatrix::scaled_sum(
                    &[(1.0, solid), (1.0, &connector)],
                    OperatorRole::Damping,
                    MatrixProperties::SYMMETRIC_POSITIVE_SEMIDEFINITE,
                )
                .map_err(|error| LinearAssemblyError::SparseOperator(format!("{error:?}")))?,
            );
        }
    }
    if let Some(csr) = stiffness_csr.as_ref() {
        for (i, diagonal) in stiffness_diag.iter_mut().enumerate().take(dof_count) {
            if i >= mesh_dof_count {
                *diagonal = 0.0;
                continue;
            }
            *diagonal = csr.get(i, i).abs().max(1.0e-12);
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
        structural_explicit_node_indices: Vec::new(),
        structural_load_plan_identity,
        structural_moment_load_count,
        structural_direct_rotational_moment_load_count,
        structural_requested_moment_norm_n_m,
        structural_moment_projection,
        spatial_projection_evidence,
        structural_rotational_constraint_count: 0,
        structural_beam_element_count: 0,
        structural_shell_element_count: 0,
        structural_connector_stiffness_count,
        structural_connector_damping_count,
        structural_connector_plan,
        structural_solid_element_count: solid_topology
            .as_ref()
            .map(|topology| topology.volume_element_count)
            .unwrap_or(0),
        structural_solid_total_mass_kg,
        structural_solid_recovery,
        structural_dof_layout: structural_dof_layout.clone(),
        structural_reduced_basis: None,
        structural_beam_recovery: Vec::new(),
        structural_shell_recovery: Vec::new(),
        structural_formulations: structural_requirement.map_or_else(Vec::new, |requirement| {
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
                            Some(StructuralRecoveryImplementation::Tet10IntegrationPointProjection),
                        ),
                    };
                    StructuralFormulationRealization::new(operator, recovery)
                })
                .or_else(|| {
                    (requirement == StructuralFormulationRequirement::SyntheticOperator).then(
                        || {
                            StructuralFormulationRealization::new(
                                StructuralOperatorImplementation::SyntheticDiagonalChain,
                                None,
                            )
                        },
                    )
                })
                .into_iter()
                .collect()
        }),
        structural_property_map_identity: solid_properties
            .as_ref()
            .map(|properties| properties.identity.clone()),
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
        physical_load,
        constraint_matrix: constraint_matrix.clone(),
        affine_constraint_reduction,
        load_count: excitations.len(),
        structural_material,
        thermo_mechanical,
        electro_thermal,
        operator: {
            let raw_stiffness = match stiffness_csr {
                Some(matrix) => matrix,
                None => {
                    let mut assembler = RealSparseAssembler::square(
                        stiffness_diag.len(),
                        OperatorRole::Stiffness,
                        MatrixProperties::SYMMETRIC_POSITIVE_DEFINITE,
                    );
                    for (row, value) in stiffness_diag.iter().copied().enumerate() {
                        assembler.add(row, row, value);
                        if let Some(coupling) = stiffness_upper.get(row).copied() {
                            assembler.add(row, row + 1, -coupling);
                            assembler.add(row + 1, row, -coupling);
                        }
                    }
                    assembler.finish().map_err(|error| {
                        LinearAssemblyError::SparseOperator(format!("{error:?}"))
                    })?
                }
            };
            let raw_mass = match mass_csr {
                Some(matrix) => matrix,
                None => RealSparseMatrix::diagonal(
                    mass_diag,
                    OperatorRole::Mass,
                    MatrixProperties::SYMMETRIC_POSITIVE_DEFINITE,
                )
                .map_err(|error| LinearAssemblyError::SparseOperator(format!("{error:?}")))?
                .with_contribution(ContributionProvenance {
                    domain: ContributionDomain::Structural,
                    contribution_id: ContributionId("lumped_structural_mass".to_string()),
                }),
            };
            let mass = raw_mass;
            let damping = match damping_csr {
                Some(matrix) => matrix,
                None => RealSparseMatrix::diagonal(
                    vec![0.0; dof_count],
                    OperatorRole::Damping,
                    MatrixProperties::SYMMETRIC_POSITIVE_SEMIDEFINITE,
                )
                .map_err(|error| LinearAssemblyError::SparseOperator(format!("{error:?}")))?
                .with_contribution(ContributionProvenance {
                    domain: ContributionDomain::Structural,
                    contribution_id: ContributionId("undamped_structural_model".to_string()),
                }),
            };
            let operators = CanonicalRealOperators::try_new(raw_stiffness, mass, damping)
                .expect("assembled structural operator roles and dimensions are valid");
            let nullspace_evidence = if let (Some(mesh), Some(plan), Some(_)) =
                (solver_mesh, constraint_plan, solid_topology.as_ref())
            {
                let resolved_storage;
                let resolved = if let Some(matrix) = constraint_matrix.as_ref() {
                    matrix
                } else {
                    resolved_storage =
                        resolve_solid_constraint_matrix(plan, mesh, &structural_dof_layout)
                            .map_err(LinearAssemblyError::ConstraintAlgebra)?;
                    &resolved_storage
                };
                let proof = if let Some(connector) = structural_connector_stiffness_csr.as_ref() {
                    let element = solid_element_stiffness_csr.as_ref().ok_or({
                        LinearAssemblyError::Structural(
                            StructuralAssemblyError::MissingDiscretization,
                        )
                    })?;
                    NullspaceEliminationEvidence::prove_connector_anchored_solid(
                        crate::algebra::ConnectorAnchoredSolidInput {
                            mesh,
                            layout: &structural_dof_layout,
                            element_operator: element,
                            connector_operator: connector,
                            combined_operator: operators.stiffness(),
                            plan,
                            constraints: resolved,
                            constrained_dofs: &constrained,
                        },
                    )
                } else {
                    NullspaceEliminationEvidence::prove_solid(
                        mesh,
                        &structural_dof_layout,
                        plan,
                        resolved,
                        operators.stiffness(),
                        &constrained,
                    )
                };
                match proof {
                    Ok(evidence) => Some(evidence),
                    Err(crate::algebra::NullspaceEvidenceError::UnderconstrainedComponent) => None,
                    Err(error) => {
                        return Err(LinearAssemblyError::SparseOperator(error.to_string()).into())
                    }
                }
            } else {
                None
            };
            AssembledRealSystem::try_new_with_nullspace_evidence(
                operators,
                constrained,
                rhs,
                nullspace_evidence,
            )
            .expect("assembled structural system is finite and dimensionally consistent")
        },
    })
}

fn electro_thermal_fingerprint(
    context: &FeaElectroThermalContext,
    dof_count: usize,
    joule_heating_scale: f64,
    conductivity_spread_ratio: f64,
) -> u64 {
    let mut hash = 1469598103934665603_u64;
    for value in [
        context.reference_temperature_k.canonical_value().to_bits(),
        context.applied_voltage_v.to_bits(),
        context.base_electrical_conductivity_s_per_m.to_bits(),
        context
            .resistive_heating_coefficient
            .canonical_value()
            .to_bits(),
        joule_heating_scale.to_bits(),
        conductivity_spread_ratio.to_bits(),
        dof_count as u64,
    ] {
        hash ^= value;
        hash = hash.wrapping_mul(1099511628211);
    }
    hash
}

fn mean(values: impl IntoIterator<Item = f64>) -> Option<f64> {
    let (sum, count) = values
        .into_iter()
        .fold((0.0, 0_usize), |(sum, count), value| {
            (sum + value, count + 1)
        });
    (count > 0).then_some(sum / count as f64)
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

fn lower_solid_direct_constraints(
    plan: &CanonicalStructuralConstraintPlan,
    mesh: &SolverMeshArtifact,
    assignments: &mut DirectConstraintAssignments<'_>,
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
        let CanonicalStructuralDofOwner::StructuralNode(node_identity) = term.owner else {
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
        assignments.assign(
            &row.constraint_id,
            node_index,
            kind,
            row.right_hand_side / term.coefficient,
        )?;
    }
    Ok(())
}

pub(crate) fn resolve_solid_constraint_matrix(
    plan: &CanonicalStructuralConstraintPlan,
    mesh: &SolverMeshArtifact,
    layout: &StructuralDofLayout,
) -> Result<ResolvedConstraintMatrix, ConstraintAlgebraError> {
    let node_identities = mesh
        .topology
        .nodes
        .iter()
        .map(|node| node.stable_identity)
        .collect::<Vec<_>>();
    let matrix = ResolvedConstraintMatrix::from_solid_plan(plan, &node_identities)?;
    if matrix.full_dof_count != layout.total_dof_count() {
        return Err(ConstraintAlgebraError::InvalidMatrix(
            "solid constraint numbering does not match the assembly layout".into(),
        ));
    }
    Ok(matrix)
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
        context.reference_temperature_k.canonical_value().to_bits(),
        context
            .applied_temperature_delta_k
            .canonical_value()
            .to_bits(),
        context
            .thermal_expansion_coefficient
            .canonical_value()
            .to_bits(),
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
        let normalized = ((region_delta.temperature_delta_k
            - context.applied_temperature_delta_k.canonical_value())
            / 240.0)
            .clamp(-0.45, 0.45);
        let start =
            ((region_hash(region_delta.region_id()) as usize).wrapping_add(idx * 5)) % dof_count;
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
    let assigned_properties = model
        .material_assignments
        .iter()
        .map(|assignment| {
            model
                .materials
                .iter()
                .find(|material| material.material_id == assignment.assigned_material_id)
                .map(|material| {
                    let modulus = material
                        .require_mechanical()
                        .map(|mechanical| mechanical.youngs_modulus_pa.canonical_value())
                        .unwrap_or(1.0e9)
                        .max(1.0);
                    let temperature_response = material
                        .thermal
                        .as_ref()
                        .map(|thermal| {
                            thermal.modulus_temp_coeff_per_k.canonical_value()
                                * (applied_temperature_delta_k
                                    + (reference_temperature_k
                                        - thermal.reference_temperature_k.canonical_value()))
                        })
                        .unwrap_or(constitutive_temperature_factor)
                        .clamp(-0.4, 0.2);
                    (modulus, temperature_response)
                })
                .unwrap_or((1.0e9, constitutive_temperature_factor))
        })
        .collect::<Vec<_>>();
    let mean_modulus = mean(assigned_properties.iter().map(|(modulus, _)| *modulus))
        .unwrap_or(1.0e9)
        .max(1.0);
    let mean_temperature_response = mean(
        assigned_properties
            .iter()
            .map(|(_, temperature_response)| *temperature_response),
    )
    .unwrap_or(constitutive_temperature_factor);
    let mut activity_sum = 0.0_f64;
    for (idx, (assignment, (assigned_modulus, assigned_temperature_response))) in model
        .material_assignments
        .iter()
        .zip(assigned_properties)
        .enumerate()
    {
        let modulus_delta_ratio =
            ((assigned_modulus - mean_modulus) / mean_modulus).clamp(-0.6, 0.6);
        let response_delta =
            (assigned_temperature_response - mean_temperature_response).clamp(-0.35, 0.35);
        let region_phase = ((region_hash(assignment.target_label()) % 11) as f64) / 10.0;
        let activity =
            (0.7 * modulus_delta_ratio.abs() + 0.3 * response_delta.abs()).clamp(0.0, 1.0);
        let signed_bias = base_amplitude
            * (0.45 * modulus_delta_ratio
                + 0.35 * response_delta
                + 0.2 * modulus_delta_ratio.signum() * region_phase);
        let stride = model.material_assignments.len().saturating_add(1).max(2);
        let start = ((region_hash(assignment.target_label()) as usize).wrapping_add(idx * 3))
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
        activity_sum += activity;
    }
    for value in dof_adjustments.iter_mut() {
        *value = value.clamp(-0.18, 0.18);
    }
    (activity_sum / model.material_assignments.len() as f64).clamp(0.0, 1.0)
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
        fixture_model, mixed_solid_beam_model, replace_fixture_excitations, FixtureExcitation,
        FixtureExcitationKind, FixtureId,
    };
    use runmat_meshing_core::{
        ElementOrder, PersistentEntityId, PersistentEntityKind, SolverMeshArtifact, StableDigest,
    };

    #[test]
    fn production_assembly_rejects_synthetic_joule_heating_rhs() {
        let model = fixture_model(FixtureId::CantileverLinearStatic);
        let result = assemble_linear_system_impl(
            &model,
            &static_excitations(&model),
            None,
            None,
            Some(FeaElectroThermalContext {
                reference_temperature_k: runmat_types::physical::Temperature::new(293.15),
                applied_voltage_v: 12.0,
                base_electrical_conductivity_s_per_m: 1.0,
                resistive_heating_coefficient: runmat_types::physical::InverseTemperature::new(0.0),
            }),
            AssemblyExecution::serial_physical(),
        );
        assert!(
            matches!(result, Err(LinearAssemblyError::CouplingRealization(message)) if message.contains("legacy synthetic Joule-heating RHS"))
        );
    }

    fn fixed_constraint(region_id: &str) -> runmat_analysis_core::AnalysisConstraint {
        runmat_analysis_core::AnalysisConstraint::Structural(
            runmat_analysis_core::StructuralConstraint {
                constraint_id: "fixed_root".to_string(),
                frame: runmat_analysis_core::CoordinateFrame::Global,
                kind: runmat_analysis_core::StructuralConstraintKind::Fixed {
                    target: runmat_analysis_core::RegionTarget::authored(
                        runmat_meshing_core::fixtures::fixture_boundary_selection(&(region_id)),
                    ),
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
        assert!(summary.operator.nullspace_evidence().is_none());
        assert_eq!(summary.structural_solid_recovery.len(), 1);
        let SolidRecoveryGeometry::Tetrahedron4 {
            node_indices,
            coordinates_m,
        } = &summary.structural_solid_recovery[0].geometry
        else {
            panic!("Tet4 assembly must retain Tet4 recovery geometry")
        };
        assert_eq!(*node_indices, [0, 1, 2, 3]);
        assert_eq!(
            coordinates_m.canonical_value(),
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        );
        let expected_mass_kg = model.materials[0]
            .require_mass_density()
            .unwrap()
            .canonical_value()
            / 6.0;
        assert!(
            (summary.structural_solid_total_mass_kg - expected_mass_kg).abs()
                <= expected_mass_kg * 1.0e-12
        );
        assert!(summary
            .operator
            .operators()
            .mass()
            .diagonal_values()
            .iter()
            .all(|mass| *mass > 0.0));
        assert!(summary
            .operator
            .operators()
            .damping()
            .values()
            .iter()
            .all(|damping| *damping == 0.0));
        let csr = summary.operator.operators().stiffness();
        assert_eq!(csr.pattern().row_offsets().len(), summary.dof_count + 1);
        assert_eq!(
            csr.pattern().row_offsets().last().copied(),
            Some(csr.values().len() as u64)
        );
        assert_eq!(csr.pattern().column_indices().len(), csr.values().len());
        assert!(summary
            .operator
            .operators()
            .stiffness()
            .diagonal_values()
            .iter()
            .all(|value| *value > 0.0));
        for row in 0..summary.dof_count {
            let diagonal = csr.get(row, row).abs();
            assert!(
                (summary.operator.operators().stiffness().diagonal_values()[row] - diagonal)
                    <= 1.0e-8
            );
        }
    }

    #[test]
    fn solver_mesh_keeps_uncoupled_solid_and_beam_components_underconstrained() {
        let mut model = fixture_model(FixtureId::CantileverLinearStatic);
        model.excitations.clear();
        model.constraints.clear();
        model.structural = Some(runmat_analysis_core::StructuralModel {
            nodes: vec![
                runmat_analysis_core::StructuralNode {
                    node_id: 1,
                    coordinates_m: runmat_types::physical::Quantity::<
                        runmat_types::physical::kind::Position,
                        _,
                    >::new([0.0, 0.0, 0.0]),
                    solid_attachment: None,
                },
                runmat_analysis_core::StructuralNode {
                    node_id: 2,
                    coordinates_m: runmat_types::physical::Quantity::<
                        runmat_types::physical::kind::Position,
                        _,
                    >::new([1.0, 0.0, 0.0]),
                    solid_attachment: None,
                },
            ],
            elements: vec![runmat_analysis_core::StructuralElement {
                element_id: "beam_1".to_string(),
                region_id: runmat_meshing_core::fixtures::fixture_edge("span"),
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
                area_m2:
                    runmat_types::physical::Quantity::<runmat_types::physical::kind::Area, _>::new(
                        1.0e-4,
                    ),
                iy_m4: runmat_types::physical::Quantity::<
                    runmat_types::physical::kind::SecondMomentOfArea,
                    _,
                >::new(1.0e-9),
                iz_m4: runmat_types::physical::Quantity::<
                    runmat_types::physical::kind::SecondMomentOfArea,
                    _,
                >::new(1.0e-9),
                torsion_j_m4: runmat_types::physical::Quantity::<
                    runmat_types::physical::kind::SecondMomentOfArea,
                    _,
                >::new(1.0e-9),
                outer_fiber_y_m: runmat_types::physical::Quantity::<
                    runmat_types::physical::kind::Length,
                    _,
                >::new(0.01),
                outer_fiber_z_m: runmat_types::physical::Quantity::<
                    runmat_types::physical::kind::Length,
                    _,
                >::new(0.01),
                torsion_outer_radius_m: runmat_types::physical::Quantity::<
                    runmat_types::physical::kind::Length,
                    _,
                >::new(0.01),
            }],
            shell_sections: Vec::new(),
        });
        model
            .material_assignments
            .push(runmat_analysis_core::MaterialAssignment {
                target: runmat_analysis_core::PropertyRegionTarget::beam(
                    runmat_meshing_core::fixtures::fixture_edge("span"),
                ),
                expected_material_id: "mat_steel".into(),
                assigned_material_id: "mat_steel".into(),
                confidence: runmat_analysis_core::EvidenceConfidence::Verified,
            });

        let summary = assemble_structural_system(
            &model,
            &static_excitations(&model),
            Some(tetrahedron4_mesh()),
            StructuralFormulationRequirement::PhysicalDiscretization,
        )
        .expect("unloaded disconnected components retain their physical semidefinite operator");

        assert_eq!(summary.structural_node_count, 6);
        assert_eq!(summary.structural_translational_dof_count, 18);
        assert_eq!(summary.structural_rotational_dof_count, 6);
        assert_eq!(summary.dof_count, 24);
        assert_eq!(summary.structural_solid_element_count, 1);
        assert_eq!(summary.structural_beam_element_count, 1);
        assert!(summary.operator.nullspace_evidence().is_none());
        assert!(
            summary
                .operator
                .operators()
                .stiffness()
                .contributions()
                .len()
                >= 2
        );
    }

    #[test]
    fn solver_mesh_and_beam_share_attached_translation_rows() {
        let mut mesh = tetrahedron4_mesh();
        let attached_coordinates = mesh.topology.nodes[0].coordinates_m.canonical_value();
        mesh.topology.nodes[0].provenance = vec![PersistentEntityId::new(
            PersistentEntityKind::Vertex,
            "beam_root".into(),
            vec!["root".into()],
        )];
        set_boundary_regions(&mut mesh, 0, &["root"]);
        set_boundary_regions(&mut mesh, 1, &["solid_load"]);
        reseal(&mut mesh);
        let model = mixed_solid_beam_model(attached_coordinates);
        let properties = crate::properties::compile_solid_property_map(&model, &mesh)
            .expect("compiled solid properties");
        let solid = solver_solid::assemble_solver_solid_operators(&mesh, &properties)
            .expect("solid operator");

        let summary = assemble_structural_system(
            &model,
            &static_excitations(&model),
            Some(mesh.clone()),
            StructuralFormulationRequirement::PhysicalDiscretization,
        )
        .expect("attached mixed topology");

        assert_eq!(summary.structural_node_count, 5);
        assert_eq!(summary.structural_translational_dof_count, 15);
        assert_eq!(summary.structural_rotational_dof_count, 6);
        assert_eq!(summary.dof_count, 21);
        let root_ux = summary
            .structural_dof_layout
            .index(0, StructuralDofKind::Ux)
            .expect("shared root translation");
        let tip_ux = summary
            .structural_dof_layout
            .index(4, StructuralDofKind::Ux)
            .expect("beam tip translation");
        let combined = summary.operator.operators().stiffness();
        assert!(combined.get(root_ux, root_ux) > solid.stiffness.get(0, 0));
        assert!(combined.get(root_ux, tip_ux).abs() > 0.0);
        assert!(summary
            .structural_dof_layout
            .index(0, StructuralDofKind::Rx)
            .is_some());
        assert!(summary
            .structural_dof_layout
            .index(1, StructuralDofKind::Rx)
            .is_none());
        assert_eq!(summary.structural_rotational_constraint_count, 3);
        assert_eq!(summary.structural_connector_stiffness_count, 1);
        assert!(summary.constrained_dof_count >= 6);
        assert!(summary.operator.nullspace_evidence().is_some());
        assert_eq!(summary.spatial_projection_evidence.len(), 2);
        assert!(summary.structural_load_plan_identity.is_some());
        assert!(summary
            .spatial_projection_evidence
            .iter()
            .all(|evidence| evidence.accepted));
        assert_eq!(
            summary.structural_sources,
            vec![
                SourceImplementation::CanonicalBoundaryProjection,
                SourceImplementation::ExplicitRegionNodalDistribution,
            ]
        );
        // The fixed solid face holds the beam's attached root node, so its support is compiled in
        // the mixed node space and also fixes that node's rotations.
        assert_eq!(
            summary.structural_boundary_transfers,
            vec![BoundaryTransferImplementation::ExplicitRegionConnectivity]
        );
        assert!(summary
            .operator
            .load()
            .iter()
            .any(|value| value.abs() > 0.0));
        summary
            .operator
            .prepare_linear()
            .expect("solid face plus attached beam-root support must be coercive");
        let solve = crate::solve::linear::solve_linear_system(
            &summary,
            crate::solve::preconditioner::SpdPreconditionerKind::Jacobi,
            crate::solve::backend::kind::LinearAlgebraBackendKind::CpuReference,
            &crate::solve::backend::cpu_reference::CpuReferenceBackend,
            None,
        )
        .unwrap();
        assert!(solve.converged, "{:?}", solve.diagnostics);
        assert!(solve.solution[tip_ux].abs() > 0.0);
        let fields = crate::post::fields::recover_result_fields(
            &summary,
            &solve,
            crate::planning::resource::LocalExecutionPlan::serial(),
        )
        .expect("mixed recovery");
        let solid_stress = fields
            .iter()
            .find(|field| field.field_id == crate::contracts::FEA_FIELD_STRUCTURAL_STRESS)
            .and_then(runmat_analysis_core::AnalysisField::as_host_f64)
            .expect("solid stress field");
        let beam_force = fields
            .iter()
            .find(|field| field.field_id == crate::contracts::FEA_FIELD_STRUCTURAL_BEAM_AXIAL_FORCE)
            .and_then(runmat_analysis_core::AnalysisField::as_host_f64)
            .expect("beam force field");
        assert_eq!(solid_stress.len(), 6);
        assert!(solid_stress.iter().all(|value| value.is_finite()));
        assert_eq!(beam_force.len(), 1);
        assert!(beam_force[0].is_finite() && beam_force[0].abs() > 0.0);

        let mut prescribed = model.clone();
        prescribed
            .constraints
            .push(runmat_analysis_core::AnalysisConstraint::Structural(
                runmat_analysis_core::StructuralConstraint {
                    constraint_id: "prescribed_tip_y".into(),
                    frame: runmat_analysis_core::CoordinateFrame::Global,
                    kind: runmat_analysis_core::StructuralConstraintKind::Prescribed(
                        runmat_analysis_core::PrescribedStructuralDofs {
                            target: runmat_analysis_core::RegionTarget::point("node:102"),
                            translations: vec![runmat_analysis_core::TranslationDofValue {
                                dof: runmat_analysis_core::StructuralTranslationDof::Uy,
                                value: runmat_analysis_core::PrescribedValue::Constant(
                                    runmat_analysis_core::Displacement::new(1.0e-3),
                                ),
                            }],
                            rotations: Vec::new(),
                        },
                    ),
                },
            ));
        let prescribed_summary = assemble_structural_system(
            &prescribed,
            &static_excitations(&prescribed),
            Some(mesh.clone()),
            StructuralFormulationRequirement::PhysicalDiscretization,
        )
        .expect("mixed prescribed value must survive solid load projection");
        let tip_uy = prescribed_summary
            .structural_dof_layout
            .index(4, StructuralDofKind::Uy)
            .expect("beam tip y translation");
        assert_eq!(prescribed_summary.operator.load()[tip_uy], 1.0e-3);

        let mut conflicting = model.clone();
        conflicting
            .constraints
            .push(runmat_analysis_core::AnalysisConstraint::Structural(
                runmat_analysis_core::StructuralConstraint {
                    constraint_id: "conflicting_shared_translation".into(),
                    frame: runmat_analysis_core::CoordinateFrame::Global,
                    kind: runmat_analysis_core::StructuralConstraintKind::Prescribed(
                        runmat_analysis_core::PrescribedStructuralDofs {
                            target: runmat_analysis_core::RegionTarget::point("node:101"),
                            translations: vec![runmat_analysis_core::TranslationDofValue {
                                dof: runmat_analysis_core::StructuralTranslationDof::Ux,
                                value: runmat_analysis_core::PrescribedValue::Constant(
                                    runmat_analysis_core::Displacement::new(1.0e-3),
                                ),
                            }],
                            rotations: Vec::new(),
                        },
                    ),
                },
            ));
        let shared_node_identity = mesh.topology.nodes[0].stable_identity;
        let error = assemble_structural_system(
            &conflicting,
            &static_excitations(&conflicting),
            Some(mesh),
            StructuralFormulationRequirement::PhysicalDiscretization,
        )
        .expect_err("solid and explicit prescriptions must not disagree on a shared row");
        assert!(matches!(
            error,
            LinearAssemblyError::ConstraintPlan(ConstraintPlanError::ConflictingRow {
                first_constraint_id,
                second_constraint_id,
                conflicting_dof: Some(crate::constraints::CanonicalStructuralDofAddress {
                    owner: CanonicalStructuralDofOwner::StructuralNode(identity),
                    dof: CanonicalStructuralDof::Translation(
                        runmat_analysis_core::StructuralTranslationDof::Ux,
                    ),
                }),
            }) if first_constraint_id == "conflicting_shared_translation"
                && second_constraint_id == "fixed_root"
                && identity == shared_node_identity
        ));
    }

    #[test]
    fn whole_model_body_force_spans_solid_and_explicit_formulations_once() {
        let mut mesh = tetrahedron4_mesh();
        let attached_coordinates = mesh.topology.nodes[0].coordinates_m.canonical_value();
        mesh.topology.nodes[0].provenance = vec![PersistentEntityId::new(
            PersistentEntityKind::Vertex,
            "beam_root".into(),
            vec!["root".into()],
        )];
        set_boundary_regions(&mut mesh, 0, &["root"]);
        reseal(&mut mesh);
        let mut model = mixed_solid_beam_model(attached_coordinates);
        model.constraints.clear();
        let mut load = static_excitations(&model).iter().next().unwrap().clone();
        load.excitation_id = runmat_analysis_core::ExcitationId("whole_body".into());
        load.quantity = runmat_analysis_core::PhysicalQuantityKind::BodyForceDensity;
        load.target = crate::excitation::RealizedExcitationTarget::WholeModel;
        load.kind = RealizedExcitationKind::BodyForce(crate::excitation::ComplexVector3::real([
            6.0, 0.0, 0.0,
        ]));
        load.policy.distribution = runmat_analysis_core::ExcitationDistribution::IntensityPerVolume;
        let solid_volume = crate::topology::AnalysisProjection::derive(&mesh)
            .unwrap()
            .elements[0]
            .signed_volume_m3;

        let summary = assemble_structural_system(
            &model,
            &RealizedExcitations::new(vec![load]),
            Some(mesh),
            StructuralFormulationRequirement::PhysicalDiscretization,
        )
        .expect("whole-model density must project into both mixed formulation domains");
        let resultant_x = (0..summary.structural_node_count)
            .map(|node| {
                summary.operator.load()[summary
                    .structural_dof_layout
                    .index(node, StructuralDofKind::Ux)
                    .unwrap()]
            })
            .sum::<f64>();
        assert_relative(resultant_x, 6.0 * (solid_volume + 1.0e-4), 1.0e-12);
        assert_eq!(summary.spatial_projection_evidence.len(), 2);
        assert!(summary
            .spatial_projection_evidence
            .iter()
            .all(|evidence| evidence.excitation_id == "whole_body" && evidence.accepted));
    }

    #[test]
    fn mixed_affine_rows_anchor_rigid_modes_without_connector_support() {
        let mut mesh = tetrahedron4_mesh();
        let attached_coordinates = mesh.topology.nodes[0].coordinates_m.canonical_value();
        mesh.topology.nodes[0].provenance = vec![PersistentEntityId::new(
            PersistentEntityKind::Vertex,
            "beam_root".into(),
            vec!["root".into()],
        )];
        set_boundary_regions(&mut mesh, 0, &["root"]);
        set_boundary_regions(&mut mesh, 1, &["solid_load"]);
        reseal(&mut mesh);

        let mut model = mixed_solid_beam_model(attached_coordinates);
        model.connectors.clear();
        model
            .constraints
            .push(runmat_analysis_core::AnalysisConstraint::Structural(
                runmat_analysis_core::StructuralConstraint {
                    constraint_id: "solid_beam_ux_relation".into(),
                    frame: runmat_analysis_core::CoordinateFrame::Global,
                    kind: runmat_analysis_core::StructuralConstraintKind::Equation(
                        runmat_analysis_core::StructuralConstraintEquation {
                            terms: vec![
                                runmat_analysis_core::StructuralConstraintEquationTerm {
                                    target: runmat_analysis_core::RegionTarget::point("node:1"),
                                    dof: runmat_analysis_core::StructuralTranslationDof::Ux,
                                    coefficient: 1.0,
                                },
                                runmat_analysis_core::StructuralConstraintEquationTerm {
                                    target: runmat_analysis_core::RegionTarget::point("node:102"),
                                    dof: runmat_analysis_core::StructuralTranslationDof::Ux,
                                    coefficient: -1.0,
                                },
                            ],
                            right_hand_side: runmat_analysis_core::Displacement::new(0.0),
                        },
                    ),
                },
            ));

        let summary = assemble_structural_system(
            &model,
            &static_excitations(&model),
            Some(mesh),
            StructuralFormulationRequirement::PhysicalDiscretization,
        )
        .expect("canonical mixed affine constraints must anchor the physical topology");

        assert!(summary.affine_constraint_reduction.is_some());
        assert!(summary.constraint_matrix.is_some());
        assert!(summary.operator.nullspace_evidence().is_some());
        let solve = crate::solve::linear::solve_linear_system(
            &summary,
            crate::solve::preconditioner::SpdPreconditionerKind::Jacobi,
            crate::solve::backend::kind::LinearAlgebraBackendKind::CpuReference,
            &crate::solve::backend::cpu_reference::CpuReferenceBackend,
            None,
        )
        .unwrap();
        assert!(solve.converged, "{:?}", solve.diagnostics);
    }

    // witness: FEA-SHARED-PROPERTIES-001/material_bearing_roles @
    //   shared_infrastructure/canonical_mixed_topology/numerical_kernel; coverage: partial;
    //   boundary: Solver-mesh material regions select the assigned solid materials; beam, shell and
    //   cell roles are not covered by this test.
    #[test]
    fn solver_mesh_material_regions_select_assigned_solid_materials() {
        let mut soft_model = fixture_model(FixtureId::CantileverLinearStatic);
        soft_model.excitations.clear();
        soft_model.constraints.clear();
        let mut hard = soft_model.materials[0].clone();
        hard.material_id = "mat_hard".to_string();
        hard.require_mechanical_mut().unwrap().youngs_modulus_pa =
            runmat_types::physical::YoungsModulus::new(200.0e9);
        hard.require_mechanical_mut().unwrap().poisson_ratio =
            runmat_types::physical::PoissonRatio::new(0.3);
        let mut soft = hard.clone();
        soft.material_id = "mat_soft".to_string();
        soft.require_mechanical_mut().unwrap().youngs_modulus_pa =
            runmat_types::physical::YoungsModulus::new(20.0e9);
        soft_model.materials = vec![hard.clone(), soft.clone()];
        soft_model.material_assignments = vec![runmat_analysis_core::MaterialAssignment {
            target: runmat_analysis_core::PropertyRegionTarget::solid(
                runmat_meshing_core::fixtures::fixture_region("soft_region"),
            ),
            expected_material_id: "mat_hard".to_string(),
            assigned_material_id: "mat_soft".to_string(),
            confidence: runmat_analysis_core::EvidenceConfidence::Verified,
        }];
        let mut soft_mesh = tetrahedron4_mesh();
        set_solver_mesh_region(&mut soft_mesh, "soft_region");

        let mut hard_model = soft_model.clone();
        hard_model.material_assignments = vec![runmat_analysis_core::MaterialAssignment {
            target: runmat_analysis_core::PropertyRegionTarget::solid(
                runmat_meshing_core::fixtures::fixture_region("soft_region"),
            ),
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
        assert_eq!(
            soft_summary.structural_solid_recovery[0]
                .material
                .youngs_modulus_pa
                .canonical_value(),
            soft.require_mechanical()
                .unwrap()
                .youngs_modulus_pa
                .canonical_value()
        );
        assert_eq!(
            hard_summary.structural_solid_recovery[0]
                .material
                .youngs_modulus_pa
                .canonical_value(),
            hard.require_mechanical()
                .unwrap()
                .youngs_modulus_pa
                .canonical_value()
        );
    }

    #[test]
    fn solver_mesh_preserves_valid_single_material_properties_exactly() {
        let mut model = fixture_model(FixtureId::CantileverLinearStatic);
        model.excitations.clear();
        model.constraints.clear();
        model.materials[0]
            .require_mechanical_mut()
            .unwrap()
            .youngs_modulus_pa = runmat_types::physical::YoungsModulus::new(0.5);
        model.materials[0]
            .require_mechanical_mut()
            .unwrap()
            .poisson_ratio = runmat_types::physical::PoissonRatio::new(-0.25);
        model.materials[0].mass_density_kg_per_m3 =
            (Some(0.125)).map(runmat_types::physical::MassDensity::new);

        let mesh = tetrahedron4_mesh();
        let expected_property_identity =
            crate::properties::compile_solid_property_map(&model, &mesh)
                .expect("solid property map")
                .identity;
        let summary = assemble_physical_structural_system(&model, Some(mesh))
            .expect("valid low-valued auxetic material should assemble without coercion");
        let recovered_material = summary.structural_solid_recovery[0].material;

        assert_eq!(recovered_material.youngs_modulus_pa.canonical_value(), 0.5);
        assert_eq!(recovered_material.poisson_ratio.canonical_value(), -0.25);
        assert_eq!(
            recovered_material.density_kg_per_m3.canonical_value(),
            0.125
        );
        assert_eq!(
            summary
                .structural_material
                .youngs_modulus_pa
                .canonical_value(),
            0.5
        );
        assert_eq!(
            summary.structural_material.poisson_ratio.canonical_value(),
            -0.25
        );
        assert_eq!(
            summary
                .structural_material
                .density_kg_per_m3
                .canonical_value(),
            0.125
        );
        assert_eq!(
            summary.structural_property_map_identity,
            Some(expected_property_identity)
        );
        assert!(summary.operator.operators().damping().values().is_empty());
    }

    #[test]
    fn thermo_material_heterogeneity_uses_assigned_physics_not_reconciliation_evidence() {
        let model = fixture_model(FixtureId::MultiMaterialAssembly);
        let mut changed_audit = model.clone();
        for assignment in &mut changed_audit.material_assignments {
            assignment.expected_material_id = "mat_aluminum".into();
            assignment.confidence = runmat_analysis_core::EvidenceConfidence::Inferred;
        }
        let mut first_adjustments = vec![0.0; 27];
        let first = apply_thermo_material_heterogeneity(
            &model,
            first_adjustments.len(),
            -0.02,
            293.15,
            80.0,
            &mut first_adjustments,
        );
        let mut second_adjustments = vec![0.0; 27];
        let second = apply_thermo_material_heterogeneity(
            &changed_audit,
            second_adjustments.len(),
            -0.02,
            293.15,
            80.0,
            &mut second_adjustments,
        );

        assert_eq!(first, second);
        assert_eq!(first_adjustments, second_adjustments);
        assert!(first > 0.0);
        assert!(first_adjustments.iter().any(|value| *value != 0.0));
    }

    #[test]
    fn solver_mesh_assembles_authored_rayleigh_damping() {
        let mut model = fixture_model(FixtureId::CantileverLinearStatic);
        model.excitations.clear();
        model.constraints.clear();
        let alpha = 1.5;
        let beta = 2.0e-6;
        model.materials[0].damping = Some(runmat_analysis_core::MechanicalDampingModel::Rayleigh {
            mass_coefficient: runmat_analysis_core::RayleighMassCoefficient::new(alpha),
            stiffness_coefficient: runmat_analysis_core::RayleighStiffnessCoefficient::new(beta),
        });

        let summary = assemble_physical_structural_system(&model, Some(tetrahedron4_mesh()))
            .expect("valid Rayleigh-damped solid should assemble");
        let operators = summary.operator.operators();

        for row in 0..summary.dof_count {
            for column in 0..summary.dof_count {
                let expected = alpha * operators.mass().get(row, column)
                    + beta * operators.stiffness().get(row, column);
                assert!(
                    (operators.damping().get(row, column) - expected).abs()
                        <= expected.abs().max(1.0) * 1.0e-12
                );
            }
        }
    }

    #[test]
    fn explicit_beam_assembles_authored_rayleigh_damping() {
        let mut model = fixture_model(FixtureId::StructuralBeamCantileverEndMomentReference);
        let alpha = 1.0;
        let beta = 2.0e-5;
        model.materials[0].damping = Some(runmat_analysis_core::MechanicalDampingModel::Rayleigh {
            mass_coefficient: runmat_analysis_core::RayleighMassCoefficient::new(alpha),
            stiffness_coefficient: runmat_analysis_core::RayleighStiffnessCoefficient::new(beta),
        });

        let summary = assemble_structural_system(
            &model,
            &static_excitations(&model),
            None,
            StructuralFormulationRequirement::PhysicalDiscretization,
        )
        .expect("physical beam damping must assemble");
        let operators = summary.operator.operators();
        for row in 0..summary.dof_count {
            for column in 0..summary.dof_count {
                let expected = alpha * operators.mass().get(row, column)
                    + beta * operators.stiffness().get(row, column);
                assert!(
                    (operators.damping().get(row, column) - expected).abs()
                        <= expected.abs().max(1.0) * 1.0e-12
                );
            }
        }
    }

    #[test]
    fn explicit_force_and_moment_publish_topology_bound_conservation_evidence() {
        let mut model = fixture_model(FixtureId::StructuralBeamCantileverEndMomentReference);
        replace_fixture_excitations(
            &mut model,
            vec![
                FixtureExcitation {
                    id: "tip_force".into(),
                    region_id: runmat_meshing_core::fixtures::fixture_edge("beam_span"),
                    kind: FixtureExcitationKind::Force([3.0, -5.0, 7.0]),
                },
                FixtureExcitation {
                    id: "tip_moment".into(),
                    region_id: runmat_meshing_core::fixtures::fixture_edge("beam_span"),
                    kind: FixtureExcitationKind::Moment([2.0, 4.0, -6.0]),
                },
            ],
        );

        let summary = assemble_structural_system(
            &model,
            &static_excitations(&model),
            None,
            StructuralFormulationRequirement::PhysicalDiscretization,
        )
        .expect("explicit load projection must conserve authored resultants");
        let structural = model.structural.as_ref().expect("explicit topology");
        let topology_digest =
            crate::physics::structural::topology::explicit_topology_digest(structural)
                .expect("topology identity");
        assert_eq!(summary.spatial_projection_evidence.len(), 2);
        assert!(summary.spatial_projection_evidence.iter().all(|evidence| {
            evidence.analysis_projection_digest == topology_digest
                && evidence.resolved_entity_count == 1
                && evidence.resolved_node_count == 2
                && evidence.accepted
                && evidence.resultant_relative_error() == 0.0
                && evidence.moment_relative_error() == Some(0.0)
        }));
        assert_eq!(summary.structural_direct_rotational_moment_load_count, 1);
        assert_eq!(
            summary.structural_sources,
            vec![SourceImplementation::ExplicitRegionNodalDistribution]
        );
    }

    // witness: FEA-STRUCTURAL-CONSTRAINT-001/typed_connector_inventory @
    //   linear_structural/beam/numerical_kernel; coverage: partial;
    //   boundary: Explicit connectors assemble physical stiffness and damping operators on a beam
    //   fixture; only the connector kinds in that fixture are exercised.
    #[test]
    fn explicit_connectors_assemble_physical_stiffness_and_damping_operators() {
        let base = fixture_model(FixtureId::StructuralBeamCantileverEndMomentReference);
        let baseline = assemble_structural_system(
            &base,
            &static_excitations(&base),
            None,
            StructuralFormulationRequirement::PhysicalDiscretization,
        )
        .expect("baseline beam assembly");
        let ux = |summary: &AssemblySummary, node| {
            summary
                .structural_dof_layout
                .index(node, StructuralDofKind::Ux)
                .expect("beam translation")
        };

        let mut spring_model = base.clone();
        spring_model.connectors = vec![runmat_analysis_core::AnalysisConnector {
            connector_id: "span_spring".into(),
            kind: runmat_analysis_core::ConnectorKind::AxialSpring {
                primary: runmat_analysis_core::RegionTarget::point("node:1"),
                secondary: runmat_analysis_core::RegionTarget::point("node:2"),
                stiffness: runmat_analysis_core::Stiffness::new(4_000.0),
            },
        }];
        let spring = assemble_structural_system(
            &spring_model,
            &static_excitations(&spring_model),
            None,
            StructuralFormulationRequirement::PhysicalDiscretization,
        )
        .expect("two-ended spring assembly");
        let first = ux(&spring, 0);
        let second = ux(&spring, 1);
        let spring_k = spring.operator.operators().stiffness();
        let baseline_k = baseline.operator.operators().stiffness();
        assert_relative(
            spring_k.get(first, first) - baseline_k.get(first, first),
            4_000.0,
            1.0e-12,
        );
        assert_relative(
            spring_k.get(first, second) - baseline_k.get(first, second),
            -4_000.0,
            1.0e-12,
        );
        assert_eq!(spring.structural_connector_stiffness_count, 1);

        let mut dashpot_model = base.clone();
        dashpot_model.connectors = vec![runmat_analysis_core::AnalysisConnector {
            connector_id: "span_dashpot".into(),
            kind: runmat_analysis_core::ConnectorKind::AxialDashpot {
                primary: runmat_analysis_core::RegionTarget::point("node:1"),
                secondary: runmat_analysis_core::RegionTarget::point("node:2"),
                damping: runmat_analysis_core::Damping::new(75.0),
            },
        }];
        let dashpot = assemble_structural_system(
            &dashpot_model,
            &static_excitations(&dashpot_model),
            None,
            StructuralFormulationRequirement::PhysicalDiscretization,
        )
        .expect("ground dashpot assembly");
        let second = ux(&dashpot, 1);
        assert_relative(
            dashpot.operator.operators().damping().get(second, second),
            75.0,
            1.0e-12,
        );
        assert_eq!(dashpot.structural_connector_damping_count, 1);

        let mut foundation_model = base;
        foundation_model.connectors = vec![runmat_analysis_core::AnalysisConnector {
            connector_id: "tip_ground_spring".into(),
            kind: runmat_analysis_core::ConnectorKind::IsotropicGroundSpring {
                target: runmat_analysis_core::RegionTarget::point("node:2"),
                stiffness: runmat_analysis_core::Stiffness::new(2_000.0),
            },
        }];
        let foundation = assemble_structural_system(
            &foundation_model,
            &static_excitations(&foundation_model),
            None,
            StructuralFormulationRequirement::PhysicalDiscretization,
        )
        .expect("ground spring assembly");
        let row = ux(&foundation, 1);
        assert_relative(
            foundation.operator.operators().stiffness().get(row, row)
                - baseline.operator.operators().stiffness().get(row, row),
            2_000.0,
            1.0e-12,
        );
    }

    // witness: FEA-STRUCTURAL-CONSTRAINT-001/typed_connector_inventory @
    //   linear_structural/beam/numerical_kernel; coverage: partial;
    //   boundary: Connector targets and foundation semantics fail closed.
    #[test]
    fn explicit_connector_targets_and_foundation_semantics_fail_closed() {
        let mut model = fixture_model(FixtureId::StructuralBeamCantileverEndMomentReference);
        model.connectors = vec![runmat_analysis_core::AnalysisConnector {
            connector_id: "missing".into(),
            kind: runmat_analysis_core::ConnectorKind::AxialSpring {
                primary: runmat_analysis_core::RegionTarget::point("missing_region"),
                secondary: runmat_analysis_core::RegionTarget::point("node:2"),
                stiffness: runmat_analysis_core::Stiffness::new(1.0),
            },
        }];
        let error = assemble_structural_system(
            &model,
            &static_excitations(&model),
            None,
            StructuralFormulationRequirement::PhysicalDiscretization,
        )
        .expect_err("unmapped connector target must fail");
        assert_eq!(
            error,
            LinearAssemblyError::Structural(StructuralAssemblyError::UnmappedExplicitConnector {
                connector_id: "missing".into(),
                target: "point",
                region_id: "missing_region".into(),
            })
        );

        model.connectors[0] = runmat_analysis_core::AnalysisConnector {
            connector_id: "invalid_foundation".into(),
            kind: runmat_analysis_core::ConnectorKind::SurfaceNormalFoundation {
                target: runmat_analysis_core::RegionTarget::point("node:1"),
                modulus: runmat_analysis_core::FoundationModulus::new(1.0),
            },
        };
        let error = assemble_structural_system(
            &model,
            &static_excitations(&model),
            None,
            StructuralFormulationRequirement::PhysicalDiscretization,
        )
        .expect_err("foundation point target must fail");
        assert!(matches!(
            error,
            LinearAssemblyError::Structural(
                StructuralAssemblyError::InvalidExplicitConnector { .. }
            )
        ));
    }

    #[test]
    fn solid_point_connectors_contribute_to_the_canonical_operator() {
        let mut model = fixture_model(FixtureId::CantileverLinearStatic);
        model.constraints.clear();
        model.connectors = vec![runmat_analysis_core::AnalysisConnector {
            connector_id: "solid_ground_spring".into(),
            kind: runmat_analysis_core::ConnectorKind::IsotropicGroundSpring {
                target: runmat_analysis_core::RegionTarget::point("node:2"),
                stiffness: runmat_analysis_core::Stiffness::new(1.0),
            },
        }];
        let summary = assemble_structural_system(
            &model,
            &RealizedExcitations::new(Vec::new()),
            Some(tetrahedron4_mesh()),
            StructuralFormulationRequirement::PhysicalDiscretization,
        )
        .expect("solid connector must share canonical solid DOFs");
        assert_eq!(summary.structural_connector_stiffness_count, 1);
        assert_eq!(summary.structural_connector_damping_count, 0);
    }

    // witness: FEA-STRUCTURAL-SOLID-001/damping_body_loads @ linear_structural/shell/numerical_kernel;
    //   coverage: partial;
    //   boundary: A shell assembles authored Rayleigh damping; solid damping is not asserted here.
    #[test]
    fn shell_assembles_authored_rayleigh_damping() {
        let mut model = fixture_model(FixtureId::StructuralShellPlateMomentReference);
        model.materials[0].damping = Some(runmat_analysis_core::MechanicalDampingModel::Rayleigh {
            mass_coefficient: runmat_analysis_core::RayleighMassCoefficient::new(1.0),
            stiffness_coefficient: runmat_analysis_core::RayleighStiffnessCoefficient::new(0.0),
        });

        let summary = assemble_structural_system(
            &model,
            &static_excitations(&model),
            None,
            StructuralFormulationRequirement::PhysicalDiscretization,
        )
        .expect("physical shell damping must assemble");
        let operators = summary.operator.operators();
        for row in 0..summary.dof_count {
            for column in 0..summary.dof_count {
                assert_relative(
                    operators.damping().get(row, column),
                    operators.mass().get(row, column),
                    1.0e-12,
                );
            }
        }
    }

    #[test]
    fn explicit_beam_elements_use_their_exact_region_materials() {
        let mut model = fixture_model(FixtureId::StructuralBeamCantileverEndMomentReference);
        model.materials[0]
            .require_mechanical_mut()
            .unwrap()
            .youngs_modulus_pa = runmat_types::physical::YoungsModulus::new(100.0e9);
        model.materials[0].mass_density_kg_per_m3 =
            (Some(1_000.0)).map(runmat_types::physical::MassDensity::new);
        model.materials[0].damping = Some(runmat_analysis_core::MechanicalDampingModel::Rayleigh {
            mass_coefficient: runmat_analysis_core::RayleighMassCoefficient::new(0.1),
            stiffness_coefficient: runmat_analysis_core::RayleighStiffnessCoefficient::new(1.0e-5),
        });
        let mut second_material = model.materials[0].clone();
        second_material.material_id = "mat_second".into();
        second_material
            .require_mechanical_mut()
            .unwrap()
            .youngs_modulus_pa = runmat_types::physical::YoungsModulus::new(200.0e9);
        second_material.mass_density_kg_per_m3 =
            (Some(2_000.0)).map(runmat_types::physical::MassDensity::new);
        second_material.damping = Some(runmat_analysis_core::MechanicalDampingModel::Rayleigh {
            mass_coefficient: runmat_analysis_core::RayleighMassCoefficient::new(0.2),
            stiffness_coefficient: runmat_analysis_core::RayleighStiffnessCoefficient::new(2.0e-5),
        });
        model.materials.push(second_material);
        model.material_assignments = vec![
            runmat_analysis_core::MaterialAssignment {
                target: runmat_analysis_core::PropertyRegionTarget::beam(
                    runmat_meshing_core::fixtures::fixture_edge("span_first"),
                ),
                expected_material_id: "mat_steel".into(),
                assigned_material_id: "mat_steel".into(),
                confidence: runmat_analysis_core::EvidenceConfidence::Verified,
            },
            runmat_analysis_core::MaterialAssignment {
                target: runmat_analysis_core::PropertyRegionTarget::beam(
                    runmat_meshing_core::fixtures::fixture_edge("span_second"),
                ),
                expected_material_id: "mat_second".into(),
                assigned_material_id: "mat_second".into(),
                confidence: runmat_analysis_core::EvidenceConfidence::Verified,
            },
        ];
        let structural = model.structural.as_mut().expect("beam topology");
        structural.elements[0].region_id =
            runmat_meshing_core::fixtures::fixture_edge("span_first");
        structural.nodes.push(runmat_analysis_core::StructuralNode {
            node_id: 3,
            coordinates_m: runmat_types::physical::Quantity::<
                runmat_types::physical::kind::Position,
                _,
            >::new([2.0, 0.0, 0.0]),
            solid_attachment: None,
        });
        let mut second_element = structural.elements[0].clone();
        second_element.element_id = "beam_2".into();
        second_element.region_id = runmat_meshing_core::fixtures::fixture_edge("span_second");
        let StructuralElementKind::Beam(beam) = &mut second_element.kind else {
            unreachable!("fixture element is a beam")
        };
        beam.node_ids = [2, 3];
        structural.elements.push(second_element);

        let summary = assemble_structural_system(
            &model,
            &static_excitations(&model),
            None,
            StructuralFormulationRequirement::PhysicalDiscretization,
        )
        .expect("multi-material beam assembly");
        assert_eq!(summary.structural_beam_recovery.len(), 2);
        assert_eq!(
            summary.structural_beam_recovery[0]
                .material
                .youngs_modulus_pa
                .canonical_value(),
            100.0e9
        );
        assert_eq!(
            summary.structural_beam_recovery[1]
                .material
                .youngs_modulus_pa
                .canonical_value(),
            200.0e9
        );

        let operators = summary.operator.operators();
        let first_endpoint = 0;
        let second_endpoint = 12;
        assert_relative(
            operators.mass().get(second_endpoint, second_endpoint)
                / operators.mass().get(first_endpoint, first_endpoint),
            2.0,
            1.0e-12,
        );
        for (dof, alpha, beta) in [
            (first_endpoint, 0.1, 1.0e-5),
            (second_endpoint, 0.2, 2.0e-5),
        ] {
            let expected =
                alpha * operators.mass().get(dof, dof) + beta * operators.stiffness().get(dof, dof);
            assert_relative(operators.damping().get(dof, dof), expected, 1.0e-12);
        }
    }

    #[test]
    fn explicit_element_with_ambiguous_unassigned_material_fails_closed() {
        let mut model = fixture_model(FixtureId::StructuralBeamCantileverEndMomentReference);
        let mut second = model.materials[0].clone();
        second.material_id = "second".into();
        model.materials.push(second);
        model.material_assignments.clear();

        let error = assemble_structural_system(
            &model,
            &static_excitations(&model),
            None,
            StructuralFormulationRequirement::PhysicalDiscretization,
        )
        .expect_err("multi-material explicit topology requires an exact region assignment");
        assert!(matches!(
            error,
            LinearAssemblyError::Structural(
                StructuralAssemblyError::InvalidMaterialCoverage(message)
            ) if message.contains("beam_1")
                && message.contains(&runmat_meshing_core::fixtures::fixture_edge("beam_span"))
                && message.contains("found 0")
        ));
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

        assert!(
            matches!(
                &err,
                LinearAssemblyError::Structural(StructuralAssemblyError::InvalidLoadPlan(message))
                    if message.contains("tip_load") && message.contains("projection")
            ),
            "{err:?}"
        );
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
                StructuralFormulationRequirement::PhysicalDiscretization,
            )
            .expect_err("Tet4 connectivity must contain exactly four nodes");
            assert!(matches!(error, LinearAssemblyError::SolidOperator(_)));
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
        assert!(summary.operator.constrained_dofs()[0]);
        assert!(summary.operator.constrained_dofs()[1]);
        assert!(summary.operator.constrained_dofs()[2]);
        assert!((summary.operator.load()[10] + 4.0).abs() <= 4.0 * f64::EPSILON);
        assert!(summary.structural_load_plan_identity.is_some());
        assert!(summary.operator.nullspace_evidence().is_some());
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
        assert_eq!(
            projection.region_id,
            runmat_meshing_core::fixtures::fixture_boundary_selection("tip")
        );
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
        let missing_tip = runmat_meshing_core::fixtures::fixture_boundary_selection("missing_tip");

        assert!(matches!(
            err,
            LinearAssemblyError::Structural(StructuralAssemblyError::InvalidLoadPlan(ref message))
                if message.contains("load_tip") && message.contains(&missing_tip)
        ));
        let message = err.to_string();
        assert!(message.contains("load_id=load_tip"));
        assert!(message.contains(&format!("region_id={missing_tip}")));
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
        assert!(message.contains(&format!(
            "`{}`",
            runmat_meshing_core::fixtures::fixture_boundary_selection("missing_root")
        )));
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
            StructuralFormulationRequirement::PhysicalDiscretization,
        )
        .expect("valid pressure boundary should assemble");

        assert_close(summary.operator.load()[1], -2.0);
        assert_close(summary.operator.load()[4], -2.0);
        assert_close(summary.operator.load()[10], -2.0);
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
                summary.operator.load()[summary
                    .structural_dof_layout
                    .index(node_index, StructuralDofKind::Ux)
                    .unwrap()],
                summary.operator.load()[summary
                    .structural_dof_layout
                    .index(node_index, StructuralDofKind::Uy)
                    .unwrap()],
                summary.operator.load()[summary
                    .structural_dof_layout
                    .index(node_index, StructuralDofKind::Uz)
                    .unwrap()],
            ];
            for axis in 0..3 {
                force[axis] += nodal[axis];
            }
            let arm = std::array::from_fn::<_, 3, _>(|axis| {
                projection.nodes[node_index].coordinates_m.canonical_value()[axis]
                    - face.centroid_m[axis]
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
            summary.operator.load()[summary
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
        assert!(evidence.resultant_relative_error() <= evidence.requirements.relative_tolerance);
        assert!(
            evidence.moment_relative_error().unwrap() <= evidence.requirements.relative_tolerance
        );
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
            assert_close(
                requested_resultant.real[axis],
                component * face.area_m2.canonical_value(),
            );
            assert_close(
                projected_resultant.real[axis],
                component * face.area_m2.canonical_value(),
            );
        }
        assert_eq!(evidence.resolved_node_count, 6);
        assert!(evidence.accepted);
        assert!(evidence.resultant_relative_error() <= evidence.requirements.relative_tolerance);
        assert!(
            evidence.moment_relative_error().unwrap() <= evidence.requirements.relative_tolerance
        );
    }

    #[test]
    fn physical_solid_uses_physical_traction_projection() {
        let mut model = fixture_model(FixtureId::CantileverLinearStatic);
        model.constraints = Vec::new();
        replace_fixture_excitations(
            &mut model,
            vec![FixtureExcitation {
                id: "traction_tip".into(),
                region_id: "tip".into(),
                kind: FixtureExcitationKind::Traction([12.0, 0.0, 0.0]),
            }],
        );
        let mut mesh = tetrahedron4_mesh();
        set_boundary_regions(&mut mesh, 1, &["tip"]);
        reseal(&mut mesh);

        let summary = assemble_structural_system(
            &model,
            &static_excitations(&model),
            Some(mesh),
            StructuralFormulationRequirement::SyntheticOperator,
        )
        .expect("admission with a solver mesh must use physical traction quadrature");

        let evidence = summary
            .spatial_projection_evidence
            .first()
            .expect("traction projection must publish evidence");
        assert_eq!(
            evidence.integration_rule,
            crate::excitation::SpatialIntegrationRule::BoundaryDunavantSeven
        );
        assert!(summary.structural_load_plan_identity.is_some());
    }

    // witness: FEA-STRUCTURAL-SOLID-001/damping_body_loads @ linear_structural/tet10/numerical_kernel;
    //   coverage: partial;
    //   boundary: Tet10 body-force quadrature preserves the integrated volume resultant.
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
                force[axis] += summary.operator.load()[summary
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
        assert!(evidence.resultant_relative_error() <= evidence.requirements.relative_tolerance);
        assert!(
            evidence.moment_relative_error().unwrap() <= evidence.requirements.relative_tolerance
        );
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
                        target: runmat_analysis_core::RegionTarget::boundary(
                            runmat_meshing_core::fixtures::fixture_face("tip"),
                        ),
                        reference_point: runmat_analysis_core::ReferencePoint {
                            reference_point_id: "tip".into(),
                            position: runmat_analysis_core::Position::new([0.0, 0.0, 0.0]),
                        },
                    },
                ),
            },
        )];
        model.constraints.push(fixed_constraint("root"));
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
        set_boundary_regions(&mut mesh, 0, &["root"]);
        set_boundary_regions(&mut mesh, 1, &["tip"]);
        reseal(&mut mesh);

        let summary = assemble_structural_system(
            &model,
            &static_excitations(&model),
            Some(mesh),
            StructuralFormulationRequirement::PhysicalDiscretization,
        )
        .expect("valid wrench boundary should assemble");

        assert!(summary.structural_moment_projection.is_empty());
        assert!(summary.operator.nullspace_evidence().is_some());
        assert_eq!(summary.structural_dof_layout.reference_point_count(), 1);
        let reference_ry = summary
            .structural_dof_layout
            .reference_point_index(0, StructuralDofKind::Ry)
            .unwrap();
        assert_close(summary.operator.load()[reference_ry], 6.0);
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
        structural.nodes[1].coordinates_m =
            runmat_types::physical::Quantity::<runmat_types::physical::kind::Position, _>::new(
                structural.nodes[0].coordinates_m.canonical_value(),
            );

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
        structural.nodes[1].coordinates_m =
            runmat_types::physical::Quantity::<runmat_types::physical::kind::Position, _>::new(
                structural.nodes[0].coordinates_m.canonical_value(),
            );
        structural.nodes[2].coordinates_m =
            runmat_types::physical::Quantity::<runmat_types::physical::kind::Position, _>::new(
                structural.nodes[0].coordinates_m.canonical_value(),
            );

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
                StructuralFormulationRequirement::SyntheticOperator,
            )
            .expect_err(name);
            assert_eq!(error, LinearAssemblyError::Structural(expected), "{name}");
        }
    }

    // witness: FEA-STRUCTURAL-BEAM-001/physical_beam_mixed_dofs @
    //   linear_structural/beam|shell/numerical_kernel; coverage: partial;
    //   boundary: Mixed beam and shell connectivity assembles one shared DOF system.
    #[test]
    fn mixed_beam_shell_connectivity_assembles_one_shared_dof_system() {
        let mut model = fixture_model(FixtureId::StructuralBeamCantileverEndMomentReference);
        let shell_model = fixture_model(FixtureId::StructuralShellPlateMomentReference);
        model
            .material_assignments
            .extend(shell_model.material_assignments.clone());
        let structural = model
            .structural
            .as_mut()
            .expect("beam fixture has structural topology");
        let shell = shell_model
            .structural
            .expect("shell fixture has structural topology");
        structural.nodes.push(shell.nodes[2].clone());
        structural.elements.push(shell.elements[0].clone());
        structural.shell_sections = shell.shell_sections;

        let summary = assemble_structural_system(
            &model,
            &static_excitations(&model),
            None,
            StructuralFormulationRequirement::PhysicalDiscretization,
        )
        .expect("shared beam-shell nodes must assemble into one physical system");

        assert_eq!(summary.dof_count, 18);
        assert_eq!(summary.structural_node_count, 3);
        assert_eq!(summary.structural_beam_element_count, 1);
        assert_eq!(summary.structural_shell_element_count, 1);
        assert_eq!(summary.structural_formulations.len(), 2);
        assert!(summary.structural_load_plan_identity.is_some());
        assert!(summary.operator.nullspace_evidence().is_some());
        summary.operator.prepare_linear().unwrap();
    }

    #[test]
    fn symmetric_orthotropic_laminate_assembles_without_homogeneous_material_fallback() {
        use runmat_analysis_core::{
            LaminatePlyModel, LaminateShellSectionModel, MaterialElasticityModel,
            MaterialOrthotropicLaminaModel, PlyOrderDirection, ShellReferenceSurface,
            ShellSectionModel,
        };

        let mut model = fixture_model(FixtureId::StructuralShellPlateMomentReference);
        let mut zero = model.materials[0].clone();
        zero.material_id = "carbon_epoxy_0".into();
        zero.name = "Carbon/epoxy 0 degree lamina".into();
        zero.mass_density_kg_per_m3 = (Some(1_600.0)).map(runmat_types::physical::MassDensity::new);
        zero.elasticity = Some(MaterialElasticityModel::OrthotropicLamina(
            MaterialOrthotropicLaminaModel {
                longitudinal_modulus_pa: runmat_types::physical::YoungsModulus::new(135.0e9),
                transverse_modulus_pa: runmat_types::physical::YoungsModulus::new(10.0e9),
                major_poisson_ratio: runmat_types::physical::PoissonRatio::new(0.3),
                in_plane_shear_modulus_pa: runmat_types::physical::ShearModulus::new(5.0e9),
                transverse_shear_modulus_13_pa: runmat_types::physical::ShearModulus::new(5.0e9),
                transverse_shear_modulus_23_pa: runmat_types::physical::ShearModulus::new(3.8e9),
            },
        ));
        zero.damping = Some(runmat_analysis_core::MechanicalDampingModel::Rayleigh {
            mass_coefficient: runmat_analysis_core::RayleighMassCoefficient::new(0.2),
            stiffness_coefficient: runmat_analysis_core::RayleighStiffnessCoefficient::new(2.0e-5),
        });
        let mut ninety = zero.clone();
        ninety.material_id = "carbon_epoxy_90".into();
        ninety.name = "Carbon/epoxy 90 degree lamina".into();
        model.materials = vec![zero, ninety];
        model.material_assignments.clear();
        let structural = model.structural.as_mut().expect("shell topology");
        let ply_thickness = 0.000_125;
        structural.shell_sections = vec![ShellSectionModel::Laminate(LaminateShellSectionModel {
            section_id: "panel_2mm".into(),
            reference_orientation_id: "global_x".into(),
            ply_order: PlyOrderDirection::AlongSurfaceNormal,
            reference_surface: ShellReferenceSurface::MidSurface,
            plies: [
                ("outer_bottom", "carbon_epoxy_0", 0.0),
                (
                    "inner_bottom",
                    "carbon_epoxy_90",
                    std::f64::consts::FRAC_PI_2,
                ),
                ("inner_top", "carbon_epoxy_90", std::f64::consts::FRAC_PI_2),
                ("outer_top", "carbon_epoxy_0", 0.0),
            ]
            .into_iter()
            .map(|(ply_id, material_id, angle_rad)| LaminatePlyModel {
                ply_id: ply_id.into(),
                material_id: material_id.into(),
                thickness_m: runmat_types::physical::Quantity::<
                    runmat_types::physical::kind::Length,
                    _,
                >::new(ply_thickness),
                angle_rad: runmat_types::physical::Angle::new(angle_rad),
            })
            .collect(),
            shear_correction: None,
        })];

        let summary = assemble_structural_system(
            &model,
            &static_excitations(&model),
            None,
            StructuralFormulationRequirement::PhysicalDiscretization,
        )
        .expect("symmetric laminate shell must assemble through the physical formulation");

        assert_eq!(summary.structural_shell_element_count, 1);
        assert_eq!(summary.structural_shell_recovery.len(), 1);
        let constitutive = &summary.structural_shell_recovery[0].constitutive;
        for row in constitutive.matrices.coupling {
            for value in row {
                assert!(value.abs() < 1.0e-8, "symmetric laminate B={value:e}");
            }
        }
        assert_eq!(constitutive.recovery_layers.len(), 4);
        assert!(summary.operator.prepare_linear().is_ok());
        assert!(summary
            .operator
            .operators()
            .damping()
            .values()
            .iter()
            .any(|value| *value != 0.0));
    }

    #[test]
    fn physical_requirement_rejects_meshless_structural_assembly() {
        let model = fixture_model(FixtureId::CantileverLinearStatic);
        let error = assemble_structural_system(
            &model,
            &static_excitations(&model),
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
    fn explicit_shell_pressure_and_traction_integrate_physical_area() {
        for (kind, expected) in [
            (FixtureExcitationKind::Pressure(12.0), [0.0, 0.0, 6.0]),
            (
                FixtureExcitationKind::Traction([2.0, 4.0, 6.0]),
                [1.0, 2.0, 3.0],
            ),
        ] {
            let mut model = fixture_model(FixtureId::StructuralShellPlateMomentReference);
            model.constraints.clear();
            replace_fixture_excitations(
                &mut model,
                vec![FixtureExcitation {
                    id: "shell_surface_load".into(),
                    region_id: runmat_meshing_core::fixtures::fixture_face("shell_panel"),
                    kind,
                }],
            );
            let summary = assemble_structural_system(
                &model,
                &static_excitations(&model),
                None,
                StructuralFormulationRequirement::PhysicalDiscretization,
            )
            .expect("explicit shell surface intensity must integrate over authored area");
            let resultant = [
                StructuralDofKind::Ux,
                StructuralDofKind::Uy,
                StructuralDofKind::Uz,
            ]
            .map(|kind| {
                (0..3)
                    .map(|node| {
                        summary.operator.load()
                            [summary.structural_dof_layout.index(node, kind).unwrap()]
                    })
                    .sum::<f64>()
            });
            for (actual, expected) in resultant.into_iter().zip(expected) {
                assert_close(actual, expected);
            }
            assert_eq!(
                summary.spatial_projection_evidence[0].integration_rule,
                crate::excitation::SpatialIntegrationRule::ExplicitConsistentElement
            );
            assert!(summary.spatial_projection_evidence[0].accepted);
        }
    }

    // witness: FEA-STRUCTURAL-SOLID-001/damping_body_loads @
    //   linear_structural/beam|shell/numerical_kernel; coverage: partial;
    //   boundary: An explicit body-force density integrates over beam and shell volumes.
    #[test]
    fn explicit_body_force_density_integrates_beam_and_shell_volume() {
        let cases = [
            (
                FixtureId::StructuralBeamCantileverEndMomentReference,
                runmat_meshing_core::fixtures::fixture_edge("beam_span"),
                [100.0, 0.0, 0.0],
                [0.02, 0.0, 0.0],
            ),
            (
                FixtureId::StructuralShellPlateMomentReference,
                runmat_meshing_core::fixtures::fixture_face("shell_panel"),
                [0.0, 0.0, -1_000.0],
                [0.0, 0.0, -1.0],
            ),
        ];
        for (fixture, region_id, density, expected) in cases {
            let mut model = fixture_model(fixture);
            model.constraints.clear();
            replace_fixture_excitations(
                &mut model,
                vec![FixtureExcitation {
                    id: "body_density".into(),
                    region_id: region_id.into(),
                    kind: FixtureExcitationKind::BodyForce(density),
                }],
            );
            let summary = assemble_structural_system(
                &model,
                &static_excitations(&model),
                None,
                StructuralFormulationRequirement::PhysicalDiscretization,
            )
            .expect("explicit body-force density must integrate formulation volume");
            let resultant = [
                StructuralDofKind::Ux,
                StructuralDofKind::Uy,
                StructuralDofKind::Uz,
            ]
            .map(|kind| {
                (0..summary.structural_node_count)
                    .map(|node| {
                        summary.operator.load()
                            [summary.structural_dof_layout.index(node, kind).unwrap()]
                    })
                    .sum::<f64>()
            });
            for (actual, expected) in resultant.into_iter().zip(expected) {
                assert_close(actual, expected);
            }
            assert!(summary.spatial_projection_evidence[0].accepted);
        }
    }

    #[test]
    fn structural_load_frames_resolve_against_boundary_and_authored_axes() {
        let mut shell = fixture_model(FixtureId::StructuralShellPlateMomentReference);
        shell.constraints.clear();
        replace_fixture_excitations(
            &mut shell,
            vec![FixtureExcitation {
                id: "normal_traction".into(),
                region_id: runmat_meshing_core::fixtures::fixture_face("shell_panel"),
                kind: FixtureExcitationKind::Traction([2.0, 0.0, 0.0]),
            }],
        );
        let shell_loads = RealizedExcitations::new(
            static_excitations(&shell)
                .iter()
                .cloned()
                .map(|mut load| {
                    load.policy.frame =
                        runmat_analysis_core::CoordinateFrame::BoundaryNormalTangent;
                    load
                })
                .collect(),
        );
        let shell_summary = assemble_structural_system(
            &shell,
            &shell_loads,
            None,
            StructuralFormulationRequirement::PhysicalDiscretization,
        )
        .expect("boundary-normal shell traction must resolve through oriented shell geometry");
        let shell_resultant = [
            StructuralDofKind::Ux,
            StructuralDofKind::Uy,
            StructuralDofKind::Uz,
        ]
        .map(|kind| {
            (0..3)
                .map(|node| {
                    shell_summary.operator.load()[shell_summary
                        .structural_dof_layout
                        .index(node, kind)
                        .unwrap()]
                })
                .sum::<f64>()
        });
        assert_eq!(shell_resultant, [0.0, 0.0, 1.0]);

        let mut solid = fixture_model(FixtureId::CantileverLinearStatic);
        solid.constraints.clear();
        let solid_loads = RealizedExcitations::new(
            static_excitations(&solid)
                .iter()
                .cloned()
                .map(|mut load| {
                    load.policy.frame = runmat_analysis_core::CoordinateFrame::Cartesian(
                        runmat_analysis_core::CartesianFrame {
                            frame_id: "rotated".into(),
                            origin: runmat_analysis_core::Position::new([0.0; 3]),
                            x_axis: [0.0, 1.0, 0.0],
                            y_axis: [-1.0, 0.0, 0.0],
                        },
                    );
                    if let RealizedExcitationKind::Force(value) = &mut load.kind {
                        value.real = [2.0, 0.0, 0.0];
                    }
                    load
                })
                .collect(),
        );
        let mut mesh = tetrahedron4_mesh();
        set_boundary_regions(&mut mesh, 1, &["tip"]);
        reseal(&mut mesh);
        let solid_summary = assemble_structural_system(
            &solid,
            &solid_loads,
            Some(mesh),
            StructuralFormulationRequirement::PhysicalDiscretization,
        )
        .expect("Cartesian load components must resolve into global solver rows");
        let solid_resultant = [
            StructuralDofKind::Ux,
            StructuralDofKind::Uy,
            StructuralDofKind::Uz,
        ]
        .map(|kind| {
            (0..solid_summary.structural_node_count)
                .map(|node| {
                    solid_summary.operator.load()[solid_summary
                        .structural_dof_layout
                        .index(node, kind)
                        .unwrap()]
                })
                .sum::<f64>()
        });
        assert_eq!(solid_resultant, [0.0, 2.0, 0.0]);
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
                    case_ids: vec![runmat_analysis_core::ExcitationCaseId("case".to_string())],
                    signal_id: runmat_analysis_core::SignalId("signal".to_string()),
                    quantity,
                    sample: crate::excitation::ProjectionSample::Static,
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

        let first = assemble_structural_system(
            &model,
            &first_sample,
            None,
            StructuralFormulationRequirement::SyntheticOperator,
        )
        .expect("first sampled load should assemble");
        let second = assemble_structural_system(
            &model,
            &second_sample,
            None,
            StructuralFormulationRequirement::SyntheticOperator,
        )
        .expect("second sampled load should assemble");

        assert_eq!(
            first.operator.operators().stiffness(),
            second.operator.operators().stiffness()
        );
        assert_eq!(first.operator.load().len(), second.operator.load().len());
        assert!(first.operator.load().iter().any(|value| value.abs() > 0.0));
        for (first_value, second_value) in first.operator.load().iter().zip(second.operator.load())
        {
            assert!((second_value - 2.0 * first_value).abs() <= 1.0e-12);
        }
    }

    #[test]
    fn structural_load_plan_rejects_unimplemented_policy_semantics_by_excitation() {
        let mut model = fixture_model(FixtureId::CantileverLinearStatic);
        model.constraints.clear();
        let baseline = static_excitations(&model);
        let cases = [
            (
                "kinematics",
                runmat_analysis_core::CoordinateFrame::Global,
                runmat_analysis_core::ExcitationDistribution::PreserveResultant,
                runmat_analysis_core::ExcitationKinematics::Follower,
            ),
            (
                "distribution",
                runmat_analysis_core::CoordinateFrame::Global,
                runmat_analysis_core::ExcitationDistribution::TopologyField,
                runmat_analysis_core::ExcitationKinematics::FixedFrame,
            ),
            (
                "frame",
                runmat_analysis_core::CoordinateFrame::Material,
                runmat_analysis_core::ExcitationDistribution::PreserveResultant,
                runmat_analysis_core::ExcitationKinematics::FixedFrame,
            ),
        ];
        for (expected_policy, frame, distribution, kinematics) in cases {
            let entries = baseline
                .iter()
                .cloned()
                .map(|mut load| {
                    load.policy.frame = frame.clone();
                    load.policy.distribution = distribution;
                    load.policy.kinematics = kinematics;
                    load
                })
                .collect();
            let error = assemble_structural_system(
                &model,
                &RealizedExcitations::new(entries),
                Some(tetrahedron4_mesh()),
                StructuralFormulationRequirement::PhysicalDiscretization,
            )
            .expect_err("unsupported load semantics must fail before solver assembly");
            assert!(
                matches!(
                    &error,
                    LinearAssemblyError::Structural(StructuralAssemblyError::InvalidLoadPlan(message))
                        if message.contains("tip_load") && message.contains(expected_policy)
                ),
                "{error:?}"
            );
        }
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
            StructuralFormulationRequirement::PhysicalDiscretization,
        )
        .expect_err("unmapped explicit source must not be dropped");
        assert!(matches!(
            error,
            LinearAssemblyError::Structural(StructuralAssemblyError::InvalidLoadPlan(message))
                if message.contains("missing_tip")
                    && message.contains(&runmat_meshing_core::fixtures::fixture_boundary_selection(
                        "not_a_connectivity_region"
                    ))
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
                        target: runmat_analysis_core::RegionTarget::authored(
                            runmat_meshing_core::fixtures::fixture_face("root"),
                        ),
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
            StructuralFormulationRequirement::PhysicalDiscretization,
        )
        .expect("Tet4 assembly should succeed");
        let tet10 = assemble_structural_system(
            &solid_model,
            &static_excitations(&solid_model),
            Some(crate::assembly::solver_solid::tests::artifact(
                ElementOrder::Tet10,
            )),
            StructuralFormulationRequirement::PhysicalDiscretization,
        )
        .expect("Tet10 assembly should succeed");
        let beam_model = fixture_model(FixtureId::StructuralBeamCantileverEndMomentReference);
        let beam = assemble_structural_system(
            &beam_model,
            &static_excitations(&beam_model),
            None,
            StructuralFormulationRequirement::PhysicalDiscretization,
        )
        .expect("beam assembly should succeed");
        assert!(beam.operator.nullspace_evidence().is_some());
        let shell_model = fixture_model(FixtureId::StructuralShellPlateMomentReference);
        let shell = assemble_structural_system(
            &shell_model,
            &static_excitations(&shell_model),
            None,
            StructuralFormulationRequirement::PhysicalDiscretization,
        )
        .expect("shell assembly should succeed");
        assert!(shell.operator.nullspace_evidence().is_some());
        let synthetic_model = fixture_model(FixtureId::CantileverLinearStatic);
        let synthetic = assemble_structural_system(
            &synthetic_model,
            &static_excitations(&synthetic_model),
            None,
            StructuralFormulationRequirement::SyntheticOperator,
        )
        .expect("typed admission should permit synthetic assembly");

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
                Some(StructuralRecoveryImplementation::Tet10IntegrationPointProjection),
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
                StructuralOperatorImplementation::Mitc3TriangularShell,
                Some(StructuralRecoveryImplementation::Mitc3ShellStressResultants),
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
            let [realization] = summary.structural_formulations.as_slice() else {
                panic!("single-formulation assembly must record exactly one realization");
            };
            assert_eq!(realization.operator, operator);
            assert_eq!(realization.recovery, recovery);
            assert_eq!(summary.structural_sources, vec![source]);
            assert_eq!(summary.structural_boundary_transfers, vec![boundary]);
        }
    }

    #[test]
    fn beam_nullspace_evidence_requires_every_disconnected_component_to_be_anchored() {
        let mut model = fixture_model(FixtureId::StructuralBeamCantileverEndMomentReference);
        model
            .material_assignments
            .push(runmat_analysis_core::MaterialAssignment {
                target: runmat_analysis_core::PropertyRegionTarget::beam(
                    runmat_meshing_core::fixtures::fixture_edge("beam_span_2"),
                ),
                expected_material_id: "mat_steel".into(),
                assigned_material_id: "mat_steel".into(),
                confidence: runmat_analysis_core::EvidenceConfidence::Verified,
            });
        let structural = model.structural.as_mut().unwrap();
        structural.nodes.extend([
            runmat_analysis_core::StructuralNode {
                node_id: 3,
                coordinates_m: runmat_types::physical::Quantity::<
                    runmat_types::physical::kind::Position,
                    _,
                >::new([0.0, 2.0, 0.0]),
                solid_attachment: None,
            },
            runmat_analysis_core::StructuralNode {
                node_id: 4,
                coordinates_m: runmat_types::physical::Quantity::<
                    runmat_types::physical::kind::Position,
                    _,
                >::new([1.0, 2.0, 0.0]),
                solid_attachment: None,
            },
        ]);
        structural
            .elements
            .push(runmat_analysis_core::StructuralElement {
                element_id: "beam_2".into(),
                region_id: runmat_meshing_core::fixtures::fixture_edge("beam_span_2"),
                kind: StructuralElementKind::Beam(runmat_analysis_core::BeamElementModel {
                    node_ids: [3, 4],
                    section_id: "rect".into(),
                    reference_axis: [0.0, 1.0, 0.0],
                }),
            });
        let unanchored = assemble_structural_system(
            &model,
            &static_excitations(&model),
            None,
            StructuralFormulationRequirement::PhysicalDiscretization,
        )
        .unwrap();
        assert!(unanchored.operator.nullspace_evidence().is_none());

        let mut second_anchor = model.constraints[0].clone();
        let structural_constraint = match &mut second_anchor {
            runmat_analysis_core::AnalysisConstraint::Structural(constraint) => constraint,
            _ => unreachable!(),
        };
        structural_constraint.constraint_id = "fixed_second_root".into();
        second_anchor.visit_targets_mut(|target| target.region_id = "node:3".into());
        model.constraints.push(second_anchor);
        let anchored = assemble_structural_system(
            &model,
            &static_excitations(&model),
            None,
            StructuralFormulationRequirement::PhysicalDiscretization,
        )
        .unwrap();
        assert!(anchored.operator.nullspace_evidence().is_some());
        anchored.operator.prepare_linear().unwrap();

        let stiffness = anchored.operator.operators().stiffness();
        let mut perturbed = RealSparseAssembler::square(
            stiffness.rows(),
            OperatorRole::Stiffness,
            MatrixProperties::SYMMETRIC_POSITIVE_SEMIDEFINITE,
        );
        for row in 0..stiffness.rows() {
            for entry in stiffness.pattern().row_range(row) {
                let column = stiffness.pattern().column(entry);
                let increment = if row == 0 && column == 0 { 1.0 } else { 0.0 };
                perturbed.add(row, column, stiffness.values()[entry] + increment);
            }
        }
        let perturbed = perturbed.finish().unwrap();
        let topology = crate::physics::structural::topology::node_space::CanonicalStructuralNodeSpace::derive_explicit(
            model.structural.as_ref().unwrap(),
        )
        .unwrap();
        let constraint_plan =
            CanonicalStructuralConstraintPlan::compile_for_model(&model, None).unwrap();
        let constraints =
            resolve_constraint_matrix(&constraint_plan, &topology, &anchored.structural_dof_layout)
                .unwrap();
        assert_eq!(
            NullspaceEliminationEvidence::prove_structural_frame(
                model.structural.as_ref().unwrap(),
                &anchored.structural_dof_layout,
                &perturbed,
                &constraint_plan,
                &constraints,
                anchored.operator.constrained_dofs(),
            ),
            Err(crate::algebra::NullspaceEvidenceError::NullModeViolation)
        );
    }

    // witness: FEA-STRUCTURAL-SHELL-001/physical_shell_operator @
    //   linear_structural/shell/numerical_kernel; coverage: partial;
    //   boundary: An unconstrained shell retains its physical rigid-body null space.
    #[test]
    fn unconstrained_shell_retains_its_physical_rigid_body_nullspace() {
        let mut model = fixture_model(FixtureId::StructuralShellPlateMomentReference);
        model.constraints.clear();
        let summary = assemble_structural_system(
            &model,
            &static_excitations(&model),
            None,
            StructuralFormulationRequirement::PhysicalDiscretization,
        )
        .unwrap();

        assert!(summary.operator.nullspace_evidence().is_none());
        let error = summary
            .operator
            .prepare_linear()
            .expect_err("an unconstrained shell must be rejected before iteration");
        assert!(
            matches!(
                error,
                crate::algebra::RealSystemError::Coercivity(
                    crate::algebra::CoercivityError::NonPositivePivot { .. }
                )
            ),
            "unexpected preparation error: {error:?}"
        );
    }

    fn tetrahedron4_mesh() -> SolverMeshArtifact {
        crate::assembly::solver_solid::tests::artifact(ElementOrder::Tet4)
    }

    fn set_boundary_regions(mesh: &mut SolverMeshArtifact, face_index: usize, regions: &[&str]) {
        mesh.topology.boundary_faces[face_index].provenance = regions
            .iter()
            .map(|region| {
                PersistentEntityId::new(
                    PersistentEntityKind::Face,
                    (*region).to_owned(),
                    vec!["root".to_owned()],
                )
            })
            .collect();
        mesh.topology.boundary_faces[face_index].provenance.sort();
    }

    fn clear_boundary_regions(mesh: &mut SolverMeshArtifact) {
        for (index, face) in mesh.topology.boundary_faces.iter_mut().enumerate() {
            face.provenance = vec![PersistentEntityId::new(
                PersistentEntityKind::Face,
                format!("unselected:{index}"),
                vec!["root".to_owned()],
            )];
        }
    }

    fn set_solver_mesh_region(mesh: &mut SolverMeshArtifact, region_id: &str) {
        let region = PersistentEntityId::new(
            PersistentEntityKind::Region,
            region_id.to_owned(),
            vec!["root".to_owned()],
        );
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

    fn assert_relative(actual: f64, expected: f64, tolerance: f64) {
        let relative = (actual - expected).abs() / expected.abs().max(1.0);
        assert!(
            relative <= tolerance,
            "actual={actual} expected={expected} relative={relative} tolerance={tolerance}"
        );
    }

    fn first_csr_diagonal(summary: &AssemblySummary) -> f64 {
        summary.operator.operators().stiffness().get(0, 0).abs()
    }
}
