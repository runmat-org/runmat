use super::{invalid, AnalysisExecutionResult, FluidSteadySolverResult, LIMITS, WORKLOAD_PREFIX};
use crate::{
    execution::{AnalysisWorkloadIdentity, SequenceIdentity, StepIdentity},
    physics::cfd::canonical::{
        CanonicalFluidInput, CanonicalFluidPolicy, CanonicalFluidProblem, CanonicalFluidReservation,
    },
    properties::{CompiledPropertyMapIdentity, PreparedProperties},
    topology::{AnalysisProjection, FiniteVolumeProjectionPolicy, FiniteVolumeTopology},
};
use runmat_analysis_core::{
    validate_model, AnalysisConstraint, AnalysisModel, AnalysisProcedure, CfdDomain,
    CoordinateFrame, ExcitationAction, ExcitationCaseActivation, ExcitationCaseSelection,
    FluidConstraint, FormulationRequirement, IndependentDomain, PhysicalSignal, PhysicsFamily,
    ProcedureSettings, ProcedureStepId, RestartPolicy, SolutionSpace, StateCompleteness, StateKind,
    StatePersistence,
};
use runmat_canonical_codec::{
    decode_contract, encode_contract, serialized_materialization_resident_bytes_estimate,
};
use runmat_execution::Digest;
use runmat_meshing_core::SolverMeshArtifact;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FluidSteadyWorkload {
    pub model: AnalysisModel,
    pub step_id: ProcedureStepId,
    pub formulation_requirement: FormulationRequirement,
    pub policy: CanonicalFluidPolicy,
    pub geometry_policy: FiniteVolumeProjectionPolicy,
}

pub(crate) struct PreparedFluidSource {
    pub source: AnalysisProjection,
    pub topology: FiniteVolumeTopology,
    pub material: CfdDomain,
    pub property_identity: CompiledPropertyMapIdentity,
    pub property_audit_digest: Digest,
    pub constraints: Vec<FluidConstraint>,
    pub preparation_resident_bytes: u64,
}

impl FluidSteadyWorkload {
    pub fn formulation(
        &self,
        mesh: &SolverMeshArtifact,
    ) -> AnalysisExecutionResult<crate::admission::AnalysisRunIdentity> {
        self.validate()?;
        crate::admission::admit_canonical_fluid_steady(mesh, self.formulation_requirement)
            .map_err(invalid)
    }
    pub(crate) fn resident_ceiling(&self) -> AnalysisExecutionResult<u64> {
        let [step] = self.model.sequence.steps.as_slice() else {
            return Err(invalid(
                "steady fluid requires exactly one resource-owning step",
            ));
        };
        if step.step_id != self.step_id {
            return Err(invalid("steady fluid resource step differs from producer"));
        }
        Ok(self
            .policy
            .maximum_resident_bytes
            .min(step.accuracy.resources.maximum_resident_bytes))
    }
    pub fn validate(&self) -> AnalysisExecutionResult<()> {
        validate_model(&self.model).map_err(invalid)?;
        let [step] = self.model.sequence.steps.as_slice() else {
            return Err(invalid("steady fluid requires exactly one authored step"));
        };
        let p = self.policy;
        if let Some(policy) = p.fixed_point_acceleration {
            policy.validate().map_err(invalid)?;
        }
        crate::results::observation::validate_sampled_requests(step).map_err(invalid)?;
        let definitions = super::fields::fluid_native_field_definition_inventory()?;
        if step
            .observations
            .iter()
            .filter_map(|request| request.spec())
            .any(|spec| {
                !definitions.iter().any(|definition| {
                    definition.quantity() == spec.quantity
                        && definition.location() == spec.source_location
                        && definition.value_shape() == &spec.value_shape
                }) || spec.frame != CoordinateFrame::Global
                    || spec.sign_convention != runmat_analysis_core::SignConvention::Native
            })
        {
            return Err(invalid(
                "fluid observations require original native cell U/p or owner-oriented face q",
            ));
        }
        if self
            .model
            .excitation_cases
            .iter()
            .any(|c| c.activation != ExcitationCaseActivation::Always)
            || self
                .model
                .excitations
                .iter()
                .any(|a| !matches!(a, ExcitationAction::BodyForce(_)))
        {
            return Err(invalid(
                "steady fluid has no scheduled/alternate excitation-case authority",
            ));
        }
        if self.formulation_requirement != FormulationRequirement::Experimental
            || step.step_id != self.step_id
            || step.family != PhysicsFamily::Fluid
            || step.procedure != AnalysisProcedure::Static
            || step.solution_space != SolutionSpace::Direct
            || step.independent_domain != IndependentDomain::Static
            || step.settings != ProcedureSettings::None
            || step.restart != RestartPolicy::Fresh
            || step.excitation_case_selection != ExcitationCaseSelection::All
            || !step.activations.is_empty()
            || !step.state_inputs.is_empty()
            || step.state_outputs.len() != 1
            || step.state_outputs.iter().any(|output| {
                output.kind != StateKind::Fluid
                    || output.completeness != StateCompleteness::Complete
                    || output.persistence != StatePersistence::UntilDependentsComplete
            })
            || !self.model.sequence.dependencies.is_empty()
            || !self.model.sequence.transitions.is_empty()
            || !self.model.sequence.branches.is_empty()
            || self.model.frame != CoordinateFrame::Global
            || self.model.structural.is_some()
            || self.model.thermal.is_some()
            || self.model.thermo_mechanical.is_some()
            || self.model.electro_thermal.is_some()
            || self.model.electromagnetic.is_some()
            || !self.model.connectors.is_empty()
            || !self.model.interfaces.is_empty()
            || !self.model.initial_conditions.is_empty()
            || !self.model.response_spectra.is_empty()
            || !self.model.random_excitations.is_empty()
            || self
                .model
                .constraints
                .iter()
                .any(|c| !matches!(c, AnalysisConstraint::Fluid(_)))
            || self.model.signals.iter().any(|s| {
                !matches!(
                    s,
                    PhysicalSignal::Velocity(_)
                        | PhysicalSignal::Pressure(_)
                        | PhysicalSignal::BodyForceDensity(_)
                )
            })
            || p.maximum_outer_iterations == 0
            || p.maximum_nonorthogonal_iterations == 0
            || p.maximum_linear_iterations == 0
            || p.gmres_restart == 0
            || p.maximum_resident_bytes == 0
            || p.maximum_operator_work == 0
            || !p.linear_relative_tolerance.is_finite()
            || !(0.0..1.0).contains(&p.linear_relative_tolerance)
            || p.linear_relative_tolerance == 0.0
            || !p.physical_relative_tolerance.is_finite()
            || !(0.0..1.0).contains(&p.physical_relative_tolerance)
            || p.physical_relative_tolerance == 0.0
            || !p.momentum_relaxation.is_finite()
            || !(0.0..=1.0).contains(&p.momentum_relaxation)
            || p.momentum_relaxation == 0.0
            || !p.pressure_relaxation.is_finite()
            || !(0.0..=1.0).contains(&p.pressure_relaxation)
            || p.pressure_relaxation == 0.0
        {
            return Err(invalid(
                "unsupported standalone experimental CPU steady fluid authority",
            ));
        }
        Ok(())
    }

    pub fn canonical_bytes(&self) -> AnalysisExecutionResult<Vec<u8>> {
        self.validate()?;
        Ok(encode_contract(
            WORKLOAD_PREFIX,
            "steady-fluid-workload",
            self,
            LIMITS,
        )?)
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> AnalysisExecutionResult<Self> {
        let value: Self = decode_contract(WORKLOAD_PREFIX, "steady-fluid-workload", bytes, LIMITS)?;
        value.validate()?;
        Ok(value)
    }

    pub fn identity(&self) -> AnalysisExecutionResult<AnalysisWorkloadIdentity> {
        AnalysisWorkloadIdentity::derive(
            "steady-fluid-workload",
            &Digest::sha256(self.canonical_bytes()?),
        )
    }

    pub fn producer_identity(&self) -> AnalysisExecutionResult<StepIdentity> {
        self.validate()?;
        StepIdentity::from_solver_contract(
            SequenceIdentity::from_solver_contract(&self.model.sequence)?,
            &self.model.sequence.steps[0],
        )
    }

    /// Local numerical execution only; no scheduler/resource-accounting attestation.
    pub fn execute(
        &self,
        mesh: &SolverMeshArtifact,
    ) -> AnalysisExecutionResult<FluidSteadySolverResult> {
        let source = self.prepare(mesh)?;
        let original = source.problem(self)?.solve().map_err(invalid)?;
        FluidSteadySolverResult::from_original(self, &source, original)
    }

    pub(crate) fn prepare(
        &self,
        mesh: &SolverMeshArtifact,
    ) -> AnalysisExecutionResult<PreparedFluidSource> {
        // Count/profile only over borrowed inputs before validation/projection/compiler expansion.
        // These are requested-heap estimates, not allocator/RSS or scheduler accounting.
        let ceiling = self.resident_ceiling()?;
        if mesh
            .topology
            .volume_elements
            .iter()
            .any(|e| e.order != runmat_meshing_core::ElementOrder::Tet4)
        {
            return Err(invalid("portable steady fluid currently admits Tet4 only"));
        }
        let geometry =
            FiniteVolumeTopology::reserve_mesh(mesh, self.geometry_policy).map_err(invalid)?;
        let numerical = CanonicalFluidReservation::for_counts(
            &geometry,
            geometry.face_incidence_count,
            self.policy,
        )
        .map_err(invalid)?;
        let projection =
            AnalysisProjection::derivation_resident_bytes_estimate(mesh).map_err(invalid)?;
        // Two whole authored metadata profiles conservatively cover cloned constraints/material,
        // shared identity serialization and validation scratch coexisting with derived records.
        let metadata =
            serialized_materialization_resident_bytes_estimate(self, LIMITS).map_err(invalid)?;
        let base = add_bytes(
            add_bytes(projection, geometry.scratch_bytes as u64)?,
            add_bytes(metadata, metadata)?,
        )?;
        let preliminary = add_bytes(base, numerical.resident_bytes_upper_bound)?;
        require_bytes(preliminary, ceiling)?;
        let prepared = PreparedProperties::prepare(&self.model, Some(mesh), ceiling - preliminary)
            .map_err(invalid)?;
        // Material source authentication independently recompiles: both maps can coexist.
        let preparation_resident_bytes = add_bytes(
            base,
            add_bytes(
                prepared.estimated_resident_bytes,
                prepared.estimated_resident_bytes,
            )?,
        )?;
        require_bytes(
            add_bytes(
                preparation_resident_bytes,
                numerical.resident_bytes_upper_bound,
            )?,
            ceiling,
        )?;
        self.validate()?;
        let source = AnalysisProjection::derive(mesh).map_err(invalid)?;
        let topology =
            FiniteVolumeTopology::derive(&source, self.geometry_policy).map_err(invalid)?;
        let properties = prepared.materialize().map_err(invalid)?;
        let material =
            crate::study::derive_canonical_fluid_material(&self.model, mesh, &properties)
                .map_err(invalid)?;
        let constraints = self
            .model
            .constraints
            .iter()
            .map(|c| match c {
                AnalysisConstraint::Fluid(c) => Ok(c.clone()),
                _ => Err(invalid("nonfluid constraint")),
            })
            .collect::<AnalysisExecutionResult<Vec<_>>>()?;
        Ok(PreparedFluidSource {
            source,
            topology,
            material,
            property_identity: properties.identity,
            property_audit_digest: properties.audit_digest,
            constraints,
            preparation_resident_bytes,
        })
    }
}

impl PreparedFluidSource {
    pub fn problem<'a>(
        &'a self,
        workload: &'a FluidSteadyWorkload,
    ) -> AnalysisExecutionResult<CanonicalFluidProblem<'a>> {
        let problem = CanonicalFluidProblem::prepare(CanonicalFluidInput {
            source: &self.source,
            topology: &self.topology,
            material: &self.material,
            constraints: &self.constraints,
            boundary_operators: &workload.model.boundaries,
            signals: &workload.model.signals,
            excitations: &workload.model.excitations,
            policy: workload.policy,
        })
        .map_err(invalid)?;
        require_bytes(
            add_bytes(
                self.preparation_resident_bytes,
                problem.reservation().resident_bytes_upper_bound,
            )?,
            workload.resident_ceiling()?,
        )?;
        Ok(problem)
    }
}

fn add_bytes(a: u64, b: u64) -> AnalysisExecutionResult<u64> {
    a.checked_add(b)
        .ok_or_else(|| invalid("steady fluid preparation byte estimate overflow"))
}
fn require_bytes(required: u64, maximum: u64) -> AnalysisExecutionResult<()> {
    if required > maximum || required > isize::MAX as u64 {
        return Err(invalid(format!(
            "steady fluid preparation requested heap {required} exceeds ceiling {maximum}"
        )));
    }
    Ok(())
}
