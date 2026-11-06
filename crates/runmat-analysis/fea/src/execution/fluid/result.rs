use super::{
    invalid, workload::PreparedFluidSource, AnalysisExecutionResult, FluidSteadyWorkload, LIMITS,
    MAX_PAYLOAD_BYTES, RESULT_PREFIX,
};
use crate::{
    execution::{
        AnalysisExecutionError, AnalysisWorkloadIdentity, NumericalPlanIdentity,
        SolverResultIdentity, StepIdentity,
    },
    physics::cfd::canonical::{
        CanonicalFluidReservation, CanonicalFluidResidual, CanonicalFluidResult,
    },
    properties::CompiledPropertyMapIdentity,
};
use runmat_canonical_codec::{
    canonical_payload_materialization_resident_bytes_estimate, decode_contract, encode_contract,
};
use runmat_execution::Digest;
use runmat_meshing_core::{SolverMeshArtifact, StableDigest};
use serde::{Deserialize, Serialize};

/// Immutable original fields. Wire decoding alone never grants source authority.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct FluidSteadySolverResult {
    /// Derived from `data` the first time it is asked for; `data` never changes afterwards.
    #[serde(skip)]
    identity: DerivedIdentity,
    data: FluidResultData,
}

/// The result identity (a digest of its whole canonical encoding), derived at most once per
/// value. It is a function of the immutable data, so it takes no part in equality.
#[derive(Clone, Debug, Default)]
struct DerivedIdentity(std::sync::OnceLock<SolverResultIdentity>);

impl PartialEq for DerivedIdentity {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}

/// A fluid result validated once against its workload's source prepared once on its mesh.
/// Only [`FluidSteadySolverResult::on_source`] and
/// [`FluidSteadySolverResult::decode_on_source`] construct it, so its field, topology and
/// request methods neither prepare the source nor validate the result again.
pub struct FluidResultOnSource<'a> {
    pub(super) result: std::borrow::Cow<'a, FluidSteadySolverResult>,
    pub(super) workload: std::borrow::Cow<'a, FluidSteadyWorkload>,
    pub(super) source: std::sync::Arc<PreparedFluidSource>,
}

impl FluidResultOnSource<'_> {
    pub fn result(&self) -> &FluidSteadySolverResult {
        &self.result
    }

    pub fn workload(&self) -> &FluidSteadyWorkload {
        &self.workload
    }

    pub fn into_result(self) -> FluidSteadySolverResult {
        self.result.into_owned()
    }
}

/// Original retained fields of the authored procedure: one steady state, or the realized
/// initial state with every backward-Euler interval of the authored time schedule.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FluidRetainedOriginal {
    Steady(CanonicalFluidResult),
    Transient(super::FluidTransientSolverResult),
    /// Converged steady flow on the deformed configuration with the structure's state.
    FluidStructure(crate::physics::fluid_structure::FluidStructureResult),
    /// Final instant of a transient fluid–structure march, with the state departing its last
    /// interval and every accepted interval's coupling and energy evidence.
    FluidStructureTransient(crate::physics::fluid_structure::FluidStructureTransientResult),
    /// Steady flow advecting a transient conjugate temperature march.
    ConjugateHeatTransient(crate::physics::conjugate_heat::ConjugateHeatTransientResult),
    /// Steady flow and the conjugate temperatures it advects.
    ConjugateHeat {
        flow: CanonicalFluidResult,
        energy: crate::physics::conjugate_heat::ConjugateHeatResult,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct FluidResultData<T = FluidRetainedOriginal> {
    schema_version: u16,
    source_workload: AnalysisWorkloadIdentity,
    producer: StepIdentity,
    mesh_digest: StableDigest,
    projection_digest: Digest,
    geometry_digest: Digest,
    properties: CompiledPropertyMapIdentity,
    property_audit_digest: Digest,
    /// Original rederived local numerical reservation, NOT a scheduler FeaResourcePlan.
    reservation: CanonicalFluidReservation,
    /// Source preparation estimate, independently rederived before owned expansion.
    preparation_resident_bytes: u64,
    numerical_plan: NumericalPlanIdentity,
    original: T,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct FluidResultWire<T = FluidRetainedOriginal> {
    pub(super) data: FluidResultData<T>,
}

impl FluidSteadySolverResult {
    pub(super) fn from_original(
        workload: &FluidSteadyWorkload,
        source: &PreparedFluidSource,
        original: CanonicalFluidResult,
    ) -> AnalysisExecutionResult<Self> {
        let problem = source.problem(workload)?;
        problem.recover(&original).map_err(invalid)?;
        let reservation = problem.reservation();
        Ok(Self {
            identity: DerivedIdentity::default(),
            data: source_data(
                workload,
                source,
                reservation,
                original.numerical_input_identity,
                FluidRetainedOriginal::Steady(original),
            )?,
        })
    }

    pub(super) fn from_transient(
        workload: &FluidSteadyWorkload,
        source: &PreparedFluidSource,
        original: super::FluidTransientSolverResult,
    ) -> AnalysisExecutionResult<Self> {
        let terminal = original
            .terminal()
            .ok_or_else(|| invalid("transient fluid retained no interval"))?
            .numerical_input_identity;
        let reservation = source.problem(workload)?.reservation();
        Ok(Self {
            identity: DerivedIdentity::default(),
            data: source_data(
                workload,
                source,
                reservation,
                terminal,
                FluidRetainedOriginal::Transient(original),
            )?,
        })
    }

    pub(super) fn from_conjugate_heat(
        workload: &FluidSteadyWorkload,
        source: &PreparedFluidSource,
        flow: CanonicalFluidResult,
        energy: crate::physics::conjugate_heat::ConjugateHeatResult,
    ) -> AnalysisExecutionResult<Self> {
        let problem = source.problem(workload)?;
        problem.recover(&flow).map_err(invalid)?;
        source
            .conjugate_heat(workload, flow.face_volume_flux_m3_per_s.canonical_values())?
            .recover(&energy)
            .map_err(invalid)?;
        let reservation = problem.reservation();
        Ok(Self {
            identity: DerivedIdentity::default(),
            data: source_data(
                workload,
                source,
                reservation,
                flow.numerical_input_identity,
                FluidRetainedOriginal::ConjugateHeat { flow, energy },
            )?,
        })
    }

    pub(super) fn from_fluid_structure(
        workload: &FluidSteadyWorkload,
        source: &PreparedFluidSource,
        mesh: &SolverMeshArtifact,
        coupled: crate::physics::fluid_structure::FluidStructureResult,
    ) -> AnalysisExecutionResult<Self> {
        source
            .fluid_structure(workload, mesh)?
            .recover(&coupled)
            .map_err(invalid)?;
        let reservation = source.problem(workload)?.reservation();
        let identity = coupled.flow.numerical_input_identity;
        Ok(Self {
            identity: DerivedIdentity::default(),
            data: source_data(
                workload,
                source,
                reservation,
                identity,
                FluidRetainedOriginal::FluidStructure(coupled),
            )?,
        })
    }

    pub(super) fn from_fluid_structure_transient(
        workload: &FluidSteadyWorkload,
        source: &PreparedFluidSource,
        mesh: &SolverMeshArtifact,
        coupled: crate::physics::fluid_structure::FluidStructureTransientResult,
    ) -> AnalysisExecutionResult<Self> {
        source
            .fluid_structure(workload, mesh)?
            .recover_transient(&coupled)
            .map_err(invalid)?;
        let reservation = source.problem(workload)?.reservation();
        let identity = coupled.flow.numerical_input_identity;
        Ok(Self {
            identity: DerivedIdentity::default(),
            data: source_data(
                workload,
                source,
                reservation,
                identity,
                FluidRetainedOriginal::FluidStructureTransient(coupled),
            )?,
        })
    }

    /// Node displacement of a fluid–structure result: the structure and its fluid mesh motion.
    pub fn coupled_displacement(&self) -> Option<&[[f64; 3]]> {
        match &self.data.original {
            FluidRetainedOriginal::FluidStructure(coupled) => Some(&coupled.displacement_m),
            FluidRetainedOriginal::FluidStructureTransient(coupled) => {
                Some(&coupled.displacement_m)
            }
            _ => None,
        }
    }

    /// The fluid–structure state retained with the flow, when the study couples a structure.
    pub fn fluid_structure(
        &self,
    ) -> Option<&crate::physics::fluid_structure::FluidStructureResult> {
        match &self.data.original {
            FluidRetainedOriginal::FluidStructure(coupled) => Some(coupled),
            _ => None,
        }
    }

    pub(super) fn from_conjugate_heat_transient(
        workload: &FluidSteadyWorkload,
        source: &PreparedFluidSource,
        coupled: crate::physics::conjugate_heat::ConjugateHeatTransientResult,
    ) -> AnalysisExecutionResult<Self> {
        let problem = source.problem(workload)?;
        Self::recover_conjugate_transient(workload, source, &problem, &coupled)?;
        let reservation = problem.reservation();
        let identity = coupled.flow.numerical_input_identity;
        Ok(Self {
            identity: DerivedIdentity::default(),
            data: source_data(
                workload,
                source,
                reservation,
                identity,
                FluidRetainedOriginal::ConjugateHeatTransient(coupled),
            )?,
        })
    }

    fn recover_conjugate_transient(
        workload: &FluidSteadyWorkload,
        source: &PreparedFluidSource,
        problem: &crate::physics::cfd::canonical::CanonicalFluidProblem<'_>,
        coupled: &crate::physics::conjugate_heat::ConjugateHeatTransientResult,
    ) -> AnalysisExecutionResult<()> {
        problem.recover(&coupled.flow).map_err(invalid)?;
        let arrival = coupled
            .steps
            .last()
            .ok_or_else(|| invalid("transient conjugate study retained no interval"))?
            .time_s;
        source
            .conjugate_heat_at(
                workload,
                coupled.flow.face_volume_flux_m3_per_s.canonical_values(),
                arrival,
            )?
            .recover_step(
                &coupled.departure,
                &coupled.energy,
                arrival - coupled.departure_time_s,
            )
            .map_err(invalid)
    }

    /// Conjugate temperatures retained with the flow, when the study couples heat transfer:
    /// the steady state or the final instant of a transient march.
    pub fn conjugate_heat(&self) -> Option<&crate::physics::conjugate_heat::ConjugateHeatResult> {
        match &self.data.original {
            FluidRetainedOriginal::ConjugateHeat { energy, .. } => Some(energy),
            FluidRetainedOriginal::ConjugateHeatTransient(coupled) => Some(&coupled.energy),
            _ => None,
        }
    }

    /// Terminal original fields: the steady state, or the last backward-Euler interval.
    pub fn original(&self) -> &CanonicalFluidResult {
        match &self.data.original {
            FluidRetainedOriginal::Steady(original) => original,
            FluidRetainedOriginal::ConjugateHeat { flow, .. } => flow,
            FluidRetainedOriginal::ConjugateHeatTransient(coupled) => &coupled.flow,
            FluidRetainedOriginal::FluidStructure(coupled) => &coupled.flow,
            FluidRetainedOriginal::FluidStructureTransient(coupled) => &coupled.flow,
            FluidRetainedOriginal::Transient(history) => history
                .terminal()
                .expect("admitted transient fluid history is nonempty"),
        }
    }
    pub fn retained(&self) -> &FluidRetainedOriginal {
        &self.data.original
    }
    /// Initial state followed by every original interval endpoint, with physical times.
    pub fn transient_history(&self) -> Option<&super::FluidTransientSolverResult> {
        match &self.data.original {
            FluidRetainedOriginal::Steady(_)
            | FluidRetainedOriginal::ConjugateHeat { .. }
            | FluidRetainedOriginal::FluidStructure(_)
            | FluidRetainedOriginal::FluidStructureTransient(_)
            | FluidRetainedOriginal::ConjugateHeatTransient(_) => None,
            FluidRetainedOriginal::Transient(history) => Some(history),
        }
    }
    pub fn source_workload(&self) -> AnalysisWorkloadIdentity {
        self.data.source_workload
    }
    pub fn producer_identity(&self) -> StepIdentity {
        self.data.producer
    }
    pub fn numerical_plan_identity(&self) -> NumericalPlanIdentity {
        self.data.numerical_plan
    }
    pub fn reservation(&self) -> CanonicalFluidReservation {
        self.data.reservation
    }
    pub fn preparation_resident_bytes(&self) -> u64 {
        self.data.preparation_resident_bytes
    }

    pub fn canonical_bytes(&self) -> AnalysisExecutionResult<Vec<u8>> {
        Ok(encode_contract(
            RESULT_PREFIX,
            "steady-fluid-retained-result",
            self,
            LIMITS,
        )?)
    }
    pub fn identity(&self) -> AnalysisExecutionResult<SolverResultIdentity> {
        if let Some(identity) = self.identity.0.get() {
            return Ok(*identity);
        }
        let identity = SolverResultIdentity::derive(
            "steady-fluid-retained-result",
            &Digest::sha256(self.canonical_bytes()?),
        )?;
        Ok(*self.identity.0.get_or_init(|| identity))
    }

    /// Verify expected transport digest before decode, then original source and equations.
    /// Retained U,p,q and face force evidence remain unchanged; no solver invocation.
    pub fn from_canonical_bytes_for(
        bytes: &[u8],
        expected: Digest,
        workload: &FluidSteadyWorkload,
        mesh: &SolverMeshArtifact,
    ) -> AnalysisExecutionResult<Self> {
        Self::decode_on_source(bytes, expected, workload, mesh)
            .map(FluidResultOnSource::into_result)
    }

    /// [`Self::from_canonical_bytes_for`], keeping the prepared source the decoded result was
    /// validated against for the caller's field, topology and request checks.
    pub fn decode_on_source<'a>(
        bytes: &[u8],
        expected: Digest,
        workload: &'a FluidSteadyWorkload,
        mesh: &SolverMeshArtifact,
    ) -> AnalysisExecutionResult<FluidResultOnSource<'a>> {
        Self::decode_with(bytes, expected, std::borrow::Cow::Borrowed(workload), mesh)
    }

    /// [`Self::decode_on_source`] that keeps the decoded workload with the result.
    pub fn decode_owned_on_source(
        bytes: &[u8],
        expected: Digest,
        workload: FluidSteadyWorkload,
        mesh: &SolverMeshArtifact,
    ) -> AnalysisExecutionResult<FluidResultOnSource<'static>> {
        Self::decode_with(bytes, expected, std::borrow::Cow::Owned(workload), mesh)
    }

    fn decode_with<'a>(
        bytes: &[u8],
        expected: Digest,
        workload: std::borrow::Cow<'a, FluidSteadyWorkload>,
        mesh: &SolverMeshArtifact,
    ) -> AnalysisExecutionResult<FluidResultOnSource<'a>> {
        if bytes.len() > MAX_PAYLOAD_BYTES {
            return Err(AnalysisExecutionError::Limit("steady fluid encoded result"));
        }
        if Digest::sha256(bytes) != expected {
            return Err(AnalysisExecutionError::Identity(
                "steady fluid result digest mismatch",
            ));
        }
        // Admit the exact source before decoding retained vector payloads.
        let source = workload.prepare(mesh)?;
        let problem = source.problem(&workload)?;
        let decode_bytes =
            canonical_payload_materialization_resident_bytes_estimate(bytes).map_err(invalid)?;
        let coexisting = problem
            .reservation()
            .resident_bytes_upper_bound
            .checked_add(decode_bytes)
            .ok_or_else(|| invalid("steady fluid cold materialization byte estimate overflow"))
            .and_then(|later| source.resident_with(later))?;
        if coexisting > workload.resident_ceiling()? || coexisting > isize::MAX as u64 {
            return Err(AnalysisExecutionError::Limit(
                "steady fluid result residency",
            ));
        }
        let wire: FluidResultWire =
            decode_contract(RESULT_PREFIX, "steady-fluid-retained-result", bytes, LIMITS)?;
        let value = Self {
            identity: DerivedIdentity::default(),
            data: wire.data,
        };
        // Validation below prepares its own original problem. Do not retain a
        // second pressure-stencil/authority cache across that preparation.
        drop(problem);
        value.validate_prepared(&workload, &source, mesh)?;
        Ok(FluidResultOnSource {
            result: std::borrow::Cow::Owned(value),
            workload,
            source,
        })
    }

    pub fn validate_for(
        &self,
        workload: &FluidSteadyWorkload,
        mesh: &SolverMeshArtifact,
    ) -> AnalysisExecutionResult<()> {
        self.on_source(workload, mesh).map(drop)
    }

    /// Validate this result once against its workload's source on `mesh`, prepared once, and
    /// keep that source for the caller's further field, topology and request checks.
    pub fn on_source<'a>(
        &'a self,
        workload: &'a FluidSteadyWorkload,
        mesh: &'a SolverMeshArtifact,
    ) -> AnalysisExecutionResult<FluidResultOnSource<'a>> {
        let source = workload.prepare(mesh)?;
        self.validate_prepared(workload, &source, mesh)?;
        Ok(FluidResultOnSource {
            result: std::borrow::Cow::Borrowed(self),
            workload: std::borrow::Cow::Borrowed(workload),
            source,
        })
    }
    pub fn recover_original(
        &self,
        workload: &FluidSteadyWorkload,
        mesh: &SolverMeshArtifact,
    ) -> AnalysisExecutionResult<CanonicalFluidResidual> {
        let source = workload.prepare(mesh)?;
        self.validate_prepared(workload, &source, mesh)?;
        // Return retained verified evidence, not a recomputed replacement field.
        Ok(self.original().residual.clone())
    }
    pub(super) fn validate_prepared(
        &self,
        workload: &FluidSteadyWorkload,
        source: &PreparedFluidSource,
        mesh: &SolverMeshArtifact,
    ) -> AnalysisExecutionResult<()> {
        let d = &self.data;
        let problem = source.problem(workload)?;
        let source_workload = workload.identity()?;
        if d.schema_version != 5
            || d.source_workload != source_workload
            || d.producer != workload.producer_identity()?
            || d.mesh_digest != source.source.mesh_digest
            || d.projection_digest != source.source.projection_digest
            || d.geometry_digest != source.topology.geometry_digest
            || d.properties != source.property_identity
            || d.property_audit_digest != source.property_audit_digest
            || d.reservation != problem.reservation()
            || d.preparation_resident_bytes != source.preparation_resident_bytes
            || d.numerical_plan
                != plan_identity(
                    source_workload,
                    source,
                    &d.reservation,
                    self.original().numerical_input_identity,
                )?
        {
            return Err(AnalysisExecutionError::Identity(
                "steady fluid source/material/resource identity mismatch",
            ));
        }
        let coupling = workload.coupling();
        let conjugate = coupling == super::workload::FluidCoupling::ConjugateHeat;
        let structural = coupling == super::workload::FluidCoupling::FluidStructure;
        match (&d.original, workload.is_transient()) {
            (FluidRetainedOriginal::FluidStructureTransient(coupled), true) if structural => {
                drop(problem);
                source
                    .fluid_structure(workload, mesh)?
                    .recover_transient(coupled)
                    .map_err(invalid)?;
            }
            (FluidRetainedOriginal::FluidStructure(coupled), false) if structural => {
                drop(problem);
                source
                    .fluid_structure(workload, mesh)?
                    .recover(coupled)
                    .map_err(invalid)?;
            }
            (FluidRetainedOriginal::Steady(original), false) if !conjugate && !structural => {
                problem.recover(original).map_err(invalid)?;
            }
            (FluidRetainedOriginal::ConjugateHeatTransient(coupled), true) if conjugate => {
                Self::recover_conjugate_transient(workload, source, &problem, coupled)?;
            }
            (FluidRetainedOriginal::ConjugateHeat { flow, energy }, false) if conjugate => {
                problem.recover(flow).map_err(invalid)?;
                source
                    .conjugate_heat(workload, flow.face_volume_flux_m3_per_s.canonical_values())?
                    .recover(energy)
                    .map_err(invalid)?;
            }
            (FluidRetainedOriginal::Transient(history), true) if !conjugate && !structural => {
                if history.terminal().is_none() {
                    return Err(invalid("transient fluid retained no interval"));
                }
                // Release the start-time problem before the kernel prepares its own chain.
                drop(problem);
                history.recover(&workload.transient_view(), mesh)?;
            }
            _ => {
                return Err(AnalysisExecutionError::Identity(
                    "retained fluid procedure differs from the authored step",
                ))
            }
        }
        Ok(())
    }
}

/// One metadata authority for actual retained results and streaming resource profiles.
pub(super) fn source_data<T>(
    workload: &FluidSteadyWorkload,
    source: &PreparedFluidSource,
    reservation: CanonicalFluidReservation,
    numerical_input: Digest,
    original: T,
) -> AnalysisExecutionResult<FluidResultData<T>> {
    let source_workload = workload.identity()?;
    let numerical_plan = plan_identity(source_workload, source, &reservation, numerical_input)?;
    Ok(FluidResultData {
        schema_version: 5,
        source_workload,
        producer: workload.producer_identity()?,
        mesh_digest: source.source.mesh_digest,
        projection_digest: source.source.projection_digest,
        geometry_digest: source.topology.geometry_digest,
        properties: source.property_identity.clone(),
        property_audit_digest: source.property_audit_digest,
        reservation,
        preparation_resident_bytes: source.preparation_resident_bytes,
        numerical_plan,
        original,
    })
}

fn plan_identity(
    workload: AnalysisWorkloadIdentity,
    source: &PreparedFluidSource,
    reservation: &CanonicalFluidReservation,
    numerical_input: Digest,
) -> AnalysisExecutionResult<NumericalPlanIdentity> {
    NumericalPlanIdentity::derive(
        "steady-fluid-local-numerical-reservation",
        &(
            workload,
            source.source.mesh_digest,
            source.topology.geometry_digest,
            &source.property_identity,
            source.property_audit_digest,
            source.preparation_resident_bytes,
            reservation,
            numerical_input,
        ),
    )
}
