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
    data: FluidResultData,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct FluidResultData {
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
    original: CanonicalFluidResult,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct FluidResultWire {
    data: FluidResultData,
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
        let source_workload = workload.identity()?;
        let numerical_plan = plan_identity(
            source_workload,
            source,
            &reservation,
            original.numerical_input_identity,
        )?;
        Ok(Self {
            data: FluidResultData {
                schema_version: 1,
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
            },
        })
    }

    pub fn original(&self) -> &CanonicalFluidResult {
        &self.data.original
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
        SolverResultIdentity::derive(
            "steady-fluid-retained-result",
            &Digest::sha256(self.canonical_bytes()?),
        )
    }

    /// Verify expected transport digest before decode, then original source and equations.
    /// Retained U,p,q and face force evidence remain unchanged; no solver invocation.
    pub fn from_canonical_bytes_for(
        bytes: &[u8],
        expected: Digest,
        workload: &FluidSteadyWorkload,
        mesh: &SolverMeshArtifact,
    ) -> AnalysisExecutionResult<Self> {
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
        let problem = source.problem(workload)?;
        let decode_bytes =
            canonical_payload_materialization_resident_bytes_estimate(bytes).map_err(invalid)?;
        let coexisting = source
            .preparation_resident_bytes
            .checked_add(problem.reservation().resident_bytes_upper_bound)
            .and_then(|v| v.checked_add(decode_bytes))
            .ok_or_else(|| invalid("steady fluid cold materialization byte estimate overflow"))?;
        if coexisting > workload.policy.maximum_resident_bytes || coexisting > isize::MAX as u64 {
            return Err(AnalysisExecutionError::Limit(
                "steady fluid result residency",
            ));
        }
        let wire: FluidResultWire =
            decode_contract(RESULT_PREFIX, "steady-fluid-retained-result", bytes, LIMITS)?;
        let value = Self { data: wire.data };
        // Validation below prepares its own original problem. Do not retain a
        // second pressure-stencil/authority cache across that preparation.
        drop(problem);
        value.validate_prepared(workload, &source)?;
        Ok(value)
    }

    pub fn validate_for(
        &self,
        workload: &FluidSteadyWorkload,
        mesh: &SolverMeshArtifact,
    ) -> AnalysisExecutionResult<()> {
        self.validate_prepared(workload, &workload.prepare(mesh)?)
    }
    pub fn recover_original(
        &self,
        workload: &FluidSteadyWorkload,
        mesh: &SolverMeshArtifact,
    ) -> AnalysisExecutionResult<CanonicalFluidResidual> {
        let source = workload.prepare(mesh)?;
        self.validate_prepared(workload, &source)?;
        // Return retained verified evidence, not a recomputed replacement field.
        Ok(self.data.original.residual.clone())
    }
    fn validate_prepared(
        &self,
        workload: &FluidSteadyWorkload,
        source: &PreparedFluidSource,
    ) -> AnalysisExecutionResult<()> {
        let d = &self.data;
        let problem = source.problem(workload)?;
        let source_workload = workload.identity()?;
        if d.schema_version != 1
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
                    d.original.numerical_input_identity,
                )?
        {
            return Err(AnalysisExecutionError::Identity(
                "steady fluid source/material/resource identity mismatch",
            ));
        }
        problem.recover(&d.original).map_err(invalid)?;
        Ok(())
    }
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
