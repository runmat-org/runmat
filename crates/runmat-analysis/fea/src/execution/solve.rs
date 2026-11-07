use runmat_analysis_core::{
    validate_model_sequence, AnalysisFieldValues, AnalysisModel, AnalysisProcedure,
    AnalysisSequence, CoupledPhysicsFamily, PhysicsFamily, ProcedureStep, ProcedureStepId,
    RestartPolicy, SolutionSpace,
};
use runmat_canonical_codec::{decode_contract, encode_contract, CanonicalLimits};
use serde::{Deserialize, Serialize};

use crate::contracts::{ComputeBackend, FeaRunResult};
use crate::results::observation::ObservationExecutionPolicy;
use crate::solve::backend::kind::LinearAlgebraBackendKind;
use crate::solve::preconditioner::SpdPreconditionerKind;

use super::{AnalysisExecutionError, AnalysisExecutionResult};
mod result_identity;
mod state_input;
use super::{
    AnalysisWorkloadIdentity, CanonicalStateTransition, NumericalReceiptIdentity,
    SolverResultIdentity,
};
use result_identity::ResultIdentitySource;
pub use state_input::StructuralLinearStateInput;

const SOLVE_PREFIX: &[u8] = b"runmat-analysis-solve/v1\0";
const SOLVE_LIMITS: CanonicalLimits = crate::contracts::canonical_digest::wire_bounded(CanonicalLimits::new(0, 0, 65_536, 64));

fn content_limits(encoded_bytes: usize) -> CanonicalLimits {
    CanonicalLimits::new(encoded_bytes, encoded_bytes, 65_536, 64)
}

/// Solver-owned, path-free input for one structural linear workload. The mesh is a separate
/// numerical object so it can be cached, transferred, and shared independently of authored state.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuralLinearWorkload {
    pub schema_version: u16,
    pub model: AnalysisModel,
    pub step_id: ProcedureStepId,
    pub backend: ComputeBackend,
    pub preconditioner: SpdPreconditionerKind,
    pub algebra_backend: LinearAlgebraBackendKind,
    pub observation_execution: ObservationExecutionPolicy,
    pub require_solver_mesh_for_solid: bool,
    pub state_input: Option<StructuralLinearStateInput>,
}

impl StructuralLinearWorkload {
    pub fn validate(&self) -> AnalysisExecutionResult<()> {
        let step = self.validate_common()?;
        self.validate_state_input(step)
    }

    /// Validates one authored sequence workload before its upstream state is bound.
    ///
    /// The coordinator accepts only the currently executable transition envelope: a fresh root
    /// step, one structural initial guess, or one solved thermal prestress source. Executable workload
    /// validation remains stricter and requires the exact canonical transition receipt.
    pub fn validate_sequence_template(&self) -> AnalysisExecutionResult<()> {
        let step = self.validate_common()?;
        crate::planning::resource::require_executable_static_backend(self.backend)
            .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
        if self.state_input.is_some() {
            return Err(AnalysisExecutionError::Invalid(
                "structural sequence template cannot contain a prebound transition".into(),
            ));
        }
        self.validate_authored_state_input(step).map(|_| ())
    }

    fn validate_common(&self) -> AnalysisExecutionResult<&ProcedureStep> {
        if self.schema_version != 2 {
            return Err(AnalysisExecutionError::Invalid(
                "unsupported structural linear workload schema".into(),
            ));
        }
        validate_model_sequence(&self.model)
            .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
        let step = self.procedure_step()?;
        if step.procedure != AnalysisProcedure::Static
            || step.solution_space != SolutionSpace::Direct
            || !matches!(
                step.family,
                PhysicsFamily::Structural
                    | PhysicsFamily::Coupled(CoupledPhysicsFamily::Thermomechanical)
            )
        {
            return Err(AnalysisExecutionError::Invalid(
                "structural linear workload requires a structural or thermomechanical static direct procedure".into(),
            ));
        }
        Ok(step)
    }

    pub fn procedure_step(&self) -> AnalysisExecutionResult<&ProcedureStep> {
        let mut matching = self
            .model
            .sequence
            .steps
            .iter()
            .filter(|step| step.step_id == self.step_id);
        let step = matching.next().ok_or_else(|| {
            AnalysisExecutionError::Invalid(format!(
                "structural linear workload references absent step `{}`",
                self.step_id.as_str()
            ))
        })?;
        if matching.next().is_some() {
            return Err(AnalysisExecutionError::Invalid(format!(
                "structural linear workload references duplicate step `{}`",
                self.step_id.as_str()
            )));
        }
        Ok(step)
    }

    /// Returns the single-procedure numerical projection consumed by the structural solver.
    /// The complete authored sequence remains in the portable workload and therefore in identity.
    pub fn solver_model(&self) -> AnalysisExecutionResult<AnalysisModel> {
        self.validate()?;
        if matches!(
            self.state_input,
            Some(StructuralLinearStateInput::ThermalPrestress(_))
        ) {
            return Err(AnalysisExecutionError::Invalid(
                "thermal prestress requires the full authored model and selected prepared thermal solve".into(),
            ));
        }
        let mut step = self.procedure_step()?.clone();
        step.state_inputs.clear();
        // Observations are part of the exact portable workload and are evaluated from this solve's
        // admitted field state. Keeping them on the single-step projection prevents publication
        // from silently losing authored outputs at a placement boundary.
        step.restart = RestartPolicy::Fresh;
        let mut model = self.model.clone();
        model.sequence = AnalysisSequence::from_steps(
            format!("{}.solver-projection", self.model.sequence.sequence_id),
            vec![step],
        );
        Ok(model)
    }

    pub fn canonical_bytes(&self) -> AnalysisExecutionResult<Vec<u8>> {
        self.validate()?;
        Ok(encode_contract(
            SOLVE_PREFIX,
            "structural-linear-workload",
            self,
            SOLVE_LIMITS,
        )?)
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> AnalysisExecutionResult<Self> {
        let workload: Self = decode_contract(
            SOLVE_PREFIX,
            "structural-linear-workload",
            bytes,
            SOLVE_LIMITS,
        )?;
        workload.validate()?;
        Ok(workload)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuralLinearWorkloadResult {
    pub schema_version: u16,
    pub run: FeaRunResult,
    pub constraint_result: crate::constraints::CanonicalStructuralConstraintResult,
    pub linearization:
        Option<crate::pipeline::structural_linearization::StructuralLinearizationArtifact>,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuralLinearNumericalReceipt {
    pub relative_equilibrium_residual: f64,
    pub equation_scale: f64,
}

/// Compact, source-bound numerical evidence published alongside a potentially large solver result.
///
/// Accuracy evaluation consumes this object without reading the solver field payload. The solver
/// result importer still validates it against the complete result before either object is trusted.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(
    deny_unknown_fields,
    try_from = "StructuralLinearNumericalReceiptArtifactWire"
)]
pub struct StructuralLinearNumericalReceiptArtifact {
    pub schema_version: u16,
    identity: NumericalReceiptIdentity,
    pub source_workload: AnalysisWorkloadIdentity,
    pub source_payload_digest: runmat_execution::Digest,
    pub source_result: SolverResultIdentity,
    source_identity: ResultIdentitySource,
    receipt: StructuralLinearNumericalReceipt,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct StructuralLinearNumericalReceiptArtifactWire {
    schema_version: u16,
    identity: NumericalReceiptIdentity,
    source_workload: AnalysisWorkloadIdentity,
    source_payload_digest: runmat_execution::Digest,
    source_result: SolverResultIdentity,
    source_identity: ResultIdentitySource,
    receipt: StructuralLinearNumericalReceipt,
}

impl TryFrom<StructuralLinearNumericalReceiptArtifactWire>
    for StructuralLinearNumericalReceiptArtifact
{
    type Error = AnalysisExecutionError;

    fn try_from(wire: StructuralLinearNumericalReceiptArtifactWire) -> Result<Self, Self::Error> {
        let artifact = Self {
            schema_version: wire.schema_version,
            identity: wire.identity,
            source_workload: wire.source_workload,
            source_payload_digest: wire.source_payload_digest,
            source_result: wire.source_result,
            source_identity: wire.source_identity,
            receipt: wire.receipt,
        };
        artifact.validate()?;
        Ok(artifact)
    }
}

impl StructuralLinearWorkloadResult {
    pub fn new(
        run: FeaRunResult,
        constraint_result: crate::constraints::CanonicalStructuralConstraintResult,
    ) -> AnalysisExecutionResult<Self> {
        let result = Self {
            schema_version: 1,
            run,
            constraint_result,
            linearization: None,
        };
        result.validate()?;
        Ok(result)
    }

    pub fn with_linearization(
        run: FeaRunResult,
        constraint_result: crate::constraints::CanonicalStructuralConstraintResult,
        linearization: crate::pipeline::structural_linearization::StructuralLinearizationArtifact,
    ) -> AnalysisExecutionResult<Self> {
        let result = Self {
            schema_version: 1,
            run,
            constraint_result,
            linearization: Some(linearization),
        };
        result.validate()?;
        Ok(result)
    }

    pub fn validate(&self) -> AnalysisExecutionResult<()> {
        if self.schema_version != 1 || self.run.fields_are_empty() {
            return Err(AnalysisExecutionError::Invalid(
                "structural linear result schema or field set is invalid".into(),
            ));
        }
        self.run
            .validate_solver_formulation()
            .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
        for evidence in &self.run.spatial_projection_evidence {
            evidence
                .validate()
                .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
        }
        self.constraint_result
            .validate()
            .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
        if self.run.identity.realization().coupling
            == runmat_analysis_core::CouplingImplementation::OneWaySolvedThermoelasticState
            && self
                .linearization
                .as_ref()
                .and_then(|value| value.thermal_source())
                .is_none()
        {
            return Err(AnalysisExecutionError::Identity(
                "solved thermal result requires retained physical source",
            ));
        }
        if self.run.structural_constraint_result.as_deref() != Some(&self.constraint_result) {
            return Err(AnalysisExecutionError::Identity(
                "portable structural constraint result differs from its canonical run payload",
            ));
        }
        if let Some(linearization) = &self.linearization {
            linearization
                .validate()
                .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
            if linearization.source_run() != &self.run.identity
                || linearization.topology_digest()
                    != self.constraint_result.topology.discretization_digest()
            {
                return Err(AnalysisExecutionError::Identity(
                    "structural linearization does not belong to its solver result",
                ));
            }
            if let Some(source) = linearization.thermal_source() {
                source
                    .validate_result(&self.run)
                    .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
            }
        }
        StructuralLinearNumericalReceipt::from_run(&self.run)?;
        let mut field_ids = std::collections::BTreeSet::new();
        for field in &self.run.fields {
            if field.field_id.trim().is_empty() || !field_ids.insert(field.field_id.as_str()) {
                return Err(AnalysisExecutionError::Invalid(
                    "portable structural result field identities must be non-empty and unique"
                        .into(),
                ));
            }
            let AnalysisFieldValues::HostF64(values) = &field.values else {
                return Err(AnalysisExecutionError::Invalid(
                    "portable structural results must contain host-backed fields".into(),
                ));
            };
            let Some(expected_values) = field
                .shape
                .iter()
                .try_fold(1usize, |count, dimension| count.checked_mul(*dimension))
            else {
                return Err(AnalysisExecutionError::Invalid(format!(
                    "portable structural field `{}` shape overflows the host address space",
                    field.field_id
                )));
            };
            if values.len() != expected_values || values.iter().any(|value| !value.is_finite()) {
                return Err(AnalysisExecutionError::Invalid(format!(
                    "portable structural field `{}` has invalid shape or non-finite values",
                    field.field_id
                )));
            }
        }
        Ok(())
    }

    /// Result fields grow with the mesh, so the codec bounds follow the content: encoding is
    /// bounded by the profiled size of this owned result, and decoding by the received bytes
    /// (every collection item occupies at least one encoded byte).
    pub fn canonical_bytes(&self) -> AnalysisExecutionResult<Vec<u8>> {
        self.validate()?;
        let profile =
            runmat_canonical_codec::profile_canonical_encoding(self, content_limits(usize::MAX))?;
        let bytes = usize::try_from(profile.encoded_bytes_upper_bound).map_err(|_| {
            AnalysisExecutionError::Invalid("structural result exceeds the address space".into())
        })?;
        Ok(encode_contract(
            SOLVE_PREFIX,
            "structural-linear-result",
            self,
            content_limits(bytes),
        )?)
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> AnalysisExecutionResult<Self> {
        let result: Self = decode_contract(
            SOLVE_PREFIX,
            "structural-linear-result",
            bytes,
            content_limits(bytes.len()),
        )?;
        result.validate()?;
        Ok(result)
    }
}

impl StructuralLinearNumericalReceiptArtifact {
    pub fn new(
        source_workload: AnalysisWorkloadIdentity,
        result: &StructuralLinearWorkloadResult,
    ) -> AnalysisExecutionResult<Self> {
        result.validate()?;
        let source_payload_digest = runmat_execution::Digest::sha256(result.canonical_bytes()?);
        let source_identity = ResultIdentitySource::from_result(result);
        let source_result = source_identity.identity(source_workload, source_payload_digest)?;
        let receipt = StructuralLinearNumericalReceipt::from_run(&result.run)?;
        let mut artifact = Self {
            schema_version: 2,
            identity: NumericalReceiptIdentity::new(source_result, source_payload_digest)?,
            source_workload,
            source_payload_digest,
            source_result,
            source_identity,
            receipt,
        };
        artifact.identity = artifact.expected_identity()?;
        artifact.validate()?;
        Ok(artifact)
    }

    pub const fn identity(&self) -> NumericalReceiptIdentity {
        self.identity
    }

    pub const fn receipt(&self) -> StructuralLinearNumericalReceipt {
        self.receipt
    }

    pub const fn source_result(&self) -> SolverResultIdentity {
        self.source_result
    }

    pub fn validate(&self) -> AnalysisExecutionResult<()> {
        self.receipt.validate()?;
        let expected_source = self
            .source_identity
            .identity(self.source_workload, self.source_payload_digest)?;
        let expected = self.expected_identity()?;
        if self.schema_version != 2
            || self.source_result != expected_source
            || self.identity != expected
        {
            return Err(AnalysisExecutionError::Identity(
                "structural numerical receipt artifact identity does not converge",
            ));
        }
        Ok(())
    }

    pub fn validate_against(
        &self,
        source_result: SolverResultIdentity,
        result: &StructuralLinearWorkloadResult,
    ) -> AnalysisExecutionResult<()> {
        self.validate()?;
        if self.source_result != source_result
            || self.source_identity != ResultIdentitySource::from_result(result)
            || self.source_payload_digest
                != runmat_execution::Digest::sha256(result.canonical_bytes()?)
            || self.receipt != StructuralLinearNumericalReceipt::from_run(&result.run)?
        {
            return Err(AnalysisExecutionError::Identity(
                "structural numerical receipt differs from its solver result",
            ));
        }
        Ok(())
    }

    pub fn canonical_bytes(&self) -> AnalysisExecutionResult<Vec<u8>> {
        self.validate()?;
        Ok(encode_contract(
            SOLVE_PREFIX,
            "structural-linear-numerical-receipt",
            self,
            SOLVE_LIMITS,
        )?)
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> AnalysisExecutionResult<Self> {
        let artifact = decode_contract(
            SOLVE_PREFIX,
            "structural-linear-numerical-receipt",
            bytes,
            SOLVE_LIMITS,
        )?;
        Ok(artifact)
    }
}

impl StructuralLinearNumericalReceipt {
    fn from_run(run: &FeaRunResult) -> AnalysisExecutionResult<Self> {
        let relative_equilibrium_residual =
            scalar_field(run, crate::contracts::FEA_FIELD_STRUCTURAL_RESIDUAL_NORM)?;
        let equation_scale =
            scalar_field(run, crate::contracts::FEA_FIELD_STRUCTURAL_EQUATION_SCALE)?;
        let receipt = Self {
            relative_equilibrium_residual,
            equation_scale,
        };
        receipt.validate()?;
        Ok(receipt)
    }

    fn validate(self) -> AnalysisExecutionResult<()> {
        if !self.relative_equilibrium_residual.is_finite()
            || self.relative_equilibrium_residual < 0.0
            || !self.equation_scale.is_finite()
            || self.equation_scale <= 0.0
        {
            return Err(AnalysisExecutionError::Invalid(
                "structural numerical receipt requires finite nonnegative residual and positive scale"
                    .into(),
            ));
        }
        Ok(())
    }
}

fn scalar_field(run: &FeaRunResult, field_id: &str) -> AnalysisExecutionResult<f64> {
    let field = run.field(field_id).ok_or_else(|| {
        AnalysisExecutionError::Invalid(format!(
            "structural result is missing numerical field `{field_id}`"
        ))
    })?;
    let values = field.as_host_f64().ok_or_else(|| {
        AnalysisExecutionError::Invalid(format!(
            "structural numerical field `{field_id}` is not host-readable"
        ))
    })?;
    let [value] = values else {
        return Err(AnalysisExecutionError::Invalid(format!(
            "structural numerical field `{field_id}` is not scalar"
        )));
    };
    Ok(*value)
}
