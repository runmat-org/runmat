use runmat_analysis_core::{
    validate_model_sequence, AnalysisFieldValues, AnalysisModel, AnalysisProcedure,
    AnalysisSequence, CoupledPhysicsFamily, FormulationRequirement, PhysicsFamily, ProcedureStep,
    ProcedureStepId, RestartPolicy, SolutionSpace, StateSource, StateUse,
};
use runmat_canonical_codec::{decode_contract, encode_contract, CanonicalLimits};
use serde::{Deserialize, Serialize};

use crate::contracts::{ComputeBackend, FeaRunResult};
use crate::results::observation::ObservationExecutionPolicy;
use crate::solve::backend::kind::LinearAlgebraBackendKind;
use crate::solve::preconditioner::SpdPreconditionerKind;

use super::{AnalysisExecutionError, AnalysisExecutionResult};
use super::{
    AnalysisWorkloadIdentity, CanonicalStateTransition, NumericalReceiptIdentity,
    SolverResultIdentity,
};

const SOLVE_PREFIX: &[u8] = b"runmat-analysis-solve/v1\0";
const SOLVE_LIMITS: CanonicalLimits = CanonicalLimits::new(64 * 1024 * 1024, 1_000_000, 65_536, 64);

/// Solver-owned, path-free input for one structural linear workload. The mesh is a separate
/// numerical object so it can be cached, transferred, and shared independently of authored state.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuralLinearWorkload {
    pub schema_version: u16,
    pub model: AnalysisModel,
    pub step_id: ProcedureStepId,
    pub backend: ComputeBackend,
    pub formulation_requirement: FormulationRequirement,
    pub preconditioner: SpdPreconditionerKind,
    pub algebra_backend: LinearAlgebraBackendKind,
    pub observation_execution: ObservationExecutionPolicy,
    pub require_solver_mesh_for_solid: bool,
    pub initial_guess_transition: Option<CanonicalStateTransition>,
}

impl StructuralLinearWorkload {
    pub fn validate(&self) -> AnalysisExecutionResult<()> {
        let step = self.validate_common()?;
        self.validate_state_input(step)
    }

    /// Validates one authored sequence workload before its upstream state is bound.
    ///
    /// The coordinator accepts only the currently executable transition envelope: a fresh root
    /// step or one upstream structural-primary state used as an initial guess. Executable workload
    /// validation remains stricter and requires the exact canonical transition receipt.
    pub fn validate_sequence_template(&self) -> AnalysisExecutionResult<()> {
        let step = self.validate_common()?;
        if self.initial_guess_transition.is_some() {
            return Err(AnalysisExecutionError::Invalid(
                "structural sequence template cannot contain a prebound transition".into(),
            ));
        }
        if !step.activations.is_empty() {
            return Err(AnalysisExecutionError::Invalid(
                "structural sequence execution does not yet implement step object activation"
                    .into(),
            ));
        }
        if step.restart != RestartPolicy::Fresh {
            return Err(AnalysisExecutionError::Invalid(
                "structural sequence execution does not yet implement restart policies".into(),
            ));
        }
        match step.state_inputs.as_slice() {
            [] => Ok(()),
            [input]
                if input.use_as == StateUse::InitialGuess
                    && input.required_kind
                        == runmat_analysis_core::StateKind::StructuralPrimary
                    && matches!(input.source, StateSource::Upstream { .. }) =>
            {
                Ok(())
            }
            _ => Err(AnalysisExecutionError::Invalid(
                "structural sequence execution currently supports at most one upstream structural-primary initial guess"
                    .into(),
            )),
        }
    }

    fn validate_common(&self) -> AnalysisExecutionResult<&ProcedureStep> {
        if self.schema_version != 1 {
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
        let mut step = self.procedure_step()?.clone();
        step.state_inputs.clear();
        // Portable solver work produces numerical state only. Authored observations are retained
        // in the workload identity and evaluated by the subsequent observation task.
        step.observations.clear();
        step.restart = RestartPolicy::Fresh;
        let mut model = self.model.clone();
        model.sequence = AnalysisSequence::from_steps(
            format!("{}.solver-projection", self.model.sequence.sequence_id),
            vec![step],
        );
        Ok(model)
    }

    fn validate_state_input(&self, step: &ProcedureStep) -> AnalysisExecutionResult<()> {
        if !step.activations.is_empty() {
            return Err(AnalysisExecutionError::Invalid(
                "structural sequence execution does not yet implement step object activation"
                    .into(),
            ));
        }
        match &self.initial_guess_transition {
            Some(transition) => {
                transition.validate()?;
                if transition.method
                    != runmat_analysis_core::StateTransitionMethod::InitialGuessOnly
                    || transition.consumer_step_id != step.step_id
                {
                    return Err(AnalysisExecutionError::Invalid(
                        "structural linear initial guess targets a different procedure step".into(),
                    ));
                }
                let matching_inputs = step
                    .state_inputs
                    .iter()
                    .filter(|input| {
                        input.use_as == StateUse::InitialGuess
                            && input.required_kind
                                == runmat_analysis_core::StateKind::StructuralPrimary
                            && matches!(
                                &input.source,
                                StateSource::Upstream {
                                    producer_step,
                                    state_id,
                                } if producer_step == &transition.producer_step_id
                                    && state_id == &transition.state_id
                            )
                    })
                    .count();
                let authored_transition = self.model.sequence.transitions.iter().any(|authored| {
                    authored.transition_id == transition.transition_id
                        && authored.producer_step == transition.producer_step_id
                        && authored.consumer_step == transition.consumer_step_id
                        && authored.state_id == transition.state_id
                        && authored.method == transition.method
                        && authored.topology == transition.topology
                        && authored.reference == transition.reference
                });
                if matching_inputs != 1 || step.state_inputs.len() != 1 || !authored_transition {
                    return Err(AnalysisExecutionError::Invalid(
                        "initial-guess transition does not exactly bind the selected step input"
                            .into(),
                    ));
                }
            }
            None if !step.state_inputs.is_empty() => {
                return Err(AnalysisExecutionError::Invalid(
                    "structural linear step has state inputs but no executable transition".into(),
                ));
            }
            None => {}
        }
        Ok(())
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
        for evidence in &self.run.spatial_projection_evidence {
            evidence
                .validate()
                .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
        }
        self.constraint_result
            .validate()
            .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
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

    pub fn canonical_bytes(&self) -> AnalysisExecutionResult<Vec<u8>> {
        self.validate()?;
        Ok(encode_contract(
            SOLVE_PREFIX,
            "structural-linear-result",
            self,
            SOLVE_LIMITS,
        )?)
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> AnalysisExecutionResult<Self> {
        let result: Self = decode_contract(
            SOLVE_PREFIX,
            "structural-linear-result",
            bytes,
            SOLVE_LIMITS,
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
        let source_result = SolverResultIdentity::new(source_workload, source_payload_digest)?;
        let receipt = StructuralLinearNumericalReceipt::from_run(&result.run)?;
        let identity = NumericalReceiptIdentity::new(source_result, receipt.content_digest()?)?;
        let artifact = Self {
            schema_version: 1,
            identity,
            source_workload,
            source_payload_digest,
            source_result,
            receipt,
        };
        artifact.validate()?;
        Ok(artifact)
    }

    pub const fn identity(&self) -> NumericalReceiptIdentity {
        self.identity
    }

    pub const fn receipt(&self) -> StructuralLinearNumericalReceipt {
        self.receipt
    }

    pub fn validate(&self) -> AnalysisExecutionResult<()> {
        self.receipt.validate()?;
        let expected_source =
            SolverResultIdentity::new(self.source_workload, self.source_payload_digest)?;
        let expected =
            NumericalReceiptIdentity::new(self.source_result, self.receipt.content_digest()?)?;
        if self.schema_version != 1
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

    fn content_digest(self) -> AnalysisExecutionResult<runmat_execution::Digest> {
        self.validate()?;
        Ok(runmat_execution::Digest::sha256(encode_contract(
            SOLVE_PREFIX,
            "structural-linear-numerical-receipt-content",
            &self,
            SOLVE_LIMITS,
        )?))
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
