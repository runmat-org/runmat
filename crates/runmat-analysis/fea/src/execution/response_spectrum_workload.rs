use std::collections::BTreeSet;

use runmat_analysis_core::{
    validate_model_sequence, AnalysisModel, AnalysisProcedure, CoordinateFrame, IndependentDomain,
    PhysicsFamily, ProcedureStep, ProcedureStepId, ResponseSpectrum, SolutionSpace, StateKind,
    StateSource, StateUse,
};
use runmat_canonical_codec::{decode_contract, encode_contract, CanonicalLimits};
use serde::{Deserialize, Serialize};

use crate::{contracts::ComputeBackend, results::observation::ObservationExecutionPolicy};

use super::{
    AnalysisExecutionError, AnalysisExecutionResult, CanonicalStateTransition, SequenceIdentity,
    StepIdentity, StructuralModalResponseSpectrumWorkloadIdentity,
};

const WORKLOAD_PREFIX: &[u8] = b"runmat-structural-response-spectrum-workload/v1\0";
const WORKLOAD_LIMITS: CanonicalLimits =
    crate::contracts::canonical_digest::wire_bounded(CanonicalLimits::new(0, 0, 65_536, 64));

/// Placement-neutral modal response-spectrum intent bound to one immutable modal basis.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(
    deny_unknown_fields,
    try_from = "StructuralModalResponseSpectrumWorkloadWire"
)]
pub struct StructuralModalResponseSpectrumWorkload {
    schema_version: u16,
    identity: StructuralModalResponseSpectrumWorkloadIdentity,
    model: AnalysisModel,
    step_id: ProcedureStepId,
    backend: ComputeBackend,
    observation_execution: ObservationExecutionPolicy,
    modal_basis_transition: CanonicalStateTransition,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct StructuralModalResponseSpectrumWorkloadWire {
    schema_version: u16,
    identity: StructuralModalResponseSpectrumWorkloadIdentity,
    model: AnalysisModel,
    step_id: ProcedureStepId,
    backend: ComputeBackend,
    observation_execution: ObservationExecutionPolicy,
    modal_basis_transition: CanonicalStateTransition,
}

pub struct StructuralModalResponseSpectrumWorkloadInput {
    pub model: AnalysisModel,
    pub step_id: ProcedureStepId,
    pub backend: ComputeBackend,
    pub observation_execution: ObservationExecutionPolicy,
    pub modal_basis_transition: CanonicalStateTransition,
}

/// Authored response-spectrum intent before the producer's immutable modal state exists.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuralModalResponseSpectrumTemplate {
    pub schema_version: u16,
    pub model: AnalysisModel,
    pub step_id: ProcedureStepId,
    pub backend: ComputeBackend,
    pub observation_execution: ObservationExecutionPolicy,
}

impl StructuralModalResponseSpectrumTemplate {
    pub fn validate(&self) -> AnalysisExecutionResult<()> {
        if self.schema_version != 1 {
            return Err(invalid(
                "modal response-spectrum template requires schema version one",
            ));
        }
        validate_common(
            self.schema_version,
            &self.model,
            &self.step_id,
            self.backend,
        )?;
        Ok(())
    }

    pub fn bind(
        self,
        modal_basis_transition: CanonicalStateTransition,
    ) -> AnalysisExecutionResult<StructuralModalResponseSpectrumWorkload> {
        self.validate()?;
        StructuralModalResponseSpectrumWorkload::new(StructuralModalResponseSpectrumWorkloadInput {
            model: self.model,
            step_id: self.step_id,
            backend: self.backend,
            observation_execution: self.observation_execution,
            modal_basis_transition,
        })
    }
}

impl TryFrom<StructuralModalResponseSpectrumWorkloadWire>
    for StructuralModalResponseSpectrumWorkload
{
    type Error = AnalysisExecutionError;

    fn try_from(wire: StructuralModalResponseSpectrumWorkloadWire) -> Result<Self, Self::Error> {
        let workload = Self {
            schema_version: wire.schema_version,
            identity: wire.identity,
            model: wire.model,
            step_id: wire.step_id,
            backend: wire.backend,
            observation_execution: wire.observation_execution,
            modal_basis_transition: wire.modal_basis_transition,
        };
        workload.validate()?;
        Ok(workload)
    }
}

impl StructuralModalResponseSpectrumWorkload {
    pub fn new(
        input: StructuralModalResponseSpectrumWorkloadInput,
    ) -> AnalysisExecutionResult<Self> {
        let mut workload = Self {
            schema_version: 1,
            identity: StructuralModalResponseSpectrumWorkloadIdentity::derive(
                "structural-response-spectrum-workload-placeholder",
                &1_u8,
            )?,
            model: input.model,
            step_id: input.step_id,
            backend: input.backend,
            observation_execution: input.observation_execution,
            modal_basis_transition: input.modal_basis_transition,
        };
        workload.identity = workload.derive_identity()?;
        workload.validate()?;
        Ok(workload)
    }

    pub const fn identity(&self) -> StructuralModalResponseSpectrumWorkloadIdentity {
        self.identity
    }

    pub fn model(&self) -> &AnalysisModel {
        &self.model
    }

    pub fn step_id(&self) -> &ProcedureStepId {
        &self.step_id
    }

    pub const fn backend(&self) -> ComputeBackend {
        self.backend
    }

    pub const fn observation_execution(&self) -> ObservationExecutionPolicy {
        self.observation_execution
    }

    pub const fn modal_basis_transition(&self) -> &CanonicalStateTransition {
        &self.modal_basis_transition
    }

    pub fn procedure_step(&self) -> AnalysisExecutionResult<&ProcedureStep> {
        validate_common(
            self.schema_version,
            &self.model,
            &self.step_id,
            self.backend,
        )
    }

    pub fn spectra(&self) -> AnalysisExecutionResult<Vec<&ResponseSpectrum>> {
        let IndependentDomain::ResponseSpectra { spectrum_ids } =
            &self.procedure_step()?.independent_domain
        else {
            return Err(invalid(
                "response-spectrum workload has an incompatible domain",
            ));
        };
        spectrum_ids
            .iter()
            .map(|id| {
                self.model
                    .response_spectra
                    .iter()
                    .find(|spectrum| spectrum.spectrum_id == *id)
                    .ok_or_else(|| invalid("response-spectrum step references an absent spectrum"))
            })
            .collect()
    }

    pub fn required_residual_directions(
        &self,
    ) -> AnalysisExecutionResult<Vec<runmat_analysis_core::SpectrumDirection>> {
        self.spectra().map(|spectra| {
            spectra
                .into_iter()
                .filter(|spectrum| {
                    spectrum.truncation.correction
                        == runmat_analysis_core::ModalTruncationCorrection::ResidualMode
                })
                .map(|spectrum| spectrum.direction.clone())
                .collect()
        })
    }

    pub fn validate(&self) -> AnalysisExecutionResult<()> {
        let step = self.procedure_step()?;
        self.modal_basis_transition.validate()?;
        let sequence = SequenceIdentity::from_solver_contract(&self.model.sequence)?;
        let consumer = StepIdentity::from_solver_contract(sequence, step)?;
        if self.modal_basis_transition.consumer != consumer
            || self.modal_basis_transition.consumer_step_id != self.step_id
        {
            return Err(AnalysisExecutionError::Identity(
                "response-spectrum modal-basis transition names a different consumer",
            ));
        }
        let spectra = self.spectra()?;
        let ids = spectra
            .iter()
            .map(|spectrum| &spectrum.spectrum_id)
            .collect::<BTreeSet<_>>();
        if spectra.len() > 3
            || ids.len() != spectra.len()
            || spectra
                .iter()
                .any(|spectrum| spectrum.validate_intrinsic().is_err())
            || spectra
                .iter()
                .any(|spectrum| spectrum.direction.frame != CoordinateFrame::Global)
            || spectra.iter().any(|spectrum| {
                spectrum.response != spectra[0].response
                    || spectrum.directional_combination != spectra[0].directional_combination
            })
        {
            return Err(invalid(
                "response-spectrum workload requires unique compatible canonical spectra",
            ));
        }
        let directions = spectra
            .iter()
            .map(|spectrum| normalized(spectrum.direction.vector))
            .collect::<AnalysisExecutionResult<Vec<_>>>()?;
        if directions.iter().enumerate().any(|(index, direction)| {
            directions[index + 1..].iter().any(|other| {
                direction
                    .iter()
                    .zip(other)
                    .map(|(left, right)| left * right)
                    .sum::<f64>()
                    .abs()
                    > 1.0e-10
            })
        }) {
            return Err(invalid(
                "response-spectrum directions must be a pairwise-orthogonal global basis",
            ));
        }
        if self.identity != self.derive_identity()? {
            return Err(AnalysisExecutionError::Identity(
                "structural response-spectrum workload identity does not match its contents",
            ));
        }
        Ok(())
    }

    pub fn canonical_bytes(&self) -> AnalysisExecutionResult<Vec<u8>> {
        self.validate()?;
        Ok(encode_contract(
            WORKLOAD_PREFIX,
            "structural-response-spectrum-workload",
            self,
            WORKLOAD_LIMITS,
        )?)
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> AnalysisExecutionResult<Self> {
        Ok(decode_contract(
            WORKLOAD_PREFIX,
            "structural-response-spectrum-workload",
            bytes,
            WORKLOAD_LIMITS,
        )?)
    }

    fn derive_identity(
        &self,
    ) -> AnalysisExecutionResult<StructuralModalResponseSpectrumWorkloadIdentity> {
        StructuralModalResponseSpectrumWorkloadIdentity::derive(
            "structural-response-spectrum-workload",
            &(
                self.schema_version,
                &self.model,
                &self.step_id,
                self.backend,
                self.observation_execution,
                &self.modal_basis_transition,
            ),
        )
    }
}

fn validate_common<'a>(
    schema_version: u16,
    model: &'a AnalysisModel,
    step_id: &ProcedureStepId,
    backend: ComputeBackend,
) -> AnalysisExecutionResult<&'a ProcedureStep> {
    if schema_version != 1 || backend != ComputeBackend::Cpu {
        return Err(invalid(
            "modal response spectrum currently requires schema one and the qualified CPU backend",
        ));
    }
    validate_model_sequence(model).map_err(|error| invalid(error.to_string()))?;
    let mut matches = model
        .sequence
        .steps
        .iter()
        .filter(|step| step.step_id == *step_id);
    let step = matches
        .next()
        .ok_or_else(|| invalid("response-spectrum workload references an absent step"))?;
    if matches.next().is_some() {
        return Err(invalid(
            "response-spectrum workload references a duplicate step",
        ));
    }
    let IndependentDomain::ResponseSpectra { spectrum_ids } = &step.independent_domain else {
        return Err(invalid(
            "modal response spectrum requires a spectrum domain",
        ));
    };
    if step.family != PhysicsFamily::Structural
        || step.procedure != AnalysisProcedure::ModalResponseSpectrum
        || step.solution_space != SolutionSpace::ModalReduced
        || spectrum_ids.is_empty()
        || !step.activations.is_empty()
        || !matches!(
            step.state_inputs.as_slice(),
            [input]
                if input.use_as == StateUse::ModalBasis
                    && input.required_kind == StateKind::ModalBasis
                    && matches!(input.source, StateSource::Upstream { .. })
        )
    {
        return Err(invalid(
            "modal response spectrum requires one modal-reduced spectrum step and one upstream modal-basis input",
        ));
    }
    Ok(step)
}

fn invalid(message: impl Into<String>) -> AnalysisExecutionError {
    AnalysisExecutionError::Invalid(message.into())
}

fn normalized(vector: [f64; 3]) -> AnalysisExecutionResult<[f64; 3]> {
    let norm = vector.iter().map(|value| value * value).sum::<f64>().sqrt();
    if !norm.is_finite() || norm <= f64::EPSILON {
        return Err(invalid(
            "response-spectrum direction is not finite and nonzero",
        ));
    }
    Ok(vector.map(|value| value / norm))
}
