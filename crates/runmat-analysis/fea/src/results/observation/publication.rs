//! Metadata-only publication authority for single-sample artifacts and complete histories.
//! Decoding establishes internal integrity; importers must validate fetched outputs separately.
use super::*;
use crate::execution::{
    AnalysisExecutionError, AnalysisExecutionResult, ObservationAssemblyRequest,
    ObservationEvaluationRequest, ObservationHistoryFinalizationRequest,
    ObservationHistorySourceContext, StepIdentity,
};
use runmat_analysis_core::{ExtremaTiePolicy, ObservationRequest, ProcedureStep, SeriesReduction};
use runmat_canonical_codec::{decode_contract, encode_contract, CanonicalLimits};
use runmat_execution::Digest;
use serde::{Deserialize, Serialize};

mod coordinates;
mod coverage;
mod resources;
mod validation;

const PREFIX: &[u8] = b"runmat-observation-publication-catalog/v2\0";
const LIMITS: CanonicalLimits = CanonicalLimits::new(64 * 1024 * 1024, 4_000_000, 65_536, 64);
fn invalid(error: impl std::fmt::Display) -> AnalysisExecutionError {
    AnalysisExecutionError::Invalid(format!("observation publication: {error}"))
}
fn digest<T: Serialize>(domain: &str, value: &T) -> AnalysisExecutionResult<Digest> {
    encode_contract(PREFIX, domain, value, LIMITS)
        .map(Digest::sha256)
        .map_err(invalid)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ObservationPublicationCatalogIdentity(Digest);
impl ObservationPublicationCatalogIdentity {
    pub const fn digest(self) -> Digest {
        self.0
    }
}

mod entry;
mod execution;
pub use entry::{
    ObservationHistoryPublication, ObservationSamplePublication, SeriesReductionPublication,
    SeriesReductionPublicationContribution,
};
use execution::ObservationResultExecution;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(
    rename_all = "snake_case",
    tag = "kind",
    content = "publication",
    deny_unknown_fields
)]
pub enum ObservationPublicationEntry {
    Sample(Box<ObservationSamplePublication>),
    History(Box<ObservationHistoryPublication>),
}
impl ObservationPublicationEntry {
    fn first_id(&self) -> (u8, &str) {
        match self {
            Self::Sample(value) => (0, &value.manifest.results[0].observation_id.0),
            Self::History(value) => (1, &value.history.observation_id.0),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, try_from = "CatalogWire")]
pub struct ObservationPublicationCatalog {
    schema_version: u16,
    identity: ObservationPublicationCatalogIdentity,
    producer: StepIdentity,
    plan: ResolvedObservationPlan,
    source: ObservationFieldCatalog,
    declarations: Vec<ObservationRequest>,
    entries: Vec<ObservationPublicationEntry>,
    execution_policy: ObservationExecutionPolicy,
    execution: Vec<ObservationResultExecution>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CatalogWire {
    schema_version: u16,
    identity: ObservationPublicationCatalogIdentity,
    producer: StepIdentity,
    plan: ResolvedObservationPlan,
    source: ObservationFieldCatalog,
    declarations: Vec<ObservationRequest>,
    entries: Vec<ObservationPublicationEntry>,
    execution_policy: ObservationExecutionPolicy,
    execution: Vec<ObservationResultExecution>,
}
impl TryFrom<CatalogWire> for ObservationPublicationCatalog {
    type Error = AnalysisExecutionError;
    fn try_from(wire: CatalogWire) -> Result<Self, Self::Error> {
        let value = Self {
            schema_version: wire.schema_version,
            identity: wire.identity,
            producer: wire.producer,
            plan: wire.plan,
            source: wire.source,
            declarations: wire.declarations,
            entries: wire.entries,
            execution_policy: wire.execution_policy,
            execution: wire.execution,
        };
        value.validate()?;
        Ok(value)
    }
}
impl ObservationPublicationCatalog {
    pub fn new(
        producer: StepIdentity,
        plan: ResolvedObservationPlan,
        source: ObservationFieldCatalog,
        step: &ProcedureStep,
        mut entries: Vec<ObservationPublicationEntry>,
        context: ObservationHistorySourceContext<'_>,
    ) -> AnalysisExecutionResult<Self> {
        // Validate entries before reading their canonical ordering keys.
        for entry in &entries {
            match entry {
                ObservationPublicationEntry::Sample(value) => value.validate()?,
                ObservationPublicationEntry::History(value) => value.validate()?,
            }
        }
        entries.sort_by(|a, b| a.first_id().cmp(&b.first_id()));
        let mut value = Self {
            schema_version: 2,
            identity: ObservationPublicationCatalogIdentity(Digest::sha256([])),
            producer,
            plan,
            source,
            declarations: step.observations.clone(),
            entries,
            execution_policy: context.observation_execution(),
            execution: Vec::new(),
        };
        value.execution = value.derive_execution(context)?;
        value.identity = value.derive_identity()?;
        value.validate_for(step, context)?;
        Ok(value)
    }
    pub const fn identity(&self) -> ObservationPublicationCatalogIdentity {
        self.identity
    }
    pub const fn producer(&self) -> StepIdentity {
        self.producer
    }
    pub fn plan(&self) -> &ResolvedObservationPlan {
        &self.plan
    }
    pub fn source(&self) -> &ObservationFieldCatalog {
        &self.source
    }
    pub fn entries(&self) -> &[ObservationPublicationEntry] {
        &self.entries
    }
    /// Compare coverage against an independently supplied authored step. The step identity
    /// alone excludes observation declarations and cannot establish this relationship.
    /// This checks authorship, not mesh-derived target indices; use `validate_for` for both.
    pub fn validate_authored_step(&self, step: &ProcedureStep) -> AnalysisExecutionResult<()> {
        self.validate_metadata_admitted(step.accuracy.resources.maximum_resident_bytes)?;
        if self.declarations != step.observations
            || self.plan.step_id != step.step_id
            || self.plan.independent_domain != step.independent_domain
        {
            return Err(invalid(
                "publication declarations differ from authored step",
            ));
        }
        Ok(())
    }

    pub fn validate_for(
        &self,
        step: &ProcedureStep,
        context: ObservationHistorySourceContext<'_>,
    ) -> AnalysisExecutionResult<()> {
        context.validate()?;
        self.validate_authored_step(step)?;
        match (self.source.source(), context) {
            (
                ObservationFieldCatalogSource::FiniteVolumeRecovery {
                    recovery_request, ..
                },
                ObservationHistorySourceContext::FluidSteady { workload, mesh },
            ) => {
                if recovery_request.producer != self.producer
                    || workload.producer_identity()? != self.producer
                    || !workload
                        .model
                        .sequence
                        .steps
                        .iter()
                        .any(|authored| authored == step)
                {
                    return Err(invalid(
                        "finite-volume publication original producer or authored step differs",
                    ));
                }
                self.source
                    .validate_finite_volume_plan(&self.plan, workload, mesh)?;
                for entry in &self.entries {
                    if let ObservationPublicationEntry::Sample(value) = entry {
                        self.source
                            .validate_finite_volume_selection(&value.request.selection)?;
                    }
                }
            }
            (
                ObservationFieldCatalogSource::SampledRecovery {
                    recovery_request,
                    manifest,
                    sampling,
                },
                context @ (ObservationHistorySourceContext::Magnetostatic { .. }
                | ObservationHistorySourceContext::MaxwellHarmonic { .. }),
            ) => {
                let (native, producer, model, step_id, mesh) = match context {
                    ObservationHistorySourceContext::Magnetostatic { workload, mesh } => (
                        crate::execution::magnetic_native_field_definitions(recovery_request)?,
                        workload.producer_identity()?,
                        &workload.model,
                        &workload.step_id,
                        mesh,
                    ),
                    ObservationHistorySourceContext::MaxwellHarmonic { workload, mesh } => (
                        crate::execution::maxwell_native_field_definitions(recovery_request)?,
                        workload.producer_identity()?,
                        &workload.model,
                        &workload.step_id,
                        mesh,
                    ),
                    _ => return Err(invalid("sampled publication source context differs")),
                };
                if sampling.rule()
                    != crate::results::field::VolumeFieldSamplingRule::TetrahedronKeastFifteen
                    || !manifest
                        .fields
                        .iter()
                        .map(|entry| &entry.definition)
                        .eq(native.iter())
                {
                    return Err(invalid(
                        "sampled publication native definition inventory or sampling rule differs",
                    ));
                }
                if recovery_request.producer != self.producer
                    || producer != self.producer
                    || &step.step_id != step_id
                    || !model.sequence.steps.iter().any(|authored| authored == step)
                {
                    return Err(invalid(
                        "sampled publication producer or original authored step differs",
                    ));
                }
                self.source
                    .validate_sampled_plan(&self.plan, step, producer, mesh)?;
                for entry in &self.entries {
                    if let ObservationPublicationEntry::Sample(value) = entry {
                        self.source
                            .validate_sampled_selection(&value.request.selection)?;
                    }
                }
            }
            (
                ObservationFieldCatalogSource::VolumeMeshRecovery {
                    recovery_request, ..
                },
                ObservationHistorySourceContext::AcousticHarmonic { workload, mesh },
            ) => {
                if recovery_request.producer != self.producer
                    || workload.producer_identity()? != self.producer
                    || workload.step_id != step.step_id
                    || !workload
                        .model
                        .sequence
                        .steps
                        .iter()
                        .any(|authored| authored == step)
                {
                    return Err(invalid(
                        "volume-mesh publication producer or authored step differs",
                    ));
                }
                self.source
                    .validate_volume_mesh_plan(&self.plan, step, mesh)?;
                for entry in &self.entries {
                    if let ObservationPublicationEntry::Sample(value) = entry {
                        self.source
                            .validate_volume_mesh_selection(&value.request.selection, mesh)?;
                    }
                }
            }
            (
                ObservationFieldCatalogSource::Thermal { manifest },
                ObservationHistorySourceContext::Thermal { workload, mesh },
            ) => {
                if manifest.producer != self.producer
                    || !workload
                        .model
                        .sequence
                        .steps
                        .iter()
                        .any(|authored| authored == step)
                {
                    return Err(invalid(
                        "thermal publication producer or authored step differs",
                    ));
                }
                let authority =
                    super::source::ThermalFieldSourceAuthority::new(workload, manifest, mesh)?;
                authority.validate_plan(&self.plan)?;
                for entry in &self.entries {
                    if let ObservationPublicationEntry::Sample(value) = entry {
                        authority.validate_selection(&value.request.selection)?;
                    }
                }
            }
            (
                ObservationFieldCatalogSource::Structural { .. },
                ObservationHistorySourceContext::Structural { source, mesh },
            ) => {
                let model = source.model();
                if source.procedure_step()? != step
                    || !model.sequence.steps.iter().any(|authored| authored == step)
                {
                    return Err(invalid("structural publication authored step differs"));
                }
                self.source
                    .validate_structural_plan(&self.plan, model, mesh)?;
            }
            (
                ObservationFieldCatalogSource::StructuralFrequencyHistory { .. },
                ObservationHistorySourceContext::Structural {
                    source: crate::execution::StructuralObservationSource::Frequency(workload),
                    mesh,
                },
            ) => {
                if workload.procedure_step()? != step {
                    return Err(invalid("frequency publication authored step differs"));
                }
                self.source
                    .validate_frequency_history_plan(&self.plan, workload, mesh)?;
            }
            _ => return Err(invalid("publication source context differs")),
        }
        self.validate_execution_for(context)?;
        Ok(())
    }
    fn derive_identity(&self) -> AnalysisExecutionResult<ObservationPublicationCatalogIdentity> {
        digest(
            "identity",
            &(
                self.schema_version,
                self.producer,
                &self.plan,
                self.source.identity(),
                &self.declarations,
                &self.entries,
                self.execution_policy,
                &self.execution,
            ),
        )
        .map(ObservationPublicationCatalogIdentity)
    }
    pub fn canonical_bytes(&self) -> AnalysisExecutionResult<Vec<u8>> {
        self.validate()?;
        encode_contract(PREFIX, "catalog", self, LIMITS).map_err(invalid)
    }
    pub fn from_canonical_bytes(bytes: &[u8]) -> AnalysisExecutionResult<Self> {
        decode_contract(PREFIX, "catalog", bytes, LIMITS).map_err(invalid)
    }
}
