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
/// Memory profiles of a catalog the run's budget already admitted: bytes and collections are
/// bounded by the platform, so a catalog of any admitted size can be sized. String and nesting
/// rules still refuse malformed content.
const PROFILE_LIMITS: CanonicalLimits =
    CanonicalLimits::platform_content(LIMITS.maximum_string_bytes, LIMITS.maximum_nesting_depth);
fn invalid(error: impl std::fmt::Display) -> AnalysisExecutionError {
    AnalysisExecutionError::Invalid(format!("observation publication: {error}"))
}
/// Catalogs grow with the observed mesh and samples, so codec bounds follow the content:
/// encoding is bounded by the profiled size of the owned value, decoding by the received bytes.
/// Never below the fixed floor.
fn content_limits(bytes: usize) -> CanonicalLimits {
    CanonicalLimits::content_bounded(
        bytes.max(LIMITS.maximum_encoded_bytes),
        LIMITS.maximum_string_bytes,
        LIMITS.maximum_nesting_depth,
    )
}
fn encode_scaled<T: Serialize>(domain: &str, value: &T) -> AnalysisExecutionResult<Vec<u8>> {
    let profile =
        runmat_canonical_codec::profile_canonical_encoding(value, content_limits(usize::MAX))
            .map_err(invalid)?;
    let bytes = usize::try_from(profile.encoded_bytes_upper_bound)
        .map_err(|_| invalid("catalog exceeds the address space"))?;
    encode_contract(PREFIX, domain, value, content_limits(bytes)).map_err(invalid)
}
/// Identities are digested from the streamed encoding; no catalog-sized buffer is built.
fn digest<T: Serialize>(domain: &str, value: &T) -> AnalysisExecutionResult<Digest> {
    crate::contracts::canonical_digest::canonical_contract_digest(PREFIX, domain, value, LIMITS)
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
        // Decoded identities are claims until checked against the decoded content.
        value.validate_metadata_uncached()?;
        remember(&VALIDATED_METADATA, value.identity.digest());
        Ok(value)
    }
}
/// Whether `validate_for_context` re-derives the execution grouping, or the caller derived it
/// from the same context in the same construction.
#[derive(Clone, Copy)]
enum ExecutionCheck {
    Rederive,
    Derived,
}

/// Whether `validate_for_context` re-resolves a volume-mesh plan against its catalog, or the
/// plan arrived sealed with that catalog on the context's mesh.
#[derive(Clone, Copy, PartialEq)]
enum PlanCheck {
    Reresolve,
    Resolved,
}

/// Successful (catalog, step, volume-mesh context) validations retained per process. One study
/// validates its catalog when it is built, and again in accuracy preparation, accuracy import
/// and publication; each is the same pure function of the keyed content, so the process runs
/// it once. Only successes are retained; a different catalog, step or mesh misses.
const VALIDATED_CAPACITY: usize = 8;
static VALIDATED_CONTEXTS: std::sync::Mutex<Vec<Digest>> = std::sync::Mutex::new(Vec::new());
/// Catalog identities whose context-free metadata validated in this process.
static VALIDATED_METADATA: std::sync::Mutex<Vec<Digest>> = std::sync::Mutex::new(Vec::new());

fn seen(memo: &std::sync::Mutex<Vec<Digest>>, key: Digest) -> bool {
    memo.lock().is_ok_and(|seen| seen.contains(&key))
}

fn remember(memo: &std::sync::Mutex<Vec<Digest>>, key: Digest) {
    if let Ok(mut seen) = memo.lock() {
        seen.retain(|k| *k != key);
        if seen.len() == VALIDATED_CAPACITY {
            seen.remove(0);
        }
        seen.push(key);
    }
}

impl ObservationPublicationCatalog {
    pub fn new(
        producer: StepIdentity,
        plan: ResolvedObservationPlan,
        source: ObservationFieldCatalog,
        step: &ProcedureStep,
        entries: Vec<ObservationPublicationEntry>,
        context: ObservationHistorySourceContext<'_>,
    ) -> AnalysisExecutionResult<Self> {
        Self::build(
            producer,
            plan,
            source,
            step,
            entries,
            context,
            PlanCheck::Reresolve,
        )
    }

    /// A volume-mesh catalog whose plan was resolved from its own field catalog: when the
    /// context is the mesh it was resolved on, the plan is not re-resolved to check it.
    pub fn on_volume_mesh(
        producer: StepIdentity,
        authority: super::VolumeMeshObservationAuthority,
        step: &ProcedureStep,
        entries: Vec<ObservationPublicationEntry>,
        context: ObservationHistorySourceContext<'_>,
    ) -> AnalysisExecutionResult<Self> {
        let check = match context {
            ObservationHistorySourceContext::VolumeMesh { mesh } if authority.resolved_on(mesh) => {
                PlanCheck::Resolved
            }
            _ => PlanCheck::Reresolve,
        };
        let (source, plan) = authority.into_parts();
        Self::build(producer, plan, source, step, entries, context, check)
    }

    fn build(
        producer: StepIdentity,
        plan: ResolvedObservationPlan,
        source: ObservationFieldCatalog,
        step: &ProcedureStep,
        mut entries: Vec<ObservationPublicationEntry>,
        context: ObservationHistorySourceContext<'_>,
        plan_check: PlanCheck,
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
        // The execution grouping was just derived from this context; validate the rest.
        context.validate()?;
        value.validate_for_context(step, context, ExecutionCheck::Derived, plan_check)?;
        if let Some(key) = value.validated_context_key(step, context)? {
            remember(&VALIDATED_CONTEXTS, key);
        }
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
        self.validate_metadata_admitted(step.accuracy.resources.resident_byte_budget())?;
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
        let key = self.validated_context_key(step, context)?;
        if key.is_some_and(|key| seen(&VALIDATED_CONTEXTS, key)) {
            return Ok(());
        }
        self.validate_for_context(
            step,
            context,
            ExecutionCheck::Rederive,
            PlanCheck::Reresolve,
        )?;
        if let Some(key) = key {
            remember(&VALIDATED_CONTEXTS, key);
        }
        Ok(())
    }

    /// The content key of one (catalog, authored step, source context) validation, when the
    /// context's identity is content-verified by `context.validate()`. The catalog's identity
    /// is a digest over its whole content, sealed at construction and decoding (fields are
    /// private and immutable), and the volume mesh's canonical digest was just checked against
    /// its payload. Other contexts carry unverified workload references and are not keyed.
    fn validated_context_key(
        &self,
        step: &ProcedureStep,
        context: ObservationHistorySourceContext<'_>,
    ) -> AnalysisExecutionResult<Option<Digest>> {
        let ObservationHistorySourceContext::VolumeMesh { mesh } = context else {
            return Ok(None);
        };
        digest(
            "validated-volume-mesh-context",
            &(self.identity, step, mesh.canonical_digest),
        )
        .map(Some)
    }

    fn validate_for_context(
        &self,
        step: &ProcedureStep,
        context: ObservationHistorySourceContext<'_>,
        execution: ExecutionCheck,
        plan_check: PlanCheck,
    ) -> AnalysisExecutionResult<()> {
        self.validate_authored_step(step)?;
        match (self.source.source(), context) {
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
                    recovery_request,
                    step: producing_step,
                    ..
                },
                ObservationHistorySourceContext::VolumeMesh { mesh },
            ) => {
                if recovery_request.producer != self.producer || producing_step.as_ref() != step {
                    return Err(invalid(
                        "volume-mesh publication producer or authored step differs",
                    ));
                }
                if plan_check == PlanCheck::Reresolve {
                    self.source.validate_volume_mesh_plan(&self.plan, mesh)?;
                }
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
        match execution {
            ExecutionCheck::Rederive => self.validate_execution_for(context)?,
            ExecutionCheck::Derived => {
                if self.execution_policy != context.observation_execution() {
                    return Err(invalid("execution policy differs from producing workload"));
                }
            }
        }
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
        encode_scaled("catalog", self)
    }
    pub fn from_canonical_bytes(bytes: &[u8]) -> AnalysisExecutionResult<Self> {
        decode_contract(PREFIX, "catalog", bytes, content_limits(bytes.len())).map_err(invalid)
    }
}
