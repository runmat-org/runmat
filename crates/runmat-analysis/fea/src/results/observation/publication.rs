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
use std::collections::BTreeSet;

const PREFIX: &[u8] = b"runmat-observation-publication-catalog/v1\0";
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
pub use entry::{
    ObservationHistoryPublication, ObservationSamplePublication, SeriesReductionPublication,
    SeriesReductionPublicationContribution,
};

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
            schema_version: 1,
            identity: ObservationPublicationCatalogIdentity(Digest::sha256([])),
            producer,
            plan,
            source,
            declarations: step.observations.clone(),
            entries,
        };
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
        self.validate()?;
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
        self.validate_authored_step(step)?;
        match (self.source.source(), context) {
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
                self.source
                    .validate_thermal_plan(&self.plan, workload, mesh)?;
                for entry in &self.entries {
                    if let ObservationPublicationEntry::Sample(value) = entry {
                        self.source.validate_thermal_selection(
                            &value.request.selection,
                            workload,
                            mesh,
                        )?;
                    }
                }
            }
            (
                ObservationFieldCatalogSource::Structural { .. },
                ObservationHistorySourceContext::Structural { model, mesh },
            ) => {
                if !model.sequence.steps.iter().any(|authored| authored == step) {
                    return Err(invalid("structural publication authored step differs"));
                }
                self.source
                    .validate_structural_plan(&self.plan, model, mesh)?;
            }
            _ => return Err(invalid("publication source context differs")),
        }
        Ok(())
    }
    pub fn validate(&self) -> AnalysisExecutionResult<()> {
        self.plan.validate_identity().map_err(invalid)?;
        self.source.validate()?;
        if self.schema_version != 1
            || self.identity != self.derive_identity()?
            || self.declarations.len() != self.plan.observations.len()
        {
            return Err(invalid(
                "catalog schema, identity, or declaration coverage differs",
            ));
        }
        let mut expected = std::collections::BTreeMap::new();
        for (declaration, resolved) in self.declarations.iter().zip(&self.plan.observations) {
            let spec = declaration.spec().ok_or_else(|| {
                invalid("diagnostic/monitor publication is not supported by this catalog")
            })?;
            if spec != &resolved.spec
                || expected
                    .insert(
                        spec.observation_id.0.as_str(),
                        matches!(declaration, ObservationRequest::History(_)),
                    )
                    .is_some()
            {
                return Err(invalid(
                    "authored declaration differs from resolved plan or duplicates ID",
                ));
            }
        }
        let mut seen = BTreeSet::new();
        let mut previous = None;
        for entry in &self.entries {
            match entry {
                ObservationPublicationEntry::Sample(value) => {
                    value.validate()?;
                    match (self.source.source(), value.request.selection.source()) {
                        (
                            ObservationFieldCatalogSource::Structural { .. },
                            ObservationFieldSource::StructuralRecovery { .. },
                        ) => {
                            self.source
                                .validate_structural_selection(&value.request.selection)?;
                        }
                        (
                            ObservationFieldCatalogSource::Thermal { manifest },
                            ObservationFieldSource::ThermalResult {
                                workload,
                                result,
                                numerical_intent,
                                mesh_digest,
                                sample_ordinal,
                                ..
                            },
                        ) => {
                            let fields = manifest.fields.get(sample_ordinal).ok_or_else(|| {
                                invalid("sample ordinal absent from source catalog")
                            })?;
                            if workload != manifest.source_workload
                                || result != manifest.identity()
                                || numerical_intent != manifest.numerical_intent
                                || Digest::from_bytes(*mesh_digest.bytes())
                                    != manifest.topology_digest
                                || manifest.producer != self.producer
                                || value
                                    .request
                                    .selection
                                    .fields()
                                    .iter()
                                    .any(|field| !fields.contains(field))
                            {
                                return Err(invalid(
                                    "sample selection differs from source catalog",
                                ));
                            }
                        }
                        _ => return Err(invalid("sample source kind differs from catalog")),
                    }
                    if value.request.producer != self.producer || value.request.plan != self.plan {
                        return Err(invalid("sample producer or plan differs"));
                    }
                    let mut published_fields = 0;
                    for result in &value.manifest.results {
                        match expected.get(result.observation_id.0.as_str()) {
                            // A sample artifact covers its entire scheduled coordinate, including
                            // history samples. Those do NOT discharge any History declaration.
                            Some(true) => {}
                            Some(false) if seen.insert(result.observation_id.0.as_str()) => {
                                published_fields += 1;
                            }
                            _ => {
                                return Err(invalid(
                                    "sample duplicates or contains an undeclared observation",
                                ))
                            }
                        }
                    }
                    if published_fields == 0 {
                        return Err(invalid("sample cannot replace a complete history"));
                    }
                }
                ObservationPublicationEntry::History(value) => {
                    value.validate()?;
                    if value.request.producer != self.producer
                        || value.request.plan != self.plan
                        || value.request.source_catalog != self.source.identity()
                        || expected.get(value.history.observation_id.0.as_str()) != Some(&true)
                        || !seen.insert(value.history.observation_id.0.as_str())
                    {
                        return Err(invalid(
                            "history producer, source, plan, or declaration differs",
                        ));
                    }
                    let observation = self
                        .plan
                        .observations
                        .iter()
                        .find(|item| item.spec.observation_id == value.history.observation_id)
                        .ok_or_else(|| invalid("history observation absent"))?;
                    let source_fields = match self.source.source() {
                        ObservationFieldCatalogSource::Structural { manifest, .. } => {
                            manifest.fields.iter().collect::<Vec<_>>()
                        }
                        ObservationFieldCatalogSource::Thermal { manifest } => {
                            if manifest.producer != self.producer {
                                return Err(invalid("thermal history producer differs"));
                            }
                            manifest.fields.iter().flatten().collect::<Vec<_>>()
                        }
                    };
                    let mut coordinates = Vec::new();
                    for field in source_fields {
                        if field.definition.identity()
                            == observation.source_field_definition_identity
                            && resolved_observation_accepts_sample(
                                &self.plan,
                                observation,
                                field.sample,
                                None,
                            )
                            .map_err(invalid)?
                        {
                            coordinates.push(field.sample);
                        }
                    }
                    if coordinates
                        != value
                            .history
                            .samples
                            .iter()
                            .map(|sample| sample.coordinate)
                            .collect::<Vec<_>>()
                    {
                        return Err(invalid(
                            "history omits or substitutes an accepted source coordinate",
                        ));
                    }
                }
            }
            if previous.is_some_and(|id| id >= entry.first_id()) {
                return Err(invalid("publication entries are not canonically ordered"));
            }
            previous = Some(entry.first_id());
        }
        if seen.len() != expected.len() {
            return Err(invalid(
                "publication does not cover every authored observation",
            ));
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
