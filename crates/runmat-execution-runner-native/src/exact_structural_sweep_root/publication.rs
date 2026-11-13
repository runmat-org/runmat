use std::collections::BTreeMap;

use runmat_analysis_fea::execution::{SweepDefinition, SweepIdentity};
use runmat_canonical_codec::{decode_contract, encode_contract};
use runmat_execution::value::{ValuePayload, ValueRef, ValueRefKind};
use runmat_execution::ProgramRevision;
use runmat_execution_artifact::cache::CacheImport;
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_execution_artifact::{
    ArtifactAccess, LogicalObject, ObjectNamespace, ProgramExecutionObject,
    ProgramExecutionRequest, ProgramExecutionResponse,
};
use serde::{Deserialize, Serialize};

use super::{
    protocol, CONTRACT_LIMITS, PUBLICATION_DOMAIN, PUBLICATION_MEDIA_TYPE, PUBLICATION_PREFIX,
    PUBLICATION_VALUE_SCHEMA,
};
use crate::{
    import_exact_structural_study_result, import_structural_study_result,
    ExactStructuralStudyResult, ExactStructuralStudyResultExpectation, NativeExecutionResult,
};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "status", deny_unknown_fields)]
enum PublishedSweepEntry {
    Succeeded { response: ProgramExecutionResponse },
    Failed { message: String },
    Cancelled,
    Indeterminate { message: String },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExactStructuralSweepPublication {
    schema_version: u16,
    program_revision: ProgramRevision,
    identity: SweepIdentity,
    entries: Vec<PublishedSweepEntry>,
}

impl ExactStructuralSweepPublication {
    fn validate(&self) -> NativeExecutionResult<()> {
        if self.schema_version != 1 || self.entries.is_empty() {
            return Err(protocol("exact structural sweep publication is invalid"));
        }
        for entry in &self.entries {
            match entry {
                PublishedSweepEntry::Succeeded { response } => {
                    if !matches!(
                        response,
                        ProgramExecutionResponse::ExternalizedSuccess { .. }
                    ) {
                        return Err(protocol("successful sweep entry is not externalized"));
                    }
                }
                PublishedSweepEntry::Failed { message }
                | PublishedSweepEntry::Indeterminate { message }
                    if message.is_empty() || message.len() > 64 * 1024 =>
                {
                    return Err(protocol("sweep failure evidence is empty or unbounded"));
                }
                _ => {}
            }
        }
        Ok(())
    }

    fn canonical_bytes(&self) -> NativeExecutionResult<Vec<u8>> {
        self.validate()?;
        encode_contract(
            PUBLICATION_PREFIX,
            PUBLICATION_DOMAIN,
            self,
            CONTRACT_LIMITS,
        )
        .map_err(protocol)
    }

    fn decode(bytes: &[u8]) -> NativeExecutionResult<Self> {
        let publication = decode_contract(
            PUBLICATION_PREFIX,
            PUBLICATION_DOMAIN,
            bytes,
            CONTRACT_LIMITS,
        )
        .map_err(protocol)?;
        Self::validate(&publication)?;
        Ok(publication)
    }
}

pub(crate) enum ExactStructuralSweepRemoteOutcome {
    Succeeded(ProgramExecutionResponse),
    Failed(String),
    Cancelled,
    Indeterminate(String),
}

#[derive(Clone, Debug, PartialEq)]
pub enum ExactStructuralSweepTerminal {
    Succeeded(Box<ExactStructuralStudyResult>),
    Failed(String),
    Cancelled,
    Indeterminate(String),
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExactStructuralSweepEntryResult {
    pub study_index: usize,
    pub terminal: ExactStructuralSweepTerminal,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExactStructuralSweepResult {
    pub identity: SweepIdentity,
    pub entries: Vec<ExactStructuralSweepEntryResult>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExactStructuralSweepEntryStatus {
    Succeeded,
    Failed,
    Cancelled,
    Indeterminate,
}

#[derive(Clone, Debug)]
pub struct ExactStructuralSweepResultCatalog {
    pub identity: SweepIdentity,
    statuses: Vec<ExactStructuralSweepEntryStatus>,
    entries: Vec<PublishedSweepEntry>,
    expected_studies: Vec<ExactStructuralStudyResultExpectation>,
}

impl ExactStructuralSweepResultCatalog {
    pub fn statuses(&self) -> &[ExactStructuralSweepEntryStatus] {
        &self.statuses
    }

    /// Materializes exactly one authored sweep entry. Successful entries read only that
    /// study's referenced result closure; terminal evidence is already present in the catalog.
    pub fn import_entry(
        &self,
        source: &impl CacheImport,
        study_index: usize,
        limits: ObjectInventoryLimits,
    ) -> NativeExecutionResult<ExactStructuralSweepEntryResult> {
        let entry = self
            .entries
            .get(study_index)
            .ok_or_else(|| protocol("structural sweep study index is out of bounds"))?;
        let expected = &self.expected_studies[study_index];
        let terminal = match entry {
            PublishedSweepEntry::Succeeded { response } => ExactStructuralSweepTerminal::Succeeded(
                Box::new(import_exact_structural_study_result(
                    source,
                    response.clone(),
                    expected,
                    limits,
                )?),
            ),
            PublishedSweepEntry::Failed { message } => {
                ExactStructuralSweepTerminal::Failed(message.clone())
            }
            PublishedSweepEntry::Cancelled => ExactStructuralSweepTerminal::Cancelled,
            PublishedSweepEntry::Indeterminate { message } => {
                ExactStructuralSweepTerminal::Indeterminate(message.clone())
            }
        };
        Ok(ExactStructuralSweepEntryResult {
            study_index,
            terminal,
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExactStructuralSweepResultExpectation {
    pub program_revision: ProgramRevision,
    pub artifact_access: ArtifactAccess,
    pub definition: SweepDefinition,
    pub studies: Vec<ExactStructuralStudyResultExpectation>,
}

impl ExactStructuralSweepResultExpectation {
    pub fn identity(&self) -> NativeExecutionResult<SweepIdentity> {
        self.definition.identity().map_err(protocol)
    }

    pub fn validate(&self) -> NativeExecutionResult<()> {
        self.artifact_access.validate().map_err(protocol)?;
        if self.studies.is_empty() {
            return Err(protocol("exact structural sweep expectation is empty"));
        }
        for study in &self.studies {
            study.validate()?;
            if study.program_revision != self.program_revision
                || study.artifact_access != self.artifact_access
            {
                return Err(protocol(
                    "sweep study expectation differs from its revision or authority",
                ));
            }
        }
        let studies = self
            .studies
            .iter()
            .map(ExactStructuralStudyResultExpectation::study_identity)
            .collect::<NativeExecutionResult<Vec<_>>>()?;
        if self.definition.studies() != studies {
            return Err(protocol(
                "sweep expectation definition differs from its ordered studies",
            ));
        }
        self.identity()?;
        Ok(())
    }
}

pub(crate) fn package_exact_structural_sweep_result(
    request: &ProgramExecutionRequest,
    identity: SweepIdentity,
    access: &ArtifactAccess,
    outcomes: Vec<ExactStructuralSweepRemoteOutcome>,
) -> NativeExecutionResult<(ProgramExecutionResponse, Vec<ProgramExecutionObject>)> {
    if outcomes.is_empty() {
        return Err(protocol("exact structural sweep produced no outcomes"));
    }
    let entries = outcomes
        .into_iter()
        .map(|outcome| match outcome {
            ExactStructuralSweepRemoteOutcome::Succeeded(response) => {
                PublishedSweepEntry::Succeeded { response }
            }
            ExactStructuralSweepRemoteOutcome::Failed(message) => {
                PublishedSweepEntry::Failed { message }
            }
            ExactStructuralSweepRemoteOutcome::Cancelled => PublishedSweepEntry::Cancelled,
            ExactStructuralSweepRemoteOutcome::Indeterminate(message) => {
                PublishedSweepEntry::Indeterminate { message }
            }
        })
        .collect::<Vec<_>>();
    let publication = ExactStructuralSweepPublication {
        schema_version: 1,
        program_revision: request.recipe.program_revision.clone(),
        identity,
        entries,
    };
    let object = LogicalObject::new(
        ObjectNamespace::ResultValue,
        "analysis/exact-structural-sweep-publications",
        PUBLICATION_MEDIA_TYPE,
        publication.canonical_bytes()?,
    )
    .map_err(protocol)?;
    let root = access
        .reference(
            &object,
            ValueRefKind::ResultObject,
            PUBLICATION_VALUE_SCHEMA,
        )
        .map_err(protocol)?;
    let mut inventory = BTreeMap::new();
    for entry in &publication.entries {
        if let PublishedSweepEntry::Succeeded {
            response: ProgramExecutionResponse::ExternalizedSuccess { result_objects, .. },
        } = entry
        {
            for reference in result_objects {
                insert_reference(&mut inventory, reference.clone())?;
            }
        }
    }
    insert_reference(&mut inventory, root.clone())?;
    let response = ProgramExecutionResponse::ExternalizedSuccess {
        outputs: vec![ValuePayload::Object(Box::new(root.clone()))],
        result_objects: inventory.into_values().collect(),
    };
    response.validate_against(request).map_err(protocol)?;
    Ok((
        response,
        vec![ProgramExecutionObject {
            reference: root,
            bytes: object.bytes,
        }],
    ))
}

pub fn import_exact_structural_sweep_result(
    source: &impl CacheImport,
    response: ProgramExecutionResponse,
    expected: &ExactStructuralSweepResultExpectation,
    limits: ObjectInventoryLimits,
) -> NativeExecutionResult<ExactStructuralSweepResult> {
    let catalog = import_exact_structural_sweep_result_catalog(source, response, expected)?;
    let entries = (0..catalog.statuses().len())
        .map(|study_index| catalog.import_entry(source, study_index, limits))
        .collect::<NativeExecutionResult<Vec<_>>>()?;
    Ok(ExactStructuralSweepResult {
        identity: catalog.identity,
        entries,
    })
}

/// Admit a sweep closure and return every entry, in authored order, with its terminal status.
/// Successful entries keep their raw response for the family importer of their study.
pub fn import_study_sweep_entries(
    source: &impl CacheImport,
    response: ProgramExecutionResponse,
    expected: &crate::StructuralSweepResultExpectation,
) -> NativeExecutionResult<(SweepIdentity, Vec<StudySweepEntry>)> {
    expected.validate()?;
    let ProgramExecutionResponse::ExternalizedSuccess {
        outputs,
        result_objects,
    } = response
    else {
        return Err(protocol("structural sweep did not return a result closure"));
    };
    let [ValuePayload::Object(root)] = outputs.as_slice() else {
        return Err(protocol("structural sweep result root is invalid"));
    };
    expected
        .artifact_access
        .validate_reference(
            root,
            ValueRefKind::ResultObject,
            PUBLICATION_MEDIA_TYPE,
            PUBLICATION_VALUE_SCHEMA,
        )
        .map_err(protocol)?;
    let publication = ExactStructuralSweepPublication::decode(&read(source, root)?)?;
    if publication.program_revision != expected.program_revision
        || publication.identity != expected.identity()?
        || publication.entries.len() != expected.studies.len()
    {
        return Err(protocol("structural sweep publication identity changed"));
    }
    let mut expected_inventory = BTreeMap::new();
    insert_reference(&mut expected_inventory, (**root).clone())?;
    for entry in &publication.entries {
        if let PublishedSweepEntry::Succeeded {
            response: ProgramExecutionResponse::ExternalizedSuccess { result_objects, .. },
        } = entry
        {
            for reference in result_objects {
                insert_reference(&mut expected_inventory, reference.clone())?;
            }
        }
    }
    if expected_inventory.into_values().collect::<Vec<_>>() != result_objects {
        return Err(protocol(
            "structural sweep result inventory differs from its publication closure",
        ));
    }
    let entries = publication
        .entries
        .into_iter()
        .enumerate()
        .map(|(study_index, entry)| StudySweepEntry {
            study_index,
            terminal: match entry {
                PublishedSweepEntry::Succeeded { response } => {
                    StudySweepTerminal::Succeeded(response)
                }
                PublishedSweepEntry::Failed { message } => StudySweepTerminal::Failed(message),
                PublishedSweepEntry::Cancelled => StudySweepTerminal::Cancelled,
                PublishedSweepEntry::Indeterminate { message } => {
                    StudySweepTerminal::Indeterminate(message)
                }
            },
        })
        .collect();
    Ok((publication.identity, entries))
}

/// Structural-only view of a sweep whose every study has structural planning.
pub fn import_structural_sweep_result(
    source: &impl CacheImport,
    response: ProgramExecutionResponse,
    expected: &crate::StructuralSweepResultExpectation,
    limits: ObjectInventoryLimits,
) -> NativeExecutionResult<ExactStructuralSweepResult> {
    let (identity, entries) = import_study_sweep_entries(source, response, expected)?;
    let entries = entries
        .into_iter()
        .zip(&expected.studies)
        .map(|(entry, study)| {
            let terminal = match entry.terminal {
                StudySweepTerminal::Succeeded(response) => {
                    ExactStructuralSweepTerminal::Succeeded(Box::new(
                        import_structural_study_result(source, response, study, limits)?,
                    ))
                }
                StudySweepTerminal::Failed(message) => {
                    ExactStructuralSweepTerminal::Failed(message)
                }
                StudySweepTerminal::Cancelled => ExactStructuralSweepTerminal::Cancelled,
                StudySweepTerminal::Indeterminate(message) => {
                    ExactStructuralSweepTerminal::Indeterminate(message)
                }
            };
            Ok(ExactStructuralSweepEntryResult {
                study_index: entry.study_index,
                terminal,
            })
        })
        .collect::<NativeExecutionResult<Vec<_>>>()?;
    Ok(ExactStructuralSweepResult { identity, entries })
}

/// Terminal of one sweep study before its family importer admits the result closure.
#[derive(Clone, Debug, PartialEq)]
pub enum StudySweepTerminal {
    Succeeded(ProgramExecutionResponse),
    Failed(String),
    Cancelled,
    Indeterminate(String),
}

#[derive(Clone, Debug, PartialEq)]
pub struct StudySweepEntry {
    pub study_index: usize,
    pub terminal: StudySweepTerminal,
}

pub fn import_exact_structural_sweep_result_catalog(
    source: &impl CacheImport,
    response: ProgramExecutionResponse,
    expected: &ExactStructuralSweepResultExpectation,
) -> NativeExecutionResult<ExactStructuralSweepResultCatalog> {
    expected.validate()?;
    let ProgramExecutionResponse::ExternalizedSuccess {
        outputs,
        result_objects,
    } = response
    else {
        return Err(protocol("structural sweep did not return a result closure"));
    };
    let [ValuePayload::Object(root)] = outputs.as_slice() else {
        return Err(protocol("structural sweep result root is invalid"));
    };
    expected
        .artifact_access
        .validate_reference(
            root,
            ValueRefKind::ResultObject,
            PUBLICATION_MEDIA_TYPE,
            PUBLICATION_VALUE_SCHEMA,
        )
        .map_err(protocol)?;
    let bytes = read(source, root)?;
    let publication = ExactStructuralSweepPublication::decode(&bytes)?;
    if publication.program_revision != expected.program_revision
        || publication.identity != expected.identity()?
        || publication.entries.len() != expected.studies.len()
    {
        return Err(protocol("structural sweep publication identity changed"));
    }
    let mut expected_inventory = BTreeMap::new();
    insert_reference(&mut expected_inventory, (**root).clone())?;
    let mut statuses = Vec::with_capacity(publication.entries.len());
    for entry in &publication.entries {
        let status = match entry {
            PublishedSweepEntry::Succeeded { response } => {
                let ProgramExecutionResponse::ExternalizedSuccess { result_objects, .. } =
                    &response
                else {
                    return Err(protocol("successful sweep entry is not externalized"));
                };
                for reference in result_objects {
                    insert_reference(&mut expected_inventory, reference.clone())?;
                }
                ExactStructuralSweepEntryStatus::Succeeded
            }
            PublishedSweepEntry::Failed { .. } => ExactStructuralSweepEntryStatus::Failed,
            PublishedSweepEntry::Cancelled => ExactStructuralSweepEntryStatus::Cancelled,
            PublishedSweepEntry::Indeterminate { .. } => {
                ExactStructuralSweepEntryStatus::Indeterminate
            }
        };
        statuses.push(status);
    }
    if expected_inventory.into_values().collect::<Vec<_>>() != result_objects {
        return Err(protocol(
            "structural sweep result inventory differs from its publication closure",
        ));
    }
    Ok(ExactStructuralSweepResultCatalog {
        identity: publication.identity,
        statuses,
        entries: publication.entries,
        expected_studies: expected.studies.clone(),
    })
}

fn read(source: &impl CacheImport, reference: &ValueRef) -> NativeExecutionResult<Vec<u8>> {
    let bytes = source
        .read_verified(reference.logical_digest)
        .map_err(protocol)?
        .ok_or_else(|| protocol("structural sweep result object is unavailable"))?;
    if reference.logical_digest != runmat_execution::Digest::sha256(&bytes)
        || reference.encoded_length != bytes.len() as u64
    {
        return Err(protocol(
            "structural sweep result differs from its object reference",
        ));
    }
    Ok(bytes)
}

fn insert_reference(
    inventory: &mut BTreeMap<runmat_execution::identity::ValueId, ValueRef>,
    reference: ValueRef,
) -> NativeExecutionResult<()> {
    if let Some(existing) = inventory.get(&reference.id) {
        return if existing == &reference {
            Ok(())
        } else {
            Err(protocol(
                "structural sweep result identity has conflicting metadata",
            ))
        };
    }
    if inventory
        .values()
        .any(|existing| existing.logical_digest == reference.logical_digest)
    {
        return Err(protocol(
            "structural sweep result inventory contains duplicate references",
        ));
    }
    inventory.insert(reference.id, reference);
    Ok(())
}
