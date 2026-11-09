//! Closed acoustic completion transport over shared immutable analysis objects.
//!
//! The trusted root digest is supplied by the caller, not learned from the publication.
//! Encoding and import are bounded, but gather metadata and mesh; this is not a peak-memory
//! qualification or a replacement for scheduler attestation of response/accounting pairs.
use std::collections::BTreeMap;

use runmat_analysis_fea::{
    execution::{
        AcousticFrequencyWorkload, AnalysisNumericalObjectKind, AnalysisObjectManifest,
        ANALYSIS_MANIFEST_MEDIA_TYPE, ANALYSIS_MANIFEST_VALUE_SCHEMA,
    },
    study::{
        plan_acoustic_study, AcousticStudyPlanningPolicy, AcousticStudyProduct,
        CompiledAnalysisStudy,
    },
};
use runmat_canonical_codec::{decode_contract, encode_contract, CanonicalLimits};
use runmat_execution::{
    value::{ValuePayload, ValueRef, ValueRefKind},
    Digest, ProgramRevision,
};
use runmat_execution_artifact::{
    cache::CacheImport, object::ObjectInventoryLimits, ArtifactAccess, LogicalObject,
    ObjectNamespace, ProgramExecutionResponse, PublishedObjectCatalog,
};
use runmat_meshing_core::SolverMeshArtifact;
use serde::{Deserialize, Serialize};

use super::completion::{
    AcousticStudyAssemblyClosure, AcousticStudyEntryClosure, AcousticStudyResultClosure,
};
use super::{import_acoustic_study_completion, AcousticStudyPreparation, CompletedAcousticStudy};
use crate::{
    import_analysis_objects, prepare_analysis_objects, AnalysisBridgeError, AnalysisBridgeResult,
    AnalysisCapabilityRequirement, AnalysisOutputPolicy,
};

const PREFIX: &[u8] = b"runmat-acoustic-study-completion\0";
// Full original task history, with the study-level history phase after the entries.
const COMPLETION_SCHEMA_VERSION: u16 = 4;
const LIMITS: CanonicalLimits = CanonicalLimits::new(256 * 1024 * 1024, 8_000_000, 65_536, 64);

#[cfg(test)]
mod tests;

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PreparationWire {
    #[serde(default = "runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial")]
    local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan,
    policy: AcousticStudyPlanningPolicy,
    compiled: CompiledAnalysisStudy,
    workload: AcousticFrequencyWorkload,
    mesh: SolverMeshArtifact,
    frozen_project_digest: Option<Digest>,
    program_revision: ProgramRevision,
    required_capabilities: std::collections::BTreeSet<AnalysisCapabilityRequirement>,
    chunk_bytes: usize,
    max_objects: usize,
    max_object_bytes: u64,
    max_total_bytes: u64,
}

impl PreparationWire {
    fn new(preparation: AcousticStudyPreparation) -> AnalysisBridgeResult<Self> {
        let limits = preparation.output_policy.inventory_limits();
        let workload = preparation.frequency_workload()?;
        Ok(Self {
            local_execution: preparation.local_execution,
            policy: preparation.policy,
            compiled: preparation.product.compiled,
            workload,
            mesh: preparation.mesh,
            frozen_project_digest: preparation.frozen_project_digest,
            program_revision: preparation.program_revision,
            required_capabilities: preparation.required_capabilities,
            chunk_bytes: preparation.output_policy.chunk_bytes(),
            max_objects: limits.max_objects,
            max_object_bytes: limits.max_object_bytes,
            max_total_bytes: limits.max_total_bytes,
        })
    }

    fn into_preparation(
        self,
        access: ArtifactAccess,
    ) -> AnalysisBridgeResult<AcousticStudyPreparation> {
        let output_policy = AnalysisOutputPolicy::new(
            self.chunk_bytes,
            ObjectInventoryLimits {
                max_objects: self.max_objects,
                max_object_bytes: self.max_object_bytes,
                max_total_bytes: self.max_total_bytes,
            },
        )?;
        if output_policy.chunk_bytes() != self.chunk_bytes {
            return Err(invalid("noncanonical acoustic output chunk size"));
        }
        let plan = plan_acoustic_study(&self.compiled, self.policy)
            .map_err(|error| invalid(&error.to_string()))?;
        let expected_workload = self.workload;
        let preparation = AcousticStudyPreparation {
            local_execution: self.local_execution,
            policy: self.policy,
            product: AcousticStudyProduct {
                compiled: self.compiled,
                plan,
            },
            mesh: self.mesh,
            frozen_project_digest: self.frozen_project_digest,
            artifact_access: access,
            program_revision: self.program_revision,
            output_policy,
            required_capabilities: self.required_capabilities,
        };
        if preparation.frequency_workload()? != expected_workload {
            return Err(invalid(
                "acoustic root workload differs from entire exact authored plan",
            ));
        }
        Ok(preparation)
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct CompletionWire {
    schema_version: u16,
    preparation: PreparationWire,
    entries: Vec<AcousticStudyEntryClosure>,
    histories: Vec<AcousticStudyResultClosure>,
    assembly: AcousticStudyAssemblyClosure,
    wave_wall_millis: u64,
    descendants: PublishedObjectCatalog,
}

/// New objects must be written to the shared store. Remaining catalog members already belong
/// to completed phase responses. Copy exactly `catalog.objects` to obtain a cold closure.
pub struct PreparedAcousticStudyRoot {
    pub root: ValueRef,
    pub objects: Vec<LogicalObject>,
    pub catalog: PublishedObjectCatalog,
}

pub fn prepare_acoustic_study_root(
    completed: &CompletedAcousticStudy,
) -> AnalysisBridgeResult<PreparedAcousticStudyRoot> {
    let preparation = completed.preparation()?;
    let access = preparation.artifact_access.clone();
    let policy = preparation.output_policy;
    let descendants = descendant_catalog(completed, &access)?;
    check_inventory(&descendants.objects, &access, policy.inventory_limits())?;
    let wire = CompletionWire {
        schema_version: COMPLETION_SCHEMA_VERSION,
        preparation: PreparationWire::new(preparation)?,
        entries: completed.entry_closures(),
        histories: completed.history_closures.clone(),
        assembly: completed.assembly_closure.clone(),
        wave_wall_millis: completed.wave_wall_millis,
        descendants,
    };
    let bytes = encode_contract(PREFIX, "acoustic-study-completion", &wire, LIMITS)?;
    let objects = prepare_analysis_objects(
        AnalysisNumericalObjectKind::AcousticStudyCompletion,
        Digest::sha256(&bytes),
        &bytes,
        policy.chunk_bytes(),
        ObjectNamespace::ResultValue,
        policy.inventory_limits(),
    )?;
    let root = objects.root_reference(&access, ValueRefKind::ResultObject)?;
    let mut inventory = wire.descendants.objects;
    inventory.extend(objects.unique_references(&access, ValueRefKind::ResultObject)?);
    let inventory = unique(inventory)?;
    check_inventory(&inventory, &access, policy.inventory_limits())?;
    let catalog = PublishedObjectCatalog::new(vec![root.clone()], inventory)?;
    let mut new_objects = objects.objects;
    for entry in &completed.entries {
        new_objects.extend(entry.baseline_usage.objects.objects.iter().cloned());
        new_objects.extend(entry.terminal_usage.objects.objects.iter().cloned());
    }
    new_objects.extend(completed.resource_usage.objects.objects.iter().cloned());
    let mut by_digest = BTreeMap::new();
    for object in new_objects {
        object.validate()?;
        if let Some(previous) = by_digest.insert(object.descriptor.digest, object.clone()) {
            if previous != object {
                return Err(invalid("conflicting acoustic closure object"));
            }
        }
    }
    Ok(PreparedAcousticStudyRoot {
        root,
        objects: by_digest.into_values().collect(),
        catalog,
    })
}

pub fn import_acoustic_study_root(
    source: &impl CacheImport,
    root: &ValueRef,
    expected_root: Digest,
    access: &ArtifactAccess,
    limits: ObjectInventoryLimits,
) -> AnalysisBridgeResult<CompletedAcousticStudy> {
    if root.logical_digest != expected_root {
        return Err(invalid("acoustic root differs from trusted digest"));
    }
    access.validate_reference(
        root,
        ValueRefKind::ResultObject,
        ANALYSIS_MANIFEST_MEDIA_TYPE,
        ANALYSIS_MANIFEST_VALUE_SCHEMA,
    )?;
    let admitted_source = crate::study_closure::CallerBoundedImport::new(source, limits)?;
    let source = &admitted_source;
    let manifest = AnalysisObjectManifest::from_canonical_bytes(&read(source, root, limits)?)?;
    let objects = import_analysis_objects(
        source,
        root,
        AnalysisNumericalObjectKind::AcousticStudyCompletion,
        manifest.logical_identity,
        manifest.payload_digest,
        access,
        limits,
    )?;
    let payload = objects.payload();
    if manifest.logical_identity != Digest::sha256(&payload) {
        return Err(invalid("acoustic completion payload identity differs"));
    }
    let wire: CompletionWire =
        decode_contract(PREFIX, "acoustic-study-completion", &payload, LIMITS)?;
    if wire.schema_version != COMPLETION_SCHEMA_VERSION {
        return Err(invalid("unsupported acoustic completion schema"));
    }
    wire.descendants.validate()?;
    let mut all = wire.descendants.objects.clone();
    all.extend(objects.unique_references(access, ValueRefKind::ResultObject)?);
    let all = unique(all)?;
    check_inventory(&all, access, limits)?;
    for reference in &wire.descendants.objects {
        read(source, reference, limits)?;
    }
    let expected_descendants = wire.descendants;
    let closed_source = admitted_source.close(&expected_descendants.objects, access)?;
    let source = &closed_source;
    let preparation = wire.preparation.into_preparation(access.clone())?;
    let completed = import_acoustic_study_completion(
        source,
        preparation,
        wire.entries,
        wire.histories,
        wire.assembly,
        wire.wave_wall_millis,
    )?;
    if descendant_catalog(&completed, access)? != expected_descendants {
        return Err(invalid(
            "acoustic completion has missing, substituted or unrelated descendants",
        ));
    }
    Ok(completed)
}

fn descendant_catalog(
    completed: &CompletedAcousticStudy,
    access: &ArtifactAccess,
) -> AnalysisBridgeResult<PublishedObjectCatalog> {
    let mut roots = Vec::new();
    let mut objects = Vec::new();
    let responses = completed
        .entries
        .iter()
        .flat_map(|entry| entry.closures.iter().map(|closure| &closure.response))
        .chain(
            completed
                .history_closures
                .iter()
                .map(|closure| &closure.response),
        )
        .chain(std::iter::once(&completed.assembly_closure.response));
    for response in responses {
        let ProgramExecutionResponse::ExternalizedSuccess {
            outputs,
            result_objects,
        } = response
        else {
            return Err(invalid(
                "acoustic completion must contain externalized phase responses",
            ));
        };
        for output in outputs {
            let ValuePayload::Object(root) = output else {
                return Err(invalid("acoustic phase root is not an object"));
            };
            roots.push(root.as_ref().clone());
        }
        objects.extend(result_objects.iter().cloned());
    }
    let usages = completed
        .entries
        .iter()
        .flat_map(|entry| [&entry.baseline_usage, &entry.terminal_usage])
        .chain(std::iter::once(&completed.resource_usage));
    for usage in usages {
        roots.push(
            usage
                .objects
                .root_reference(access, ValueRefKind::ResultObject)?,
        );
        objects.extend(
            usage
                .objects
                .unique_references(access, ValueRefKind::ResultObject)?,
        );
    }
    Ok(PublishedObjectCatalog::new(
        unique(roots)?,
        unique(objects)?,
    )?)
}

fn unique(references: Vec<ValueRef>) -> AnalysisBridgeResult<Vec<ValueRef>> {
    let mut by_id = BTreeMap::new();
    for reference in references {
        if let Some(previous) = by_id.insert(reference.id, reference.clone()) {
            if previous != reference {
                return Err(invalid("acoustic closure reference identity conflict"));
            }
        }
    }
    Ok(by_id.into_values().collect())
}

fn check_inventory(
    references: &[ValueRef],
    access: &ArtifactAccess,
    limits: ObjectInventoryLimits,
) -> AnalysisBridgeResult<()> {
    if references.len() > limits.max_objects {
        return Err(invalid("acoustic closure object limit exceeded"));
    }
    let mut total = 0_u64;
    for reference in references {
        access.validate_reference(
            reference,
            ValueRefKind::ResultObject,
            &reference.media_type,
            &reference.value_schema,
        )?;
        total = total
            .checked_add(reference.encoded_length)
            .ok_or_else(|| invalid("acoustic closure byte count overflow"))?;
        if reference.encoded_length > limits.max_object_bytes || total > limits.max_total_bytes {
            return Err(invalid("acoustic closure byte limit exceeded"));
        }
    }
    Ok(())
}

fn read(
    source: &impl CacheImport,
    reference: &ValueRef,
    limits: ObjectInventoryLimits,
) -> AnalysisBridgeResult<Vec<u8>> {
    if reference.encoded_length > limits.max_object_bytes {
        return Err(invalid("acoustic object byte limit exceeded"));
    }
    let bytes = source
        .read_verified_bounded(reference.logical_digest, reference.encoded_length)?
        .ok_or_else(|| invalid("acoustic closure child is missing"))?;
    if bytes.len() as u64 != reference.encoded_length
        || Digest::sha256(&bytes) != reference.logical_digest
    {
        return Err(invalid(
            "acoustic closure child bytes differ from reference",
        ));
    }
    Ok(bytes)
}

fn invalid(message: &str) -> AnalysisBridgeError {
    AnalysisBridgeError::Invalid(message.into())
}
