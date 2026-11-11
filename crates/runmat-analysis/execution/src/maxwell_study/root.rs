//! Closed maxwell completion transport over shared immutable analysis objects.
//!
//! The trusted root digest is supplied by the caller, not learned from the publication.
//! Encoding and import are bounded, but gather metadata and mesh; this is not a peak-memory
//! qualification or a replacement for scheduler attestation of response/accounting pairs.
use std::collections::BTreeMap;

use crate::study_closure::completion_codec_limits;
use runmat_analysis_fea::{
    execution::{
        AnalysisNumericalObjectKind, AnalysisObjectManifest, MaxwellHarmonicWorkload,
        ANALYSIS_MANIFEST_MEDIA_TYPE, ANALYSIS_MANIFEST_VALUE_SCHEMA,
    },
    study::{
        plan_maxwell_study, CompiledAnalysisStudy, MaxwellStudyPlanningPolicy, MaxwellStudyProduct,
    },
};
use runmat_canonical_codec::{decode_contract, encode_contract};
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

use super::{
    import_maxwell_study_completion, CompletedMaxwellStudy, MaxwellStudyPreparation,
    MaxwellStudyResultClosure,
};
use crate::{
    import_analysis_objects, prepare_analysis_objects, AnalysisBridgeError, AnalysisBridgeResult,
    AnalysisCapabilityRequirement, AnalysisOutputPolicy,
};

const PREFIX: &[u8] = b"runmat-maxwell-study-completion\0";
// Full original task history replaces the previous winning-attempt payload.
const COMPLETION_SCHEMA_VERSION: u16 = 2;

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PreparationWire {
    #[serde(default = "runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial")]
    local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan,
    policy: MaxwellStudyPlanningPolicy,
    compiled: CompiledAnalysisStudy,
    workload: MaxwellHarmonicWorkload,
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
    fn new(preparation: MaxwellStudyPreparation) -> Self {
        let limits = preparation.output_policy.inventory_limits();
        let workload = preparation.workload().clone();
        Self {
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
        }
    }

    fn into_preparation(
        self,
        access: ArtifactAccess,
    ) -> AnalysisBridgeResult<MaxwellStudyPreparation> {
        let output_policy = AnalysisOutputPolicy::new(
            self.chunk_bytes,
            ObjectInventoryLimits {
                max_objects: self.max_objects,
                max_object_bytes: self.max_object_bytes,
                max_total_bytes: self.max_total_bytes,
            },
        )?;
        if output_policy.chunk_bytes() != self.chunk_bytes {
            return Err(invalid("noncanonical maxwell output chunk size"));
        }
        let workload = plan_maxwell_study(&self.compiled, self.policy)
            .map_err(|error| invalid(&error.to_string()))?;
        if workload != self.workload {
            return Err(invalid("maxwell root workload differs from authored plan"));
        }
        Ok(MaxwellStudyPreparation {
            local_execution: self.local_execution,
            policy: self.policy,
            product: MaxwellStudyProduct {
                compiled: self.compiled,
                workload,
            },
            mesh: self.mesh,
            frozen_project_digest: self.frozen_project_digest,
            artifact_access: access,
            program_revision: self.program_revision,
            output_policy,
            required_capabilities: self.required_capabilities,
        })
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct CompletionWire {
    schema_version: u16,
    preparation: PreparationWire,
    closures: Vec<MaxwellStudyResultClosure>,
    descendants: PublishedObjectCatalog,
}

/// New objects must be written to the shared store. Remaining catalog members already belong
/// to completed phase responses. Copy exactly `catalog.objects` to obtain a cold closure.
pub struct PreparedMaxwellStudyRoot {
    pub root: ValueRef,
    pub objects: Vec<LogicalObject>,
    pub catalog: PublishedObjectCatalog,
}

pub fn prepare_maxwell_study_root(
    completed: &CompletedMaxwellStudy,
) -> AnalysisBridgeResult<PreparedMaxwellStudyRoot> {
    let preparation = completed.preparation()?;
    let access = preparation.artifact_access.clone();
    let policy = preparation.output_policy;
    let descendants = descendant_catalog(completed, &access)?;
    check_inventory(&descendants.objects, &access, policy.inventory_limits())?;
    let wire = CompletionWire {
        schema_version: COMPLETION_SCHEMA_VERSION,
        preparation: PreparationWire::new(preparation),
        closures: completed.result_closures().to_vec(),
        descendants,
    };
    let bytes = encode_contract(
        PREFIX,
        "maxwell-study-completion",
        &wire,
        completion_codec_limits(policy.inventory_limits()),
    )?;
    let objects = prepare_analysis_objects(
        AnalysisNumericalObjectKind::MaxwellStudyCompletion,
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
    new_objects.extend(completed.baseline_usage.objects.objects.iter().cloned());
    new_objects.extend(completed.terminal_usage.objects.objects.iter().cloned());
    let mut by_digest = BTreeMap::new();
    for object in new_objects {
        object.validate()?;
        if let Some(previous) = by_digest.insert(object.descriptor.digest, object.clone()) {
            if previous != object {
                return Err(invalid("conflicting maxwell closure object"));
            }
        }
    }
    Ok(PreparedMaxwellStudyRoot {
        root,
        objects: by_digest.into_values().collect(),
        catalog,
    })
}

pub fn import_maxwell_study_root(
    source: &impl CacheImport,
    root: &ValueRef,
    expected_root: Digest,
    access: &ArtifactAccess,
    limits: ObjectInventoryLimits,
) -> AnalysisBridgeResult<CompletedMaxwellStudy> {
    if root.logical_digest != expected_root {
        return Err(invalid("maxwell root differs from trusted digest"));
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
        AnalysisNumericalObjectKind::MaxwellStudyCompletion,
        manifest.logical_identity,
        manifest.payload_digest,
        access,
        limits,
    )?;
    let payload = objects.payload();
    if manifest.logical_identity != Digest::sha256(&payload) {
        return Err(invalid("maxwell completion payload identity differs"));
    }
    let wire: CompletionWire = decode_contract(
        PREFIX,
        "maxwell-study-completion",
        &payload,
        completion_codec_limits(limits),
    )?;
    if wire.schema_version != COMPLETION_SCHEMA_VERSION {
        return Err(invalid("unsupported maxwell completion schema"));
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
    let completed = import_maxwell_study_completion(source, preparation, wire.closures)?;
    if descendant_catalog(&completed, access)? != expected_descendants {
        return Err(invalid(
            "maxwell completion has missing, substituted or unrelated descendants",
        ));
    }
    Ok(completed)
}

fn descendant_catalog(
    completed: &CompletedMaxwellStudy,
    access: &ArtifactAccess,
) -> AnalysisBridgeResult<PublishedObjectCatalog> {
    let mut roots = Vec::new();
    let mut objects = Vec::new();
    for closure in completed.result_closures() {
        let ProgramExecutionResponse::ExternalizedSuccess {
            outputs,
            result_objects,
        } = &closure.response
        else {
            return Err(invalid(
                "maxwell completion must contain externalized phase responses",
            ));
        };
        for output in outputs {
            let ValuePayload::Object(root) = output else {
                return Err(invalid("maxwell phase root is not an object"));
            };
            roots.push(root.as_ref().clone());
        }
        objects.extend(result_objects.iter().cloned());
    }
    for usage in [&completed.baseline_usage, &completed.terminal_usage] {
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
                return Err(invalid("maxwell closure reference identity conflict"));
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
        return Err(invalid("maxwell closure object limit exceeded"));
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
            .ok_or_else(|| invalid("maxwell closure byte count overflow"))?;
        if reference.encoded_length > limits.max_object_bytes || total > limits.max_total_bytes {
            return Err(invalid("maxwell closure byte limit exceeded"));
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
        return Err(invalid("maxwell object byte limit exceeded"));
    }
    let bytes = source
        .read_verified_bounded(reference.logical_digest, reference.encoded_length)?
        .ok_or_else(|| invalid("maxwell closure child is missing"))?;
    if bytes.len() as u64 != reference.encoded_length
        || Digest::sha256(&bytes) != reference.logical_digest
    {
        return Err(invalid("maxwell closure child bytes differ from reference"));
    }
    Ok(bytes)
}

fn invalid(message: &str) -> AnalysisBridgeError {
    AnalysisBridgeError::Invalid(message.into())
}
