//! Closed nonlinear completion transport over shared immutable analysis objects.
//!
//! The trusted root digest is supplied by the caller, not learned from the publication.
//! Encoding and import are bounded, but gather metadata and mesh; this is not a peak-memory
//! qualification or a replacement for scheduler attestation of response/accounting pairs.
use crate::study_closure::{check_inventory, completion_codec_limits, read, unique};
use std::collections::BTreeMap;

use runmat_analysis_fea::{
    execution::{
        AnalysisNumericalObjectKind, AnalysisObjectManifest, NonlinearSolidWorkload,
        ANALYSIS_MANIFEST_MEDIA_TYPE, ANALYSIS_MANIFEST_VALUE_SCHEMA,
    },
    study::{
        plan_nonlinear_solid_study_product, CompiledAnalysisStudy, NonlinearSolidStudyProduct,
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

use super::{
    limit, CompletedNonlinearSolidStudy, NonlinearSolidStudyPreparation,
    NonlinearSolidStudyResultClosure,
};
use crate::{
    import_analysis_objects, prepare_analysis_objects, AnalysisBridgeError, AnalysisBridgeResult,
    AnalysisOutputPolicy,
};

mod resources;
mod wire;
pub(super) use resources::{admit_encoding, retained_storage};
pub(super) use wire::ReplayContext;
use wire::{CompletionWire, PreparationWire};

const PREFIX: &[u8] = b"runmat-nonlinear-solid-study-completion\0";
// Full original task history replaces the previous winning-attempt payload.
const COMPLETION_SCHEMA_VERSION: u16 = 2;
const LIMITS: CanonicalLimits = CanonicalLimits::new(256 * 1024 * 1024, 8_000_000, 65_536, 64);

/// New objects must be written to the shared store. Remaining catalog members already belong
/// to completed phase responses. Copy exactly `catalog.objects` to obtain a cold closure.
pub struct PreparedNonlinearSolidStudyRoot {
    pub root: ValueRef,
    pub objects: Vec<LogicalObject>,
    pub catalog: PublishedObjectCatalog,
}

pub fn prepare_nonlinear_solid_study_root(
    completed: &CompletedNonlinearSolidStudy,
) -> AnalysisBridgeResult<PreparedNonlinearSolidStudyRoot> {
    admit_encoding(completed)?;
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
        "nonlinear-solid-study-completion",
        &wire,
        completion_codec_limits(policy.inventory_limits()),
    )?;
    let objects = prepare_analysis_objects(
        AnalysisNumericalObjectKind::NonlinearSolidStudyCompletion,
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
        if let Some(previous) = by_digest.get(&object.descriptor.digest) {
            if previous != &object {
                return Err(invalid("conflicting nonlinear closure object"));
            }
        } else {
            by_digest.insert(object.descriptor.digest, object);
        }
    }
    Ok(PreparedNonlinearSolidStudyRoot {
        root,
        objects: by_digest.into_values().collect(),
        catalog,
    })
}

pub fn import_nonlinear_solid_study_root(
    source: &impl CacheImport,
    root: &ValueRef,
    expected_root: Digest,
    access: &ArtifactAccess,
    limits: ObjectInventoryLimits,
    maximum_resident_bytes: u64,
) -> AnalysisBridgeResult<CompletedNonlinearSolidStudy> {
    if maximum_resident_bytes == 0 {
        return Err(invalid("nonlinear root requires caller decode admission"));
    }
    if root.logical_digest != expected_root {
        return Err(invalid("nonlinear root differs from trusted digest"));
    }
    access.validate_reference(
        root,
        ValueRefKind::ResultObject,
        ANALYSIS_MANIFEST_MEDIA_TYPE,
        ANALYSIS_MANIFEST_VALUE_SCHEMA,
    )?;
    let read_limits = ObjectInventoryLimits {
        max_object_bytes: limits.max_object_bytes.min(maximum_resident_bytes),
        ..limits
    };
    let manifest_bytes = read(source, root, read_limits)?;
    let manifest_resident =
        runmat_canonical_codec::canonical_payload_typed_resident_bytes_estimate(&manifest_bytes)?;
    if manifest_resident > maximum_resident_bytes {
        return Err(limit(
            "nonlinear root manifest exceeds caller decode admission",
        ));
    }
    let manifest = AnalysisObjectManifest::from_canonical_bytes(&manifest_bytes)?;
    let payload_limits = ObjectInventoryLimits {
        max_total_bytes: limits.max_total_bytes.min(maximum_resident_bytes),
        ..limits
    };
    let objects = import_analysis_objects(
        source,
        root,
        AnalysisNumericalObjectKind::NonlinearSolidStudyCompletion,
        manifest.logical_identity,
        manifest.payload_digest,
        access,
        payload_limits,
    )?;
    let payload = objects.payload();
    if manifest.logical_identity != Digest::sha256(&payload) {
        return Err(invalid("nonlinear completion payload identity differs"));
    }
    let decode_bytes =
        runmat_canonical_codec::canonical_payload_typed_resident_bytes_estimate(&payload)?
            .checked_add(objects.retained_storage_bytes()?)
            .and_then(|n| n.checked_add(manifest_resident))
            .ok_or_else(|| limit("nonlinear root decode storage overflow"))?;
    if decode_bytes > maximum_resident_bytes {
        return Err(limit(
            "nonlinear root decode exceeds caller memory admission",
        ));
    }
    let wire: CompletionWire = decode_contract(
        PREFIX,
        "nonlinear-solid-study-completion",
        &payload,
        completion_codec_limits(limits),
    )?;
    if wire.schema_version != COMPLETION_SCHEMA_VERSION {
        return Err(invalid("unsupported nonlinear completion schema"));
    }
    let [step] = wire.preparation.workload.model.sequence.steps.as_slice() else {
        return Err(invalid("nonlinear root requires original standalone step"));
    };
    if decode_bytes > step.accuracy.resources.resident_byte_budget() {
        return Err(limit(
            "nonlinear root decode exceeds original authored memory",
        ));
    }
    let unavailable_bytes = step
        .accuracy
        .resources
        .resident_byte_budget()
        .saturating_sub(maximum_resident_bytes);
    if wire.preparation.max_objects > limits.max_objects
        || wire.preparation.max_object_bytes > limits.max_object_bytes
        || wire.preparation.max_total_bytes > limits.max_total_bytes
    {
        return Err(invalid(
            "nonlinear replay policy exceeds caller import limits",
        ));
    }
    wire.descendants.validate()?;
    let mut all = wire.descendants.objects.clone();
    all.extend(objects.unique_references(access, ValueRefKind::ResultObject)?);
    let all = unique(all)?;
    check_inventory(&all, access, limits)?;
    for reference in &wire.descendants.objects {
        read(source, reference, read_limits)?;
    }
    // Decode has finished. Wire preparation/closures transfer ownership to the
    // coordinator; they are not a second retained value tree. Release raw buffers
    // before replay and keep the still-owned comparison catalog independently charged.
    drop(all);
    drop(payload);
    drop(objects);
    drop(manifest);
    drop(manifest_bytes);
    let expected_descendants = wire.descendants;
    let catalog_resident = runmat_canonical_codec::profile_serialized(
        &expected_descendants,
        completion_codec_limits(limits),
    )?
    .materialization_resident_bytes_estimate;
    let replay_retained = unavailable_bytes
        .checked_add(catalog_resident)
        .ok_or_else(|| limit("nonlinear root retained replay overflow"))?;
    let preparation = wire.preparation.into_preparation(access.clone())?;
    let completed = super::completion::import_with_retained(
        source,
        preparation,
        wire.closures,
        replay_retained,
    )?;
    if descendant_catalog(&completed, access)? != expected_descendants {
        return Err(invalid(
            "nonlinear completion has missing, substituted or unrelated descendants",
        ));
    }
    Ok(completed)
}

fn descendant_catalog(
    completed: &CompletedNonlinearSolidStudy,
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
                "nonlinear completion must contain externalized phase responses",
            ));
        };
        for output in outputs {
            let ValuePayload::Object(root) = output else {
                return Err(invalid("nonlinear phase root is not an object"));
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

fn invalid(message: &str) -> AnalysisBridgeError {
    AnalysisBridgeError::Invalid(message.into())
}
