//! Closed thermal completion transport over shared immutable analysis objects.
//!
//! The trusted root digest is supplied by the caller, not learned from the publication.
//! Encoding and import are bounded, but gather metadata and mesh; this is not a peak-memory
//! qualification or a replacement for scheduler attestation of response/accounting pairs.
use std::collections::BTreeMap;

use crate::study_closure::completion_codec_limits;
use runmat_analysis_fea::{
    execution::{
        AnalysisNumericalObjectKind, AnalysisObjectManifest, RetainedJouleHeatSource,
        ThermalWorkload, ANALYSIS_MANIFEST_MEDIA_TYPE, ANALYSIS_MANIFEST_VALUE_SCHEMA,
    },
    study::{CompiledAnalysisStudy, ThermalStudyProduct},
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
    import_thermal_study_completion, CompletedThermalStudy, ThermalStudyPreparation,
    ThermalStudyResultClosure, ThermalStudyScope,
};
use crate::{
    import_analysis_objects, prepare_analysis_objects, AnalysisBridgeError, AnalysisBridgeResult,
    AnalysisCapabilityRequirement, AnalysisOutputPolicy, PreparedAnalysisObjects,
};

const PREFIX: &[u8] = b"runmat-thermal-study-completion\0";

#[cfg(test)]
mod tests;

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PreparationWire {
    #[serde(default = "runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial")]
    local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan,
    field_source: Option<FieldSourceWire>,
    scope: ThermalStudyScope,
    compiled: CompiledAnalysisStudy,
    workload: ThermalWorkload,
    mesh: SolverMeshArtifact,
    frozen_project_digest: Option<Digest>,
    program_revision: ProgramRevision,
    required_capabilities: std::collections::BTreeSet<AnalysisCapabilityRequirement>,
    chunk_bytes: usize,
    max_objects: usize,
    max_object_bytes: u64,
    max_total_bytes: u64,
}

/// Only immutable object authority belongs in the completion metadata. Numerical arrays remain
/// in the separately chunked source object, covered by the trusted root and descendant catalog.
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct FieldSourceWire {
    root: ValueRef,
    identity: Digest,
    payload_digest: Digest,
}

impl PreparationWire {
    fn new(preparation: ThermalStudyPreparation) -> Self {
        let limits = preparation.output_policy.inventory_limits();
        Self {
            local_execution: preparation.local_execution,
            field_source: None,
            scope: preparation.scope,
            compiled: preparation.product.compiled,
            workload: preparation.product.workload,
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

    #[cfg(test)]
    fn into_preparation(
        self,
        access: ArtifactAccess,
    ) -> AnalysisBridgeResult<ThermalStudyPreparation> {
        self.into_preparation_with_source(access, None)
    }

    fn into_preparation_with_source(
        self,
        access: ArtifactAccess,
        field_source: Option<RetainedJouleHeatSource>,
    ) -> AnalysisBridgeResult<ThermalStudyPreparation> {
        match (&self.field_source, &field_source) {
            (None, None) => {}
            (Some(binding), Some(source)) if binding.identity == source.identity()? => {}
            _ => {
                return Err(invalid(
                    "thermal completion source binding is missing or substituted",
                ))
            }
        }
        let output_policy = AnalysisOutputPolicy::new(
            self.chunk_bytes,
            ObjectInventoryLimits {
                max_objects: self.max_objects,
                max_object_bytes: self.max_object_bytes,
                max_total_bytes: self.max_total_bytes,
            },
        )?;
        if output_policy.chunk_bytes() != self.chunk_bytes {
            return Err(invalid("noncanonical thermal output chunk size"));
        }
        Ok(ThermalStudyPreparation {
            local_execution: self.local_execution,
            field_source,
            scope: self.scope,
            product: ThermalStudyProduct {
                compiled: self.compiled,
                workload: self.workload,
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
    closures: Vec<ThermalStudyResultClosure>,
    descendants: PublishedObjectCatalog,
}

/// New objects must be written to the shared store. Remaining catalog members already belong
/// to completed phase responses. Copy exactly `catalog.objects` to obtain a cold closure.
pub struct PreparedThermalStudyRoot {
    pub root: ValueRef,
    pub objects: Vec<LogicalObject>,
    pub catalog: PublishedObjectCatalog,
}

pub fn prepare_thermal_study_root(
    completed: &CompletedThermalStudy,
) -> AnalysisBridgeResult<PreparedThermalStudyRoot> {
    let preparation = completed.preparation();
    let access = preparation.artifact_access.clone();
    let policy = preparation.output_policy;
    let field_objects = prepare_field_source(&preparation)?;
    let descendants = descendant_catalog(completed, &access, field_objects.as_ref())?;
    check_inventory(&descendants.objects, &access, policy.inventory_limits())?;
    let mut preparation_wire = PreparationWire::new(preparation);
    preparation_wire.field_source = field_objects
        .as_ref()
        .map(|objects| -> AnalysisBridgeResult<FieldSourceWire> {
            Ok(FieldSourceWire {
                root: objects.root_reference(&access, ValueRefKind::ResultObject)?,
                identity: objects.manifest.logical_identity,
                payload_digest: objects.manifest.payload_digest,
            })
        })
        .transpose()?;
    let wire = CompletionWire {
        schema_version: 4,
        preparation: preparation_wire,
        closures: completed.result_closures().to_vec(),
        descendants,
    };
    let bytes = encode_contract(
        PREFIX,
        "thermal-study-completion",
        &wire,
        completion_codec_limits(policy.inventory_limits()),
    )?;
    let objects = prepare_analysis_objects(
        AnalysisNumericalObjectKind::ThermalStudyCompletion,
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
    if let Some(field_objects) = field_objects {
        new_objects.extend(field_objects.objects);
    }
    new_objects.extend(completed.baseline_usage.objects.objects.iter().cloned());
    new_objects.extend(completed.terminal_usage.objects.objects.iter().cloned());
    let mut by_digest = BTreeMap::new();
    for object in new_objects {
        object.validate()?;
        if let Some(previous) = by_digest.insert(object.descriptor.digest, object.clone()) {
            if previous != object {
                return Err(invalid("conflicting thermal closure object"));
            }
        }
    }
    Ok(PreparedThermalStudyRoot {
        root,
        objects: by_digest.into_values().collect(),
        catalog,
    })
}

pub fn import_thermal_study_root(
    source: &impl CacheImport,
    root: &ValueRef,
    expected_root: Digest,
    access: &ArtifactAccess,
    limits: ObjectInventoryLimits,
) -> AnalysisBridgeResult<CompletedThermalStudy> {
    if root.logical_digest != expected_root {
        return Err(invalid("thermal root differs from trusted digest"));
    }
    access.validate_reference(
        root,
        ValueRefKind::ResultObject,
        ANALYSIS_MANIFEST_MEDIA_TYPE,
        ANALYSIS_MANIFEST_VALUE_SCHEMA,
    )?;
    let manifest = AnalysisObjectManifest::from_canonical_bytes(&read(source, root, limits)?)?;
    let objects = import_analysis_objects(
        source,
        root,
        AnalysisNumericalObjectKind::ThermalStudyCompletion,
        manifest.logical_identity,
        manifest.payload_digest,
        access,
        limits,
    )?;
    let payload = objects.payload();
    if manifest.logical_identity != Digest::sha256(&payload) {
        return Err(invalid("thermal completion payload identity differs"));
    }
    let wire: CompletionWire = decode_contract(
        PREFIX,
        "thermal-study-completion",
        &payload,
        completion_codec_limits(limits),
    )?;
    if wire.schema_version != 4 {
        return Err(invalid("unsupported thermal completion schema"));
    }
    if wire.preparation.max_objects > limits.max_objects
        || wire.preparation.max_object_bytes > limits.max_object_bytes
        || wire.preparation.max_total_bytes > limits.max_total_bytes
    {
        return Err(invalid(
            "thermal replay policy exceeds caller import limits",
        ));
    }
    wire.descendants.validate()?;
    let mut all = wire.descendants.objects.clone();
    all.extend(objects.unique_references(access, ValueRefKind::ResultObject)?);
    let all = unique(all)?;
    check_inventory(&all, access, limits)?;
    for reference in &wire.descendants.objects {
        read(source, reference, limits)?;
    }
    let field_source = match &wire.preparation.field_source {
        None => None,
        Some(binding) => {
            if !wire.descendants.roots.contains(&binding.root) {
                return Err(invalid("thermal source root is missing from descendants"));
            }
            let source_objects = import_analysis_objects(
                source,
                &binding.root,
                AnalysisNumericalObjectKind::FieldSourceBinding,
                binding.identity,
                binding.payload_digest,
                access,
                limits,
            )?;
            let decoded = RetainedJouleHeatSource::from_canonical_bytes(&source_objects.payload())?;
            if decoded.identity()? != binding.identity {
                return Err(invalid("thermal field source identity differs"));
            }
            Some(decoded)
        }
    };
    let expected_descendants = wire.descendants;
    let preparation = wire
        .preparation
        .into_preparation_with_source(access.clone(), field_source)?;
    let completed = import_thermal_study_completion(source, preparation, wire.closures)?;
    let field_objects = prepare_field_source(&completed.preparation())?;
    if descendant_catalog(&completed, access, field_objects.as_ref())? != expected_descendants {
        return Err(invalid(
            "thermal completion has missing, substituted or unrelated descendants",
        ));
    }
    Ok(completed)
}

fn descendant_catalog(
    completed: &CompletedThermalStudy,
    access: &ArtifactAccess,
    field_source: Option<&PreparedAnalysisObjects>,
) -> AnalysisBridgeResult<PublishedObjectCatalog> {
    let mut roots = Vec::new();
    let mut objects = Vec::new();
    if let Some(field_source) = field_source {
        roots.push(field_source.root_reference(access, ValueRefKind::ResultObject)?);
        objects.extend(field_source.unique_references(access, ValueRefKind::ResultObject)?);
    }
    for closure in completed.result_closures() {
        let ProgramExecutionResponse::ExternalizedSuccess {
            outputs,
            result_objects,
        } = &closure.response
        else {
            return Err(invalid(
                "thermal completion must contain externalized phase responses",
            ));
        };
        for output in outputs {
            let ValuePayload::Object(root) = output else {
                return Err(invalid("thermal phase root is not an object"));
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

fn prepare_field_source(
    preparation: &ThermalStudyPreparation,
) -> AnalysisBridgeResult<Option<PreparedAnalysisObjects>> {
    preparation
        .field_source
        .as_ref()
        .map(|source| {
            source.validate_for(&preparation.product.workload, &preparation.mesh)?;
            prepare_analysis_objects(
                AnalysisNumericalObjectKind::FieldSourceBinding,
                source.identity()?,
                &source.canonical_bytes()?,
                preparation.output_policy.chunk_bytes(),
                ObjectNamespace::ResultValue,
                preparation.output_policy.inventory_limits(),
            )
        })
        .transpose()
}

fn unique(references: Vec<ValueRef>) -> AnalysisBridgeResult<Vec<ValueRef>> {
    let mut by_id = BTreeMap::new();
    for reference in references {
        if let Some(previous) = by_id.insert(reference.id, reference.clone()) {
            if previous != reference {
                return Err(invalid("thermal closure reference identity conflict"));
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
        return Err(invalid("thermal closure object limit exceeded"));
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
            .ok_or_else(|| invalid("thermal closure byte count overflow"))?;
        if reference.encoded_length > limits.max_object_bytes || total > limits.max_total_bytes {
            return Err(invalid("thermal closure byte limit exceeded"));
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
        return Err(invalid("thermal object byte limit exceeded"));
    }
    let bytes = source
        .read_verified(reference.logical_digest)?
        .ok_or_else(|| invalid("thermal closure child is missing"))?;
    if bytes.len() as u64 != reference.encoded_length
        || Digest::sha256(&bytes) != reference.logical_digest
    {
        return Err(invalid("thermal closure child bytes differ from reference"));
    }
    Ok(bytes)
}

fn invalid(message: &str) -> AnalysisBridgeError {
    AnalysisBridgeError::Invalid(message.into())
}
