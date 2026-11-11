//! Closed fluid completion transport over shared immutable analysis objects.
//!
//! The trusted root digest is supplied by the caller, not learned from the publication.
//! Encoding and import are bounded, but gather metadata and mesh; this is not a peak-memory
//! qualification or a replacement for scheduler attestation of response/accounting pairs.
use crate::study_closure::{check_inventory, completion_codec_limits, read, unique};
use std::collections::BTreeMap;

use runmat_analysis_fea::{
    execution::{
        AnalysisNumericalObjectKind, AnalysisObjectManifest, FluidSteadyWorkload,
        ANALYSIS_MANIFEST_MEDIA_TYPE, ANALYSIS_MANIFEST_VALUE_SCHEMA,
    },
    study::{plan_fluid_study_product, CompiledAnalysisStudy, FluidStudyPlanningPolicy},
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
    import_fluid_study_completion, CompletedFluidStudy, FluidStudyPreparation,
    FluidStudyResultClosure,
};
use crate::{
    import_analysis_objects, prepare_analysis_objects, AnalysisBridgeError, AnalysisBridgeResult,
    AnalysisCapabilityRequirement, AnalysisOutputPolicy,
};

const PREFIX: &[u8] = b"runmat-fluid-study-completion\0";
// Full original task history replaces the previous winning-attempt payload.
const COMPLETION_SCHEMA_VERSION: u16 = 2;
/// The completion carries the solver mesh and workload of any admitted mesh, so collections are
/// bounded by the encoded-byte ceiling rather than a fixed item count.

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PreparationWire {
    #[serde(default = "runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial")]
    local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan,
    policy: FluidStudyPlanningPolicy,
    compiled: CompiledAnalysisStudy,
    workload: FluidSteadyWorkload,
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
    fn new(preparation: FluidStudyPreparation) -> Self {
        let limits = preparation.output_policy.inventory_limits();
        let workload = preparation.workload().clone();
        Self {
            local_execution: preparation.local_execution,
            policy: preparation.policy,
            compiled: preparation.product.into_compiled(),
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
    ) -> AnalysisBridgeResult<FluidStudyPreparation> {
        let output_policy = AnalysisOutputPolicy::new(
            self.chunk_bytes,
            ObjectInventoryLimits {
                max_objects: self.max_objects,
                max_object_bytes: self.max_object_bytes,
                max_total_bytes: self.max_total_bytes,
            },
        )?;
        if output_policy.chunk_bytes() != self.chunk_bytes {
            return Err(invalid("noncanonical fluid output chunk size"));
        }
        let product = plan_fluid_study_product(self.compiled, &self.mesh, self.policy)
            .map_err(|error| invalid(&error.to_string()))?;
        if *product.workload() != self.workload {
            return Err(invalid("fluid root workload differs from authored plan"));
        }
        Ok(FluidStudyPreparation {
            local_execution: self.local_execution,
            policy: self.policy,
            product,
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
    closures: Vec<FluidStudyResultClosure>,
    descendants: PublishedObjectCatalog,
}

/// New objects must be written to the shared store. Remaining catalog members already belong
/// to completed phase responses. Copy exactly `catalog.objects` to obtain a cold closure.
pub struct PreparedFluidStudyRoot {
    pub root: ValueRef,
    pub objects: Vec<LogicalObject>,
    pub catalog: PublishedObjectCatalog,
}

pub fn prepare_fluid_study_root(
    completed: &CompletedFluidStudy,
) -> AnalysisBridgeResult<PreparedFluidStudyRoot> {
    prepare_fluid_study_root_for(completed, completed.preparation()?)
}

/// As `prepare_fluid_study_root`, from the study's already validated preparation.
pub fn prepare_fluid_study_root_for(
    completed: &CompletedFluidStudy,
    preparation: FluidStudyPreparation,
) -> AnalysisBridgeResult<PreparedFluidStudyRoot> {
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
        "fluid-study-completion",
        &wire,
        completion_codec_limits(policy.inventory_limits()),
    )?;
    let objects = prepare_analysis_objects(
        AnalysisNumericalObjectKind::FluidStudyCompletion,
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
                return Err(invalid("conflicting fluid closure object"));
            }
        }
    }
    Ok(PreparedFluidStudyRoot {
        root,
        objects: by_digest.into_values().collect(),
        catalog,
    })
}

pub fn import_fluid_study_root(
    source: &impl CacheImport,
    root: &ValueRef,
    expected_root: Digest,
    access: &ArtifactAccess,
    limits: ObjectInventoryLimits,
) -> AnalysisBridgeResult<CompletedFluidStudy> {
    if root.logical_digest != expected_root {
        return Err(invalid("fluid root differs from trusted digest"));
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
        AnalysisNumericalObjectKind::FluidStudyCompletion,
        manifest.logical_identity,
        manifest.payload_digest,
        access,
        limits,
    )?;
    let payload = objects.payload();
    if manifest.logical_identity != Digest::sha256(&payload) {
        return Err(invalid("fluid completion payload identity differs"));
    }
    let wire: CompletionWire = decode_contract(
        PREFIX,
        "fluid-study-completion",
        &payload,
        completion_codec_limits(limits),
    )?;
    if wire.schema_version != COMPLETION_SCHEMA_VERSION {
        return Err(invalid("unsupported fluid completion schema"));
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
    let completed = import_fluid_study_completion(source, preparation, wire.closures)?;
    if descendant_catalog(&completed, access)? != expected_descendants {
        return Err(invalid(
            "fluid completion has missing, substituted or unrelated descendants",
        ));
    }
    Ok(completed)
}

fn descendant_catalog(
    completed: &CompletedFluidStudy,
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
                "fluid completion must contain externalized phase responses",
            ));
        };
        for output in outputs {
            let ValuePayload::Object(root) = output else {
                return Err(invalid("fluid phase root is not an object"));
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
