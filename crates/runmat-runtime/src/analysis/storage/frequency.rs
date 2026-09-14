use std::collections::{BTreeMap, BTreeSet};

use runmat_analysis_fea::execution::{
    AnalysisObjectManifest, ANALYSIS_CHUNK_MEDIA_TYPE, ANALYSIS_CHUNK_VALUE_SCHEMA,
    ANALYSIS_MANIFEST_MEDIA_TYPE, ANALYSIS_MANIFEST_VALUE_SCHEMA,
};
use runmat_execution::value::{PublishedObjectCatalog, PublishedObjectSource, ValueRef};
use runmat_execution::Digest;
use serde::{Deserialize, Serialize};

use crate::analysis::contracts::AnalysisArtifactRecord;
use crate::analysis::frequency_result::StructuralFrequencyArtifactRootIndex;
use crate::analysis::{
    result_policy, run_operation_for_identity, AnalysisFieldRequestOptions, AnalysisFieldSelection,
    AnalysisStudyStepNumericalResultData, AnalysisStudyStepRunData, QualityGate,
};

use super::{validate_run_id_token, AnalysisFieldPageLookup};

pub(super) const FREQUENCY_ARTIFACT_SCHEMA_VERSION: &str =
    "analysis_structural_frequency_artifact/2026-09";

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct PersistedStructuralFrequencyArtifact {
    pub schema_version: String,
    pub created_at: String,
    pub op_version: String,
    pub step: AnalysisStudyStepRunData,
    pub object_catalog: PublishedObjectCatalog,
    pub artifact_roots: StructuralFrequencyArtifactRootIndex,
}

pub(super) fn prepare_artifact(
    step: &AnalysisStudyStepRunData,
    object_catalog: &PublishedObjectCatalog,
    artifact_roots: &StructuralFrequencyArtifactRootIndex,
    created_at: String,
) -> Result<PersistedStructuralFrequencyArtifact, String> {
    let artifact = PersistedStructuralFrequencyArtifact {
        schema_version: FREQUENCY_ARTIFACT_SCHEMA_VERSION.to_string(),
        created_at,
        op_version: step.run_op_version.clone(),
        step: step.clone(),
        object_catalog: object_catalog.clone(),
        artifact_roots: artifact_roots.clone(),
    };
    validate_artifact(&artifact, &step.run_id)?;
    Ok(artifact)
}

pub(super) fn decode_artifact(
    bytes: &[u8],
    expected_run_id: &str,
) -> Result<PersistedStructuralFrequencyArtifact, String> {
    let artifact = serde_json::from_slice::<PersistedStructuralFrequencyArtifact>(bytes)
        .map_err(|error| format!("failed to parse structural frequency artifact: {error}"))?;
    validate_artifact(&artifact, expected_run_id)?;
    Ok(artifact)
}

pub(super) fn validate_artifact(
    artifact: &PersistedStructuralFrequencyArtifact,
    expected_run_id: &str,
) -> Result<(), String> {
    validate_run_id_token(expected_run_id)?;
    if artifact.schema_version != FREQUENCY_ARTIFACT_SCHEMA_VERSION
        || artifact.step.run_id != expected_run_id
        || artifact.op_version != artifact.step.run_op_version
    {
        return Err(
            "structural frequency artifact schema, key, or operation version is invalid".into(),
        );
    }
    let AnalysisStudyStepNumericalResultData::StructuralFrequency(result) =
        &artifact.step.numerical_result
    else {
        return Err("structural frequency artifact contains another numerical result kind".into());
    };
    result
        .publication
        .validate()
        .map_err(|error| error.to_string())?;
    artifact
        .artifact_roots
        .validate(&result.publication, &artifact.object_catalog)?;
    let operation = run_operation_for_identity(&artifact.step.identity).ok_or_else(|| {
        "structural frequency result identity has no runtime operation".to_string()
    })?;
    if operation.name() != artifact.step.run_operation
        || operation.version() != artifact.step.run_op_version
    {
        return Err("structural frequency artifact operation differs from its run identity".into());
    }
    if artifact.step.numerically_acceptable
        != (artifact.step.solver_convergence == QualityGate::Pass
            && artifact.step.result_quality == QualityGate::Pass)
        || artifact.step.run_status
            != result_policy::classify_run_status(
                artifact.step.result_quality,
                artifact.step.numerically_acceptable,
            )
    {
        return Err("structural frequency artifact quality classification is inconsistent".into());
    }
    let (publishable, reasons) = result_policy::evaluate_identity_production_publication(
        &artifact.step.identity,
        artifact.step.numerically_acceptable,
    );
    if artifact.step.production_publishable != publishable
        || artifact.step.production_publication_reasons != reasons
    {
        return Err(
            "structural frequency artifact production classification is inconsistent".into(),
        );
    }
    Ok(())
}

pub(super) fn ingest_objects(
    artifact: &PersistedStructuralFrequencyArtifact,
    source: &dyn PublishedObjectSource,
) -> Result<BTreeMap<Digest, Vec<u8>>, String> {
    validate_artifact(artifact, &artifact.step.run_id)?;
    let mut objects = BTreeMap::new();
    for reference in &artifact.object_catalog.objects {
        let bytes = source.read_verified(reference)?;
        if bytes.len() as u64 != reference.encoded_length
            || Digest::sha256(&bytes) != reference.logical_digest
            || objects.insert(reference.logical_digest, bytes).is_some()
        {
            return Err(
                "structural frequency source differs from its declared object catalog".into(),
            );
        }
    }
    validate_closure(artifact, |digest| objects.get(&digest).cloned())?;
    Ok(objects)
}

pub(super) fn validate_closure(
    artifact: &PersistedStructuralFrequencyArtifact,
    mut read: impl FnMut(Digest) -> Option<Vec<u8>>,
) -> Result<(), String> {
    let mut expected = BTreeMap::<Digest, (u64, &str, &str)>::new();
    for root in all_roots(&artifact.artifact_roots) {
        let reference = root.reference();
        if reference.value_schema != ANALYSIS_MANIFEST_VALUE_SCHEMA
            || reference.media_type != ANALYSIS_MANIFEST_MEDIA_TYPE
        {
            return Err("structural frequency root is not an analysis manifest object".into());
        }
        insert_expected(&mut expected, reference)?;
        let bytes = read(reference.logical_digest)
            .ok_or_else(|| "structural frequency artifact root is missing".to_string())?;
        if bytes.len() as u64 != reference.encoded_length
            || Digest::sha256(&bytes) != reference.logical_digest
        {
            return Err("structural frequency artifact root failed content verification".into());
        }
        let manifest = AnalysisObjectManifest::from_canonical_bytes(&bytes)
            .map_err(|error| error.to_string())?;
        if manifest.kind != root.expected_kind()
            || manifest.logical_identity != root.expected_logical_identity()
            || manifest.payload_digest != root.expected_payload_digest()
        {
            return Err(
                "structural frequency artifact root differs from its typed expectation".into(),
            );
        }
        for chunk in &manifest.chunks {
            let declared = artifact
                .object_catalog
                .objects
                .iter()
                .find(|candidate| candidate.logical_digest == chunk.digest)
                .ok_or_else(|| {
                    "structural frequency manifest names an undeclared chunk".to_string()
                })?;
            if declared.encoded_length != chunk.encoded_length
                || declared.value_schema != ANALYSIS_CHUNK_VALUE_SCHEMA
                || declared.media_type != ANALYSIS_CHUNK_MEDIA_TYPE
            {
                return Err(
                    "structural frequency chunk reference differs from its manifest".into(),
                );
            }
            insert_expected(&mut expected, declared)?;
        }
    }
    let declared = artifact
        .object_catalog
        .objects
        .iter()
        .map(|reference| reference.logical_digest)
        .collect::<BTreeSet<_>>();
    if declared != expected.keys().copied().collect() {
        return Err("structural frequency catalog contains missing or unreferenced objects".into());
    }
    Ok(())
}

pub(super) fn same_content(
    left: &PersistedStructuralFrequencyArtifact,
    right: &PersistedStructuralFrequencyArtifact,
) -> bool {
    left.schema_version == right.schema_version
        && left.op_version == right.op_version
        && left.step == right.step
        && left.object_catalog == right.object_catalog
        && left.artifact_roots == right.artifact_roots
}

pub(super) fn field_page(
    artifact: &PersistedStructuralFrequencyArtifact,
    selection: &AnalysisFieldSelection,
    options: AnalysisFieldRequestOptions,
    read: impl FnMut(Digest) -> Result<Option<Vec<u8>>, String>,
) -> Result<AnalysisFieldPageLookup, String> {
    validate_artifact(artifact, &artifact.step.run_id)?;
    super::validate_field_id(&selection.field_id)?;
    let AnalysisStudyStepNumericalResultData::StructuralFrequency(result) =
        &artifact.step.numerical_result
    else {
        return Err("structural frequency artifact contains another result kind".into());
    };
    super::frequency_page::load_page(
        &artifact.step.run_id,
        &artifact.artifact_roots,
        &result.publication,
        selection,
        options,
        read,
    )
}

pub(super) fn artifact_record(
    artifact: &PersistedStructuralFrequencyArtifact,
) -> AnalysisArtifactRecord {
    let field_ids = match &artifact.step.numerical_result {
        AnalysisStudyStepNumericalResultData::StructuralFrequency(result) => result
            .publication
            .samples()
            .iter()
            .flat_map(|sample| sample.fields.fields.iter())
            .map(|entry| entry.definition.field_id().to_string())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect(),
        AnalysisStudyStepNumericalResultData::Standard => Vec::new(),
    };
    AnalysisArtifactRecord {
        run_id: artifact.step.run_id.clone(),
        created_at: artifact.created_at.clone(),
        identity: artifact.step.identity.clone(),
        op_version: artifact.op_version.clone(),
        field_ids,
    }
}

fn all_roots(
    index: &StructuralFrequencyArtifactRootIndex,
) -> Vec<&crate::analysis::StructuralFrequencyArtifactRoot> {
    let mut roots = vec![index.result(), index.receipt(), index.resource_usage()];
    for sample in index.samples() {
        roots.push(sample.fields());
        if let Some(observations) = sample.observations() {
            roots.push(observations);
        }
    }
    roots.extend(index.histories().iter().map(|history| history.checkpoint()));
    roots
}

fn insert_expected<'a>(
    expected: &mut BTreeMap<Digest, (u64, &'a str, &'a str)>,
    reference: &'a ValueRef,
) -> Result<(), String> {
    let facts = (
        reference.encoded_length,
        reference.media_type.as_str(),
        reference.value_schema.as_str(),
    );
    match expected.insert(reference.logical_digest, facts) {
        Some(existing) if existing != facts => {
            Err("structural frequency catalog aliases one digest with different metadata".into())
        }
        _ => Ok(()),
    }
}
