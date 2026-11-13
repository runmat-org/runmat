//! Shared complete immutable manifest/chunk inventory admission; numerical chunks remain lazy.
use super::AnalysisArtifactRoot;
use runmat_analysis_fea::execution::{
    AnalysisObjectManifest, ANALYSIS_CHUNK_MEDIA_TYPE, ANALYSIS_CHUNK_VALUE_SCHEMA,
    ANALYSIS_MANIFEST_MEDIA_TYPE, ANALYSIS_MANIFEST_VALUE_SCHEMA,
};
use runmat_execution::{
    value::{PublishedObjectCatalog, ValueRef},
    Digest,
};
use std::collections::{BTreeMap, BTreeSet};

pub(in crate::analysis) fn validate_closure(
    roots: &[&AnalysisArtifactRoot],
    catalog: &PublishedObjectCatalog,
    mut read: impl FnMut(Digest) -> Result<Option<Vec<u8>>, String>,
) -> Result<(), String> {
    // Different semantic roles may reuse one immutable physical object. The catalog
    // declares each object once; every role's typed expectation is still checked below.
    let mut declared_roots = roots
        .iter()
        .map(|root| root.reference().clone())
        .collect::<Vec<_>>();
    declared_roots.sort_by_key(|reference| reference.id);
    declared_roots.dedup();
    super::validate_complete_root_inventory(declared_roots, catalog)?;
    let objects = catalog
        .objects
        .iter()
        .map(|reference| (reference.logical_digest, reference))
        .collect::<BTreeMap<_, _>>();
    let mut expected = BTreeMap::<Digest, (u64, &str, &str)>::new();
    for root in roots {
        let reference = root.reference();
        if reference.value_schema != ANALYSIS_MANIFEST_VALUE_SCHEMA
            || reference.media_type != ANALYSIS_MANIFEST_MEDIA_TYPE
        {
            return Err("analysis artifact root is not an analysis manifest object".into());
        }
        insert_expected(&mut expected, reference)?;
        let bytes = read(reference.logical_digest)?
            .ok_or_else(|| "analysis artifact root is missing".to_string())?;
        if bytes.len() as u64 != reference.encoded_length
            || Digest::sha256(&bytes) != reference.logical_digest
        {
            return Err("analysis artifact root failed content verification".into());
        }
        let manifest = AnalysisObjectManifest::from_canonical_bytes(&bytes)
            .map_err(|error| error.to_string())?;
        if manifest.kind != root.expected_kind()
            || manifest.logical_identity != root.expected_logical_identity()
            || manifest.payload_digest != root.expected_payload_digest()
        {
            return Err("analysis artifact root differs from its typed expectation".into());
        }
        for chunk in &manifest.chunks {
            let declared = objects.get(&chunk.digest).copied().ok_or_else(|| {
                "analysis artifact manifest names an undeclared chunk".to_string()
            })?;
            if declared.encoded_length != chunk.encoded_length
                || declared.value_schema != ANALYSIS_CHUNK_VALUE_SCHEMA
                || declared.media_type != ANALYSIS_CHUNK_MEDIA_TYPE
            {
                return Err("analysis artifact chunk reference differs from its manifest".into());
            }
            insert_expected(&mut expected, declared)?;
        }
    }
    let declared = catalog
        .objects
        .iter()
        .map(|reference| reference.logical_digest)
        .collect::<BTreeSet<_>>();
    if declared != expected.keys().copied().collect() {
        return Err("analysis artifact catalog contains missing or unreferenced objects".into());
    }
    Ok(())
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
            Err("analysis artifact catalog aliases one digest with different metadata".into())
        }
        _ => Ok(()),
    }
}
