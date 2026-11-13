//! Shared path-free frequency publication metadata on the generic object transport.
//! Physical arrays remain in their original independently addressable field objects.

use runmat_analysis_fea::execution::{
    AnalysisNumericalObjectKind, FrequencySamplePublicationAuthority,
    FrequencySamplePublicationManifest,
};
use runmat_execution_artifact::{object::ObjectInventoryLimits, ObjectNamespace};

use crate::{
    prepare_analysis_objects, AnalysisBridgeError, AnalysisBridgeResult, PreparedAnalysisObjects,
};

pub fn prepare_frequency_sample_publication_objects(
    publication: &FrequencySamplePublicationManifest,
    chunk_bytes: usize,
    namespace: ObjectNamespace,
    limits: ObjectInventoryLimits,
) -> AnalysisBridgeResult<PreparedAnalysisObjects> {
    let bytes = publication.canonical_bytes()?;
    prepare_analysis_objects(
        AnalysisNumericalObjectKind::FrequencySamplePublication,
        publication.identity().digest(),
        &bytes,
        chunk_bytes,
        namespace,
        limits,
    )
}

pub fn decode_prepared_frequency_sample_publication(
    prepared: &PreparedAnalysisObjects,
    authority: &FrequencySamplePublicationAuthority,
) -> AnalysisBridgeResult<FrequencySamplePublicationManifest> {
    if prepared.manifest.kind != AnalysisNumericalObjectKind::FrequencySamplePublication {
        return Err(AnalysisBridgeError::Invalid(
            "expected a frequency-sample publication object".into(),
        ));
    }
    let publication =
        FrequencySamplePublicationManifest::from_canonical_bytes(&prepared.payload())?;
    if prepared.manifest.logical_identity != publication.identity().digest() {
        return Err(AnalysisBridgeError::Identity(
            "frequency-sample publication has the wrong logical identity",
        ));
    }
    publication.validate_for(authority)?;
    Ok(publication)
}
