use super::{
    admit_exact_geometry_closure, encode_exact_evaluators, encode_exact_topology,
    encode_geometry_healing_report, EncodedExactGeometryClosure, ExactGeometryManifest,
    EXACT_EVALUATOR_MEDIA_TYPE, EXACT_GEOMETRY_MANIFEST_SCHEMA_VERSION, EXACT_TOPOLOGY_MEDIA_TYPE,
    GEOMETRY_HEALING_MEDIA_TYPE, KERNEL_REPRESENTATION_MEDIA_TYPE,
};
use crate::{
    ExactBRepTopology, ExactEvaluatorRegistry, GeometryContractError, GeometryDocument,
    GeometryHealingReport, GeometryModel, GeometryObjectRef, EXACT_BREP_MEDIA_TYPE,
    GEOMETRY_PRIMARY_ARTIFACT_SCHEMA_VERSION,
};

pub fn build_exact_geometry_closure(
    mut document: GeometryDocument,
    topology: &ExactBRepTopology,
    evaluators: &ExactEvaluatorRegistry,
    kernel_representation: Option<&[u8]>,
    healing_report: Option<&GeometryHealingReport>,
) -> Result<EncodedExactGeometryClosure, GeometryContractError> {
    let GeometryModel::ExactBRep { model } = &document.model else {
        return Err(GeometryContractError::invalid(
            "exact geometry closure",
            "faceted geometry cannot be packaged as an exact closure",
        ));
    };
    let topology_bytes = encode_exact_topology(topology, model)?;
    let evaluator_bytes = encode_exact_evaluators(evaluators, topology, model)?;
    let healing_bytes = healing_report
        .map(encode_geometry_healing_report)
        .transpose()?;
    let manifest = ExactGeometryManifest {
        schema_version: EXACT_GEOMETRY_MANIFEST_SCHEMA_VERSION,
        source_digest: document.source.content_digest,
        revision: document.revision.clone(),
        kernel_abi: model.kernel_abi.clone(),
        topology: reference(&topology_bytes, EXACT_TOPOLOGY_MEDIA_TYPE)?,
        evaluators: reference(&evaluator_bytes, EXACT_EVALUATOR_MEDIA_TYPE)?,
        kernel_representation: kernel_representation
            .map(|bytes| reference(bytes, KERNEL_REPRESENTATION_MEDIA_TYPE))
            .transpose()?,
        healing_report: healing_bytes
            .as_deref()
            .map(|bytes| reference(bytes, GEOMETRY_HEALING_MEDIA_TYPE))
            .transpose()?,
    };
    let manifest_bytes = manifest.canonical_encode()?;
    let GeometryModel::ExactBRep { model } = &mut document.model else {
        unreachable!()
    };
    model.artifact = reference(&manifest_bytes, EXACT_BREP_MEDIA_TYPE)?;
    document.validate()?;
    admit_exact_geometry_closure(
        &document,
        &manifest_bytes,
        &topology_bytes,
        &evaluator_bytes,
        kernel_representation,
        healing_bytes.as_deref(),
    )?;
    Ok(EncodedExactGeometryClosure {
        document,
        manifest,
        manifest_bytes,
        topology_bytes,
        evaluator_bytes,
        healing_bytes,
    })
}

fn reference(bytes: &[u8], media_type: &str) -> Result<GeometryObjectRef, GeometryContractError> {
    Ok(GeometryObjectRef {
        digest: crate::model::canonical::digest(bytes)?,
        encoded_length: bytes.len() as u64,
        media_type: media_type.into(),
        schema_version: GEOMETRY_PRIMARY_ARTIFACT_SCHEMA_VERSION,
    })
}
