use std::collections::BTreeMap;

use runmat_execution::value::{ValuePayload, ValueRefKind};
use runmat_execution::Digest;
use runmat_execution_artifact::cache::CacheImport;
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_execution_artifact::ProgramExecutionResponse;

use super::{
    protocol, read, ExactStructuralStudyPublication, ExactStructuralStudyResult,
    ExactStructuralStudyResultExpectation, PUBLICATION_MEDIA_TYPE, PUBLICATION_VALUE_SCHEMA,
};
use crate::{NativeExecutionResult, StructuralStudyResultExpectation};

/// Imports a result whose exact workloads were derived remotely after geometry materialization.
///
/// The client retains the source contract. The returned geometry receipt is admitted against its
/// exact client-held request, then the study is recompiled locally over the imported mesh before
/// the ordinary exact-result importer admits any numerical closure.
pub fn import_structural_study_result(
    source: &impl CacheImport,
    response: ProgramExecutionResponse,
    expected: &StructuralStudyResultExpectation,
    limits: ObjectInventoryLimits,
) -> NativeExecutionResult<ExactStructuralStudyResult> {
    expected.validate()?;
    let planning = expected
        .planning
        .structural()
        .ok_or_else(|| protocol("family study results use the family importer"))?;
    let ProgramExecutionResponse::ExternalizedSuccess { outputs, .. } = &response else {
        return Err(protocol(
            "structural study execution did not return a result closure",
        ));
    };
    let [ValuePayload::Object(root)] = outputs.as_slice() else {
        return Err(protocol("structural study result root is invalid"));
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
    let publication =
        ExactStructuralStudyPublication::decode(&read(source, root)?, &expected.artifact_access)?;
    if publication.program_revision != expected.program_revision
        || publication.authored_study_digest != expected.authored_study_digest
    {
        return Err(protocol(
            "structural study result differs from its submitted source identity",
        ));
    }
    let compiled = publication
        .compiled_study
        .as_ref()
        .ok_or_else(|| protocol("structural study result omits its compiled study definition"))?;
    let geometry_output = publication
        .compiled_geometry_output
        .as_ref()
        .ok_or_else(|| {
            protocol("structural study result omits its geometry materialization receipt")
        })?;
    geometry_output
        .validate_against(&expected.geometry_request)
        .map_err(protocol)?;
    let mesh = runmat_meshing_execution::import_exact_meshing_result(
        source,
        publication.meshing_root.clone(),
        publication.meshing_objects.clone(),
        expected.artifact_access.clone(),
        limits,
    )
    .map_err(protocol)?;
    compiled
        .validate_against(&mesh.artifact)
        .map_err(protocol)?;
    let materials = expected
        .materials
        .iter()
        .map(|entry| (entry.reference.clone(), entry.material.clone()))
        .collect::<BTreeMap<_, _>>();
    let recompiled = runmat_analysis_fea::study::build_structural_study_product_with_fields(
        runmat_analysis_fea::study::StructuralStudyProductInput {
            study: &expected.study,
            geometry: &geometry_output.document,
            regions: &geometry_output.regions,
            materials: &materials,
            mesh: &mesh.artifact,
            planning,
        },
        &expected.initial_fields,
    )
    .map_err(protocol)?;
    if &recompiled.compiled != compiled {
        return Err(protocol(
            "structural study result differs from the independently compiled source contract",
        ));
    }
    let exact = ExactStructuralStudyResultExpectation {
        program_revision: expected.program_revision.clone(),
        artifact_access: expected.artifact_access.clone(),
        authored_study_digest: expected.authored_study_digest,
        frozen_project_digest: expected.frozen_project_digest,
        geometry_source_digest: Digest::from_bytes(*mesh.artifact.geometry.source_digest.bytes()),
        workloads: recompiled.workloads,
        local_execution: expected.local_execution,
        analysis_accelerator_requests: expected.analysis_accelerator_requests.clone(),
        input_materializations: expected.input_materializations.clone(),
        expected_meshing_request: expected.expected_meshing_request.clone(),
    };
    let result = super::import_exact_structural_study_result(source, response, &exact, limits)?;
    if result
        .compiled_study
        .as_ref()
        .map(|compiled| &compiled.model)
        != Some(
            result
                .resolved_workloads
                .authored_model()
                .map_err(protocol)?,
        )
    {
        return Err(protocol(
            "imported structural study differs from its materialized workload model",
        ));
    }
    Ok(result)
}
