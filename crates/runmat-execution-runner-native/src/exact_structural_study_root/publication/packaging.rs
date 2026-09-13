use super::*;

pub(crate) fn package_exact_structural_study_result(
    request: &ProgramExecutionRequest,
    decoded: &DecodedExactStructuralStudyRoot,
    meshing: &ExactMeshingDagRunResult,
    boundary: &runmat_analysis_execution::PreparedAnalysisObjects,
    analysis: ExactStructuralAnalysisPublication<'_>,
) -> NativeExecutionResult<(ProgramExecutionResponse, Vec<ProgramExecutionObject>)> {
    let boundary_objects = boundary
        .references(&decoded.artifact_access, ValueRefKind::ResultObject)
        .map_err(protocol)?;
    let boundary_root = boundary_objects
        .first()
        .ok_or_else(|| protocol("boundary resolution closure is empty"))?
        .clone();
    let signal_closures = analysis
        .signals()
        .iter()
        .map(|signal| {
            signal
                .objects
                .references(&decoded.artifact_access, ValueRefKind::ResultObject)
                .map_err(protocol)
        })
        .collect::<NativeExecutionResult<Vec<_>>>()?;
    let signal_roots = signal_closures
        .iter()
        .map(|objects| {
            objects
                .first()
                .cloned()
                .ok_or_else(|| protocol("signal realization closure is empty"))
        })
        .collect::<NativeExecutionResult<Vec<_>>>()?;
    let signal_objects = unique_references(signal_closures.into_iter().flatten())?;
    let unresolved_workloads_digest = analysis.unresolved_workloads_digest();
    let analysis = match analysis {
        ExactStructuralAnalysisPublication::LinearSequence { sequence, .. } => {
            let closures = sequence.result_closures().map_err(protocol)?;
            let mut closure_objects = Vec::new();
            for closure in &closures {
                closure_objects.extend(response_objects(&closure.response)?);
            }
            let objects = unique_references(closure_objects)?;
            ExactStructuralStudyAnalysisClosure::LinearSequence { closures, objects }
        }
        ExactStructuralAnalysisPublication::DirectTransient { completion, .. } => {
            let result_root = completion
                .result
                .result_objects
                .root_reference(&decoded.artifact_access, ValueRefKind::ResultObject)
                .map_err(protocol)?;
            let receipt_root = completion
                .result
                .receipt_objects
                .root_reference(&decoded.artifact_access, ValueRefKind::ResultObject)
                .map_err(protocol)?;
            let objects = unique_references(
                completion
                    .result
                    .result_objects
                    .unique_references(&decoded.artifact_access, ValueRefKind::ResultObject)
                    .map_err(protocol)?
                    .into_iter()
                    .chain(
                        completion
                            .result
                            .receipt_objects
                            .unique_references(&decoded.artifact_access, ValueRefKind::ResultObject)
                            .map_err(protocol)?,
                    ),
            )?;
            ExactStructuralStudyAnalysisClosure::DirectTransient {
                result_root: Box::new(result_root),
                receipt_root: Box::new(receipt_root),
                objects,
            }
        }
    };
    let publication = ExactStructuralStudyPublication {
        schema_version: 1,
        program_revision: request.recipe.program_revision.clone(),
        authored_study_digest: decoded.authored_study_digest,
        unresolved_workloads_digest,
        meshing_request_digest: Digest::from_bytes(
            *decoded
                .meshing_request
                .canonical_digest()
                .map_err(protocol)?
                .bytes(),
        ),
        meshing_root: meshing.root.clone(),
        meshing_objects: meshing.result_objects.clone(),
        boundary_root,
        boundary_objects: boundary_objects.clone(),
        signal_roots,
        signal_objects: signal_objects.clone(),
        analysis,
    };
    let mut new_objects = boundary
        .objects
        .iter()
        .zip(&boundary_objects)
        .map(|(object, reference)| ProgramExecutionObject {
            reference: reference.clone(),
            bytes: object.bytes.clone(),
        })
        .collect::<Vec<_>>();
    let publication_object = LogicalObject::new(
        ObjectNamespace::ResultValue,
        "analysis/exact-structural-study-publications",
        PUBLICATION_MEDIA_TYPE,
        publication.canonical_bytes(&decoded.artifact_access)?,
    )
    .map_err(protocol)?;
    let publication_root = decoded
        .artifact_access
        .reference(
            &publication_object,
            ValueRefKind::ResultObject,
            PUBLICATION_VALUE_SCHEMA,
        )
        .map_err(protocol)?;
    let mut inventory = BTreeMap::new();
    let mut logical_digests = BTreeSet::new();
    for reference in meshing
        .result_objects
        .iter()
        .chain(&boundary_objects)
        .chain(&signal_objects)
        .chain(publication.analysis.objects())
        .chain(std::iter::once(&publication_root))
    {
        if inventory.insert(reference.id, reference.clone()).is_some()
            || !logical_digests.insert(reference.logical_digest)
        {
            return Err(protocol(
                "whole-study result inventory contains duplicate object references",
            ));
        }
    }
    let response = ProgramExecutionResponse::ExternalizedSuccess {
        outputs: vec![ValuePayload::Object(Box::new(publication_root.clone()))],
        result_objects: inventory.into_values().collect(),
    };
    response.validate_against(request).map_err(protocol)?;
    new_objects.push(ProgramExecutionObject {
        reference: publication_root,
        bytes: publication_object.bytes,
    });
    for object in &new_objects {
        object.validate().map_err(protocol)?;
    }
    Ok((response, new_objects))
}
