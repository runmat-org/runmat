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
        ExactStructuralAnalysisPublication::StructuralTransient { completion, .. } => {
            transient_closure(
                &completion.result.result_objects,
                &completion.result.receipt_objects,
                &decoded.artifact_access,
            )?
        }
        ExactStructuralAnalysisPublication::StructuralModal { completion, .. } => {
            modal_closure(completion, &decoded.artifact_access)?
        }
        ExactStructuralAnalysisPublication::StructuralModalTransient { completion, .. } => {
            modal_transient_closure(completion, &decoded.artifact_access)?
        }
        ExactStructuralAnalysisPublication::StructuralBuckling { completion, .. } => {
            buckling_closure(completion, &decoded.artifact_access)?
        }
        ExactStructuralAnalysisPublication::StructuralFrequency { completion, .. } => {
            frequency_closure(completion, &decoded.artifact_access)?
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

fn buckling_closure(
    completion: &crate::CompletedStructuralBucklingSequenceStudy,
    access: &ArtifactAccess,
) -> NativeExecutionResult<ExactStructuralStudyAnalysisClosure> {
    let components = [
        &completion.preload.objects,
        &completion.preload.numerical_receipt_objects,
        &completion.prestress_state.objects,
        &completion.buckling.objects,
        &completion.buckling.numerical_receipt_objects,
    ];
    let mut roots = Vec::with_capacity(components.len());
    let mut objects = Vec::new();
    for component in components {
        roots.push(
            component
                .root_reference(access, ValueRefKind::ResultObject)
                .map_err(protocol)?,
        );
        objects.extend(
            component
                .unique_references(access, ValueRefKind::ResultObject)
                .map_err(protocol)?,
        );
    }
    let [preload_result_root, preload_receipt_root, prestress_state_root, buckling_result_root, buckling_receipt_root] =
        roots.as_slice()
    else {
        return Err(protocol("buckling closure has an invalid root inventory"));
    };
    Ok(ExactStructuralStudyAnalysisClosure::StructuralBuckling {
        preload_result_root: Box::new(preload_result_root.clone()),
        preload_receipt_root: Box::new(preload_receipt_root.clone()),
        prestress_state_root: Box::new(prestress_state_root.clone()),
        buckling_result_root: Box::new(buckling_result_root.clone()),
        buckling_receipt_root: Box::new(buckling_receipt_root.clone()),
        objects: unique_references(objects)?,
    })
}

fn modal_closure(
    completion: &crate::CompletedStructuralModalStudy,
    access: &ArtifactAccess,
) -> NativeExecutionResult<ExactStructuralStudyAnalysisClosure> {
    let result_root = completion
        .result
        .objects
        .root_reference(access, ValueRefKind::ResultObject)
        .map_err(protocol)?;
    let receipt_root = completion
        .result
        .numerical_receipt_objects
        .root_reference(access, ValueRefKind::ResultObject)
        .map_err(protocol)?;
    let objects = unique_references(
        completion
            .result
            .objects
            .unique_references(access, ValueRefKind::ResultObject)
            .map_err(protocol)?
            .into_iter()
            .chain(
                completion
                    .result
                    .numerical_receipt_objects
                    .unique_references(access, ValueRefKind::ResultObject)
                    .map_err(protocol)?,
            ),
    )?;
    Ok(ExactStructuralStudyAnalysisClosure::StructuralModal {
        result_root: Box::new(result_root),
        receipt_root: Box::new(receipt_root),
        objects,
    })
}

fn modal_transient_closure(
    completion: &crate::CompletedStructuralModalTransientStudy,
    access: &ArtifactAccess,
) -> NativeExecutionResult<ExactStructuralStudyAnalysisClosure> {
    let modal = modal_closure(
        &crate::CompletedStructuralModalStudy {
            result: completion.modal.clone(),
        },
        access,
    )?;
    let ExactStructuralStudyAnalysisClosure::StructuralModal {
        result_root: modal_result_root,
        receipt_root: modal_receipt_root,
        objects: modal_objects,
    } = modal
    else {
        return Err(protocol("modal closure has an invalid kind"));
    };
    let basis_state_root = completion
        .basis_state
        .objects
        .root_reference(access, ValueRefKind::ResultObject)
        .map_err(protocol)?;
    let transient_result_root = completion
        .transient
        .result
        .result_objects
        .root_reference(access, ValueRefKind::ResultObject)
        .map_err(protocol)?;
    let transient_receipt_root = completion
        .transient
        .result
        .receipt_objects
        .root_reference(access, ValueRefKind::ResultObject)
        .map_err(protocol)?;
    let objects = unique_references(
        modal_objects
            .into_iter()
            .chain(
                completion
                    .basis_state
                    .objects
                    .unique_references(access, ValueRefKind::ResultObject)
                    .map_err(protocol)?,
            )
            .chain(
                completion
                    .transient
                    .result
                    .result_objects
                    .unique_references(access, ValueRefKind::ResultObject)
                    .map_err(protocol)?,
            )
            .chain(
                completion
                    .transient
                    .result
                    .receipt_objects
                    .unique_references(access, ValueRefKind::ResultObject)
                    .map_err(protocol)?,
            ),
    )?;
    Ok(
        ExactStructuralStudyAnalysisClosure::StructuralModalTransient {
            modal_result_root,
            modal_receipt_root,
            basis_state_root: Box::new(basis_state_root),
            transient_result_root: Box::new(transient_result_root),
            transient_receipt_root: Box::new(transient_receipt_root),
            objects,
        },
    )
}

fn transient_closure(
    result_objects: &runmat_analysis_execution::PreparedAnalysisObjects,
    receipt_objects: &runmat_analysis_execution::PreparedAnalysisObjects,
    access: &ArtifactAccess,
) -> NativeExecutionResult<ExactStructuralStudyAnalysisClosure> {
    let result_root = result_objects
        .root_reference(access, ValueRefKind::ResultObject)
        .map_err(protocol)?;
    let receipt_root = receipt_objects
        .root_reference(access, ValueRefKind::ResultObject)
        .map_err(protocol)?;
    let objects = unique_references(
        result_objects
            .unique_references(access, ValueRefKind::ResultObject)
            .map_err(protocol)?
            .into_iter()
            .chain(
                receipt_objects
                    .unique_references(access, ValueRefKind::ResultObject)
                    .map_err(protocol)?,
            ),
    )?;
    Ok(ExactStructuralStudyAnalysisClosure::StructuralTransient {
        result_root: Box::new(result_root),
        receipt_root: Box::new(receipt_root),
        objects,
    })
}

fn frequency_closure(
    completion: &crate::CompletedStructuralFrequencyStudy,
    access: &ArtifactAccess,
) -> NativeExecutionResult<ExactStructuralStudyAnalysisClosure> {
    if completion.entries.len() != completion.observation_samples.len() {
        return Err(protocol(
            "structural frequency fields and observation samples have different inventories",
        ));
    }
    let (catalog, _) = completion.published_artifacts(access)?;
    let result_root = completion
        .result
        .result_objects
        .root_reference(access, ValueRefKind::ResultObject)
        .map_err(protocol)?;
    let receipt_root = completion
        .result
        .receipt_objects
        .root_reference(access, ValueRefKind::ResultObject)
        .map_err(protocol)?;
    let resource_usage_root = completion
        .resource_usage
        .objects
        .root_reference(access, ValueRefKind::ResultObject)
        .map_err(protocol)?;
    let mut entries = Vec::with_capacity(completion.entries.len());
    for (entry, observations) in completion
        .entries
        .iter()
        .zip(&completion.observation_samples)
    {
        let sample = runmat_analysis_fea::results::field::RecoveredFieldSample::FrequencyHertz(
            entry.entry.result.request().entry().frequency_hz(),
        );
        if observations.sample != sample {
            return Err(protocol(
                "structural frequency observations differ from their numerical entry",
            ));
        }
        let observations_root = observations
            .observations
            .as_ref()
            .map(|observations| observations.output.root.clone());
        entries.push(ExactStructuralFrequencyEntryClosure {
            entry: entry.entry.result.request().entry(),
            source_entry: entry.entry.result.manifest().clone(),
            physical_sample: entry.entry.result.sample().identity(),
            sample,
            fields_root: entry.recovered_fields.output.root.clone(),
            observations_root,
        });
    }
    let mut histories = Vec::with_capacity(completion.histories.len());
    for history in &completion.histories {
        let checkpoint_root = history
            .checkpoint
            .objects
            .root_reference(access, ValueRefKind::ResultObject)
            .map_err(protocol)?;
        histories.push(ExactStructuralFrequencyHistoryClosure {
            observation_id: history.observation_id.clone(),
            checkpoint_root,
        });
    }
    Ok(ExactStructuralStudyAnalysisClosure::StructuralFrequency {
        result_root: Box::new(result_root),
        receipt_root: Box::new(receipt_root),
        publication: Box::new(completion.publication.clone()),
        entries,
        histories,
        resource_usage_root: Box::new(resource_usage_root),
        objects: catalog.objects,
    })
}
