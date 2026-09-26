use super::*;

pub(crate) fn package_exact_structural_study_result(
    request: &ProgramExecutionRequest,
    decoded: &DecodedExactStructuralStudyRoot,
    meshing: &ExactMeshingDagRunResult,
    region_resolution: &runmat_analysis_execution::PreparedAnalysisObjects,
    analysis: ExactStructuralAnalysisPublication<'_>,
) -> NativeExecutionResult<(ProgramExecutionResponse, Vec<ProgramExecutionObject>)> {
    if let Some(compiled) = &decoded.compiled_study {
        compiled
            .validate_against(&meshing.artifact)
            .map_err(protocol)?;
        if decoded.workloads.authored_model().map_err(protocol)? != &compiled.model {
            return Err(protocol(
                "compiled study model differs from its executed structural workloads",
            ));
        }
    }
    let region_resolution_objects = region_resolution
        .references(&decoded.artifact_access, ValueRefKind::ResultObject)
        .map_err(protocol)?;
    let region_resolution_root = region_resolution_objects
        .first()
        .ok_or_else(|| protocol("topology-region resolution closure is empty"))?
        .clone();
    let input_closures = analysis
        .inputs()
        .iter()
        .map(|input| {
            input
                .objects
                .references(&decoded.artifact_access, ValueRefKind::ResultObject)
                .map_err(protocol)
        })
        .collect::<NativeExecutionResult<Vec<_>>>()?;
    let input_roots = input_closures
        .iter()
        .map(|objects| {
            objects
                .first()
                .cloned()
                .ok_or_else(|| protocol("input materialization closure is empty"))
        })
        .collect::<NativeExecutionResult<Vec<_>>>()?;
    let input_objects = unique_references(input_closures.into_iter().flatten())?;
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
        ExactStructuralAnalysisPublication::StructuralModalResponseSpectrum {
            completion, ..
        } => modal_response_spectrum_closure(completion, &decoded.artifact_access)?,
        ExactStructuralAnalysisPublication::StructuralRandomResponse { completion, .. } => {
            random_response_closure(completion, &decoded.artifact_access)?
        }
        ExactStructuralAnalysisPublication::StructuralBuckling { completion, .. } => {
            buckling_closure(completion, &decoded.artifact_access)?
        }
        ExactStructuralAnalysisPublication::StructuralPrestressedModal { completion, .. } => {
            prestressed_modal_closure(completion, &decoded.artifact_access)?
        }
        ExactStructuralAnalysisPublication::StructuralPrestressedFrequency {
            completion, ..
        } => prestressed_frequency_closure(completion, &decoded.artifact_access)?,
        ExactStructuralAnalysisPublication::StructuralPrestressedModalFrequency {
            completion,
            ..
        } => prestressed_modal_frequency_closure(completion, &decoded.artifact_access)?,
        ExactStructuralAnalysisPublication::StructuralFrequency { completion, .. } => {
            frequency_closure(completion, &decoded.artifact_access)?
        }
    };
    let publication = ExactStructuralStudyPublication {
        schema_version: 2,
        program_revision: request.recipe.program_revision.clone(),
        authored_study_digest: decoded.authored_study_digest,
        compiled_study: decoded.compiled_study.clone(),
        compiled_geometry_output: decoded.compiled_geometry_output.clone(),
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
        region_resolution_root,
        region_resolution_objects: region_resolution_objects.clone(),
        input_roots,
        input_objects: input_objects.clone(),
        analysis,
    };
    let mut new_objects = region_resolution
        .objects
        .iter()
        .zip(&region_resolution_objects)
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
        .chain(&region_resolution_objects)
        .chain(&input_objects)
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
    let (roots, objects) = numerical_closure_roots(
        [
            &completion.preload.objects,
            &completion.preload.numerical_receipt_objects,
            &completion.prestress_state.objects,
            &completion.buckling.objects,
            &completion.buckling.numerical_receipt_objects,
        ],
        access,
    )?;
    let [preload_result_root, preload_receipt_root, prestress_state_root, buckling_result_root, buckling_receipt_root] =
        roots;
    Ok(ExactStructuralStudyAnalysisClosure::StructuralBuckling {
        preload_result_root: Box::new(preload_result_root),
        preload_receipt_root: Box::new(preload_receipt_root),
        prestress_state_root: Box::new(prestress_state_root),
        buckling_result_root: Box::new(buckling_result_root),
        buckling_receipt_root: Box::new(buckling_receipt_root),
        objects,
    })
}

fn prestressed_modal_closure(
    completion: &crate::CompletedStructuralPrestressedModalStudy,
    access: &ArtifactAccess,
) -> NativeExecutionResult<ExactStructuralStudyAnalysisClosure> {
    let (roots, objects) = numerical_closure_roots(
        [
            &completion.preload.objects,
            &completion.preload.numerical_receipt_objects,
            &completion.prestress_state.objects,
            &completion.modal.objects,
            &completion.modal.numerical_receipt_objects,
        ],
        access,
    )?;
    let [preload_result_root, preload_receipt_root, prestress_state_root, modal_result_root, modal_receipt_root] =
        roots;
    Ok(
        ExactStructuralStudyAnalysisClosure::StructuralPrestressedModal {
            preload_result_root: Box::new(preload_result_root),
            preload_receipt_root: Box::new(preload_receipt_root),
            prestress_state_root: Box::new(prestress_state_root),
            modal_result_root: Box::new(modal_result_root),
            modal_receipt_root: Box::new(modal_receipt_root),
            objects,
        },
    )
}

fn prestressed_frequency_closure(
    completion: &crate::CompletedStructuralPrestressedFrequencyStudy,
    access: &ArtifactAccess,
) -> NativeExecutionResult<ExactStructuralStudyAnalysisClosure> {
    let preload_result_root = completion
        .preload
        .objects
        .root_reference(access, ValueRefKind::ResultObject)
        .map_err(protocol)?;
    let preload_receipt_root = completion
        .preload
        .numerical_receipt_objects
        .root_reference(access, ValueRefKind::ResultObject)
        .map_err(protocol)?;
    let prestress_state_root = completion
        .prestress_state
        .objects
        .root_reference(access, ValueRefKind::ResultObject)
        .map_err(protocol)?;
    let (frequency, frequency_objects) = frequency_components(&completion.frequency, access)?;
    let objects = unique_references(
        completion
            .preload
            .objects
            .unique_references(access, ValueRefKind::ResultObject)
            .map_err(protocol)?
            .into_iter()
            .chain(
                completion
                    .preload
                    .numerical_receipt_objects
                    .unique_references(access, ValueRefKind::ResultObject)
                    .map_err(protocol)?,
            )
            .chain(
                completion
                    .prestress_state
                    .objects
                    .unique_references(access, ValueRefKind::ResultObject)
                    .map_err(protocol)?,
            )
            .chain(frequency_objects),
    )?;
    Ok(
        ExactStructuralStudyAnalysisClosure::StructuralPrestressedFrequency {
            preload_result_root: Box::new(preload_result_root),
            preload_receipt_root: Box::new(preload_receipt_root),
            prestress_state_root: Box::new(prestress_state_root),
            frequency: Box::new(frequency),
            objects,
        },
    )
}

fn prestressed_modal_frequency_closure(
    completion: &crate::CompletedStructuralPrestressedModalFrequencyStudy,
    access: &ArtifactAccess,
) -> NativeExecutionResult<ExactStructuralStudyAnalysisClosure> {
    let (roots, prefix_objects) = numerical_closure_roots(
        [
            &completion.preload.objects,
            &completion.preload.numerical_receipt_objects,
            &completion.prestress_state.objects,
            &completion.modal.objects,
            &completion.modal.numerical_receipt_objects,
            &completion.basis_state.objects,
        ],
        access,
    )?;
    let [preload_result_root, preload_receipt_root, prestress_state_root, modal_result_root, modal_receipt_root, basis_state_root] =
        roots;
    let (frequency, frequency_objects) = frequency_components(&completion.frequency, access)?;
    let objects = unique_references(prefix_objects.into_iter().chain(frequency_objects))?;
    Ok(
        ExactStructuralStudyAnalysisClosure::StructuralPrestressedModalFrequency {
            preload_result_root: Box::new(preload_result_root),
            preload_receipt_root: Box::new(preload_receipt_root),
            prestress_state_root: Box::new(prestress_state_root),
            modal_result_root: Box::new(modal_result_root),
            modal_receipt_root: Box::new(modal_receipt_root),
            basis_state_root: Box::new(basis_state_root),
            frequency: Box::new(frequency),
            objects,
        },
    )
}

fn numerical_closure_roots<const N: usize>(
    components: [&runmat_analysis_execution::PreparedAnalysisObjects; N],
    access: &ArtifactAccess,
) -> NativeExecutionResult<([ValueRef; N], Vec<ValueRef>)> {
    let roots = components
        .iter()
        .map(|component| {
            component
                .root_reference(access, ValueRefKind::ResultObject)
                .map_err(protocol)
        })
        .collect::<NativeExecutionResult<Vec<_>>>()?
        .try_into()
        .map_err(|_| protocol("numerical closure has an invalid root inventory"))?;
    let objects = unique_references(
        components
            .into_iter()
            .map(|component| {
                component
                    .unique_references(access, ValueRefKind::ResultObject)
                    .map_err(protocol)
            })
            .collect::<NativeExecutionResult<Vec<_>>>()?
            .into_iter()
            .flatten(),
    )?;
    Ok((roots, objects))
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

fn modal_response_spectrum_closure(
    completion: &crate::CompletedStructuralModalResponseSpectrumStudy,
    access: &ArtifactAccess,
) -> NativeExecutionResult<ExactStructuralStudyAnalysisClosure> {
    let (roots, objects) = numerical_closure_roots(
        [
            &completion.modal.objects,
            &completion.modal.numerical_receipt_objects,
            &completion.basis_state.objects,
            &completion.response.result_objects,
            &completion.response.receipt_objects,
            &completion.accuracy.objects,
            &completion.publication_baseline_usage.objects,
            &completion.publication.objects,
            &completion.resource_usage.objects,
            &completion.response.field_recovery_objects,
            &completion.envelope_state.objects,
        ],
        access,
    )?;
    let [modal_result_root, modal_receipt_root, basis_state_root, response_result_root, response_receipt_root, response_accuracy_root, response_publication_baseline_usage_root, response_publication_root, response_resource_usage_root, field_recovery_root, envelope_state_root] =
        roots;
    Ok(
        ExactStructuralStudyAnalysisClosure::StructuralModalResponseSpectrum {
            modal_result_root: Box::new(modal_result_root),
            modal_receipt_root: Box::new(modal_receipt_root),
            basis_state_root: Box::new(basis_state_root),
            response_result_root: Box::new(response_result_root),
            response_receipt_root: Box::new(response_receipt_root),
            response_accuracy_root: Box::new(response_accuracy_root),
            response_publication_baseline_usage_root: Box::new(
                response_publication_baseline_usage_root,
            ),
            response_resource_usage_root: Box::new(response_resource_usage_root),
            response_publication_root: Box::new(response_publication_root),
            field_recovery_root: Box::new(field_recovery_root),
            envelope_state_root: Box::new(envelope_state_root),
            objects,
        },
    )
}

fn random_response_closure(
    completion: &crate::CompletedStructuralRandomResponseStudy,
    access: &ArtifactAccess,
) -> NativeExecutionResult<ExactStructuralStudyAnalysisClosure> {
    let (roots, objects) = numerical_closure_roots(
        [
            &completion.modal.objects,
            &completion.modal.numerical_receipt_objects,
            &completion.basis_state.objects,
            &completion.response.result_objects,
            &completion.response.receipt_objects,
            &completion.accuracy.objects,
            &completion.publication_baseline_usage.objects,
            &completion.publication.objects,
            &completion.resource_usage.objects,
            &completion.statistics_state.objects,
        ],
        access,
    )?;
    let [modal_result_root, modal_receipt_root, basis_state_root, response_result_root, response_receipt_root, response_accuracy_root, response_publication_baseline_usage_root, response_publication_root, response_resource_usage_root, statistics_state_root] =
        roots;
    Ok(
        ExactStructuralStudyAnalysisClosure::StructuralRandomResponse {
            modal_result_root: Box::new(modal_result_root),
            modal_receipt_root: Box::new(modal_receipt_root),
            basis_state_root: Box::new(basis_state_root),
            response_result_root: Box::new(response_result_root),
            response_receipt_root: Box::new(response_receipt_root),
            response_accuracy_root: Box::new(response_accuracy_root),
            response_publication_baseline_usage_root: Box::new(
                response_publication_baseline_usage_root,
            ),
            response_resource_usage_root: Box::new(response_resource_usage_root),
            response_publication_root: Box::new(response_publication_root),
            statistics_state_root: Box::new(statistics_state_root),
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
    let (frequency, objects) = frequency_components(completion, access)?;
    Ok(ExactStructuralStudyAnalysisClosure::StructuralFrequency {
        frequency: Box::new(frequency),
        objects,
    })
}

fn frequency_components(
    completion: &crate::CompletedStructuralFrequencyStudy,
    access: &ArtifactAccess,
) -> NativeExecutionResult<(ExactStructuralFrequencyClosure, Vec<ValueRef>)> {
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
    Ok((
        ExactStructuralFrequencyClosure {
            result_root: Box::new(result_root),
            receipt_root: Box::new(receipt_root),
            publication: Box::new(completion.publication.clone()),
            entries,
            histories,
            resource_usage_root: Box::new(resource_usage_root),
            objects: catalog.objects.clone(),
        },
        catalog.objects,
    ))
}
