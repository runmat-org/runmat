use super::*;

pub fn import_exact_structural_study_result(
    source: &impl CacheImport,
    response: ProgramExecutionResponse,
    expected: &ExactStructuralStudyResultExpectation,
    limits: ObjectInventoryLimits,
) -> NativeExecutionResult<ExactStructuralStudyResult> {
    expected.validate()?;
    let ProgramExecutionResponse::ExternalizedSuccess {
        outputs,
        result_objects,
    } = response
    else {
        return Err(protocol(
            "whole-study execution did not return a result closure",
        ));
    };
    let [ValuePayload::Object(root)] = outputs.as_slice() else {
        return Err(protocol("whole-study result root is invalid"));
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
    if publication.program_revision != expected.program_revision {
        return Err(protocol("whole-study program revision changed"));
    }
    if publication.authored_study_digest != expected.authored_study_digest {
        return Err(protocol("whole-study authored study identity changed"));
    }
    let expected_workloads_digest = structural_workload_inventory_digest(&expected.workloads)?;
    if publication.unresolved_workloads_digest != expected_workloads_digest {
        return Err(protocol("whole-study workload identity changed"));
    }
    let expected_meshing_request_digest = Digest::from_bytes(
        *expected
            .expected_meshing_request
            .canonical_digest()
            .map_err(protocol)?
            .bytes(),
    );
    if publication.meshing_request_digest != expected_meshing_request_digest {
        return Err(protocol("whole-study meshing request identity changed"));
    }
    let expected_inventory = unique_references(
        publication
            .meshing_objects
            .iter()
            .chain(&publication.region_resolution_objects)
            .chain(&publication.input_objects)
            .chain(publication.analysis.objects())
            .chain(
                publication
                    .thermal
                    .iter()
                    .flat_map(|thermal| thermal.catalog.objects.iter()),
            )
            .cloned()
            .chain(std::iter::once((**root).clone())),
    )?;
    if reference_inventory(&expected_inventory)? != reference_inventory(&result_objects)? {
        return Err(protocol(
            "whole-study result inventory differs from its canonical closure",
        ));
    }
    let mesh = runmat_meshing_execution::import_exact_meshing_result(
        source,
        publication.meshing_root,
        publication.meshing_objects,
        expected.artifact_access.clone(),
        limits,
    )
    .map_err(protocol)?;
    let mut expected_resolved_request = expected.expected_meshing_request.clone();
    if !crate::exact_geometry_admission::same_requested_tolerance(
        &expected_resolved_request.tolerance,
        &mesh.artifact.resolved_request.tolerance,
    ) {
        return Err(protocol(
            "whole-study mesh tolerance differs from the authored meshing request",
        ));
    }
    expected_resolved_request.tolerance = mesh.artifact.resolved_request.tolerance;
    if mesh.artifact.resolved_request != expected_resolved_request {
        return Err(protocol(
            "whole-study mesh differs from the authored meshing request",
        ));
    }
    if let Some(compiled) = &publication.compiled_study {
        compiled
            .validate_against(&mesh.artifact)
            .map_err(protocol)?;
    }
    let region_resolution_manifest_bytes = read(source, &publication.region_resolution_root)?;
    let region_resolution_manifest =
        runmat_analysis_fea::execution::AnalysisObjectManifest::from_canonical_bytes(
            &region_resolution_manifest_bytes,
        )
        .map_err(protocol)?;
    let region_resolution_objects = runmat_analysis_execution::import_analysis_objects(
        source,
        &publication.region_resolution_root,
        AnalysisNumericalObjectKind::TopologyResolution,
        region_resolution_manifest.logical_identity,
        region_resolution_manifest.payload_digest,
        &expected.artifact_access,
        limits,
    )
    .map_err(protocol)?;
    let expected_region_resolution_objects = region_resolution_objects
        .references(&expected.artifact_access, ValueRefKind::ResultObject)
        .map_err(protocol)?;
    if expected_region_resolution_objects != publication.region_resolution_objects {
        return Err(protocol(
            "topology-region resolution inventory differs from its canonical closure",
        ));
    }
    let region_resolution =
        TopologyRegionResolution::from_canonical_bytes(&region_resolution_objects.payload())?;
    if region_resolution.geometry_source_digest != expected.geometry_source_digest
        || region_resolution.solver_mesh_digest
            != Digest::from_bytes(*mesh.artifact.canonical_digest.bytes())
    {
        return Err(protocol(
            "topology-region resolution differs from the returned geometry or mesh",
        ));
    }
    if publication.input_roots.len() != expected.input_materializations.len() {
        return Err(protocol(
            "whole-study input result roots differ from the expected materializations",
        ));
    }
    let mut materialized_signals = BTreeMap::new();
    let mut materialized_responses = BTreeMap::new();
    let mut materialized_random = BTreeMap::new();
    let mut admitted_input_objects = Vec::new();
    for (root, prepared) in publication
        .input_roots
        .iter()
        .zip(&expected.input_materializations)
    {
        // Resolve from the retained original intent, never from returned admission facts.
        let prepared = prepared.resolve(&mesh.artifact).map_err(protocol)?;
        let manifest =
            runmat_analysis_fea::execution::AnalysisObjectManifest::from_canonical_bytes(&read(
                source, root,
            )?)
            .map_err(protocol)?;
        let kind = match &prepared {
            runmat_analysis_execution::InputMaterializationExpectation::Signal(_) => {
                AnalysisNumericalObjectKind::Signal
            }
            runmat_analysis_execution::InputMaterializationExpectation::Numerical(expected) => {
                match &expected.authored {
                    runmat_analysis_execution::NumericalGeneratorSource::ResponseSpectrum(_) => {
                        AnalysisNumericalObjectKind::ResponseSpectrum
                    }
                    runmat_analysis_execution::NumericalGeneratorSource::RandomSpectrum(_) => {
                        AnalysisNumericalObjectKind::RandomSpectrum
                    }
                }
            }
        };
        let objects = runmat_analysis_execution::import_analysis_objects(
            source,
            root,
            kind,
            manifest.logical_identity,
            manifest.payload_digest,
            &expected.artifact_access,
            limits,
        )
        .map_err(protocol)?;
        let references = objects
            .references(&expected.artifact_access, ValueRefKind::ResultObject)
            .map_err(protocol)?;
        for reference in references {
            if !publication.input_objects.contains(&reference) {
                return Err(protocol(
                    "input materialization closure contains an undeclared object",
                ));
            }
            admitted_input_objects.push(reference);
        }
        match &prepared {
            runmat_analysis_execution::InputMaterializationExpectation::Signal(prepared) => {
                prepared
                    .validate_for_mesh(&mesh.artifact)
                    .map_err(protocol)?;
                let realization =
                    runmat_analysis_fea::execution::CanonicalSignalRealization::from_canonical_bytes(
                        &objects.payload(),
                    )
                    .map_err(protocol)?;
                prepared
                    .validate_realization(&realization)
                    .map_err(protocol)?;
                let signal = runmat_analysis_execution::materialize_signal_for_solver(
                    &prepared.authored,
                    &prepared.admitted,
                    &realization,
                )
                .map_err(protocol)?;
                materialized_signals.insert(prepared.authored.signal_id().clone(), signal);
            }
            runmat_analysis_execution::InputMaterializationExpectation::Numerical(prepared) => {
                match &prepared.authored {
                    runmat_analysis_execution::NumericalGeneratorSource::ResponseSpectrum(
                        authored,
                    ) => {
                        let spectrum =
                            runmat_analysis_execution::decode_materialized_response_spectrum(
                                &objects.payload(),
                            )
                            .map_err(protocol)?;
                        let mut expected_spectrum = authored.as_ref().clone();
                        expected_spectrum.ordinates = spectrum.ordinates.clone();
                        if spectrum != expected_spectrum
                            || !matches!(
                                &spectrum.ordinates,
                                runmat_analysis_core::NumericalDataSource::Materialized {
                                    provenance,
                                    content_digest,
                                    ..
                                } if provenance == &prepared.admitted.invocation.provenance
                                    && content_digest == &manifest.logical_identity.to_string()
                            )
                        {
                            return Err(protocol(
                                "materialized response spectrum differs from its admission",
                            ));
                        }
                        materialized_responses.insert(spectrum.spectrum_id.clone(), spectrum);
                    }
                    runmat_analysis_execution::NumericalGeneratorSource::RandomSpectrum(
                        authored,
                    ) => {
                        let spectrum =
                            runmat_analysis_execution::decode_materialized_random_spectrum(
                                &objects.payload(),
                            )
                            .map_err(protocol)?;
                        let mut expected_spectrum = authored.as_ref().clone();
                        expected_spectrum.spectral_density = spectrum.spectral_density.clone();
                        if spectrum != expected_spectrum
                            || !matches!(
                                &spectrum.spectral_density,
                                runmat_analysis_core::NumericalDataSource::Materialized {
                                    provenance,
                                    content_digest,
                                    ..
                                } if provenance == &prepared.admitted.invocation.provenance
                                    && content_digest == &manifest.logical_identity.to_string()
                            )
                        {
                            return Err(protocol(
                                "materialized random spectrum differs from its admission",
                            ));
                        }
                        materialized_random.insert(spectrum.excitation_id.clone(), spectrum);
                    }
                }
            }
        }
    }
    let admitted_input_objects = unique_references(admitted_input_objects)?;
    if reference_inventory(&admitted_input_objects)?
        != reference_inventory(&publication.input_objects)?
    {
        return Err(protocol(
            "input materialization inventory contains unreferenced objects",
        ));
    }
    let mut resolved_model = expected
        .workloads
        .authored_model()
        .map_err(protocol)?
        .clone();
    resolved_model.signals = resolved_model
        .signals
        .iter()
        .map(|signal| {
            materialized_signals
                .get(signal.signal_id())
                .cloned()
                .unwrap_or_else(|| signal.clone())
        })
        .collect();
    if resolved_model
        .signals
        .iter()
        .any(|signal| signal.generator().is_some())
    {
        return Err(protocol(
            "whole-study result retained an unmaterialized signal generator",
        ));
    }
    for spectrum in &mut resolved_model.response_spectra {
        if let Some(materialized) = materialized_responses.get(&spectrum.spectrum_id) {
            *spectrum = materialized.clone();
        }
    }
    for spectrum in &mut resolved_model.random_excitations {
        if let Some(materialized) = materialized_random.get(&spectrum.excitation_id) {
            *spectrum = materialized.clone();
        }
    }
    if resolved_model
        .response_spectra
        .iter()
        .any(|spectrum| spectrum.ordinates.generator().is_some())
        || resolved_model
            .random_excitations
            .iter()
            .any(|spectrum| spectrum.spectral_density.generator().is_some())
    {
        return Err(protocol(
            "whole-study result retained an unmaterialized spectral generator",
        ));
    }
    let resolved_compiled_model = resolved_model.clone();
    let mut resolved_workloads = expected.workloads.clone();
    resolved_workloads
        .replace_model(resolved_model)
        .map_err(protocol)?;
    let retained_workloads = resolved_workloads.clone();
    let thermal = super::thermal_prefix::import(super::thermal_prefix::ImportContext {
        source,
        compiled: publication.compiled_study.as_ref(),
        analysis: &publication.analysis,
        thermal: publication.thermal.as_ref(),
        expected,
        workloads: &resolved_workloads,
        mesh: &mesh.artifact,
        limits,
    })?;
    let analysis = match (resolved_workloads, publication.analysis) {
        (
            StructuralStudyWorkloads::LinearSequence { workloads },
            ExactStructuralStudyAnalysisClosure::LinearSequence { closures, .. },
        ) => {
            let sequence = runmat_analysis_execution::import_structural_sequence_completion(
                source,
                StructuralSequencePreparation {
                    authored_study_digest: expected.authored_study_digest,
                    frozen_project_digest: expected.frozen_project_digest,
                    program_revision: expected.program_revision.clone(),
                    artifact_access: expected.artifact_access.clone(),
                    workloads: workloads.clone(),
                    mesh: Some(mesh.artifact.clone()),
                    chunk_bytes: runmat_analysis_fea::execution::DEFAULT_ANALYSIS_CHUNK_BYTES,
                    inventory_limits: limits,
                    required_capabilities: BTreeSet::new(),
                    accelerator_requests: expected.analysis_accelerator_requests.clone(),
                    local_execution: expected.local_execution,
                    solve_selection:
                        runmat_analysis_execution::StructuralSequenceSolveSelection::recorded(
                            &closures,
                        )
                        .map_err(protocol)?,
                },
                closures,
            )
            .map_err(protocol)?;
            ExactStructuralStudyAnalysisResult::LinearSequence {
                resolved_workloads: workloads,
                sequence,
            }
        }
        (
            StructuralStudyWorkloads::StructuralTransient {
                workload,
                block_plan: _,
            },
            ExactStructuralStudyAnalysisClosure::StructuralTransient {
                result_root,
                receipt_root,
                objects,
            },
        ) => {
            let result_objects = import_numerical_analysis_objects(
                source,
                &result_root,
                &objects,
                AnalysisNumericalObjectKind::SolverResult,
                &expected.artifact_access,
                limits,
            )?;
            let receipt_objects = import_numerical_analysis_objects(
                source,
                &receipt_root,
                &objects,
                AnalysisNumericalObjectKind::NumericalReceipt,
                &expected.artifact_access,
                limits,
            )?;
            validate_numerical_closure_inventory(
                &result_objects,
                &receipt_objects,
                &objects,
                &expected.artifact_access,
            )?;
            let result =
                runmat_analysis_execution::decode_prepared_structural_transient_solver_result(
                    &result_objects,
                )
                .map_err(protocol)?;
            let numerical_receipt =
                runmat_analysis_execution::decode_prepared_structural_transient_numerical_receipt(
                    &receipt_objects,
                )
                .map_err(protocol)?;
            numerical_receipt
                .validate_for_result(&result)
                .map_err(protocol)?;
            let expected_scope =
                runmat_analysis_fea::pipeline::transient::prepare_transient_checkpoint_scope(
                    &workload,
                    &mesh.artifact,
                    None,
                )
                .map_err(protocol)?;
            if result.manifest().scope != expected_scope {
                return Err(protocol(
                    "transient result names a different resolved workload",
                ));
            }
            ExactStructuralStudyAnalysisResult::StructuralTransient {
                resolved_workload: workload,
                result: Box::new(result),
                numerical_receipt,
            }
        }
        (
            StructuralStudyWorkloads::StructuralModal { workload },
            ExactStructuralStudyAnalysisClosure::StructuralModal {
                result_root,
                receipt_root,
                objects,
            },
        ) => super::modal_import::import_modal_result(super::modal_import::ModalResultImport {
            source,
            workload: *workload,
            result_root: *result_root,
            receipt_root: *receipt_root,
            objects,
            access: &expected.artifact_access,
            limits,
        })?,
        (
            StructuralStudyWorkloads::HeterogeneousSequence { workloads },
            ExactStructuralStudyAnalysisClosure::StructuralModalResponseSpectrum {
                modal_result_root,
                modal_receipt_root,
                basis_state_root,
                response_result_root,
                response_receipt_root,
                response_accuracy_root,
                response_publication_baseline_usage_root,
                response_resource_usage_root,
                response_publication_root,
                field_recovery_root,
                envelope_state_root,
                objects,
            },
        ) => super::response_spectrum_import::import_response_spectrum_result(
            super::response_spectrum_import::ResponseSpectrumResultImport {
                source,
                workloads,
                modal_result_root: *modal_result_root,
                modal_receipt_root: *modal_receipt_root,
                basis_state_root: *basis_state_root,
                response_result_root: *response_result_root,
                response_receipt_root: *response_receipt_root,
                response_accuracy_root: *response_accuracy_root,
                response_publication_baseline_usage_root: *response_publication_baseline_usage_root,
                response_resource_usage_root: *response_resource_usage_root,
                response_publication_root: *response_publication_root,
                field_recovery_root: *field_recovery_root,
                envelope_state_root: *envelope_state_root,
                objects,
                mesh: &mesh.artifact,
                access: &expected.artifact_access,
                limits,
            },
        )?,
        (
            StructuralStudyWorkloads::HeterogeneousSequence { workloads },
            ExactStructuralStudyAnalysisClosure::StructuralRandomResponse {
                modal_result_root,
                modal_receipt_root,
                basis_state_root,
                response_result_root,
                response_receipt_root,
                response_accuracy_root,
                response_publication_baseline_usage_root,
                response_resource_usage_root,
                response_publication_root,
                statistics_state_root,
                objects,
            },
        ) => super::random_response_import::import_random_response_result(
            super::random_response_import::RandomResponseResultImport {
                source,
                workloads,
                modal_result_root: *modal_result_root,
                modal_receipt_root: *modal_receipt_root,
                basis_state_root: *basis_state_root,
                response_result_root: *response_result_root,
                response_receipt_root: *response_receipt_root,
                response_accuracy_root: *response_accuracy_root,
                response_publication_baseline_usage_root: *response_publication_baseline_usage_root,
                response_resource_usage_root: *response_resource_usage_root,
                response_publication_root: *response_publication_root,
                statistics_state_root: *statistics_state_root,
                objects,
                mesh: &mesh.artifact,
                access: &expected.artifact_access,
                limits,
            },
        )?,
        (
            StructuralStudyWorkloads::HeterogeneousSequence { workloads },
            ExactStructuralStudyAnalysisClosure::StructuralPrestressedModal {
                preload_result_root,
                preload_receipt_root,
                prestress_state_root,
                modal_result_root,
                modal_receipt_root,
                objects,
            },
        ) => super::prestressed_modal_import::import_prestressed_modal_result(
            super::prestressed_modal_import::PrestressedModalResultImport {
                source,
                workloads,
                preload_result_root: *preload_result_root,
                preload_receipt_root: *preload_receipt_root,
                prestress_state_root: *prestress_state_root,
                modal_result_root: *modal_result_root,
                modal_receipt_root: *modal_receipt_root,
                objects,
                mesh: &mesh.artifact,
                access: &expected.artifact_access,
                limits,
            },
        )?,
        (
            StructuralStudyWorkloads::HeterogeneousSequence { workloads },
            ExactStructuralStudyAnalysisClosure::StructuralPrestressedFrequency {
                preload_result_root,
                preload_receipt_root,
                prestress_state_root,
                frequency,
                objects,
            },
        ) => super::prestressed_frequency_import::import_prestressed_frequency_result(
            super::prestressed_frequency_import::PrestressedFrequencyResultImport {
                source,
                workloads,
                preload_result_root: *preload_result_root,
                preload_receipt_root: *preload_receipt_root,
                prestress_state_root: *prestress_state_root,
                frequency: *frequency,
                objects,
                mesh: &mesh.artifact,
                access: &expected.artifact_access,
                limits,
            },
        )?,
        (
            StructuralStudyWorkloads::HeterogeneousSequence { workloads },
            ExactStructuralStudyAnalysisClosure::StructuralPrestressedModalFrequency {
                preload_result_root,
                preload_receipt_root,
                prestress_state_root,
                modal_result_root,
                modal_receipt_root,
                basis_state_root,
                frequency,
                objects,
            },
        ) => super::prestressed_modal_frequency_import::import_prestressed_modal_frequency_result(
            super::prestressed_modal_frequency_import::PrestressedModalFrequencyResultImport {
                source,
                workloads,
                preload_result_root: *preload_result_root,
                preload_receipt_root: *preload_receipt_root,
                prestress_state_root: *prestress_state_root,
                modal_result_root: *modal_result_root,
                modal_receipt_root: *modal_receipt_root,
                basis_state_root: *basis_state_root,
                frequency: *frequency,
                objects,
                mesh: &mesh.artifact,
                access: &expected.artifact_access,
                limits,
            },
        )?,
        (
            StructuralStudyWorkloads::HeterogeneousSequence { workloads },
            ExactStructuralStudyAnalysisClosure::StructuralBuckling {
                preload_result_root,
                preload_receipt_root,
                prestress_state_root,
                buckling_result_root,
                buckling_receipt_root,
                objects,
            },
        ) => super::buckling_import::import_buckling_result(
            super::buckling_import::BucklingResultImport {
                source,
                workloads,
                preload_result_root: *preload_result_root,
                preload_receipt_root: *preload_receipt_root,
                prestress_state_root: *prestress_state_root,
                buckling_result_root: *buckling_result_root,
                buckling_receipt_root: *buckling_receipt_root,
                objects,
                mesh: &mesh.artifact,
                access: &expected.artifact_access,
                limits,
            },
        )?,
        (
            StructuralStudyWorkloads::HeterogeneousSequence { workloads },
            ExactStructuralStudyAnalysisClosure::StructuralModalTransient {
                modal_result_root,
                modal_receipt_root,
                basis_state_root,
                transient_result_root,
                transient_receipt_root,
                objects,
            },
        ) => super::modal_import::import_modal_transient_result(
            super::modal_import::ModalTransientResultImport {
                source,
                workloads,
                modal_result_root: *modal_result_root,
                modal_receipt_root: *modal_receipt_root,
                basis_state_root: *basis_state_root,
                transient_result_root: *transient_result_root,
                transient_receipt_root: *transient_receipt_root,
                objects,
                mesh: &mesh.artifact,
                access: &expected.artifact_access,
                limits,
            },
        )?,
        (
            StructuralStudyWorkloads::StructuralFrequency { workload },
            ExactStructuralStudyAnalysisClosure::StructuralFrequency { frequency, objects },
        ) => ExactStructuralStudyAnalysisResult::StructuralFrequency(Box::new(
            super::frequency_closure_import::import_frequency_closure(
                super::frequency_closure_import::FrequencyClosureImport {
                    source,
                    workload: *workload,
                    closure: *frequency,
                    objects,
                    mesh: &mesh.artifact,
                    prestress_state: None,
                    modal_state: None,
                    access: &expected.artifact_access,
                    limits,
                },
            )?,
        )),
        _ => {
            return Err(protocol(
                "whole-study analysis result kind differs from its authored workload kind",
            ))
        }
    };
    let mut compiled_study = publication.compiled_study;
    if let Some(compiled) = &mut compiled_study {
        compiled.model = resolved_compiled_model;
        compiled
            .validate_against(&mesh.artifact)
            .map_err(protocol)?;
    }
    Ok(ExactStructuralStudyResult {
        frozen_project_digest: expected.frozen_project_digest,
        compiled_study,
        mesh,
        region_resolution,
        resolved_workloads: retained_workloads,
        analysis,
        thermal,
    })
}

pub(super) fn runtime_artifact_root(
    objects: &runmat_analysis_execution::PreparedAnalysisObjects,
    reference: ValueRef,
) -> NativeExecutionResult<runmat_runtime::analysis::AnalysisArtifactRoot> {
    runmat_runtime::analysis::AnalysisArtifactRoot::new(
        reference,
        objects.manifest.kind,
        objects.manifest.logical_identity,
        objects.manifest.payload_digest,
    )
    .map_err(protocol)
}

pub(super) fn import_numerical_analysis_objects(
    source: &impl CacheImport,
    root: &ValueRef,
    declared_objects: &[ValueRef],
    kind: AnalysisNumericalObjectKind,
    access: &ArtifactAccess,
    limits: ObjectInventoryLimits,
) -> NativeExecutionResult<runmat_analysis_execution::PreparedAnalysisObjects> {
    if !declared_objects.contains(root) {
        return Err(protocol(
            "structural numerical object root is absent from its declared closure",
        ));
    }
    let manifest = runmat_analysis_fea::execution::AnalysisObjectManifest::from_canonical_bytes(
        &read(source, root)?,
    )
    .map_err(protocol)?;
    if manifest.kind != kind {
        return Err(protocol(
            "structural numerical object kind differs from its declared role",
        ));
    }
    let imported = runmat_analysis_execution::import_analysis_objects(
        source,
        root,
        kind,
        manifest.logical_identity,
        manifest.payload_digest,
        access,
        limits,
    )
    .map_err(protocol)?;
    for reference in imported
        .unique_references(access, ValueRefKind::ResultObject)
        .map_err(protocol)?
    {
        if !declared_objects.contains(&reference) {
            return Err(protocol(
                "structural numerical object references an undeclared closure member",
            ));
        }
    }
    Ok(imported)
}

pub(super) fn validate_numerical_closure_inventory(
    result_objects: &runmat_analysis_execution::PreparedAnalysisObjects,
    receipt_objects: &runmat_analysis_execution::PreparedAnalysisObjects,
    declared_objects: &[ValueRef],
    access: &ArtifactAccess,
) -> NativeExecutionResult<()> {
    let admitted_objects = unique_references(
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
    if reference_inventory(&admitted_objects)? != reference_inventory(declared_objects)? {
        return Err(protocol(
            "structural numerical closure contains missing or unreferenced objects",
        ));
    }
    Ok(())
}

pub(super) fn validate_complete_numerical_inventory(
    components: &[&runmat_analysis_execution::PreparedAnalysisObjects],
    declared_objects: &[ValueRef],
    access: &ArtifactAccess,
) -> NativeExecutionResult<()> {
    let mut admitted = Vec::new();
    for component in components {
        admitted.extend(
            component
                .unique_references(access, ValueRefKind::ResultObject)
                .map_err(protocol)?,
        );
    }
    let admitted = unique_references(admitted)?;
    if reference_inventory(&admitted)? != reference_inventory(declared_objects)? {
        return Err(protocol(
            "structural numerical closure contains missing or unrelated objects",
        ));
    }
    Ok(())
}
