use runmat_analysis_core::ObservationRequest;
use runmat_analysis_execution::{
    analysis_object_references, decode_prepared_observation_history_checkpoint,
    import_analysis_objects, import_analysis_partition, AnalysisOutputRoot,
    AnalysisPartitionImport, ImportedObservations, ImportedRecoveredFields,
    PreparedAnalysisObjects, PreparedResourceUsage,
};
use runmat_analysis_fea::execution::{
    AnalysisNumericalObjectKind, FieldRecoverySource, StructuralFrequencyNumericalReceipt,
    StructuralFrequencyPublicationHistory, StructuralFrequencyPublicationInput,
    StructuralFrequencyPublicationManifest, StructuralFrequencyPublicationSample,
    StructuralFrequencySolverResult, StructuralFrequencyWorkload,
};
use runmat_analysis_fea::planning::resource::ResourceUsageEvidence;
use runmat_analysis_fea::results::field::RecoveredFieldArtifactManifest;
use runmat_analysis_fea::results::observation::{
    ObservationArtifactManifest, ObservationFieldSource,
};
use runmat_execution::value::{ValueRef, ValueRefKind};
use runmat_execution_artifact::cache::CacheImport;
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_execution_artifact::ArtifactAccess;

use super::{
    protocol, read, reference_inventory, unique_references, ExactStructuralFrequencyEntryClosure,
    ExactStructuralFrequencyEntryResult, ExactStructuralFrequencyHistoryClosure,
    ExactStructuralFrequencyHistoryResult,
};
use crate::NativeExecutionResult;

pub(super) struct FrequencyArtifactImport<'a, Source> {
    pub source: &'a Source,
    pub workload: &'a StructuralFrequencyWorkload,
    pub result: &'a StructuralFrequencySolverResult,
    pub receipt: &'a StructuralFrequencyNumericalReceipt,
    pub declared_entries: &'a [ExactStructuralFrequencyEntryClosure],
    pub declared_histories: &'a [ExactStructuralFrequencyHistoryClosure],
    pub resource_usage_root: &'a ValueRef,
    pub result_objects: &'a PreparedAnalysisObjects,
    pub receipt_objects: &'a PreparedAnalysisObjects,
    pub declared_objects: &'a [ValueRef],
    pub access: &'a ArtifactAccess,
    pub limits: ObjectInventoryLimits,
    pub declared_publication: &'a StructuralFrequencyPublicationManifest,
    pub mesh: &'a runmat_meshing_core::SolverMeshArtifact,
}

pub(super) fn import_frequency_artifacts<Source: CacheImport>(
    import: FrequencyArtifactImport<'_, Source>,
) -> NativeExecutionResult<(
    Vec<ExactStructuralFrequencyEntryResult>,
    Vec<ExactStructuralFrequencyHistoryResult>,
    PreparedResourceUsage,
    StructuralFrequencyPublicationManifest,
)> {
    let FrequencyArtifactImport {
        source,
        workload,
        result,
        receipt,
        declared_entries,
        declared_histories,
        resource_usage_root,
        result_objects,
        receipt_objects,
        declared_objects,
        access,
        limits,
        declared_publication,
        mesh,
    } = import;
    if declared_entries.len() != result.samples().len() {
        return Err(protocol(
            "structural frequency artifact entries do not exactly cover the result",
        ));
    }
    let mut admitted = result_objects
        .unique_references(access, ValueRefKind::ResultObject)
        .map_err(protocol)?;
    admitted.extend(
        receipt_objects
            .unique_references(access, ValueRefKind::ResultObject)
            .map_err(protocol)?,
    );
    let mut entries = Vec::with_capacity(declared_entries.len());
    for (declared, physical) in declared_entries.iter().zip(result.samples()) {
        if declared.entry != physical.entry()
            || declared.physical_sample != physical.identity()
            || declared.sample
                != runmat_analysis_fea::results::field::RecoveredFieldSample::FrequencyHertz(
                    physical.entry().frequency_hz(),
                )
        {
            return Err(protocol(
                "structural frequency artifact entry differs from its physical sample",
            ));
        }
        declared.source_entry.validate().map_err(protocol)?;
        if declared.source_entry.workload() != workload.identity()
            || declared.source_entry.scope() != result.manifest().scope().identity()
            || declared.source_entry.entry() != physical.entry()
            || declared.source_entry.sample() != physical.identity()
        {
            return Err(protocol(
                "structural frequency source entry differs from its workload or physical sample",
            ));
        }
        let (catalog_objects, output) = import_partitioned_catalog(
            source,
            &declared.fields_root,
            AnalysisNumericalObjectKind::Field,
            declared_objects,
            access,
            limits,
            &mut admitted,
        )?;
        let manifest =
            RecoveredFieldArtifactManifest::from_canonical_bytes(&catalog_objects.payload())
                .map_err(protocol)?;
        if manifest.identity.digest() != output.logical_identity
            || manifest.source
                != FieldRecoverySource::StructuralFrequencyEntry(declared.source_entry.identity())
            || manifest
                .fields
                .iter()
                .any(|field| field.sample != declared.sample)
        {
            return Err(protocol(
                "structural frequency recovered fields differ from their declared entry",
            ));
        }
        let recovered_fields = ImportedRecoveredFields {
            manifest,
            catalog_objects,
            output,
        };
        let observations = declared
            .observations_root
            .as_ref()
            .map(|root| {
                let (catalog_objects, output) = import_partitioned_catalog(
                    source,
                    root,
                    AnalysisNumericalObjectKind::Observation,
                    declared_objects,
                    access,
                    limits,
                    &mut admitted,
                )?;
                let manifest =
                    ObservationArtifactManifest::from_canonical_bytes(&catalog_objects.payload())
                        .map_err(protocol)?;
                if manifest.identity.digest() != output.logical_identity
                    || !matches!(manifest.source,
                        ObservationFieldSource::StructuralRecovery {
                            artifact, recovery_request, numerical_source, topology,
                        } if artifact == recovered_fields.manifest.identity
                            && recovery_request == recovered_fields.manifest.recovery_request
                            && numerical_source == FieldRecoverySource::StructuralFrequencyEntry(declared.source_entry.identity())
                            && topology.discretization_digest() == recovered_fields.manifest.topology_digest
                    )
                    || manifest
                        .results
                        .iter()
                        .any(|entry| entry.sample != declared.sample)
                {
                    return Err(protocol(
                        "structural frequency observations differ from their recovered fields",
                    ));
                }
                Ok(ImportedObservations {
                    manifest,
                    catalog_objects,
                    output,
                    access: access.clone(),
                })
            })
            .transpose()?;
        entries.push(ExactStructuralFrequencyEntryResult {
            sample: declared.sample,
            recovered_fields,
            observations,
        });
    }

    let requested_histories = workload
        .procedure_step()
        .map_err(protocol)?
        .observations
        .iter()
        .filter_map(|request| match request {
            ObservationRequest::History(spec) => Some(spec.observation_id.clone()),
            _ => None,
        })
        .collect::<Vec<_>>();
    if declared_histories
        .iter()
        .map(|history| history.observation_id.clone())
        .collect::<Vec<_>>()
        != requested_histories
    {
        return Err(protocol(
            "structural frequency history closure differs from the authored histories",
        ));
    }
    let mut histories = Vec::with_capacity(declared_histories.len());
    for (declared, published) in declared_histories
        .iter()
        .zip(declared_publication.histories())
    {
        let objects = import_whole_objects(
            source,
            &declared.checkpoint_root,
            AnalysisNumericalObjectKind::ObservationHistoryCheckpoint,
            declared_objects,
            access,
            limits,
            &mut admitted,
        )?;
        let checkpoint =
            decode_prepared_observation_history_checkpoint(&objects).map_err(protocol)?;
        if checkpoint.manifest().stream.observation_id != declared.observation_id {
            return Err(protocol(
                "structural frequency history checkpoint names another observation",
            ));
        }
        let finalization = match (&declared.finalization, &published.finalization) {
            (None, None) => None,
            (Some(roots), Some(publication)) => {
                let catalog = import_whole_objects(
                    source,
                    roots.catalog.reference(),
                    AnalysisNumericalObjectKind::ObservationFieldCatalog,
                    declared_objects,
                    access,
                    limits,
                    &mut admitted,
                )?;
                let decoded_catalog = runmat_analysis_fea::results::observation::ObservationFieldCatalog::from_canonical_bytes(&catalog.payload()).map_err(protocol)?;
                // Authenticate the aggregate against fields already admitted from the
                // original physical result, rather than trusting a matching schedule.
                let runmat_analysis_fea::results::observation::ObservationFieldCatalogSource::EntryHistory { samples } = decoded_catalog.source() else {
                    return Err(protocol("frequency history requires its complete source catalog"));
                };
                if samples.len() != entries.len() || samples.iter().zip(&entries).any(|(sample, entry)| {
                    !matches!(sample.source(), runmat_analysis_fea::results::observation::ObservationFieldCatalogSource::Structural { manifest, .. }
                        if manifest.as_ref() == &entry.recovered_fields.manifest)
                }) {
                    return Err(protocol("frequency history catalog differs from admitted physical fields"));
                }
                let expected =
                    runmat_analysis_fea::results::observation::BoundedObservationHistory::resume(
                        &publication.request.plan,
                        &declared.observation_id,
                        checkpoint.clone(),
                    )
                    .map_err(protocol)?
                    .finish()
                    .map_err(protocol)?;
                let history = runmat_analysis_execution::import_observation_history(
                    source,
                    runmat_analysis_execution::ObservationHistoryImport {
                        root: roots.history.reference(),
                        expected_identity: publication.history.identity,
                        expected_payload_digest: roots.history.expected_payload_digest(),
                        partial_access: publication.history.partial_access,
                        access,
                        limits,
                    },
                )
                .map_err(protocol)?;
                if history.manifest != *expected.manifest() {
                    return Err(protocol(
                        "frequency finalized history differs from terminal checkpoint",
                    ));
                }
                for sample in expected.samples() {
                    let actual = runmat_analysis_execution::import_observation_history_sample(
                        source,
                        &history,
                        sample.coordinate,
                        limits,
                    )
                    .map_err(protocol)?;
                    if actual != *sample {
                        return Err(protocol(
                            "frequency finalized sample differs from terminal checkpoint",
                        ));
                    }
                }
                let series_objects = roots
                    .series
                    .as_ref()
                    .map(|root| {
                        import_whole_objects(
                            source,
                            root.reference(),
                            AnalysisNumericalObjectKind::SeriesReductionResult,
                            declared_objects,
                            access,
                            limits,
                            &mut admitted,
                        )
                    })
                    .transpose()?;
                let series_reduction = series_objects.as_ref().map(|objects| runmat_analysis_fea::results::observation::SeriesReductionResult::from_canonical_bytes(&objects.payload()).map_err(protocol)).transpose()?;
                publication.validate_history(&checkpoint, &decoded_catalog,
                    runmat_analysis_fea::execution::ObservationHistorySourceContext::Structural { source: runmat_analysis_fea::execution::StructuralObservationSource::Frequency(workload), mesh: Some(mesh) },
                    &expected, series_reduction.as_ref()).map_err(protocol)?;
                let completion = crate::analysis_frequency_observations::CompletedStructuralFrequencyHistoryFinalization {
                    catalog, result: runmat_analysis_execution::ImportedObservationHistoryFinalization { publication: publication.clone(), history, series_reduction, series_objects },
                };
                let (actual_roots, references) = completion.published_artifacts(access)?;
                if actual_roots != *roots
                    || references
                        .iter()
                        .any(|reference| !declared_objects.contains(reference))
                {
                    return Err(protocol(
                        "frequency history finalization roots differ from closure",
                    ));
                }
                admitted.extend(references);
                Some(completion)
            }
            _ => return Err(protocol("frequency history finalization inventory differs")),
        };
        histories.push(ExactStructuralFrequencyHistoryResult {
            observation_id: declared.observation_id.clone(),
            checkpoint: runmat_analysis_execution::ImportedObservationHistoryCheckpointTask {
                checkpoint,
                objects,
            },
            finalization,
        });
    }

    let resource_objects = import_whole_objects(
        source,
        resource_usage_root,
        AnalysisNumericalObjectKind::ResourceUsage,
        declared_objects,
        access,
        limits,
        &mut admitted,
    )?;
    let evidence = ResourceUsageEvidence::from_canonical_bytes(&resource_objects.payload())
        .map_err(protocol)?;
    if evidence.identity().digest() != resource_objects.manifest.logical_identity
        || evidence.result_digest != result.manifest().identity().digest()
        || !evidence.overruns.is_empty()
    {
        return Err(protocol(
            "structural frequency resource evidence differs from its result or exceeded admission",
        ));
    }
    if reference_inventory(&unique_references(admitted)?)? != reference_inventory(declared_objects)?
    {
        return Err(protocol(
            "structural frequency closure contains missing or unreferenced objects",
        ));
    }
    let prepared_resource_usage = PreparedResourceUsage {
        evidence,
        objects: resource_objects,
    };
    let publication =
        StructuralFrequencyPublicationManifest::new(StructuralFrequencyPublicationInput {
            workload,
            result,
            receipt: receipt.clone(),
            samples: declared_entries
                .iter()
                .zip(&entries)
                .zip(result.manifest().samples())
                .map(
                    |((declared, imported), result)| StructuralFrequencyPublicationSample {
                        result: *result,
                        source_entry: declared.source_entry.clone(),
                        coordinate: declared.sample,
                        fields: imported.recovered_fields.manifest.clone(),
                        observations: imported
                            .observations
                            .as_ref()
                            .map(|observations| observations.manifest.clone()),
                    },
                )
                .collect(),
            histories: histories
                .iter()
                .map(|history| StructuralFrequencyPublicationHistory {
                    observation_id: history.observation_id.clone(),
                    checkpoint: history.checkpoint.checkpoint.manifest().clone(),
                    finalization: history
                        .finalization
                        .as_ref()
                        .map(|finalization| finalization.result.publication.clone()),
                })
                .collect(),
            resource_usage: prepared_resource_usage.evidence.clone(),
        })
        .map_err(protocol)?;
    if &publication != declared_publication {
        return Err(protocol(
            "structural frequency semantic publication differs from its object closure",
        ));
    }
    Ok((entries, histories, prepared_resource_usage, publication))
}

fn import_partitioned_catalog(
    source: &impl CacheImport,
    root: &ValueRef,
    kind: AnalysisNumericalObjectKind,
    declared_objects: &[ValueRef],
    access: &ArtifactAccess,
    limits: ObjectInventoryLimits,
    admitted: &mut Vec<ValueRef>,
) -> NativeExecutionResult<(
    runmat_analysis_execution::ImportedAnalysisPartition,
    AnalysisOutputRoot,
)> {
    if !declared_objects.contains(root) {
        return Err(protocol(
            "structural frequency artifact root is absent from its closure",
        ));
    }
    let manifest = runmat_analysis_fea::execution::AnalysisObjectManifest::from_canonical_bytes(
        &read(source, root)?,
    )
    .map_err(protocol)?;
    if manifest.kind != kind {
        return Err(protocol(
            "structural frequency artifact root has the wrong numerical kind",
        ));
    }
    let references =
        analysis_object_references(&manifest, root, access, ValueRefKind::ResultObject)
            .map_err(protocol)?;
    if references
        .iter()
        .any(|reference| !declared_objects.contains(reference))
    {
        return Err(protocol(
            "structural frequency artifact references an undeclared closure member",
        ));
    }
    admitted.extend(references);
    let imported = import_analysis_partition(
        source,
        AnalysisPartitionImport {
            root,
            expected_kind: kind,
            expected_logical_identity: manifest.logical_identity,
            expected_payload_digest: manifest.payload_digest,
            partition_identity: manifest.logical_identity,
            access,
            limits,
        },
    )
    .map_err(protocol)?;
    let output = AnalysisOutputRoot {
        kind,
        logical_identity: manifest.logical_identity,
        payload_digest: manifest.payload_digest,
        root: root.clone(),
    };
    Ok((imported, output))
}

fn import_whole_objects(
    source: &impl CacheImport,
    root: &ValueRef,
    kind: AnalysisNumericalObjectKind,
    declared_objects: &[ValueRef],
    access: &ArtifactAccess,
    limits: ObjectInventoryLimits,
    admitted: &mut Vec<ValueRef>,
) -> NativeExecutionResult<PreparedAnalysisObjects> {
    if !declared_objects.contains(root) {
        return Err(protocol(
            "structural frequency numerical root is absent from its closure",
        ));
    }
    let manifest = runmat_analysis_fea::execution::AnalysisObjectManifest::from_canonical_bytes(
        &read(source, root)?,
    )
    .map_err(protocol)?;
    if manifest.kind != kind {
        return Err(protocol(
            "structural frequency numerical root has the wrong kind",
        ));
    }
    let imported = import_analysis_objects(
        source,
        root,
        kind,
        manifest.logical_identity,
        manifest.payload_digest,
        access,
        limits,
    )
    .map_err(protocol)?;
    let references = imported
        .unique_references(access, ValueRefKind::ResultObject)
        .map_err(protocol)?;
    if references
        .iter()
        .any(|reference| !declared_objects.contains(reference))
    {
        return Err(protocol(
            "structural frequency numerical object references an undeclared member",
        ));
    }
    admitted.extend(references);
    Ok(imported)
}
