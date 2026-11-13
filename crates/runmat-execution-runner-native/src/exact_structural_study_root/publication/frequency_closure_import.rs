use runmat_analysis_fea::execution::{
    AnalysisNumericalObjectKind, CanonicalSemanticState, StructuralFrequencyWorkload,
};
use runmat_execution::value::{ValueRef, ValueRefKind};
use runmat_execution_artifact::cache::CacheImport;
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_execution_artifact::ArtifactAccess;

use super::frequency_import::{import_frequency_artifacts, FrequencyArtifactImport};
use super::{protocol, ExactStructuralFrequencyAnalysisResult, ExactStructuralFrequencyClosure};
use crate::NativeExecutionResult;

pub(super) struct FrequencyClosureImport<'a, Source> {
    pub source: &'a Source,
    pub workload: StructuralFrequencyWorkload,
    pub closure: ExactStructuralFrequencyClosure,
    pub objects: Vec<ValueRef>,
    pub mesh: &'a runmat_meshing_core::SolverMeshArtifact,
    pub prestress_state: Option<&'a CanonicalSemanticState>,
    pub modal_state: Option<&'a CanonicalSemanticState>,
    pub access: &'a ArtifactAccess,
    pub limits: ObjectInventoryLimits,
}

pub(super) fn import_frequency_closure<Source: CacheImport>(
    input: FrequencyClosureImport<'_, Source>,
) -> NativeExecutionResult<ExactStructuralFrequencyAnalysisResult> {
    let FrequencyClosureImport {
        source,
        workload,
        closure,
        objects: enclosing_objects,
        mesh,
        prestress_state,
        modal_state,
        access,
        limits,
    } = input;
    if closure
        .objects
        .iter()
        .any(|reference| !enclosing_objects.contains(reference))
    {
        return Err(protocol(
            "structural frequency object closure escapes its enclosing result",
        ));
    }
    let objects = closure.objects.clone();
    let result_objects = super::result_import::import_numerical_analysis_objects(
        source,
        &closure.result_root,
        &objects,
        AnalysisNumericalObjectKind::SolverResult,
        access,
        limits,
    )?;
    let receipt_objects = super::result_import::import_numerical_analysis_objects(
        source,
        &closure.receipt_root,
        &objects,
        AnalysisNumericalObjectKind::NumericalReceipt,
        access,
        limits,
    )?;
    let result = runmat_analysis_execution::decode_prepared_structural_frequency_solver_result(
        &result_objects,
    )
    .map_err(protocol)?;
    let numerical_receipt =
        runmat_analysis_execution::decode_prepared_structural_frequency_numerical_receipt(
            &receipt_objects,
        )
        .map_err(protocol)?;
    numerical_receipt
        .validate_for_result(&result)
        .map_err(protocol)?;
    let prepared = match prestress_state {
        Some(state) => {
            runmat_analysis_fea::pipeline::structural_frequency::prepare_prestressed_structural_frequency(
                &workload, mesh, state,
            )
        }
        None => {
            runmat_analysis_fea::pipeline::structural_frequency::prepare_structural_frequency(
                &workload, mesh,
            )
        }
    }
    .map_err(protocol)?;
    if result.manifest().scope() != &prepared.scope {
        return Err(protocol(
            "structural frequency result names a different resolved workload",
        ));
    }
    let mut semantic_roots = vec![
        (*closure.result_root).clone(),
        (*closure.receipt_root).clone(),
        (*closure.resource_usage_root).clone(),
    ];
    semantic_roots.extend(closure.entries.iter().flat_map(|entry| {
        std::iter::once(entry.fields_root.clone()).chain(entry.observations_root.iter().cloned())
    }));
    for history in &closure.histories {
        semantic_roots.push(history.checkpoint_root.clone());
        if let Some(finalization) = &history.finalization {
            semantic_roots.push(finalization.catalog.reference().clone());
            semantic_roots.push(finalization.history.reference().clone());
            semantic_roots.extend(
                finalization
                    .series
                    .iter()
                    .map(|root| root.reference().clone()),
            );
        }
    }
    let object_catalog = runmat_execution::value::PublishedObjectCatalog::new_with_shared_objects(
        semantic_roots,
        objects.clone(),
    )
    .map_err(protocol)?;
    let (entries, histories, resource_usage, publication) =
        import_frequency_artifacts(FrequencyArtifactImport {
            source,
            workload: &workload,
            result: &result,
            receipt: &numerical_receipt,
            declared_entries: &closure.entries,
            declared_histories: &closure.histories,
            resource_usage_root: &closure.resource_usage_root,
            result_objects: &result_objects,
            receipt_objects: &receipt_objects,
            declared_objects: &objects,
            access,
            limits,
            declared_publication: &closure.publication,
            mesh,
        })?;
    admit_publication(
        &workload,
        mesh,
        prestress_state,
        modal_state,
        &result,
        &publication,
    )?;
    let sample_roots = entries
        .iter()
        .map(|entry| {
            let fields = runmat_runtime::analysis::AnalysisArtifactRoot::new(
                entry.recovered_fields.output.root.clone(),
                entry.recovered_fields.output.kind,
                entry.recovered_fields.output.logical_identity,
                entry.recovered_fields.output.payload_digest,
            )
            .map_err(protocol)?;
            let observations = entry
                .observations
                .as_ref()
                .map(|observations| {
                    runmat_runtime::analysis::AnalysisArtifactRoot::new(
                        observations.output.root.clone(),
                        observations.output.kind,
                        observations.output.logical_identity,
                        observations.output.payload_digest,
                    )
                    .map_err(protocol)
                })
                .transpose()?;
            Ok(runmat_runtime::analysis::FrequencySampleArtifactRoots::new(
                fields,
                observations,
            ))
        })
        .collect::<NativeExecutionResult<Vec<_>>>()?;
    let history_roots = histories
        .iter()
        .map(|history| {
            let reference = history
                .checkpoint
                .objects
                .root_reference(access, ValueRefKind::ResultObject)
                .map_err(protocol)?;
            let mut binding = runmat_runtime::analysis::StructuralFrequencyHistoryArtifactRoot::new(
                super::result_import::runtime_artifact_root(
                    &history.checkpoint.objects,
                    reference,
                )?,
            );
            if let Some(finalization) = &history.finalization {
                binding = binding.with_finalization(finalization.published_artifacts(access)?.0);
            }
            Ok(binding)
        })
        .collect::<NativeExecutionResult<Vec<_>>>()?;
    let artifact_roots = runmat_runtime::analysis::StructuralFrequencyArtifactRootIndex::new(
        super::result_import::runtime_artifact_root(
            &result_objects,
            (*closure.result_root).clone(),
        )?,
        super::result_import::runtime_artifact_root(
            &receipt_objects,
            (*closure.receipt_root).clone(),
        )?,
        super::result_import::runtime_artifact_root(
            &resource_usage.objects,
            (*closure.resource_usage_root).clone(),
        )?,
        sample_roots,
        history_roots,
        &publication,
        &object_catalog,
    )
    .map_err(protocol)?;
    Ok(ExactStructuralFrequencyAnalysisResult {
        resolved_workload: Box::new(workload),
        result: Box::new(result),
        numerical_receipt,
        entries,
        histories,
        resource_usage,
        publication: Box::new(publication),
        object_catalog,
        artifact_roots,
    })
}

fn admit_publication(
    workload: &StructuralFrequencyWorkload,
    mesh: &runmat_meshing_core::SolverMeshArtifact,
    prestress_state: Option<&CanonicalSemanticState>,
    modal_state: Option<&CanonicalSemanticState>,
    result: &runmat_analysis_fea::execution::StructuralFrequencySolverResult,
    publication: &runmat_analysis_fea::execution::StructuralFrequencyPublicationManifest,
) -> NativeExecutionResult<()> {
    match (prestress_state, modal_state) {
        (Some(prestress), Some(modal)) => runmat_analysis_fea::pipeline::structural_frequency::admit_prestressed_modal_structural_frequency_publication(
            workload,
            mesh,
            prestress,
            modal,
            result,
            publication,
        ),
        (Some(state), None) => runmat_analysis_fea::pipeline::structural_frequency::admit_prestressed_structural_frequency_publication(
            workload,
            mesh,
            state,
            result,
            publication,
        ),
        (None, None) => runmat_analysis_fea::pipeline::structural_frequency::admit_structural_frequency_publication(
            workload,
            mesh,
            result,
            publication,
        ),
        (None, Some(_)) => {
            return Err(protocol(
                "modal structural frequency publication is missing its prestress state",
            ))
        }
    }
    .map(|_| ())
    .map_err(protocol)
}
