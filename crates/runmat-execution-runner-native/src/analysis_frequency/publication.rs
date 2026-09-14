use runmat_analysis_execution::PreparedAnalysisObjects;
use runmat_analysis_fea::execution::AnalysisNumericalObjectKind;
use runmat_execution::value::{ValueRef, ValueRefKind};
use runmat_execution_artifact::{ArtifactAccess, PublishedObjectCatalog};
use runmat_runtime::analysis::{
    StructuralFrequencyArtifactRoot, StructuralFrequencyArtifactRootIndex,
    StructuralFrequencyHistoryArtifactRoot, StructuralFrequencySampleArtifactRoots,
};

use super::CompletedStructuralFrequencyStudy;
use crate::analysis_task_pool::protocol;
use crate::NativeExecutionResult;

impl CompletedStructuralFrequencyStudy {
    pub fn published_artifacts(
        &self,
        access: &ArtifactAccess,
    ) -> NativeExecutionResult<(PublishedObjectCatalog, StructuralFrequencyArtifactRootIndex)> {
        let result_root = self
            .result
            .result_objects
            .root_reference(access, ValueRefKind::ResultObject)
            .map_err(protocol)?;
        let receipt_root = self
            .result
            .receipt_objects
            .root_reference(access, ValueRefKind::ResultObject)
            .map_err(protocol)?;
        let resource_root = self
            .resource_usage
            .objects
            .root_reference(access, ValueRefKind::ResultObject)
            .map_err(protocol)?;
        let result_binding = artifact_root(&self.result.result_objects, result_root.clone())?;
        let receipt_binding = artifact_root(&self.result.receipt_objects, receipt_root.clone())?;
        let resource_binding = artifact_root(&self.resource_usage.objects, resource_root.clone())?;
        let mut roots = vec![result_root, receipt_root, resource_root];
        let mut objects = self
            .result
            .result_objects
            .unique_references(access, ValueRefKind::ResultObject)
            .map_err(protocol)?;
        objects.extend(
            self.result
                .receipt_objects
                .unique_references(access, ValueRefKind::ResultObject)
                .map_err(protocol)?,
        );
        objects.extend(
            self.resource_usage
                .objects
                .unique_references(access, ValueRefKind::ResultObject)
                .map_err(protocol)?,
        );
        for entry in &self.entries {
            roots.push(entry.recovered_fields.output.root.clone());
            objects.extend(
                runmat_analysis_execution::analysis_object_references(
                    &entry.recovered_fields.catalog_objects.manifest,
                    &entry.recovered_fields.output.root,
                    access,
                    ValueRefKind::ResultObject,
                )
                .map_err(protocol)?,
            );
        }
        for observations in &self.observation_samples {
            if let Some(observations) = &observations.observations {
                roots.push(observations.output.root.clone());
                objects.extend(
                    runmat_analysis_execution::analysis_object_references(
                        &observations.catalog_objects.manifest,
                        &observations.output.root,
                        access,
                        ValueRefKind::ResultObject,
                    )
                    .map_err(protocol)?,
                );
            }
        }
        let sample_roots = self
            .entries
            .iter()
            .zip(&self.observation_samples)
            .map(|(entry, observations)| {
                let fields = StructuralFrequencyArtifactRoot::new(
                    entry.recovered_fields.output.root.clone(),
                    AnalysisNumericalObjectKind::Field,
                    entry.recovered_fields.output.logical_identity,
                    entry.recovered_fields.output.payload_digest,
                )
                .map_err(protocol)?;
                let observations = observations
                    .observations
                    .as_ref()
                    .map(|observations| {
                        StructuralFrequencyArtifactRoot::new(
                            observations.output.root.clone(),
                            AnalysisNumericalObjectKind::Observation,
                            observations.output.logical_identity,
                            observations.output.payload_digest,
                        )
                        .map_err(protocol)
                    })
                    .transpose()?;
                Ok(StructuralFrequencySampleArtifactRoots::new(
                    fields,
                    observations,
                ))
            })
            .collect::<NativeExecutionResult<Vec<_>>>()?;
        for history in &self.histories {
            roots.push(
                history
                    .checkpoint
                    .objects
                    .root_reference(access, ValueRefKind::ResultObject)
                    .map_err(protocol)?,
            );
            objects.extend(
                history
                    .checkpoint
                    .objects
                    .unique_references(access, ValueRefKind::ResultObject)
                    .map_err(protocol)?,
            );
        }
        let history_roots = self
            .histories
            .iter()
            .map(|history| {
                let reference = history
                    .checkpoint
                    .objects
                    .root_reference(access, ValueRefKind::ResultObject)
                    .map_err(protocol)?;
                Ok(StructuralFrequencyHistoryArtifactRoot::new(artifact_root(
                    &history.checkpoint.objects,
                    reference,
                )?))
            })
            .collect::<NativeExecutionResult<Vec<_>>>()?;
        let catalog = PublishedObjectCatalog::new(roots, objects).map_err(protocol)?;
        let index = StructuralFrequencyArtifactRootIndex::new(
            result_binding,
            receipt_binding,
            resource_binding,
            sample_roots,
            history_roots,
            &self.publication,
            &catalog,
        )
        .map_err(protocol)?;
        Ok((catalog, index))
    }
}

fn artifact_root(
    objects: &PreparedAnalysisObjects,
    reference: ValueRef,
) -> NativeExecutionResult<StructuralFrequencyArtifactRoot> {
    StructuralFrequencyArtifactRoot::new(
        reference,
        objects.manifest.kind,
        objects.manifest.logical_identity,
        objects.manifest.payload_digest,
    )
    .map_err(protocol)
}
