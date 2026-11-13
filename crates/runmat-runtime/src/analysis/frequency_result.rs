use std::fmt;
use std::sync::Arc;

use runmat_analysis_core::ProcedureStepId;
use runmat_analysis_fea::execution::{
    AnalysisNumericalObjectKind, StructuralFrequencyPublicationManifest,
    StructuralFrequencySolverResult, StructuralFrequencyWorkload,
};
use runmat_execution::value::{PublishedObjectCatalog, PublishedObjectSource};
use runmat_execution::Digest;

mod admission;

pub(in crate::analysis) use admission::{
    admit_modal_structural_frequency_result, admit_prestressed_modal_structural_frequency_result,
    admit_prestressed_structural_frequency_result, admit_structural_frequency_result,
    CompletedStructuralFrequencyStep, StructuralFrequencyPersistence,
};

use super::{AnalysisArtifactRoot, FrequencySampleArtifactRoots};

#[derive(Debug, Clone, serde::Serialize)]
pub enum StructuralStudyPrestressedFrequencyReductionResult {
    Direct,
    ModalReduced(Box<StructuralStudyPrestressedModalFrequencyReductionResult>),
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct StructuralStudyPrestressedModalFrequencyReductionResult {
    pub modal_step_id: ProcedureStepId,
    pub modal_result: runmat_analysis_fea::execution::StructuralModalSolverResult,
    pub modal_numerical_receipt: runmat_analysis_fea::execution::StructuralModalNumericalReceipt,
    pub basis_state: runmat_analysis_fea::execution::CanonicalSemanticState,
}

/// Ordered artifact root for one published structural-frequency history.
#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuralFrequencyHistoryArtifactRoot {
    checkpoint: AnalysisArtifactRoot,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    finalization: Option<StructuralFrequencyFinalizedHistoryArtifactRoots>,
}

#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuralFrequencyFinalizedHistoryArtifactRoots {
    pub catalog: AnalysisArtifactRoot,
    pub history: AnalysisArtifactRoot,
    pub series: Option<AnalysisArtifactRoot>,
}

impl StructuralFrequencyHistoryArtifactRoot {
    pub const fn new(checkpoint: AnalysisArtifactRoot) -> Self {
        Self {
            checkpoint,
            finalization: None,
        }
    }

    pub fn with_finalization(
        mut self,
        roots: StructuralFrequencyFinalizedHistoryArtifactRoots,
    ) -> Self {
        self.finalization = Some(roots);
        self
    }

    pub const fn finalization(&self) -> Option<&StructuralFrequencyFinalizedHistoryArtifactRoots> {
        self.finalization.as_ref()
    }

    pub const fn checkpoint(&self) -> &AnalysisArtifactRoot {
        &self.checkpoint
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct StructuralStudyPrestressedFrequencyNumericalResult {
    pub frozen_project_digest: Option<runmat_execution::Digest>,
    pub preload_step_id: ProcedureStepId,
    pub frequency_step_id: ProcedureStepId,
    pub preload_result: runmat_analysis_fea::execution::StructuralLinearWorkloadResult,
    pub preload_solver_result_identity: runmat_analysis_fea::execution::SolverResultIdentity,
    pub preload_numerical_receipt:
        runmat_analysis_fea::execution::StructuralLinearNumericalReceiptArtifact,
    pub prestress_state: runmat_analysis_fea::execution::CanonicalSemanticState,
    pub reduction: StructuralStudyPrestressedFrequencyReductionResult,
    pub frequency: StructuralStudyFrequencyNumericalResult,
}

/// A modal-extraction step and the modal-reduced frequency sweep (`modal_harmonic`) that
/// consumes its published basis.
#[derive(Debug, Clone, serde::Serialize)]
pub struct StructuralStudyModalFrequencyNumericalResult {
    pub frozen_project_digest: Option<runmat_execution::Digest>,
    pub modal_step_id: ProcedureStepId,
    pub frequency_step_id: ProcedureStepId,
    pub modal_result: runmat_analysis_fea::execution::StructuralModalSolverResult,
    pub modal_numerical_receipt: runmat_analysis_fea::execution::StructuralModalNumericalReceipt,
    pub basis_state: runmat_analysis_fea::execution::CanonicalSemanticState,
    pub frequency: StructuralStudyFrequencyNumericalResult,
}

/// Typed storage index over every semantic root in a structural-frequency closure.
///
/// Sample and history vectors preserve publication order. The generic catalog remains canonically
/// ordered by authority-bound value identity, so construction compares the complete declared root
/// inventory after canonical ordering rather than searching the catalog for roots by untrusted
/// metadata. These physical references do not participate in the placement-neutral numerical
/// result identity.
#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuralFrequencyArtifactRootIndex {
    result: AnalysisArtifactRoot,
    receipt: AnalysisArtifactRoot,
    resource_usage: AnalysisArtifactRoot,
    samples: Vec<FrequencySampleArtifactRoots>,
    histories: Vec<StructuralFrequencyHistoryArtifactRoot>,
}

impl StructuralFrequencyArtifactRootIndex {
    pub fn new(
        result: AnalysisArtifactRoot,
        receipt: AnalysisArtifactRoot,
        resource_usage: AnalysisArtifactRoot,
        samples: Vec<FrequencySampleArtifactRoots>,
        histories: Vec<StructuralFrequencyHistoryArtifactRoot>,
        publication: &StructuralFrequencyPublicationManifest,
        object_catalog: &PublishedObjectCatalog,
    ) -> Result<Self, String> {
        let index = Self {
            result,
            receipt,
            resource_usage,
            samples,
            histories,
        };
        index.validate(publication, object_catalog)?;
        Ok(index)
    }

    pub const fn result(&self) -> &AnalysisArtifactRoot {
        &self.result
    }

    pub const fn receipt(&self) -> &AnalysisArtifactRoot {
        &self.receipt
    }

    pub const fn resource_usage(&self) -> &AnalysisArtifactRoot {
        &self.resource_usage
    }

    pub fn samples(&self) -> &[FrequencySampleArtifactRoots] {
        &self.samples
    }

    pub fn histories(&self) -> &[StructuralFrequencyHistoryArtifactRoot] {
        &self.histories
    }

    pub fn validate(
        &self,
        publication: &StructuralFrequencyPublicationManifest,
        object_catalog: &PublishedObjectCatalog,
    ) -> Result<(), String> {
        publication.validate().map_err(|error| error.to_string())?;
        object_catalog
            .validate()
            .map_err(|error| error.to_string())?;
        self.result.validate_role(
            AnalysisNumericalObjectKind::SolverResult,
            publication.result().identity().digest(),
            "solver-result",
        )?;
        self.receipt.validate_role(
            AnalysisNumericalObjectKind::NumericalReceipt,
            publication.receipt().identity().digest(),
            "numerical-receipt",
        )?;
        self.resource_usage.validate_role(
            AnalysisNumericalObjectKind::ResourceUsage,
            publication.resource_usage().identity().digest(),
            "resource-usage",
        )?;

        if self.samples.len() != publication.samples().len() {
            return Err(
                "structural frequency artifact roots do not exactly cover published samples".into(),
            );
        }
        for (roots, sample) in self.samples.iter().zip(publication.samples()) {
            roots.validate_catalogs(&sample.fields, sample.observations.as_ref())?;
        }

        if self.histories.len() != publication.histories().len() {
            return Err(
                "structural frequency artifact roots do not exactly cover published histories"
                    .into(),
            );
        }
        for (root, history) in self.histories.iter().zip(publication.histories()) {
            root.checkpoint.validate_role(
                AnalysisNumericalObjectKind::ObservationHistoryCheckpoint,
                history.checkpoint.identity.digest(),
                "history-checkpoint",
            )?;
            match (&root.finalization, &history.finalization) {
                (Some(roots), Some(publication)) => {
                    roots.catalog.validate_role(
                        AnalysisNumericalObjectKind::ObservationFieldCatalog,
                        publication.request.source_catalog,
                        "history source catalog",
                    )?;
                    roots.history.validate_role(
                        AnalysisNumericalObjectKind::History,
                        publication.history.identity.digest(),
                        "finalized history",
                    )?;
                    match (&roots.series, &publication.series) {
                        (Some(root), Some(series)) => root.validate_role(
                            AnalysisNumericalObjectKind::SeriesReductionResult,
                            series.identity.digest(),
                            "history series reduction",
                        )?,
                        (None, None) => {}
                        _ => return Err("frequency series roots differ from publication".into()),
                    }
                }
                (None, None) => {}
                _ => return Err("frequency finalization roots differ from publication".into()),
            }
        }

        let mut declared_roots =
            Vec::with_capacity(3 + self.samples.len() * 2 + self.histories.len());
        declared_roots.push(self.result.reference.clone());
        declared_roots.push(self.receipt.reference.clone());
        declared_roots.push(self.resource_usage.reference.clone());
        for sample in &self.samples {
            declared_roots.push(sample.fields.reference.clone());
            if let Some(observations) = &sample.observations {
                declared_roots.push(observations.reference.clone());
            }
        }
        for history in &self.histories {
            declared_roots.push(history.checkpoint.reference.clone());
            if let Some(finalization) = &history.finalization {
                declared_roots.push(finalization.catalog.reference.clone());
                declared_roots.push(finalization.history.reference.clone());
                if let Some(series) = &finalization.series {
                    declared_roots.push(series.reference.clone());
                }
            }
        }
        declared_roots.sort_by_key(|reference| reference.id);
        declared_roots.dedup();
        super::artifact_root::validate_complete_root_inventory(declared_roots, object_catalog)
    }
}

/// Numerical and storage handoff for a completed structural frequency study.
///
/// The physical result and semantic publication are strongly typed and placement-neutral. The
/// authority-bound object catalog and source are erased only at the generic content-addressed
/// storage boundary, allowing local filesystem and remote downloaded sources to share one
/// admission and persistence path without embedding paths in FEA.
#[derive(Clone, serde::Serialize)]
pub struct StructuralStudyFrequencyNumericalResult {
    pub frozen_project_digest: Option<Digest>,
    pub step_id: ProcedureStepId,
    pub workload: StructuralFrequencyWorkload,
    pub result: StructuralFrequencySolverResult,
    pub publication: StructuralFrequencyPublicationManifest,
    pub object_catalog: PublishedObjectCatalog,
    artifact_roots: StructuralFrequencyArtifactRootIndex,
    #[serde(skip_serializing)]
    source: Arc<dyn PublishedObjectSource>,
}

impl fmt::Debug for StructuralStudyFrequencyNumericalResult {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("StructuralStudyFrequencyNumericalResult")
            .field("frozen_project_digest", &self.frozen_project_digest)
            .field("step_id", &self.step_id)
            .field("workload", &self.workload)
            .field("result", &self.result)
            .field("publication", &self.publication)
            .field("object_catalog", &self.object_catalog)
            .field("artifact_roots", &self.artifact_roots)
            .finish_non_exhaustive()
    }
}

impl StructuralStudyFrequencyNumericalResult {
    pub fn new<Source>(
        frozen_project_digest: Option<Digest>,
        workload: StructuralFrequencyWorkload,
        result: StructuralFrequencySolverResult,
        publication: StructuralFrequencyPublicationManifest,
        object_catalog: PublishedObjectCatalog,
        artifact_roots: StructuralFrequencyArtifactRootIndex,
        source: Source,
    ) -> Result<Self, String>
    where
        Source: PublishedObjectSource + 'static,
    {
        workload.validate().map_err(|error| error.to_string())?;
        result.validate().map_err(|error| error.to_string())?;
        publication
            .validate_for_workload(&workload)
            .map_err(|error| error.to_string())?;
        publication
            .receipt()
            .validate_for_result(&result)
            .map_err(|error| error.to_string())?;
        object_catalog
            .validate()
            .map_err(|error| error.to_string())?;
        if publication.result() != result.manifest() {
            return Err(
                "structural frequency publication differs from its numerical result".into(),
            );
        }
        artifact_roots.validate(&publication, &object_catalog)?;
        let step_id = workload.step_id().clone();
        Ok(Self {
            frozen_project_digest,
            step_id,
            workload,
            result,
            publication,
            object_catalog,
            artifact_roots,
            source: Arc::new(source),
        })
    }

    pub const fn artifact_roots(&self) -> &StructuralFrequencyArtifactRootIndex {
        &self.artifact_roots
    }

    pub const fn step_id(&self) -> &runmat_analysis_core::ProcedureStepId {
        &self.step_id
    }

    pub(in crate::analysis) fn persistence(&self) -> StructuralFrequencyPersistence {
        StructuralFrequencyPersistence {
            object_catalog: self.object_catalog.clone(),
            artifact_roots: self.artifact_roots.clone(),
            source: Arc::clone(&self.source),
        }
    }
}
