use std::fmt;
use std::sync::Arc;

use runmat_analysis_core::ProcedureStepId;
use runmat_analysis_fea::execution::{
    AnalysisNumericalObjectKind, StructuralFrequencyPublicationManifest,
    StructuralFrequencySolverResult, StructuralFrequencyWorkload,
};
use runmat_execution::value::{PublishedObjectCatalog, PublishedObjectSource, ValueRef};
use runmat_execution::Digest;

use super::contracts::{
    AnalysisStructuralFrequencyRunOptions, AnalysisStudyStepNumericalResultData, FallbackEvent,
    QualityGate, QualityReason, QualityReasonCode, RunProvenance, RunStatus,
    StructuralFrequencyResultsData,
};
use super::result_policy::{classify_run_status, evaluate_identity_production_publication};
use super::{contracts, AnalysisRunIdentity};

/// One explicitly assigned semantic root in a structural-frequency object closure.
///
/// This is a process-local handoff contract. The reference locates the immutable analysis-object
/// manifest, while the remaining fields state exactly what that manifest must describe when the
/// runtime admits it. Root roles are assigned by [`StructuralFrequencyArtifactRootIndex`]; callers
/// must not infer them from media types, schemas, or catalog position.
#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuralFrequencyArtifactRoot {
    reference: ValueRef,
    expected_kind: AnalysisNumericalObjectKind,
    expected_logical_identity: Digest,
    expected_payload_digest: Digest,
}

impl StructuralFrequencyArtifactRoot {
    pub fn new(
        reference: ValueRef,
        expected_kind: AnalysisNumericalObjectKind,
        expected_logical_identity: Digest,
        expected_payload_digest: Digest,
    ) -> Result<Self, String> {
        if expected_logical_identity == Digest::sha256([])
            || expected_payload_digest == Digest::sha256([])
        {
            return Err(
                "structural frequency artifact root requires nonempty logical and payload identities"
                    .into(),
            );
        }
        Ok(Self {
            reference,
            expected_kind,
            expected_logical_identity,
            expected_payload_digest,
        })
    }

    pub fn reference(&self) -> &ValueRef {
        &self.reference
    }

    pub const fn expected_kind(&self) -> AnalysisNumericalObjectKind {
        self.expected_kind
    }

    pub const fn expected_logical_identity(&self) -> Digest {
        self.expected_logical_identity
    }

    pub const fn expected_payload_digest(&self) -> Digest {
        self.expected_payload_digest
    }

    fn validate_role(
        &self,
        expected_kind: AnalysisNumericalObjectKind,
        expected_logical_identity: Digest,
        role: &str,
    ) -> Result<(), String> {
        if self.expected_kind != expected_kind
            || self.expected_logical_identity != expected_logical_identity
            || self.expected_payload_digest == Digest::sha256([])
        {
            return Err(format!(
                "structural frequency {role} root differs from its semantic publication"
            ));
        }
        Ok(())
    }
}

/// Ordered artifact roots for one published structural-frequency sample.
#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuralFrequencySampleArtifactRoots {
    fields: StructuralFrequencyArtifactRoot,
    observations: Option<StructuralFrequencyArtifactRoot>,
}

impl StructuralFrequencySampleArtifactRoots {
    pub fn new(
        fields: StructuralFrequencyArtifactRoot,
        observations: Option<StructuralFrequencyArtifactRoot>,
    ) -> Self {
        Self {
            fields,
            observations,
        }
    }

    pub const fn fields(&self) -> &StructuralFrequencyArtifactRoot {
        &self.fields
    }

    pub const fn observations(&self) -> Option<&StructuralFrequencyArtifactRoot> {
        self.observations.as_ref()
    }
}

/// Ordered artifact root for one published structural-frequency history.
#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuralFrequencyHistoryArtifactRoot {
    checkpoint: StructuralFrequencyArtifactRoot,
}

impl StructuralFrequencyHistoryArtifactRoot {
    pub const fn new(checkpoint: StructuralFrequencyArtifactRoot) -> Self {
        Self { checkpoint }
    }

    pub const fn checkpoint(&self) -> &StructuralFrequencyArtifactRoot {
        &self.checkpoint
    }
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
    result: StructuralFrequencyArtifactRoot,
    receipt: StructuralFrequencyArtifactRoot,
    resource_usage: StructuralFrequencyArtifactRoot,
    samples: Vec<StructuralFrequencySampleArtifactRoots>,
    histories: Vec<StructuralFrequencyHistoryArtifactRoot>,
}

impl StructuralFrequencyArtifactRootIndex {
    pub fn new(
        result: StructuralFrequencyArtifactRoot,
        receipt: StructuralFrequencyArtifactRoot,
        resource_usage: StructuralFrequencyArtifactRoot,
        samples: Vec<StructuralFrequencySampleArtifactRoots>,
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

    pub const fn result(&self) -> &StructuralFrequencyArtifactRoot {
        &self.result
    }

    pub const fn receipt(&self) -> &StructuralFrequencyArtifactRoot {
        &self.receipt
    }

    pub const fn resource_usage(&self) -> &StructuralFrequencyArtifactRoot {
        &self.resource_usage
    }

    pub fn samples(&self) -> &[StructuralFrequencySampleArtifactRoots] {
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
            roots.fields.validate_role(
                AnalysisNumericalObjectKind::Field,
                sample.fields.identity.digest(),
                "field",
            )?;
            match (&roots.observations, &sample.observations) {
                (Some(root), Some(observations)) => root.validate_role(
                    AnalysisNumericalObjectKind::Observation,
                    observations.identity.digest(),
                    "observation",
                )?,
                (None, None) => {}
                _ => return Err(
                    "structural frequency observation roots differ from the published sample order"
                        .into(),
                ),
            }
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
        declared_roots.extend(
            self.histories
                .iter()
                .map(|history| history.checkpoint.reference.clone()),
        );
        declared_roots.sort_by_key(|reference| reference.id);
        if declared_roots != object_catalog.roots {
            return Err(
                "structural frequency artifact-root index differs from the published root inventory"
                    .into(),
            );
        }
        Ok(())
    }
}

/// Numerical and storage handoff for a completed structural frequency study.
///
/// The physical result and semantic publication are strongly typed and placement-neutral. The
/// authority-bound object catalog and source are erased only at the generic content-addressed
/// storage boundary, allowing local filesystem and remote downloaded sources to share one
/// admission and persistence path without embedding paths in FEA.
#[derive(Clone)]
pub struct StructuralStudyFrequencyNumericalResult {
    pub frozen_project_digest: Option<Digest>,
    pub step_id: ProcedureStepId,
    pub workload: StructuralFrequencyWorkload,
    pub result: StructuralFrequencySolverResult,
    pub publication: StructuralFrequencyPublicationManifest,
    pub object_catalog: PublishedObjectCatalog,
    artifact_roots: StructuralFrequencyArtifactRootIndex,
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

    pub(in crate::analysis) fn persistence(&self) -> StructuralFrequencyPersistence {
        StructuralFrequencyPersistence {
            object_catalog: self.object_catalog.clone(),
            artifact_roots: self.artifact_roots.clone(),
            source: Arc::clone(&self.source),
        }
    }
}

#[derive(Clone)]
pub(in crate::analysis) struct StructuralFrequencyPersistence {
    pub object_catalog: PublishedObjectCatalog,
    pub artifact_roots: StructuralFrequencyArtifactRootIndex,
    pub source: Arc<dyn PublishedObjectSource>,
}

pub(in crate::analysis) struct CompletedStructuralFrequencyStep {
    pub identity: AnalysisRunIdentity,
    pub run_id: String,
    pub run_status: RunStatus,
    pub numerically_acceptable: bool,
    pub production_publishable: bool,
    pub solver_convergence: QualityGate,
    pub result_quality: QualityGate,
    pub quality_reasons: Vec<QualityReason>,
    pub production_publication_reasons: Vec<super::ProductionPublicationReason>,
    pub provenance: RunProvenance,
    pub numerical_result: AnalysisStudyStepNumericalResultData,
    pub persistence: StructuralFrequencyPersistence,
}

pub(in crate::analysis) fn admit_structural_frequency_result(
    numerical: &StructuralStudyFrequencyNumericalResult,
    solver_mesh: &runmat_meshing_core::SolverMeshArtifact,
    options: &AnalysisStructuralFrequencyRunOptions,
    assigned_run_id: String,
) -> Result<CompletedStructuralFrequencyStep, String> {
    admit_structural_frequency_result_with_state(
        numerical,
        solver_mesh,
        options,
        assigned_run_id,
        None,
        None,
    )
}

pub(in crate::analysis) fn admit_prestressed_structural_frequency_result(
    numerical: &StructuralStudyFrequencyNumericalResult,
    solver_mesh: &runmat_meshing_core::SolverMeshArtifact,
    options: &AnalysisStructuralFrequencyRunOptions,
    assigned_run_id: String,
    prestress_state: &runmat_analysis_fea::execution::CanonicalSemanticState,
) -> Result<CompletedStructuralFrequencyStep, String> {
    admit_structural_frequency_result_with_state(
        numerical,
        solver_mesh,
        options,
        assigned_run_id,
        Some(prestress_state),
        None,
    )
}

pub(in crate::analysis) fn admit_prestressed_modal_structural_frequency_result(
    numerical: &StructuralStudyFrequencyNumericalResult,
    solver_mesh: &runmat_meshing_core::SolverMeshArtifact,
    options: &AnalysisStructuralFrequencyRunOptions,
    assigned_run_id: String,
    prestress_state: &runmat_analysis_fea::execution::CanonicalSemanticState,
    modal_state: &runmat_analysis_fea::execution::CanonicalSemanticState,
) -> Result<CompletedStructuralFrequencyStep, String> {
    admit_structural_frequency_result_with_state(
        numerical,
        solver_mesh,
        options,
        assigned_run_id,
        Some(prestress_state),
        Some(modal_state),
    )
}

fn admit_structural_frequency_result_with_state(
    numerical: &StructuralStudyFrequencyNumericalResult,
    solver_mesh: &runmat_meshing_core::SolverMeshArtifact,
    options: &AnalysisStructuralFrequencyRunOptions,
    assigned_run_id: String,
    prestress_state: Option<&runmat_analysis_fea::execution::CanonicalSemanticState>,
    modal_state: Option<&runmat_analysis_fea::execution::CanonicalSemanticState>,
) -> Result<CompletedStructuralFrequencyStep, String> {
    let identity = match (prestress_state, modal_state) {
        (Some(prestress), Some(modal)) => runmat_analysis_fea::pipeline::structural_frequency::admit_prestressed_modal_structural_frequency_publication(
            &numerical.workload,
            solver_mesh,
            prestress,
            modal,
            &numerical.result,
            &numerical.publication,
        ),
        (Some(state), None) => runmat_analysis_fea::pipeline::structural_frequency::admit_prestressed_structural_frequency_publication(
            &numerical.workload,
            solver_mesh,
            state,
            &numerical.result,
            &numerical.publication,
        ),
        (None, None) => runmat_analysis_fea::pipeline::structural_frequency::admit_structural_frequency_publication(
            &numerical.workload,
            solver_mesh,
            &numerical.result,
            &numerical.publication,
        ),
        (None, Some(_)) => return Err(
            "structural modal-frequency admission requires its prestress state".into(),
        ),
    }
    .map_err(|error| error.to_string())?;
    options
        .direct_solver
        .validate()
        .map_err(|error| error.to_string())?;

    let receipt = numerical.publication.receipt();
    let evidence = receipt.samples();
    let first = evidence
        .first()
        .ok_or_else(|| "structural frequency receipt contains no sample evidence".to_string())?;
    if evidence.iter().any(|sample| {
        sample.solver_method() != first.solver_method()
            || sample.solver_backend() != first.solver_backend()
            || sample.preconditioner() != first.preconditioner()
    }) {
        return Err(
            "structural frequency samples use inconsistent solver provenance within one run".into(),
        );
    }
    if first.solver_backend()
        != runmat_analysis_fea::solve::backend::kind::LinearAlgebraBackendKind::CpuReference
        || numerical.workload.backend() != runmat_analysis_fea::ComputeBackend::Cpu
    {
        return Err(
            "current structural frequency runtime publication admits only governed CPU execution"
                .into(),
        );
    }

    let relative_tolerance = options.direct_solver.relative_tolerance;
    let maximum_solver_residual = evidence
        .iter()
        .map(|sample| sample.solver_relative_residual())
        .fold(0.0_f64, f64::max);
    let maximum_balance_error = evidence
        .iter()
        .map(|sample| sample.physical_relative_balance_error())
        .fold(0.0_f64, f64::max);
    let solver_convergence = if maximum_solver_residual <= relative_tolerance {
        QualityGate::Pass
    } else {
        QualityGate::Fail
    };
    let result_quality = if maximum_balance_error <= relative_tolerance {
        QualityGate::Pass
    } else {
        QualityGate::Fail
    };
    let mut quality_reasons = Vec::new();
    if solver_convergence != QualityGate::Pass {
        quality_reasons.push(QualityReason {
            code: QualityReasonCode::StructuralFrequencyResidualExceeded,
            detail: format!(
                "maximum structural frequency relative solver residual {maximum_solver_residual} exceeds {relative_tolerance}"
            ),
        });
    }
    if result_quality != QualityGate::Pass {
        quality_reasons.push(QualityReason {
            code: QualityReasonCode::StructuralFrequencyBalanceExceeded,
            detail: format!(
                "maximum structural frequency physical balance error {maximum_balance_error} exceeds {relative_tolerance}"
            ),
        });
    }
    let numerically_acceptable =
        solver_convergence == QualityGate::Pass && result_quality == QualityGate::Pass;
    let (production_publishable, production_publication_reasons) =
        evaluate_identity_production_publication(&identity, numerically_acceptable);
    let run_status = classify_run_status(result_quality, numerically_acceptable);

    Ok(CompletedStructuralFrequencyStep {
        identity,
        run_id: assigned_run_id,
        run_status,
        numerically_acceptable,
        production_publishable,
        solver_convergence,
        result_quality,
        quality_reasons,
        production_publication_reasons,
        provenance: RunProvenance {
            requested_backend: numerical.workload.backend(),
            execution_backend: runmat_analysis_fea::ExecutionBackend::Cpu,
            solver_backend: first.solver_backend(),
            solver_device_apply_k_ratio: 0.0,
            solver_host_sync_count: 0,
            precision_mode: contracts::format_precision_mode(options.precision_mode),
            deterministic_mode: options.deterministic_mode,
            solver_method: first.solver_method(),
            preconditioner: first.preconditioner(),
            quality_policy: contracts::format_quality_policy(options.quality_policy),
            fallback_events: Vec::<FallbackEvent>::new(),
        },
        numerical_result: AnalysisStudyStepNumericalResultData::StructuralFrequency(Box::new(
            StructuralFrequencyResultsData {
                publication: numerical.publication.clone(),
            },
        )),
        persistence: numerical.persistence(),
    })
}
