//! The thermal solve's retained time samples as a sampled-history observation source.
use crate::{
    import_analysis_output_set, sampled_history_observations::SampledHistorySource,
    AnalysisBridgeError, AnalysisBridgeResult, AnalysisHostWorkload, AnalysisOutputPolicy,
    ImportedThermalResult, ObservationHistoryFinalizationSource, ObservationPreparation,
    ObservationPublishedSource, ObservationWorkloadSource, PreparedAnalysisObjects,
    PreparedThermalTask, ThermalStudyPreparation,
};
use runmat_analysis_core::ProcedureStep;
use runmat_analysis_fea::{
    execution::{
        AnalysisNumericalObjectKind as Kind, ObservationHistorySourceContext, SequenceIdentity,
    },
    results::{
        field::RecoveredFieldSample,
        observation::{ObservationFieldCatalog, ObservationLimits},
    },
};
use runmat_execution::ProgramRevision;
use runmat_execution_artifact::cache::CacheImport;

pub(super) struct Context<'a> {
    pub setup: &'a ThermalStudyPreparation,
    pub solve: &'a PreparedThermalTask,
    pub result: &'a ImportedThermalResult,
}
impl Context<'_> {
    fn input(&self, kind: Kind) -> AnalysisBridgeResult<PreparedAnalysisObjects> {
        self.solve
            .inputs
            .iter()
            .find(|input| input.manifest.kind == kind)
            .cloned()
            .ok_or(AnalysisBridgeError::Identity(
                "thermal observation source input absent",
            ))
    }
}

impl SampledHistorySource for Context<'_> {
    fn step(&self) -> &ProcedureStep {
        self.setup.selected_step()
    }
    fn solve_host(&self) -> &AnalysisHostWorkload {
        &self.solve.host
    }
    fn program_revision(&self) -> &ProgramRevision {
        &self.setup.program_revision
    }
    fn output_policy(&self) -> AnalysisOutputPolicy {
        self.setup.output_policy
    }
    fn sample_count(&self) -> usize {
        self.result.manifest.fields.len()
    }
    fn observation_preparation(
        &self,
        source: &impl CacheImport,
        ordinal: usize,
    ) -> AnalysisBridgeResult<(ObservationPreparation, RecoveredFieldSample)> {
        let output =
            import_analysis_output_set(source, self.result.response.clone(), &self.solve.host)?
                .manifest
                .outputs
                .into_iter()
                .find(|output| output.kind == Kind::SolverResult)
                .ok_or(AnalysisBridgeError::Identity(
                    "thermal solver result output absent",
                ))?;
        let coordinate = self.result.manifest.fields[ordinal]
            .first()
            .ok_or(AnalysisBridgeError::Identity(
                "thermal sample has no fields",
            ))?
            .sample;
        let policy = self.setup.output_policy;
        Ok((
            ObservationPreparation {
                source_host: self.solve.host.clone(),
                source: ObservationPublishedSource::Thermal {
                    manifest: self.result.manifest.clone(),
                    output,
                    access: self.solve.host.artifact_access.clone(),
                    sample_ordinal: ordinal,
                },
                source_workload: ObservationWorkloadSource::Thermal(
                    self.input(Kind::ThermalWorkload)?,
                ),
                mesh: Some(self.input(Kind::Mesh)?),
                accelerator_requests: Vec::new(),
                sequence: SequenceIdentity::from_solver_contract(
                    &self.setup.product.workload.model.sequence,
                )?,
                step: self.step().clone(),
                limits: ObservationLimits::default(),
                program_revision: self.setup.program_revision.clone(),
                chunk_bytes: policy.chunk_bytes(),
                inventory_limits: policy.inventory_limits(),
                local_execution: self.setup.local_execution,
            },
            coordinate,
        ))
    }
    fn field_catalog(&self) -> AnalysisBridgeResult<ObservationFieldCatalog> {
        Ok(ObservationFieldCatalog::thermal(
            self.result.manifest.clone(),
        )?)
    }
    fn finalization_source(&self) -> AnalysisBridgeResult<ObservationHistoryFinalizationSource> {
        Ok(ObservationHistoryFinalizationSource::Thermal {
            workload: self.input(Kind::ThermalWorkload)?,
            mesh: self.input(Kind::Mesh)?,
        })
    }
    fn history_context(&self) -> ObservationHistorySourceContext<'_> {
        ObservationHistorySourceContext::Thermal {
            workload: &self.setup.product.workload,
            mesh: &self.setup.mesh,
        }
    }
}
