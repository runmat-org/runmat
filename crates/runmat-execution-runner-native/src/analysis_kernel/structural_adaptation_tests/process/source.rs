//! Import original request-bound closures without preparing or re-solving the model.
use super::*;
use runmat_analysis_execution::{
    import_analysis_objects, import_analysis_output_set, AnalysisBridgeError, AnalysisBridgeResult,
    AnalysisHostWorkload, PreparedAnalysisObjects,
};
use runmat_execution_artifact::{ProgramExecutionRequest, ProgramExecutionResponse};

pub(super) struct OriginalSolve {
    pub host: AnalysisHostWorkload,
    pub inputs: Vec<PreparedAnalysisObjects>,
    pub result: PreparedAnalysisObjects,
    pub receipt: PreparedAnalysisObjects,
}
impl OriginalSolve {
    pub(super) fn import(
        store: &impl runmat_execution_artifact::cache::CacheImport,
        request: &ProgramExecutionRequest,
        response: &ProgramExecutionResponse,
    ) -> AnalysisBridgeResult<Self> {
        let host = AnalysisHostWorkload::from_program_request(request, store)?;
        let roots = AnalysisHostWorkload::semantic_input_roots(request)?;
        let inputs = host
            .workload
            .inputs
            .iter()
            .zip(&roots)
            .filter(|(input, _)| {
                matches!(
                    input.kind,
                    AnalysisNumericalObjectKind::SolverInvocation
                        | AnalysisNumericalObjectKind::Mesh
                )
            })
            .map(|(input, root)| {
                import_analysis_objects(
                    store,
                    root,
                    input.kind,
                    input.logical_identity,
                    input.payload_digest,
                    &host.artifact_access,
                    ObjectInventoryLimits::default(),
                )
            })
            .collect::<AnalysisBridgeResult<Vec<_>>>()?;
        let completed = import_analysis_output_set(store, response.clone(), &host)?;
        let load = |kind| -> AnalysisBridgeResult<PreparedAnalysisObjects> {
            let mut matches = completed.manifest.outputs.iter().filter(|o| o.kind == kind);
            let output = matches.next().ok_or(AnalysisBridgeError::Identity(
                "original cold native output is absent",
            ))?;
            if matches.next().is_some() {
                return Err(AnalysisBridgeError::Identity(
                    "original cold native output is duplicated",
                ));
            }
            import_analysis_objects(
                store,
                &output.root,
                output.kind,
                output.logical_identity,
                output.payload_digest,
                &host.artifact_access,
                ObjectInventoryLimits::default(),
            )
        };
        Ok(Self {
            result: load(AnalysisNumericalObjectKind::SolverResult)?,
            receipt: load(AnalysisNumericalObjectKind::NumericalReceipt)?,
            host,
            inputs,
        })
    }
    pub(super) fn source(&self) -> StructuralAdaptationNativeSource<'_> {
        let input = |kind| {
            self.inputs
                .iter()
                .find(|i| i.manifest.kind == kind)
                .unwrap()
        };
        StructuralAdaptationNativeSource {
            host: &self.host,
            invocation: input(AnalysisNumericalObjectKind::SolverInvocation),
            mesh: input(AnalysisNumericalObjectKind::Mesh),
            result: &self.result,
            receipt: &self.receipt,
        }
    }
    pub(super) fn verify_affine_original(&self) {
        let mesh = SolverMeshArtifact::canonical_decode(&self.source().mesh.payload()).unwrap();
        let result =
            StructuralLinearWorkloadResult::from_canonical_bytes(&self.result.payload()).unwrap();
        fixture::assert_affine_original(&mesh, &result.run);
        assert!(result.run.structural_static_convergence.unwrap().converged);
    }
}
