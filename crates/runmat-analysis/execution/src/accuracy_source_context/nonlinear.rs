//! Original terminal block and authored accuracy controls; no solve or interpolation.
use crate::{AnalysisBridgeError, AnalysisBridgeResult, PreparedAnalysisObjects};
use runmat_analysis_fea::{
    execution::{
        AccuracyEvaluationRequest, AnalysisNumericalObjectKind as Kind,
        NonlinearSolidBlockArtifact, NonlinearSolidNumericalReceiptArtifact,
        NonlinearSolidWorkload,
    },
    results::observation::ObservationPublicationCatalog,
};
use runmat_execution::Digest;
use runmat_meshing_core::{CanonicalMeshingContract, SolverMeshArtifact};

pub(super) fn validate(
    request: &AccuracyEvaluationRequest,
    receipt: &NonlinearSolidNumericalReceiptArtifact,
    observations: Option<&ObservationPublicationCatalog>,
    inputs: &[&PreparedAnalysisObjects],
) -> AnalysisBridgeResult<()> {
    if inputs.len() != 3 {
        return Err(AnalysisBridgeError::Identity(
            "nonlinear accuracy requires original workload, mesh and terminal block",
        ));
    }
    let unique = |kind| -> AnalysisBridgeResult<&PreparedAnalysisObjects> {
        let mut found = inputs.iter().copied().filter(|i| i.manifest.kind == kind);
        match (found.next(), found.next()) {
            (Some(i), None) => Ok(i),
            _ => Err(AnalysisBridgeError::Identity(
                "nonlinear accuracy repeats or lacks original source",
            )),
        }
    };
    for input in inputs {
        crate::accuracy_resource::authenticate_accuracy_objects(input)?;
    }
    let w = unique(Kind::NonlinearSolidWorkload)?;
    let m = unique(Kind::Mesh)?;
    let b = unique(Kind::NonlinearSolidBlockArtifact)?;
    let workload = NonlinearSolidWorkload::from_canonical_bytes(&w.payload())?;
    let mesh = SolverMeshArtifact::canonical_decode(&m.payload())
        .map_err(|e| AnalysisBridgeError::Invalid(e.to_string()))?;
    mesh.validate_canonical()
        .map_err(|e| AnalysisBridgeError::Invalid(e.to_string()))?;
    let block = NonlinearSolidBlockArtifact::from_canonical_bytes(
        &b.payload(),
        b.manifest.logical_identity,
    )?;
    receipt.validate_against(&workload, &block)?;
    let step = &workload.model.sequence.steps[0];
    if w.manifest.logical_identity != workload.identity()?.digest()
        || m.manifest.logical_identity != Digest::from_bytes(*mesh.canonical_digest.bytes())
        || m.manifest.logical_identity != block.request.topology
        || !block.complete
        || receipt.source_result() != request.source_result
        || workload.producer_identity()? != request.producer
        || request.plan != step.accuracy
        || observations.is_some() == step.observations.is_empty()
    {
        return Err(AnalysisBridgeError::Identity(
            "nonlinear accuracy original source or authored controls differ",
        ));
    }
    if let Some(catalog) = observations {
        catalog.validate_source_result(
            request.producer,
            request.source_kind,
            request.source_result,
        )?;
        catalog.validate_for(
            step,
            runmat_analysis_fea::execution::ObservationHistorySourceContext::Structural {
                source: runmat_analysis_fea::execution::StructuralObservationSource::NonlinearSolid(
                    &workload,
                ),
                mesh: Some(&mesh),
            },
        )?;
        let runmat_analysis_fea::results::observation::ObservationFieldCatalogSource::Structural {
            recovery_request,
            ..
        } = catalog.source().source()
        else {
            return Err(AnalysisBridgeError::Identity(
                "nonlinear accuracy requires original structural recovery catalog",
            ));
        };
        if recovery_request.as_ref() != &block.native_field_request(&workload, &mesh)? {
            return Err(AnalysisBridgeError::Identity(
                "nonlinear observation recovery differs from original terminal block",
            ));
        }
    }
    Ok(())
}
