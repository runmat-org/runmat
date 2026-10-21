//! Structural receipts name the producing host contract. Authenticate its selected original
//! inputs before using their authored requirements; a catalog cannot supply its own authority.
use runmat_analysis_fea::execution::{
    AccuracyEvaluationRequest, AccuracySourceKind, AccuracySourceReceipt,
    AnalysisNumericalObjectKind as Kind, AnalysisWorkKind, ObservationHistorySourceContext,
    SequenceIdentity, StepIdentity, StructuralLinearWorkload,
    StructuralModalResponseSpectrumWorkload, StructuralObservationSource,
    StructuralRandomResponseWorkload,
};
use runmat_analysis_fea::results::observation::ObservationPublicationCatalog;
use runmat_canonical_codec::{decode_contract, encode_contract, CanonicalLimits};
use runmat_execution::Digest;
use runmat_execution_artifact::{object::ObjectInventoryLimits, ObjectNamespace};
use runmat_meshing_core::{CanonicalMeshingContract, SolverMeshArtifact};

use crate::{
    prepare_analysis_objects, AnalysisBridgeError, AnalysisBridgeResult, AnalysisWorkloadRequest,
    PreparedAnalysisObjects,
};

const PREFIX: &[u8] = b"runmat-analysis-accuracy-source-context/v1\0";
const LIMITS: CanonicalLimits = CanonicalLimits::new(16 * 1024 * 1024, 1_000_000, 65_536, 64);

pub(crate) fn prepare_contract(
    source: &AnalysisWorkloadRequest,
    chunk_bytes: usize,
    limits: ObjectInventoryLimits,
) -> AnalysisBridgeResult<PreparedAnalysisObjects> {
    source.validate()?;
    let bytes = encode_contract(PREFIX, "accuracy-source-context", source, LIMITS)
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    prepare_analysis_objects(
        Kind::AccuracySourceContext,
        source.identity.digest(),
        &bytes,
        chunk_bytes,
        ObjectNamespace::InputValue,
        limits,
    )
}

pub(super) fn validate(
    request: &AccuracyEvaluationRequest,
    receipt: &AccuracySourceReceipt,
    observations: Option<&ObservationPublicationCatalog>,
    inputs: &[&PreparedAnalysisObjects],
) -> AnalysisBridgeResult<()> {
    let unique = |kind| -> AnalysisBridgeResult<&PreparedAnalysisObjects> {
        let mut matches = inputs
            .iter()
            .copied()
            .filter(|input| input.manifest.kind == kind);
        match (matches.next(), matches.next()) {
            (Some(input), None) => Ok(input),
            _ => Err(AnalysisBridgeError::Identity(
                "structural accuracy requires unique source inputs",
            )),
        }
    };
    let descriptor = unique(Kind::AccuracySourceContext)?;
    let source: AnalysisWorkloadRequest = decode_contract(
        PREFIX,
        "accuracy-source-context",
        &descriptor.payload(),
        LIMITS,
    )
    .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    source.validate()?;
    let (work_kind, input_kind) = match request.source_kind {
        AccuracySourceKind::StructuralLinear => (AnalysisWorkKind::Solve, Kind::SolverInvocation),
        AccuracySourceKind::StructuralModalResponseSpectrum => (
            AnalysisWorkKind::AssembleStructuralModalResponseSpectrumResult,
            Kind::StructuralModalResponseSpectrumWorkload,
        ),
        AccuracySourceKind::StructuralRandomResponse => (
            AnalysisWorkKind::AssembleStructuralRandomResponseResult,
            Kind::StructuralRandomResponseWorkload,
        ),
        AccuracySourceKind::NonlinearSolid
        | AccuracySourceKind::FluidSteady
        | AccuracySourceKind::MaxwellHarmonic
        | AccuracySourceKind::Thermal
        | AccuracySourceKind::Magnetostatic
        | AccuracySourceKind::AcousticHarmonic
        | AccuracySourceKind::AcousticResponse
        | AccuracySourceKind::ElectricalStatic => {
            return Err(AnalysisBridgeError::Identity(
                "nonstructural source used structural context",
            ))
        }
    };
    if source.identity != receipt.source_workload()
        || descriptor.manifest.logical_identity != source.identity.digest()
        || source.step != request.producer
        || source.kind != work_kind
    {
        return Err(AnalysisBridgeError::Identity(
            "structural accuracy context differs from numerical receipt",
        ));
    }
    let workload = unique(input_kind)?;
    let mesh_required = source.inputs.iter().any(|input| input.kind == Kind::Mesh);
    if inputs.len() != 2 + usize::from(mesh_required) {
        return Err(AnalysisBridgeError::Identity(
            "structural accuracy context inventory differs from source",
        ));
    }
    // The receipt-bound workload identity authenticates semantic identities AND payload digests.
    // Root fields check internal handle consistency, not immutable transport provenance: the
    // same canonical payload may be rechunked without changing the producing workload identity.
    for input in inputs
        .iter()
        .filter(|input| input.manifest.kind != Kind::AccuracySourceContext)
    {
        let mut contracts = source
            .inputs
            .iter()
            .filter(|contract| contract.kind == input.manifest.kind);
        let contract = contracts.next().ok_or(AnalysisBridgeError::Identity(
            "accuracy input absent from source",
        ))?;
        if contracts.next().is_some()
            || contract.logical_identity != input.manifest.logical_identity
            || contract.payload_digest != input.manifest.payload_digest
            || contract.manifest_digest != input.root.digest
            || contract.manifest_encoded_length != input.root.encoded_length
        {
            return Err(AnalysisBridgeError::Identity(
                "accuracy input differs from original source contract",
            ));
        }
    }
    let mesh = if mesh_required {
        let objects = unique(Kind::Mesh)?;
        let mesh = SolverMeshArtifact::canonical_decode(&objects.payload())
            .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
        if Digest::from_bytes(*mesh.canonical_digest.bytes()) != objects.manifest.logical_identity {
            return Err(AnalysisBridgeError::Identity(
                "accuracy mesh differs from canonical identity",
            ));
        }
        Some(mesh)
    } else {
        None
    };
    match request.source_kind {
        AccuracySourceKind::StructuralLinear => {
            let workload = StructuralLinearWorkload::from_canonical_bytes(&workload.payload())?;
            validate_authored(
                request,
                observations,
                StructuralObservationSource::Linear(&workload),
                mesh.as_ref(),
            )
        }
        AccuracySourceKind::StructuralModalResponseSpectrum => {
            let workload =
                StructuralModalResponseSpectrumWorkload::from_canonical_bytes(&workload.payload())?;
            validate_authored(
                request,
                observations,
                StructuralObservationSource::ModalResponseSpectrum(&workload),
                mesh.as_ref(),
            )
        }
        AccuracySourceKind::StructuralRandomResponse => {
            let workload =
                StructuralRandomResponseWorkload::from_canonical_bytes(&workload.payload())?;
            validate_authored(
                request,
                observations,
                StructuralObservationSource::RandomResponse(&workload),
                mesh.as_ref(),
            )
        }
        AccuracySourceKind::NonlinearSolid
        | AccuracySourceKind::FluidSteady
        | AccuracySourceKind::MaxwellHarmonic
        | AccuracySourceKind::Thermal
        | AccuracySourceKind::Magnetostatic
        | AccuracySourceKind::AcousticHarmonic
        | AccuracySourceKind::AcousticResponse
        | AccuracySourceKind::ElectricalStatic => {
            unreachable!("rejected above")
        }
    }
}

fn validate_authored(
    request: &AccuracyEvaluationRequest,
    observations: Option<&ObservationPublicationCatalog>,
    source: StructuralObservationSource<'_>,
    mesh: Option<&SolverMeshArtifact>,
) -> AnalysisBridgeResult<()> {
    source.validate()?;
    let model = source.model();
    let step = source.procedure_step()?;
    let producer = StepIdentity::from_solver_contract(
        SequenceIdentity::from_solver_contract(&model.sequence)?,
        step,
    )?;
    if producer != request.producer
        || request.plan != step.accuracy
        || observations.is_some() == step.observations.is_empty()
    {
        return Err(AnalysisBridgeError::Identity(
            "structural accuracy differs from authored requirements",
        ));
    }
    if let Some(catalog) = observations {
        catalog.validate_source_result(
            request.producer,
            request.source_kind,
            request.source_result,
        )?;
        // Statistical steps publish through the family-neutral native mesh catalog.
        let context = match (catalog.source().source(), mesh) {
            (
                runmat_analysis_fea::results::observation::ObservationFieldCatalogSource::VolumeMeshRecovery { .. },
                Some(mesh),
            ) => ObservationHistorySourceContext::VolumeMesh { mesh },
            _ => ObservationHistorySourceContext::Structural { source, mesh },
        };
        catalog.validate_for(step, context)?;
    }
    Ok(())
}
