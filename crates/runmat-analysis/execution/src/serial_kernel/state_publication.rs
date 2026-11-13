use super::*;
use runmat_analysis_fea::execution::SemanticStatePublicationProfileSource;
mod fluid;
mod nonlinear;

pub(crate) fn execute(
    invocation: AnalysisKernelInvocation<'_>,
) -> AnalysisBridgeResult<Vec<AnalysisKernelOutput>> {
    if optional_unique_input(
        &invocation,
        AnalysisNumericalObjectKind::NonlinearSolidWorkload,
    )?
    .is_some()
    {
        return nonlinear::execute(invocation);
    }
    if optional_unique_input(
        &invocation,
        AnalysisNumericalObjectKind::FluidSteadyWorkload,
    )?
    .is_some()
    {
        return fluid::execute(invocation);
    }
    let publication = unique_input(
        &invocation,
        AnalysisNumericalObjectKind::StatePublicationRequest,
    )?;
    let solver_result = unique_input(&invocation, AnalysisNumericalObjectKind::SolverResult)?;
    if invocation.inputs.len() != 4
        || invocation.host.workload.outputs != vec![AnalysisNumericalObjectKind::SemanticState]
    {
        return Err(AnalysisBridgeError::Invalid(
            "state publication requires request, result, profile catalog, resource plan and one state output"
                .into(),
        ));
    }
    let publication =
        SemanticStatePublicationRequest::from_canonical_bytes(&publication.payload())?;
    if solver_result.manifest.logical_identity != publication.source_result.digest() {
        return Err(AnalysisBridgeError::Identity(
            "state publication received a substituted solver result",
        ));
    }
    let profiles = unique_input(
        &invocation,
        AnalysisNumericalObjectKind::StatePublicationProfiles,
    )?;
    let catalog = runmat_analysis_fea::execution::StatePublicationProfiles::from_canonical_bytes(
        &profiles.payload(),
    )?;
    let plan_input = unique_input(&invocation, AnalysisNumericalObjectKind::ResourcePlan)?;
    let plan = FeaResourcePlan::from_canonical_bytes(&plan_input.payload())?;
    if invocation.host.workload.step != publication.producer
        || plan_input.manifest.logical_identity != plan.identity.digest()
        || plan
            != crate::build_prepared_state_publication_resource_plan(
                &publication,
                &catalog,
                invocation.inputs,
            )?
    {
        return Err(AnalysisBridgeError::Identity(
            "state publication resource plan does not converge from authoritative inputs",
        ));
    }
    let check_profile = |source| -> AnalysisBridgeResult<()> {
        if publication.encoding_profile(source)? != catalog.entry(&publication)?.profile() {
            return Err(AnalysisBridgeError::Identity(
                "state publication profile differs from its numerical source",
            ));
        }
        Ok(())
    };
    let state = match publication.source {
        SemanticStatePublicationSource::NonlinearSolidOriginalState => {
            return Err(AnalysisBridgeError::Identity(
                "nonlinear material history requires its original typed workload/block closure",
            ));
        }
        SemanticStatePublicationSource::FluidPrimaryFields => {
            return Err(AnalysisBridgeError::Identity(
                "fluid state requires its original immutable closure",
            ));
        }
        SemanticStatePublicationSource::MaxwellNativeFields => {
            let result =
                runmat_analysis_fea::execution::MaxwellHarmonicSolverResult::from_canonical_bytes(
                    &solver_result.payload(),
                )?;
            result.validate_publication_request(&publication)?;
            check_profile(SemanticStatePublicationProfileSource::MaxwellNativeFields(
                &result,
            ))?;
            CanonicalSemanticState::from_maxwell_publication(&publication, &result)?
        }
        SemanticStatePublicationSource::MagnetostaticNativeFields => {
            let result =
                runmat_analysis_fea::execution::MagnetostaticSolverResult::from_canonical_bytes(
                    &solver_result.payload(),
                )?;
            result.validate_publication_request(&publication)?;
            check_profile(
                SemanticStatePublicationProfileSource::MagnetostaticNativeFields(&result),
            )?;
            CanonicalSemanticState::from_magnetostatic_publication(&publication, &result)?
        }
        SemanticStatePublicationSource::StaticElectricalNativeFields => {
            let result =
                runmat_analysis_fea::execution::ElectricalStaticSolverResult::from_canonical_bytes(
                    &solver_result.payload(),
                )?;
            result.validate_publication_request(&publication)?;
            check_profile(
                SemanticStatePublicationProfileSource::StaticElectricalNativeFields(&result),
            )?;
            CanonicalSemanticState::from_electrical_static_publication(&publication, &result)?
        }
        SemanticStatePublicationSource::AcousticNativeFields => {
            let result =
                runmat_analysis_fea::execution::AcousticHarmonicSolverResult::from_canonical_bytes(
                    &solver_result.payload(),
                )?;
            result.validate_publication_request(&publication)?;
            check_profile(SemanticStatePublicationProfileSource::AcousticNativeFields(
                &result,
            ))?;
            CanonicalSemanticState::from_acoustic_publication(&publication, &result)?
        }
        SemanticStatePublicationSource::AcousticResponseFields => {
            let result =
                runmat_analysis_fea::execution::AcousticResponseSolverResult::from_canonical_bytes(
                    &solver_result.payload(),
                )?;
            result.validate_publication_request(&publication)?;
            let fields = result.state_fields();
            check_profile(SemanticStatePublicationProfileSource::AcousticResponseFields(&fields))?;
            CanonicalSemanticState::from_acoustic_response_publication(&publication, &result)?
        }
        SemanticStatePublicationSource::ThermalPrimaryField => {
            let manifest = crate::decode_prepared_thermal_manifest(solver_result)?;
            check_profile(SemanticStatePublicationProfileSource::Thermal(&manifest))?;
            crate::decode_prepared_thermal_state(&publication, solver_result)?
        }
        SemanticStatePublicationSource::StructuralPrimaryFields => {
            let solver_result =
                StructuralLinearWorkloadResult::from_canonical_bytes(&solver_result.payload())?;
            solver_result
                .validate_thermal_source(publication.producer, publication.source_result)?;
            if solver_result
                .constraint_result
                .topology
                .discretization_digest()
                != publication.topology_digest
            {
                return Err(AnalysisBridgeError::Identity(
                    "structural state publication topology differs from its solver result",
                ));
            }
            check_profile(
                SemanticStatePublicationProfileSource::StructuralPrimaryFields(
                    &solver_result.run.fields,
                ),
            )?;
            CanonicalSemanticState::from_publication(&publication, &solver_result.run.fields)?
        }
        SemanticStatePublicationSource::StructuralLinearization => {
            let solver_result =
                StructuralLinearWorkloadResult::from_canonical_bytes(&solver_result.payload())?;
            solver_result
                .validate_thermal_source(publication.producer, publication.source_result)?;
            let linearization = solver_result.linearization.ok_or_else(|| {
                AnalysisBridgeError::Invalid(
                    "solver result did not publish its declared structural linearization".into(),
                )
            })?;
            check_profile(
                SemanticStatePublicationProfileSource::StructuralLinearization(&linearization),
            )?;
            CanonicalSemanticState::from_linearization_publication(&publication, linearization)?
        }
        SemanticStatePublicationSource::ModalBasis => {
            let solver_result =
                StructuralModalSolverResult::from_canonical_bytes(&solver_result.payload())?;
            if solver_result.topology_digest() != publication.topology_digest {
                return Err(AnalysisBridgeError::Identity(
                    "modal state publication topology differs from its solver result",
                ));
            }
            check_profile(SemanticStatePublicationProfileSource::ModalBasis(
                solver_result.basis(),
            ))?;
            CanonicalSemanticState::from_modal_publication(
                &publication,
                solver_result.basis().clone(),
            )?
        }
        SemanticStatePublicationSource::StructuralResponseEnvelope => {
            let solver_result = StructuralModalResponseSpectrumSolverResult::from_canonical_bytes(
                &solver_result.payload(),
            )?;
            check_profile(
                SemanticStatePublicationProfileSource::StructuralResponseEnvelope(&solver_result),
            )?;
            CanonicalSemanticState::from_structural_response_envelope_publication(
                &publication,
                &solver_result,
            )?
        }
        SemanticStatePublicationSource::StructuralRandomStatistics => {
            let solver_result = StructuralRandomResponseSolverResult::from_canonical_bytes(
                &solver_result.payload(),
            )?;
            check_profile(
                SemanticStatePublicationProfileSource::StructuralRandomStatistics(&solver_result),
            )?;
            CanonicalSemanticState::from_structural_random_statistics_publication(
                &publication,
                &solver_result,
            )?
        }
    };
    Ok(vec![AnalysisKernelOutput::whole(
        AnalysisNumericalObjectKind::SemanticState,
        state.identity.digest(),
        state.canonical_bytes()?,
    )])
}
