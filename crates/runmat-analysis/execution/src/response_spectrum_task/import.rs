use super::*;

pub fn import_structural_modal_response_spectrum_result(
    source: &impl CacheImport,
    response: ProgramExecutionResponse,
    prepared: &PreparedStructuralModalResponseSpectrumAssemblyTask,
    limits: ObjectInventoryLimits,
) -> AnalysisBridgeResult<ImportedStructuralModalResponseSpectrumResult> {
    response.validate_against(&prepared.program.request)?;
    let retained_response = response.clone();
    let output_set = import_analysis_output_set(source, response, &prepared.host)?;
    let [result_output, receipt_output, field_recovery_output, profiles_output] =
        output_set.manifest.outputs.as_slice()
    else {
        return Err(AnalysisBridgeError::Invalid(
            "response-spectrum fan-in must return result, receipt, field recovery and publication profiles"
                .into(),
        ));
    };
    let identity = SolverResultIdentity::new(
        prepared.host.workload.identity,
        result_output.payload_digest,
    )?;
    if result_output.logical_identity != identity.digest() {
        return Err(AnalysisBridgeError::Identity(
            "response-spectrum result root differs from its workload-bound identity",
        ));
    }
    let result_objects = import_analysis_objects(
        source,
        &result_output.root,
        AnalysisNumericalObjectKind::SolverResult,
        result_output.logical_identity,
        result_output.payload_digest,
        &prepared.host.artifact_access,
        limits,
    )?;
    let receipt_objects = import_analysis_objects(
        source,
        &receipt_output.root,
        AnalysisNumericalObjectKind::NumericalReceipt,
        receipt_output.logical_identity,
        receipt_output.payload_digest,
        &prepared.host.artifact_access,
        limits,
    )?;
    let field_recovery_objects = import_analysis_objects(
        source,
        &field_recovery_output.root,
        AnalysisNumericalObjectKind::StructuralModalResponseSpectrumFieldRecovery,
        field_recovery_output.logical_identity,
        field_recovery_output.payload_digest,
        &prepared.host.artifact_access,
        limits,
    )?;
    let result = decode_prepared_structural_modal_response_spectrum_solver_result(&result_objects)?;
    result.validate_for(&prepared.workload)?;
    if result.assembly() != &prepared.request {
        return Err(AnalysisBridgeError::Identity(
            "response-spectrum result differs from its prepared fan-in",
        ));
    }
    let numerical_receipt =
        decode_prepared_structural_modal_response_spectrum_numerical_receipt(&receipt_objects)?;
    numerical_receipt.validate_against(identity, &result)?;
    if receipt_output.logical_identity != numerical_receipt.identity().digest() {
        return Err(AnalysisBridgeError::Identity(
            "response-spectrum receipt root differs from its canonical identity",
        ));
    }
    let field_recovery =
        decode_prepared_structural_modal_response_spectrum_field_recovery(&field_recovery_objects)?;
    field_recovery
        .validate_for_result(result.numerical())
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    if field_recovery_output.logical_identity
        != field_recovery
            .identity()
            .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?
    {
        return Err(AnalysisBridgeError::Identity(
            "response-spectrum field-recovery root differs from its canonical identity",
        ));
    }
    let state_publication_profile_objects = import_analysis_objects(
        source,
        &profiles_output.root,
        AnalysisNumericalObjectKind::StatePublicationProfiles,
        profiles_output.logical_identity,
        profiles_output.payload_digest,
        &prepared.host.artifact_access,
        limits,
    )?;
    let state_publication_profiles = crate::structural_state_profiles::validate_profiles(
        &state_publication_profile_objects,
        crate::structural_state_publication_profiles(
            prepared.workload.model(),
            prepared.workload.procedure_step()?,
            identity,
            result_objects.manifest.payload_digest,
            crate::StructuralStateProfileResult::ResponseSpectrum(&result),
        )?,
    )?;
    let mut expected = result_objects
        .unique_references(&prepared.host.artifact_access, ValueRefKind::ResultObject)?;
    expected.extend(
        receipt_objects
            .unique_references(&prepared.host.artifact_access, ValueRefKind::ResultObject)?,
    );
    expected.extend(
        field_recovery_objects
            .unique_references(&prepared.host.artifact_access, ValueRefKind::ResultObject)?,
    );
    expected.extend(
        state_publication_profile_objects
            .unique_references(&prepared.host.artifact_access, ValueRefKind::ResultObject)?,
    );
    crate::object::retain_unique_references(&mut expected);
    expected.push(output_set.root);
    if output_set.result_objects != expected {
        return Err(AnalysisBridgeError::Identity(
            "response-spectrum result inventory differs from its canonical closure",
        ));
    }
    Ok(ImportedStructuralModalResponseSpectrumResult {
        state_publication_profiles,
        state_publication_profile_objects,
        identity,
        result,
        result_objects,
        numerical_receipt,
        receipt_objects,
        field_recovery,
        field_recovery_objects,
        response: retained_response,
    })
}
