use super::*;

pub fn import_structural_linear_result(
    source: &impl CacheImport,
    response: ProgramExecutionResponse,
    prepared: &PreparedStructuralLinearProgram,
    limits: ObjectInventoryLimits,
) -> AnalysisBridgeResult<StructuralLinearWorkloadResult> {
    Ok(import_structural_linear_completion(source, response, prepared, limits)?.result)
}

pub fn import_structural_linear_completion(
    source: &impl CacheImport,
    response: ProgramExecutionResponse,
    prepared: &PreparedStructuralLinearProgram,
    limits: ObjectInventoryLimits,
) -> AnalysisBridgeResult<ImportedStructuralLinearResult> {
    response.validate_against(&prepared.program.request)?;
    let output_set = import_analysis_output_set(source, response, &prepared.host)?;
    let [result_output, receipt_output, profiles_output] = output_set.manifest.outputs.as_slice()
    else {
        return Err(AnalysisBridgeError::Invalid(
            "structural analysis must return result, numerical receipt and publication profiles"
                .into(),
        ));
    };
    let imported = import_analysis_objects(
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
    let state_publication_profile_objects = import_analysis_objects(
        source,
        &profiles_output.root,
        AnalysisNumericalObjectKind::StatePublicationProfiles,
        profiles_output.logical_identity,
        profiles_output.payload_digest,
        &prepared.host.artifact_access,
        limits,
    )?;
    let mut expected =
        imported.unique_references(&prepared.host.artifact_access, ValueRefKind::ResultObject)?;
    expected.extend(
        receipt_objects
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
            "analysis result inventory differs from its canonical closure",
        ));
    }
    let result_object_bytes = output_set
        .result_objects
        .iter()
        .try_fold(0_u64, |total, reference| {
            total.checked_add(reference.encoded_length)
        })
        .ok_or_else(|| {
            AnalysisBridgeError::Invalid("analysis result byte accounting overflowed".into())
        })?;
    let result = StructuralLinearWorkloadResult::from_canonical_bytes(&imported.payload())?;
    let expected_result = result.result_identity(prepared.host.workload.identity)?;
    if expected_result.digest() != result_output.logical_identity {
        return Err(AnalysisBridgeError::Identity(
            "structural result identity differs from its canonical source",
        ));
    }
    let numerical_receipt =
        StructuralLinearNumericalReceiptArtifact::from_canonical_bytes(&receipt_objects.payload())?;
    if numerical_receipt.identity().digest() != receipt_output.logical_identity
        || numerical_receipt.source_workload != prepared.host.workload.identity
    {
        return Err(AnalysisBridgeError::Identity(
            "numerical receipt manifest or source workload differs from its canonical identity",
        ));
    }
    numerical_receipt.validate_against(expected_result, &result)?;
    let constraint_plan = prepared
        .inputs
        .iter()
        .find(|input| input.manifest.kind == AnalysisNumericalObjectKind::ConstraintPlan)
        .ok_or_else(|| {
            AnalysisBridgeError::Invalid(
                "prepared structural program has no canonical constraint plan".into(),
            )
        })?;
    validate_result_semantics(&prepared.inputs[0], constraint_plan, &result)?;
    let workload = StructuralLinearWorkload::from_canonical_bytes(&prepared.inputs[0].payload())?;
    let state_publication_profiles = crate::structural_state_profiles::validate_profiles(
        &state_publication_profile_objects,
        crate::structural_state_publication_profiles(
            &workload.model,
            workload.procedure_step()?,
            expected_result,
            imported.manifest.payload_digest,
            crate::StructuralStateProfileResult::Linear(&result),
        )?,
    )?;
    Ok(ImportedStructuralLinearResult {
        state_publication_profiles,
        state_publication_profile_objects,
        identity: expected_result,
        result,
        objects: imported,
        numerical_receipt,
        numerical_receipt_objects: receipt_objects,
        result_object_bytes,
    })
}
