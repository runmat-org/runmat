//! Retained original completion storage and additional root-encoding work.
//! All typed values, response metadata and transport remain charged; no wire copy
//! exists during read-only block replay. These shared codec bounds are not RSS.
use super::*;
use crate::nonlinear_study::CompletedNonlinearSolidStudy;

fn estimates(completed: &CompletedNonlinearSolidStudy) -> AnalysisBridgeResult<(u64, u64)> {
    let [step] = completed.workload.model.sequence.steps.as_slice() else {
        return Err(invalid(
            "nonlinear completion requires original standalone step",
        ));
    };
    let ceiling = step.accuracy.resources.resident_byte_budget();
    let profile = runmat_canonical_codec::profile_serialized(
        &(
            &completed.compiled,
            &completed.workload,
            &completed.replay.mesh,
            &completed.closures,
            &completed.terminal_block.block,
            &completed.terminal_block.numerical_receipt,
            &completed.publication.result,
            (
                &completed.replay.program_revision,
                &completed.replay.artifact_access,
                completed.replay.accepted_block_limit,
            ),
            &completed.accuracy.artifact,
            completed.fields.as_ref().map(|f| (&f.manifest, &f.output)),
            completed
                .observations
                .as_ref()
                .map(|o| (&o.manifest, &o.output)),
            &completed.observation_catalog,
            &completed.baseline_usage.evidence,
            &completed.terminal_usage.evidence,
        ),
        CanonicalLimits::content_bounded(
            usize::try_from(ceiling).map_err(|_| limit("nonlinear encoding limit overflow"))?,
            65_536,
            64,
        ),
    )?;
    let encoding_work = profile
        .materialization_resident_bytes_estimate
        .checked_add(
            profile
                .encoded_bytes_upper_bound
                .checked_mul(2)
                .ok_or_else(|| limit("nonlinear encoding storage overflow"))?,
        )
        .ok_or_else(|| limit("nonlinear encoding storage overflow"))?;
    let mut bytes = profile
        .materialization_resident_bytes_estimate
        .checked_add(std::mem::size_of::<CompletedNonlinearSolidStudy>() as u64)
        .ok_or_else(|| limit("nonlinear retained storage overflow"))?;
    for objects in [
        &completed.terminal_block.objects,
        &completed.terminal_block.receipt_objects,
        &completed.accuracy.objects,
        &completed.baseline_usage.objects,
        &completed.terminal_usage.objects,
        &completed.publication.objects,
    ] {
        bytes = bytes
            .checked_add(objects.retained_storage_bytes()?)
            .ok_or_else(|| limit("nonlinear encoding storage overflow"))?;
    }
    if let Some(fields) = &completed.fields {
        bytes = bytes
            .checked_add(fields.catalog_objects.retained_storage_bytes()?)
            .ok_or_else(|| limit("nonlinear field storage overflow"))?;
    }
    if let Some(observations) = &completed.observations {
        bytes = bytes
            .checked_add(observations.catalog_objects.retained_storage_bytes()?)
            .ok_or_else(|| limit("nonlinear observation storage overflow"))?;
    }
    for state in &completed.states {
        let state_profile = runmat_canonical_codec::profile_serialized(&state.response, LIMITS)?;
        bytes = bytes
            .checked_add(state.objects.retained_storage_bytes()?)
            .and_then(|n| n.checked_add(state_profile.materialization_resident_bytes_estimate))
            .ok_or_else(|| limit("nonlinear encoding storage overflow"))?;
    }
    bytes = bytes
        .checked_add(
            runmat_analysis_fea::execution::semantic_state_retained_storage_bytes(
                completed.states.iter().map(|s| &s.state),
                LIMITS,
            )?,
        )
        .ok_or_else(|| limit("nonlinear state retained storage overflow"))?;
    let encoding = bytes
        .checked_add(encoding_work)
        .ok_or_else(|| limit("nonlinear encoding storage overflow"))?;
    Ok((bytes, encoding))
}

pub(in crate::nonlinear_study) fn retained_storage(
    completed: &CompletedNonlinearSolidStudy,
) -> AnalysisBridgeResult<u64> {
    let (bytes, _) = estimates(completed)?;
    admit(completed, bytes)?;
    Ok(bytes)
}

pub(in crate::nonlinear_study) fn admit_encoding(
    completed: &CompletedNonlinearSolidStudy,
) -> AnalysisBridgeResult<u64> {
    let (_, bytes) = estimates(completed)?;
    admit(completed, bytes)?;
    Ok(bytes)
}

fn admit(completed: &CompletedNonlinearSolidStudy, bytes: u64) -> AnalysisBridgeResult<()> {
    if bytes
        > completed.workload.model.sequence.steps[0]
            .accuracy
            .resources
            .resident_byte_budget()
    {
        return Err(limit(
            "nonlinear completion storage exceeds original authored memory",
        ));
    }
    Ok(())
}
