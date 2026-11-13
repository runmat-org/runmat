//! Admission bounds for finalizing an existing checkpoint, never a second solve.
use super::post_process::{
    copied_output_payload_bytes_upper_bound, post_process_ceiling, post_process_worker,
    validate_projected_output,
};
use super::{
    build_resource_plan, ByteCount, Count, FeaResourceDemand, FeaResourcePhase, FeaResourcePlan,
    FeaResourcePlanningInput, ObservationHistoryFinalizationResourceDemand,
    PreparedInputResourceFacts, PreparedOutputResourceFacts, PreparedPhaseResourceFacts,
    ResourcePlanningError, RESOURCE_ESTIMATION_MODEL_REVISION,
};
use crate::execution::{ObservationHistoryFinalizationRequest, ObservationHistorySourceContext};
use crate::results::observation::{ObservationFieldCatalog, ObservationHistoryCheckpoint};

fn invalid(error: impl std::fmt::Display) -> ResourcePlanningError {
    ResourcePlanningError::Invalid(error.to_string())
}
fn add(a: u64, b: u64) -> Result<u64, ResourcePlanningError> {
    a.checked_add(b).ok_or(ResourcePlanningError::Overflow(
        "history finalization bytes",
    ))
}
fn multiply(a: u64, b: u64) -> Result<u64, ResourcePlanningError> {
    a.checked_mul(b).ok_or(ResourcePlanningError::Overflow(
        "history finalization bytes",
    ))
}
fn count(value: usize) -> Result<u64, ResourcePlanningError> {
    u64::try_from(value)
        .map_err(|_| ResourcePlanningError::Overflow("history finalization cardinality"))
}
fn values(checkpoint: &ObservationHistoryCheckpoint) -> Result<u64, ResourcePlanningError> {
    checkpoint.samples().iter().try_fold(0, |total, sample| {
        let values = sample
            .result
            .field
            .as_host_f64()
            .ok_or_else(|| invalid("history finalization requires host f64 samples"))?;
        add(total, count(values.len())?)
    })
}

pub fn observation_history_finalization_payload_bytes_upper_bound(
    request: &ObservationHistoryFinalizationRequest,
    checkpoint: &ObservationHistoryCheckpoint,
) -> Result<u64, ResourcePlanningError> {
    request.validate().map_err(invalid)?;
    checkpoint.validate().map_err(invalid)?;
    if request.terminal_checkpoint != checkpoint.manifest().identity {
        return Err(invalid(
            "history finalization checkpoint differs from request",
        ));
    }
    // The published history holds the checkpoint's retained samples and its manifest, plus one
    // record envelope per sample and one for the manifest.
    checkpoint_output_payload_bytes_upper_bound(
        checkpoint,
        add(count(checkpoint.samples().len())?, 1)?,
        0,
    )
}

/// Output ceiling of a payload built from a retained checkpoint: the checkpoint's own encoded
/// samples and manifest, the record envelopes, and generated values.
fn checkpoint_output_payload_bytes_upper_bound(
    checkpoint: &ObservationHistoryCheckpoint,
    records: u64,
    generated_values: u64,
) -> Result<u64, ResourcePlanningError> {
    copied_output_payload_bytes_upper_bound(
        super::post_process::observation_history_checkpoint_payload_bytes(checkpoint)?,
        records,
        generated_values,
    )
}

pub fn observation_history_series_reduction_payload_bytes_upper_bound(
    request: &ObservationHistoryFinalizationRequest,
    checkpoint: &ObservationHistoryCheckpoint,
) -> Result<Option<u64>, ResourcePlanningError> {
    observation_history_finalization_payload_bytes_upper_bound(request, checkpoint)?;
    if request.series_reduction.is_none() {
        return Ok(None);
    }
    let retained_values = values(checkpoint)?;
    let samples = count(checkpoint.samples().len())?;
    let lanes = retained_values
        .checked_div(samples)
        .ok_or_else(|| invalid("series output has no retained samples"))?;
    if lanes == 0 || retained_values % samples != 0 {
        return Err(invalid(
            "series output samples have inconsistent value widths",
        ));
    }
    // Statistical reductions emit one field and no extrema locations. Their original
    // contributions still retain every source/value, covered by the checkpoint's bytes.
    // Extrema additionally retain actual governing locations, including all ties if requested.
    let operation = request
        .series_reduction
        .ok_or_else(|| invalid("series operation absent"))?;
    let extrema_lanes = match operation {
        runmat_analysis_core::SeriesReduction::Mean { .. }
        | runmat_analysis_core::SeriesReduction::RootMeanSquare { .. }
        | runmat_analysis_core::SeriesReduction::Variance { .. } => 0,
        runmat_analysis_core::SeriesReduction::None => {
            return Err(invalid("series operation absent"))
        }
        runmat_analysis_core::SeriesReduction::Minimum
        | runmat_analysis_core::SeriesReduction::Maximum
        | runmat_analysis_core::SeriesReduction::AbsoluteMaximum
        | runmat_analysis_core::SeriesReduction::Envelope => {
            let locations = if request
                .plan
                .observations
                .iter()
                .find(|o| o.spec.observation_id == request.observation_id)
                .and_then(|o| o.spec.reduction.as_ref())
                .ok_or_else(|| invalid("series observation absent"))?
                .extrema_ties
                == runmat_analysis_core::ExtremaTiePolicy::AllLocations
            {
                retained_values
            } else {
                lanes
            };
            // Envelope emits minimum and maximum. Other extrema emit one location inventory.
            if operation == runmat_analysis_core::SeriesReduction::Envelope {
                multiply(locations, 2)?
            } else {
                locations
            }
        }
    };
    let records = add(add(samples, extrema_lanes)?, 1)?;
    // Covers the complete contribution values and either one or both terminal fields.
    checkpoint_output_payload_bytes_upper_bound(checkpoint, records, multiply(retained_values, 2)?)
        .map(Some)
}

pub fn build_observation_history_finalization_resource_plan(
    request: &ObservationHistoryFinalizationRequest,
    checkpoint: &ObservationHistoryCheckpoint,
    catalog: &ObservationFieldCatalog,
    context: ObservationHistorySourceContext<'_>,
    phase: PreparedPhaseResourceFacts,
    series_output: Option<PreparedOutputResourceFacts>,
) -> Result<FeaResourcePlan, ResourcePlanningError> {
    let PreparedPhaseResourceFacts {
        inputs,
        output: history_output,
        output_set,
    } = phase;
    request
        .validate_inputs(checkpoint, catalog, context)
        .map_err(invalid)?;
    let history_bytes =
        observation_history_finalization_payload_bytes_upper_bound(request, checkpoint)?;
    validate_projected_output(&history_output, history_bytes, "history finalization")?;
    let series_bytes =
        observation_history_series_reduction_payload_bytes_upper_bound(request, checkpoint)?;
    match (&series_output, series_bytes) {
        (Some(output), Some(bound)) => {
            validate_projected_output(output, bound, "history series reduction")?
        }
        (None, None) => {}
        _ => {
            return Err(invalid(
                "series output presence differs from authored reduction",
            ))
        }
    }
    let retained_values = values(checkpoint)?;
    // Resume/finish validate and serialize retained history; the serial reducer carries
    // compensated accumulators, extrema locations, and sample evidence.
    let workspace = add(
        source_validation_workspace(request, catalog, context, &inputs)?,
        add(
            multiply(retained_values, 256)?,
            add(history_bytes, series_bytes.unwrap_or(0))?,
        )?,
    )?;
    build_resource_plan(FeaResourcePlanningInput {
        estimation_model_revision: RESOURCE_ESTIMATION_MODEL_REVISION,
        phase: FeaResourcePhase::ObservationHistoryFinalization(request.identity()),
        demand: FeaResourceDemand::ObservationHistoryFinalization(
            ObservationHistoryFinalizationResourceDemand {
                request: request.identity(),
                topology_digest: request.plan.topology_digest,
                retained_sample_count: Count(count(checkpoint.samples().len())?),
                retained_value_count: Count(retained_values),
                inputs,
                workspace_bytes_upper_bound: ByteCount(workspace),
                history_output,
                series_output,
            },
        ),
        output_set: Some(output_set),
        worker: post_process_worker(),
        accuracy_ceiling: post_process_ceiling(),
    })
}

/// Storage of validating the history's source beside the decoded inputs.
/// - Thermal: the shared thermal source-validation law, plus one more decoded copy of the
///   inputs (finalization consumes a copy of the checkpoint while the catalog and plan stay
///   borrowed).
/// - Other families have no counted source law yet and keep four decoded copies of the inputs,
///   which covers their mesh and catalog validation.
fn source_validation_workspace(
    request: &ObservationHistoryFinalizationRequest,
    catalog: &ObservationFieldCatalog,
    context: ObservationHistorySourceContext<'_>,
    inputs: &PreparedInputResourceFacts,
) -> Result<u64, ResourcePlanningError> {
    let decoded = inputs.decoded_resident_bytes_admission_budget.0;
    match (catalog.source(), context) {
        (
            crate::results::observation::ObservationFieldCatalogSource::Thermal { manifest },
            ObservationHistorySourceContext::Thermal { workload, mesh },
        ) => add(
            crate::results::observation::thermal_source_validation_resident_bytes(
                workload,
                manifest,
                &request.plan,
                mesh,
                u64::MAX,
            )
            .map_err(invalid)?,
            decoded,
        ),
        _ => multiply(decoded, 4),
    }
}
