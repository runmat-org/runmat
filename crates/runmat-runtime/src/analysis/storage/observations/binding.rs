use super::*;
use runmat_analysis_fea::results::field::reconstruct_recovered_field;

pub(super) fn validate_run(
    run: &AnalysisRunResult,
    completed: &ThermalStudyCompletion,
    source: &ThermalStudyObjectSource,
) -> Result<(), String> {
    let mut projected = completed.manifest.run.clone();
    crate::analysis::study_result::thermal_observation_admission::project(
        completed,
        source,
        &mut projected,
    )?;
    if run.run.observations != projected.observations
        || run.run.series_reductions != projected.series_reductions
        || run.run.observation_histories != projected.observation_histories
    {
        return Err(
            "persisted run observation projection differs from authenticated catalog".into(),
        );
    }
    let thermal = run
        .thermal_results
        .as_ref()
        .ok_or("observation run has no thermal fields")?;
    let manifest = &completed.manifest;
    if run.run.identity != manifest.run.identity
        || thermal.time_points_s != manifest.time_points_s
        || thermal.residual_norms != manifest.residual_norms
        || thermal.reference_temperature_k.canonical_value()
            != manifest.reference_temperature_k.canonical_value()
        || thermal.initial_state != manifest.initial_state
    {
        return Err("persisted thermal run differs from observation source".into());
    }
    let histories = [
        &thermal.temperature_snapshots,
        &thermal.temperature_gradient_snapshots,
        &thermal.heat_flux_snapshots,
        &thermal.energy_storage_rate_snapshots,
        &thermal.interface_temperature_snapshots,
    ];
    let field_count = manifest.fields.first().map_or(4, Vec::len);
    for history in &histories[..field_count] {
        if history.len() != manifest.fields.len() {
            return Err("thermal source snapshot inventory differs".into());
        }
    }
    for (sample, entries) in manifest.fields.iter().enumerate() {
        for (kind, entry) in entries.iter().enumerate() {
            let partitions = entry
                .partitions
                .iter()
                .map(|descriptor| {
                    completed
                        .field_partitions
                        .iter()
                        .find(|partition| partition.identity == descriptor.partition)
                        .cloned()
                        .ok_or_else(|| "thermal source partition missing".to_string())
                })
                .collect::<Result<Vec<_>, _>>()?;
            let recovered = reconstruct_recovered_field(
                entry.field,
                entry.definition.clone(),
                entry.sample,
                partitions,
                match kind {
                    0 => manifest.ordered_node_ids.len(),
                    4 => manifest.interface_node_ids.len(),
                    _ => manifest.ordered_element_ids.len(),
                },
            )
            .map_err(|error| error.to_string())?
            .into_storage();
            if recovered != histories[kind][sample] {
                return Err(
                    "persisted thermal snapshot differs from exact observation source".into(),
                );
            }
        }
    }
    Ok(())
}
