//! Original field-history transport mapping; no numerical reconstruction.
use crate::analysis::{NonlinearMethod, NonlinearResultsData};

pub(super) fn data(
    nonlinear_run: runmat_analysis_fea::FeaNonlinearRunResult,
) -> (runmat_analysis_fea::FeaRunResult, NonlinearResultsData) {
    let run = nonlinear_run.run;
    (
        run,
        NonlinearResultsData {
            nonlinear_payload_version: "nonlinear_results/v1".to_string(),
            load_factors: nonlinear_run.load_factors,
            displacement_snapshots: nonlinear_run.displacement_snapshots,
            rotation_snapshots: nonlinear_run.rotation_snapshots,
            von_mises_snapshots: nonlinear_run.von_mises_snapshots,
            plastic_strain_snapshots: nonlinear_run.plastic_strain_snapshots,
            equivalent_plastic_strain_snapshots: nonlinear_run.equivalent_plastic_strain_snapshots,
            contact_pressure_snapshots: nonlinear_run.contact_pressure_snapshots,
            contact_gap_snapshots: nonlinear_run.contact_gap_snapshots,
            load_factor_snapshots: nonlinear_run.load_factor_snapshots,
            residual_norm_snapshots: nonlinear_run.residual_norm_snapshots,
            thermo_mechanical_temperature_snapshots: nonlinear_run
                .thermo_mechanical_temperature_snapshots,
            thermo_mechanical_thermal_strain_snapshots: nonlinear_run
                .thermo_mechanical_thermal_strain_snapshots,
            thermo_mechanical_thermal_stress_snapshots: nonlinear_run
                .thermo_mechanical_thermal_stress_snapshots,
            thermo_mechanical_displacement_snapshots: nonlinear_run
                .thermo_mechanical_displacement_snapshots,
            thermo_mechanical_von_mises_snapshots: nonlinear_run
                .thermo_mechanical_von_mises_snapshots,
            thermo_mechanical_coupling_residual_snapshots: nonlinear_run
                .thermo_mechanical_coupling_residual_snapshots,
            electro_thermal_temperature_snapshots: nonlinear_run
                .electro_thermal_temperature_snapshots,
            electro_thermal_thermal_residual_snapshots: nonlinear_run
                .electro_thermal_thermal_residual_snapshots,
            residual_norms: nonlinear_run.residual_norms,
            increment_norms: nonlinear_run.increment_norms,
            iteration_counts: nonlinear_run.iteration_counts,
            failed_increments: nonlinear_run.failed_increments,
            line_search_backtracks: nonlinear_run.line_search_backtracks,
            max_line_search_backtracks_per_increment: nonlinear_run
                .max_line_search_backtracks_per_increment,
            tangent_rebuild_count: nonlinear_run.tangent_rebuild_count,
            iteration_spike_count: nonlinear_run.iteration_spike_count,
            convergence_stall_count: nonlinear_run.convergence_stall_count,
            backtrack_burst_count: nonlinear_run.backtrack_burst_count,
            method: NonlinearMethod::IncrementalNewtonRaphson,
        },
    )
}

pub(super) fn append(
    into: &mut NonlinearResultsData,
    next: NonlinearResultsData,
) -> Result<(), String> {
    let failed_increments = into
        .failed_increments
        .checked_add(next.failed_increments)
        .ok_or("original nonlinear counter overflow")?;
    let line_search_backtracks = into
        .line_search_backtracks
        .checked_add(next.line_search_backtracks)
        .ok_or("original nonlinear counter overflow")?;
    let tangent_rebuild_count = into
        .tangent_rebuild_count
        .checked_add(next.tangent_rebuild_count)
        .ok_or("original nonlinear counter overflow")?;
    let iteration_spike_count = into
        .iteration_spike_count
        .checked_add(next.iteration_spike_count)
        .ok_or("original nonlinear counter overflow")?;
    let convergence_stall_count = into
        .convergence_stall_count
        .checked_add(next.convergence_stall_count)
        .ok_or("original nonlinear counter overflow")?;
    let backtrack_burst_count = into
        .backtrack_burst_count
        .checked_add(next.backtrack_burst_count)
        .ok_or("original nonlinear counter overflow")?;
    into.load_factors.extend(next.load_factors);
    into.displacement_snapshots
        .extend(next.displacement_snapshots);
    into.rotation_snapshots.extend(next.rotation_snapshots);
    into.von_mises_snapshots.extend(next.von_mises_snapshots);
    into.plastic_strain_snapshots
        .extend(next.plastic_strain_snapshots);
    into.equivalent_plastic_strain_snapshots
        .extend(next.equivalent_plastic_strain_snapshots);
    into.contact_pressure_snapshots
        .extend(next.contact_pressure_snapshots);
    into.contact_gap_snapshots
        .extend(next.contact_gap_snapshots);
    into.load_factor_snapshots
        .extend(next.load_factor_snapshots);
    into.residual_norm_snapshots
        .extend(next.residual_norm_snapshots);
    into.thermo_mechanical_temperature_snapshots
        .extend(next.thermo_mechanical_temperature_snapshots);
    into.thermo_mechanical_thermal_strain_snapshots
        .extend(next.thermo_mechanical_thermal_strain_snapshots);
    into.thermo_mechanical_thermal_stress_snapshots
        .extend(next.thermo_mechanical_thermal_stress_snapshots);
    into.thermo_mechanical_displacement_snapshots
        .extend(next.thermo_mechanical_displacement_snapshots);
    into.thermo_mechanical_von_mises_snapshots
        .extend(next.thermo_mechanical_von_mises_snapshots);
    into.thermo_mechanical_coupling_residual_snapshots
        .extend(next.thermo_mechanical_coupling_residual_snapshots);
    into.electro_thermal_temperature_snapshots
        .extend(next.electro_thermal_temperature_snapshots);
    into.electro_thermal_thermal_residual_snapshots
        .extend(next.electro_thermal_thermal_residual_snapshots);
    into.residual_norms.extend(next.residual_norms);
    into.increment_norms.extend(next.increment_norms);
    into.iteration_counts.extend(next.iteration_counts);
    into.failed_increments = failed_increments;
    into.line_search_backtracks = line_search_backtracks;
    into.tangent_rebuild_count = tangent_rebuild_count;
    into.iteration_spike_count = iteration_spike_count;
    into.convergence_stall_count = convergence_stall_count;
    into.backtrack_burst_count = backtrack_burst_count;
    into.max_line_search_backtracks_per_increment = into
        .max_line_search_backtracks_per_increment
        .max(next.max_line_search_backtracks_per_increment);
    Ok(())
}
