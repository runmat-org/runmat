//! The fields a run result holds, visited in publication order and collected once per id.
use super::*;
use runmat_analysis_fea::results::observation::SeriesReductionPayload;

macro_rules! visit_analysis_result_fields {
    ($run_result:expr, $as_ref:ident, [$($borrow:tt)*], $visit:expr) => {{
        for field in $($borrow)* $run_result.run.fields {
            $visit(field);
        }
        for observation in $($borrow)* $run_result.run.observations {
            $visit($($borrow)* observation.field);
        }
        for reduction in $($borrow)* $run_result.run.series_reductions {
            match $($borrow)* reduction.payload {
                SeriesReductionPayload::Values { field, .. } => $visit(field),
                SeriesReductionPayload::Envelope {
                    minimum, maximum, ..
                } => {
                    $visit(minimum);
                    $visit(maximum);
                }
            }
        }
        if let Some(modal) = $run_result.modal_results.$as_ref() {
            for field in $($borrow)* modal.mode_shapes {
                $visit(field);
            }
        }
        if let Some(buckling) = $run_result.buckling_results.$as_ref() {
            for fields in [
                $($borrow)* buckling.mode_shapes,
                $($borrow)* buckling.rotation_mode_shapes,
            ] {
                for field in fields {
                    $visit(field);
                }
            }
        }
        if let Some(thermal) = $run_result.thermal_results.$as_ref() {
            for fields in [
                $($borrow)* thermal.temperature_snapshots,
                $($borrow)* thermal.temperature_gradient_snapshots,
                $($borrow)* thermal.heat_flux_snapshots,
                $($borrow)* thermal.energy_storage_rate_snapshots,
                $($borrow)* thermal.interface_temperature_snapshots,
            ] {
                for field in fields {
                    $visit(field);
                }
            }
        }
        if let Some(transient) = $run_result.transient_results.$as_ref() {
            for fields in [
                $($borrow)* transient.displacement_snapshots,
                $($borrow)* transient.rotation_snapshots,
                $($borrow)* transient.velocity_snapshots,
                $($borrow)* transient.angular_velocity_snapshots,
                $($borrow)* transient.acceleration_snapshots,
                $($borrow)* transient.angular_acceleration_snapshots,
                $($borrow)* transient.strain_snapshots,
                $($borrow)* transient.stress_snapshots,
                $($borrow)* transient.strain_energy_density_snapshots,
                $($borrow)* transient.von_mises_snapshots,
                $($borrow)* transient.kinetic_energy_snapshots,
                $($borrow)* transient.strain_energy_snapshots,
                $($borrow)* transient.residual_norm_snapshots,
                $($borrow)* transient.reaction_force_snapshots,
                $($borrow)* transient.reaction_moment_snapshots,
                $($borrow)* transient.external_work_snapshots,
                $($borrow)* transient.support_work_snapshots,
                $($borrow)* transient.damping_dissipation_snapshots,
                $($borrow)* transient.energy_balance_residual_snapshots,
                $($borrow)* transient.relative_energy_balance_error_snapshots,
            ] {
                for field in fields {
                    $visit(field);
                }
            }
        }
        if let Some(nonlinear) = $run_result.nonlinear_results.$as_ref() {
            for fields in [
                $($borrow)* nonlinear.displacement_snapshots,
                $($borrow)* nonlinear.rotation_snapshots,
                $($borrow)* nonlinear.von_mises_snapshots,
                $($borrow)* nonlinear.plastic_strain_snapshots,
                $($borrow)* nonlinear.equivalent_plastic_strain_snapshots,
                $($borrow)* nonlinear.contact_pressure_snapshots,
                $($borrow)* nonlinear.contact_gap_snapshots,
                $($borrow)* nonlinear.load_factor_snapshots,
                $($borrow)* nonlinear.residual_norm_snapshots,
            ] {
                for field in fields {
                    $visit(field);
                }
            }
        }
    }};
}

pub(crate) fn visit_analysis_result_fields(
    run_result: &AnalysisRunResult,
    mut visit: impl FnMut(&AnalysisField),
) {
    visit_analysis_result_fields!(run_result, as_ref, [&], visit);
}

pub(crate) fn visit_analysis_result_fields_mut(
    run_result: &mut AnalysisRunResult,
    mut visit: impl FnMut(&mut AnalysisField),
) {
    visit_analysis_result_fields!(run_result, as_mut, [& mut], visit);
}

pub(crate) fn collect_analysis_result_fields(run_result: &AnalysisRunResult) -> Vec<AnalysisField> {
    let mut fields = Vec::new();
    let mut seen = HashSet::new();
    visit_analysis_result_fields(run_result, |field| {
        push_analysis_result_field(&mut fields, &mut seen, field);
    });
    fields
}

fn push_analysis_result_field(
    fields: &mut Vec<AnalysisField>,
    seen: &mut HashSet<String>,
    field: &AnalysisField,
) {
    if !seen.insert(field.field_id.clone()) {
        return;
    }
    fields.push(field.clone());
}

pub(crate) fn analysis_run_field_ids(run_result: &AnalysisRunResult) -> Vec<String> {
    collect_analysis_result_fields(run_result)
        .into_iter()
        .map(|field| field.field_id)
        .collect()
}
