use runmat_analysis_core::{AnalysisField, AnalysisModel, ObservationRequest};

use crate::{
    contracts::{
        ComputeBackend, ElectromagneticSolveOptions, FeaElectromagneticRunResult, FeaRunError,
    },
    diagnostics::{FeaDiagnostic, FeaDiagnosticSeverity},
    topology::AnalysisProjection,
};

pub use super::electromagnetic_frequency_contract::{
    ElectromagneticFrequencyPlan, ElectromagneticFrequencyPlanEntry,
    ElectromagneticFrequencyPlanEntryIdentity, ElectromagneticFrequencyPlanIdentity,
    ElectromagneticFrequencySummary, FeaElectromagneticFrequencyResponse,
};

use super::{
    electromagnetic_frequency_history::complex_history_series,
    observation_history::{
        evaluate_field_histories, required_frequency_history_samples, HistoryTopology,
    },
};

pub fn run_electromagnetic_frequency_response(
    model: &AnalysisModel,
    backend: ComputeBackend,
    options: ElectromagneticSolveOptions,
    requested_frequencies_hz: &[f64],
) -> Result<FeaElectromagneticFrequencyResponse, FeaRunError> {
    let step = super::electromagnetic::matching_step(model, options.procedure)?;
    let observation_frequencies_hz = required_frequency_history_samples(step)
        .map_err(|error| FeaRunError::Observation(error.to_string()))?;
    let reference_frequency_hz = model
        .electromagnetic
        .as_ref()
        .ok_or_else(|| FeaRunError::InvalidModel("electromagnetic domain is absent".into()))?
        .reference_frequency
        .hertz;
    let plan = ElectromagneticFrequencyPlan::new(
        reference_frequency_hz,
        requested_frequencies_hz,
        &observation_frequencies_hz,
    )?;
    let mut runs = Vec::with_capacity(plan.frequencies_hz().len());
    for entry in plan.entries()? {
        let frequency_hz = entry.frequency_hz();
        if !super::electromagnetic::has_nonzero_source_at_frequency(
            model,
            options.procedure,
            frequency_hz,
        )? {
            return Err(FeaRunError::MissingElectromagneticSource { frequency_hz });
        }
        let mut sample_model = model.clone();
        sample_model
            .electromagnetic
            .as_mut()
            .expect("frequency plan validated the electromagnetic domain")
            .reference_frequency
            .hertz = frequency_hz;
        runs.push(super::electromagnetic::run_electromagnetic_with_options(
            &sample_model,
            backend,
            options.clone(),
        )?);
    }
    assemble_electromagnetic_frequency_response(model, plan, runs, options.solver_mesh.as_ref())
}

pub(crate) fn assemble_electromagnetic_frequency_response(
    model: &AnalysisModel,
    plan: ElectromagneticFrequencyPlan,
    mut runs: Vec<FeaElectromagneticRunResult>,
    solver_mesh: Option<&runmat_meshing_core::SolverMeshArtifact>,
) -> Result<FeaElectromagneticFrequencyResponse, FeaRunError> {
    plan.validate()?;
    if runs.len() != plan.frequencies_hz().len()
        || runs
            .iter()
            .zip(plan.frequencies_hz())
            .any(|(run, frequency)| run.reference_frequency_hz.to_bits() != frequency.to_bits())
    {
        return Err(FeaRunError::InvalidFrequencyPlan(
            "electromagnetic frequency results do not exactly cover their canonical plan".into(),
        ));
    }
    let summary = summarize(plan.frequencies_hz(), &runs)?;
    let reference_index = plan.reference_index();
    let step = super::electromagnetic::matching_step(
        model,
        runs[reference_index].run.identity.procedure(),
    )?;
    if step
        .observations
        .iter()
        .any(|request| matches!(request, ObservationRequest::History(_)))
    {
        let mesh = solver_mesh.ok_or_else(|| {
            FeaRunError::Observation(
                "electromagnetic frequency histories require canonical solver-mesh topology".into(),
            )
        })?;
        let projection = AnalysisProjection::derive(mesh)
            .map_err(|error| FeaRunError::Observation(error.to_string()))?;
        let owned_series = complex_history_series(mesh, &runs)?;
        let series = owned_series
            .iter()
            .map(|series| series.as_history_series())
            .collect::<Vec<_>>();
        let coordinates = plan
            .frequencies_hz()
            .iter()
            .copied()
            .map(crate::results::field::RecoveredFieldSample::FrequencyHertz)
            .collect::<Vec<_>>();
        let evaluated = evaluate_field_histories(
            step,
            HistoryTopology::Solid(&projection),
            &coordinates,
            None,
            &series,
        )
        .map_err(|error| FeaRunError::Observation(error.to_string()))?;
        runs[reference_index].run.observation_histories = evaluated.histories;
        runs[reference_index].run.series_reductions = evaluated.series_reductions;
    }
    runs[reference_index]
        .run
        .diagnostics
        .push(frequency_evidence(&plan, &summary));
    if runs.len() > 1 {
        runs[reference_index]
            .run
            .diagnostics
            .push(frequency_known_answer_evidence(&plan, &summary));
    }
    Ok(FeaElectromagneticFrequencyResponse {
        plan,
        runs,
        summary,
    })
}

fn peak_abs(field: &AnalysisField) -> Result<f64, FeaRunError> {
    let values = field
        .as_host_f64()
        .ok_or_else(|| FeaRunError::Observation("frequency field is not host real64".into()))?;
    if values.is_empty() || values.iter().any(|value| !value.is_finite()) {
        return Err(FeaRunError::Observation(
            "frequency field is empty or contains non-finite values".into(),
        ));
    }
    Ok(values.iter().copied().map(f64::abs).fold(0.0_f64, f64::max))
}

fn summarize(
    frequencies_hz: &[f64],
    runs: &[FeaElectromagneticRunResult],
) -> Result<ElectromagneticFrequencySummary, FeaRunError> {
    let peak_flux_density = runs
        .iter()
        .map(|run| peak_abs(&run.magnetic_flux_density_magnitude_field))
        .collect::<Result<Vec<_>, _>>()?;
    let solve_quality = runs.iter().map(|run| run.solve_quality).collect::<Vec<_>>();
    if solve_quality.iter().any(|value| !value.is_finite()) {
        return Err(FeaRunError::Observation(
            "electromagnetic solve quality contains a non-finite value".into(),
        ));
    }
    let (peak_index, peak_value) = peak_flux_density
        .iter()
        .copied()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .unwrap_or((0, 0.0));
    let peak_frequency_hz = frequencies_hz[peak_index];
    let minimum = peak_flux_density
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    let half_power = peak_value * std::f64::consts::FRAC_1_SQRT_2;
    let mut left = peak_index;
    while left > 0 && peak_flux_density[left - 1] >= half_power {
        left -= 1;
    }
    let mut right = peak_index;
    while right + 1 < peak_flux_density.len() && peak_flux_density[right + 1] >= half_power {
        right += 1;
    }
    let resonance_bandwidth_hz =
        (right > left).then(|| (frequencies_hz[right] - frequencies_hz[left]).max(0.0));
    Ok(ElectromagneticFrequencySummary {
        peak_flux_density,
        solve_quality,
        resonance_peak_frequency_hz: Some(peak_frequency_hz),
        resonance_peak_flux_density: Some(peak_value),
        resonance_bandwidth_hz,
        resonance_quality_factor: resonance_bandwidth_hz
            .filter(|bandwidth| *bandwidth > 0.0)
            .map(|bandwidth| (peak_frequency_hz / bandwidth).max(0.0)),
        resonance_flux_gain: Some((peak_value / minimum.max(1.0e-12)).max(1.0)),
    })
}

fn frequency_evidence(
    plan: &ElectromagneticFrequencyPlan,
    summary: &ElectromagneticFrequencySummary,
) -> FeaDiagnostic {
    FeaDiagnostic {
        code: "FEA_EM_SWEEP".into(),
        severity: if plan.frequencies_hz().len() > 1 {
            FeaDiagnosticSeverity::Info
        } else {
            FeaDiagnosticSeverity::Warning
        },
        message: format!(
            "frequency_plan_digest={} sweep_count={} reference_frequency_hz={} frequency_min_hz={} frequency_max_hz={} resonance_peak_frequency_hz={} resonance_peak_flux_density={} resonance_bandwidth_hz={} resonance_quality_factor={} resonance_flux_gain={}",
            plan.identity().digest(),
            plan.frequencies_hz().len(),
            plan.reference_frequency_hz(),
            plan.frequencies_hz().first().copied().unwrap_or(0.0),
            plan.frequencies_hz().last().copied().unwrap_or(0.0),
            summary.resonance_peak_frequency_hz.unwrap_or(0.0),
            summary.resonance_peak_flux_density.unwrap_or(0.0),
            summary.resonance_bandwidth_hz.unwrap_or(0.0),
            summary.resonance_quality_factor.unwrap_or(0.0),
            summary.resonance_flux_gain.unwrap_or(0.0),
        ),
    }
}

fn frequency_known_answer_evidence(
    plan: &ElectromagneticFrequencyPlan,
    summary: &ElectromagneticFrequencySummary,
) -> FeaDiagnostic {
    let minimum = plan.frequencies_hz().first().copied().unwrap_or(0.0);
    let maximum = plan.frequencies_hz().last().copied().unwrap_or(0.0);
    let reference = plan.reference_frequency_hz();
    let reference_scale = reference.abs().max(1.0e-9);
    let reference_is_bracketed = minimum <= reference && reference <= maximum;
    let normalized_peak_error = summary
        .resonance_peak_frequency_hz
        .map(|peak| ((peak - reference).abs() / reference_scale).clamp(0.0, 1.0))
        .unwrap_or(1.0);
    let gain = summary.resonance_flux_gain.unwrap_or(0.0);
    let quality_factor = summary.resonance_quality_factor.unwrap_or(0.0);
    let passes = plan.frequencies_hz().len() >= 3
        && reference_is_bracketed
        && normalized_peak_error <= 0.25
        && gain >= 1.0
        && quality_factor >= 1.5;
    FeaDiagnostic {
        code: "FEA_EM_SWEEP_KNOWN_ANSWER".into(),
        severity: if passes {
            FeaDiagnosticSeverity::Info
        } else {
            FeaDiagnosticSeverity::Warning
        },
        message: format!(
            "basis=bracketed_reference_frequency frequency_plan_digest={} sweep_count={} reference_frequency_hz={} sweep_frequency_min_hz={} sweep_frequency_max_hz={} reference_frequency_in_sweep_ratio=1 reference_frequency_bracket_ratio={} normalized_peak_frequency_error_ratio={} resonance_flux_gain={} resonance_quality_factor={} sweep_known_answer_coverage_ratio={}",
            plan.identity().digest(),
            plan.frequencies_hz().len(),
            reference,
            minimum,
            maximum,
            u8::from(reference_is_bracketed),
            normalized_peak_error,
            gain,
            quality_factor,
            u8::from(passes),
        ),
    }
}
