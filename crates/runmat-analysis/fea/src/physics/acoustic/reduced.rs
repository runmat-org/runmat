use runmat_analysis_core::{AnalysisField, AnalysisModel};

use crate::{
    diagnostics::{FeaDiagnostic, FeaDiagnosticSeverity},
    solve::backend::kind::LinearAlgebraBackendKind,
    AnalysisRunIdentity, ComputeBackend, ExecutionBackend, FeaRunResult, SolverMethod,
    SolverPreconditioner, FEA_FIELD_ACOUSTIC_PARTICLE_VELOCITY, FEA_FIELD_ACOUSTIC_PHASE,
    FEA_FIELD_ACOUSTIC_PRESSURE_IMAG, FEA_FIELD_ACOUSTIC_PRESSURE_MAGNITUDE,
    FEA_FIELD_ACOUSTIC_PRESSURE_REAL, FEA_FIELD_ACOUSTIC_SOUND_PRESSURE_LEVEL_DB,
};

use super::{domain, linear, medium, recovery};

/// Nodal sound pressure levels of peak pressure phasor magnitudes: the shared RMS definition
/// of the distributed observations, so both result paths publish one level.
pub(crate) fn nodal_sound_pressure_levels_db(pressure_magnitude_pa: &[f64]) -> Vec<f64> {
    pressure_magnitude_pa
        .iter()
        .copied()
        .map(runmat_analysis_core::sound_pressure_level_db)
        .collect()
}

/// Executes the graph-domain acoustic proxy.
///
/// Keeping this function in the acoustic physics domain prevents runtime adapters from owning
/// numerical assembly, solution, recovery, or physics diagnostics. Production acoustic
/// admission remains blocked until the finite-element Helmholtz formulation replaces this proxy.
pub fn solve_reduced_harmonic(
    model: &AnalysisModel,
    backend: ComputeBackend,
    identity: AnalysisRunIdentity,
    mode_count: usize,
    residual_warn_threshold: f64,
) -> Result<FeaRunResult, String> {
    let admission = crate::contracts::accelerator_admission::admit_accelerator(
        crate::contracts::accelerator_admission::AcceleratorFamily::Acoustic,
        backend,
    )
    .map_err(|error| error.to_string())?;
    let node_count = domain::node_count(model);
    let material = medium::material_summary(model, mode_count);
    let boundary = medium::boundary_summary(model, material.characteristic_impedance_pa_s_per_m());
    let drive_frequency_hz = domain::drive_frequency_hz(mode_count, node_count);
    let (source_real, source_imag) = domain::source_vectors(model, node_count, drive_frequency_hz)?;
    let topology = domain::topology(node_count);
    let system = domain::helmholtz_operator(
        topology,
        drive_frequency_hz,
        material.speed_of_sound_m_per_s.canonical_value(),
        material.damping_ratio.canonical_value(),
        &boundary,
    );
    let (pressure_real, pressure_imag) =
        linear::solve_complex_graph_operator(&system, &source_real, &source_imag);
    let normalized_residual_norm = domain::residual_norm(
        &system,
        &pressure_real,
        &pressure_imag,
        &source_real,
        &source_imag,
    );
    let pressure_magnitude = pressure_real
        .iter()
        .zip(&pressure_imag)
        .map(|(real, imaginary)| real.hypot(*imaginary))
        .collect::<Vec<_>>();
    let phase = pressure_real
        .iter()
        .zip(&pressure_imag)
        .map(|(real, imaginary)| imaginary.atan2(*real))
        .collect::<Vec<_>>();
    let sound_pressure_level_db = nodal_sound_pressure_levels_db(&pressure_magnitude);
    let particle_velocity = recovery::particle_velocity(
        &pressure_real,
        topology,
        drive_frequency_hz,
        material.density_kg_per_m3.canonical_value(),
    );
    let peak_pressure_pa = pressure_magnitude.iter().copied().fold(0.0_f64, f64::max);
    let sweep_frequencies_hz = domain::sweep_frequencies_hz(drive_frequency_hz);
    let mut frequency_response_fields = Vec::with_capacity(sweep_frequencies_hz.len());
    let mut sweep_peak_pressure_pa = 0.0_f64;
    let mut sweep_residual_norm = 0.0_f64;
    for frequency_hz in &sweep_frequencies_hz {
        let (sweep_source_real, sweep_source_imag) =
            domain::source_vectors(model, node_count, *frequency_hz)?;
        let sweep_system = domain::helmholtz_operator(
            topology,
            *frequency_hz,
            material.speed_of_sound_m_per_s.canonical_value(),
            material.damping_ratio.canonical_value(),
            &boundary,
        );
        let (sweep_real, sweep_imag) = linear::solve_complex_graph_operator(
            &sweep_system,
            &sweep_source_real,
            &sweep_source_imag,
        );
        let sweep_magnitude = sweep_real
            .iter()
            .zip(&sweep_imag)
            .map(|(real, imaginary)| real.hypot(*imaginary))
            .collect::<Vec<_>>();
        sweep_peak_pressure_pa =
            sweep_peak_pressure_pa.max(sweep_magnitude.iter().copied().fold(0.0_f64, f64::max));
        sweep_residual_norm = sweep_residual_norm.max(domain::residual_norm(
            &sweep_system,
            &sweep_real,
            &sweep_imag,
            &sweep_source_real,
            &sweep_source_imag,
        ));
        frequency_response_fields.push(AnalysisField::host_f64(
            crate::fea_acoustic_frequency_response_field_id(*frequency_hz),
            vec![node_count],
            sweep_magnitude,
        ));
    }
    let sweep_frequency_min_hz = sweep_frequencies_hz
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    let sweep_frequency_max_hz = sweep_frequencies_hz.iter().copied().fold(0.0_f64, f64::max);
    let sweep_bandwidth_hz = if sweep_frequency_min_hz.is_finite() {
        (sweep_frequency_max_hz - sweep_frequency_min_hz).max(0.0)
    } else {
        0.0
    };
    let known_answer = recovery::known_answer_metrics(
        topology,
        &pressure_magnitude,
        drive_frequency_hz,
        material.speed_of_sound_m_per_s.canonical_value(),
    );
    let mut fields = vec![
        AnalysisField::host_f64(
            FEA_FIELD_ACOUSTIC_PRESSURE_REAL,
            vec![node_count],
            pressure_real,
        ),
        AnalysisField::host_f64(
            FEA_FIELD_ACOUSTIC_PRESSURE_IMAG,
            vec![node_count],
            pressure_imag,
        ),
        AnalysisField::host_f64(
            FEA_FIELD_ACOUSTIC_PRESSURE_MAGNITUDE,
            vec![node_count],
            pressure_magnitude,
        ),
        AnalysisField::host_f64(FEA_FIELD_ACOUSTIC_PHASE, vec![node_count], phase),
        AnalysisField::host_f64(
            FEA_FIELD_ACOUSTIC_SOUND_PRESSURE_LEVEL_DB,
            vec![node_count],
            sound_pressure_level_db,
        ),
        AnalysisField::host_f64(
            FEA_FIELD_ACOUSTIC_PARTICLE_VELOCITY,
            vec![node_count, 3],
            particle_velocity,
        ),
    ];
    let acoustic_field_count = fields.len() + frequency_response_fields.len();
    fields.extend(frequency_response_fields);
    let mut run = FeaRunResult {
        identity,
        requested_backend: backend,
        execution_backend: ExecutionBackend::Cpu,
        solver_backend: LinearAlgebraBackendKind::CpuReference,
        solver_device_apply_k_ratio: 0.0,
        solver_method: SolverMethod::AcousticDomainGraphHelmholtzHarmonic,
        preconditioner: SolverPreconditioner::None,
        solver_host_sync_count: 0,
        accelerator_evidence: Vec::new(),
        spatial_projection_evidence: Vec::new(),
        structural_static_convergence: None,
        structural_static_balance: None,
        structural_constraint_result: None,
        structural_connector_result: None,
        observations: Vec::new(),
        observation_histories: Vec::new(),
        series_reductions: Vec::new(),
        diagnostics: diagnostics(DiagnosticInputs {
            topology,
            source_real: &source_real,
            normalized_residual_norm,
            residual_warn_threshold,
            drive_frequency_hz,
            acoustic_field_count,
            peak_pressure_pa,
            material,
            boundary,
            sweep_frequencies_hz: &sweep_frequencies_hz,
            sweep_frequency_min_hz,
            sweep_frequency_max_hz,
            sweep_bandwidth_hz,
            sweep_peak_pressure_pa,
            sweep_residual_norm,
            known_answer,
        }),
        fields,
    };
    run.diagnostics.extend(admission.fallback);
    run.validate_solver_formulation()
        .map_err(|error| error.to_string())?;
    Ok(run)
}

struct DiagnosticInputs<'a> {
    topology: domain::AcousticDomainTopology,
    source_real: &'a [f64],
    normalized_residual_norm: f64,
    residual_warn_threshold: f64,
    drive_frequency_hz: f64,
    acoustic_field_count: usize,
    peak_pressure_pa: f64,
    material: medium::AcousticMaterialSummary,
    boundary: medium::AcousticBoundarySummary,
    sweep_frequencies_hz: &'a [f64],
    sweep_frequency_min_hz: f64,
    sweep_frequency_max_hz: f64,
    sweep_bandwidth_hz: f64,
    sweep_peak_pressure_pa: f64,
    sweep_residual_norm: f64,
    known_answer: recovery::AcousticKnownAnswerMetrics,
}

fn diagnostics(input: DiagnosticInputs<'_>) -> Vec<FeaDiagnostic> {
    let residual_severity = if input.normalized_residual_norm <= input.residual_warn_threshold {
        FeaDiagnosticSeverity::Info
    } else {
        FeaDiagnosticSeverity::Warning
    };
    vec![
        FeaDiagnostic {
            code: "FEA_ACOUSTIC_HELMHOLTZ_RESIDUAL".to_string(),
            severity: residual_severity,
            message: format!(
                "normalized_residual_norm={} equation_scale={} residual_warn_threshold={}",
                input.normalized_residual_norm,
                domain::source_norm(input.source_real, &[]).max(1.0),
                input.residual_warn_threshold,
            ),
        },
        FeaDiagnostic {
            code: "FEA_ACOUSTIC_DOMAIN_ASSEMBLY".to_string(),
            severity: if input.topology.edge_count > 0
                && input.topology.active_dimension_count >= 2
            {
                FeaDiagnosticSeverity::Info
            } else {
                FeaDiagnosticSeverity::Warning
            },
            message: format!(
                "domain_node_count={} domain_edge_count={} domain_active_dimension_count={} domain_dim_x={} domain_dim_y={} domain_dim_z={} domain_spacing_x_m={} domain_spacing_y_m={} domain_spacing_z_m={} boundary_node_count={} average_node_degree={} source_node_count={} domain_volume_m3={}",
                input.topology.node_count,
                input.topology.edge_count,
                input.topology.active_dimension_count,
                input.topology.dims[0],
                input.topology.dims[1],
                input.topology.dims[2],
                input.topology.spacing[0],
                input.topology.spacing[1],
                input.topology.spacing[2],
                input.topology.boundary_node_count,
                input.topology.average_node_degree(),
                input.source_real.iter().filter(|value| value.abs() > 1.0e-12).count(),
                input.topology.volume_m3(),
            ),
        },
        FeaDiagnostic {
            code: "FEA_ACOUSTIC_HARMONIC_RESPONSE".to_string(),
            severity: FeaDiagnosticSeverity::Info,
            message: format!(
                "drive_frequency_hz={} speed_of_sound_m_per_s={} density_kg_per_m3={} damping_ratio={} acoustic_node_count={} acoustic_field_count={} peak_pressure_pa={} acoustic_material_count={} acoustic_material_coverage_ratio={}",
                input.drive_frequency_hz,
                input.material.speed_of_sound_m_per_s.canonical_value(),
                input.material.density_kg_per_m3.canonical_value(),
                input.material.damping_ratio.canonical_value(),
                input.topology.node_count,
                input.acoustic_field_count,
                input.peak_pressure_pa,
                input.material.explicit_material_count,
                input.material.coverage_ratio,
            ),
        },
        FeaDiagnostic {
            code: "FEA_ACOUSTIC_BOUNDARY_MODEL".to_string(),
            severity: if input.boundary.has_acoustic_boundary_data() {
                FeaDiagnosticSeverity::Info
            } else {
                FeaDiagnosticSeverity::Warning
            },
            message: format!(
                "acoustic_boundary_count={} rigid_wall_count={} radiation_boundary_count={} impedance_boundary_count={} acoustic_boundary_coverage_ratio={} mean_specific_impedance_pa_s_per_m={} radiation_loss_factor={} impedance_loss_factor={}",
                input.boundary.acoustic_boundary_count,
                input.boundary.rigid_wall_count,
                input.boundary.radiation_boundary_count,
                input.boundary.impedance_boundary_count,
                input.boundary.coverage_ratio,
                input.boundary.mean_specific_impedance_pa_s_per_m,
                input.boundary.radiation_loss_factor,
                input.boundary.impedance_loss_factor,
            ),
        },
        FeaDiagnostic {
            code: "FEA_ACOUSTIC_FREQUENCY_RESPONSE".to_string(),
            severity: if input.sweep_residual_norm <= input.residual_warn_threshold {
                FeaDiagnosticSeverity::Info
            } else {
                FeaDiagnosticSeverity::Warning
            },
            message: format!(
                "sweep_count={} sweep_frequency_min_hz={} sweep_frequency_max_hz={} sweep_bandwidth_hz={} sweep_peak_pressure_pa={} sweep_max_residual_norm={} response_coverage_ratio={}",
                input.sweep_frequencies_hz.len(),
                input.sweep_frequency_min_hz,
                input.sweep_frequency_max_hz,
                input.sweep_bandwidth_hz,
                input.sweep_peak_pressure_pa,
                input.sweep_residual_norm,
                if input.sweep_frequencies_hz.is_empty() { 0.0 } else { 1.0 },
            ),
        },
        FeaDiagnostic {
            code: "FEA_ACOUSTIC_KNOWN_ANSWER".to_string(),
            severity: if input.known_answer.known_answer_coverage_ratio >= 1.0
                && input.known_answer.tube_mode_alignment_error_ratio <= 0.5
                && input.known_answer.tube_pressure_variation_ratio > 1.0e-12
                && input.known_answer.cavity_mode_spacing_ratio.is_finite()
                && input.known_answer.cavity_mode_spacing_ratio > 0.0
            {
                FeaDiagnosticSeverity::Info
            } else {
                FeaDiagnosticSeverity::Warning
            },
            message: format!(
                "tube_mode_alignment_error_ratio={} tube_pressure_variation_ratio={} cavity_mode_spacing_ratio={} cavity_reference_mode_count={} known_answer_coverage_ratio={}",
                input.known_answer.tube_mode_alignment_error_ratio,
                input.known_answer.tube_pressure_variation_ratio,
                input.known_answer.cavity_mode_spacing_ratio,
                input.known_answer.cavity_reference_mode_count,
                input.known_answer.known_answer_coverage_ratio,
            ),
        },
    ]
}
