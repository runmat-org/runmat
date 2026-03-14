use runmat_analysis_core::{validate_model, AnalysisField, AnalysisModel};

use crate::{
    assembly::assemble_linear_system,
    contracts::{
        ComputeBackend, ElectromagneticSolveOptions, FeaElectromagneticRunResult, FeaRunError,
        FeaRunResult,
    },
    diagnostics::{FeaDiagnostic, FeaDiagnosticSeverity},
    operator::OperatorSystem,
    solve::{
        backend::{cpu_reference::CpuReferenceBackend, kind::LinearAlgebraBackendKind},
        linear::solve_linear_system,
        preconditioner::SpdPreconditionerKind,
    },
};

pub fn run_electromagnetic(
    model: &AnalysisModel,
    backend: ComputeBackend,
) -> Result<FeaElectromagneticRunResult, FeaRunError> {
    run_electromagnetic_with_options(model, backend, ElectromagneticSolveOptions::default())
}

pub fn run_electromagnetic_with_options(
    model: &AnalysisModel,
    backend: ComputeBackend,
    options: ElectromagneticSolveOptions,
) -> Result<FeaElectromagneticRunResult, FeaRunError> {
    validate_model(model).map_err(|err| FeaRunError::InvalidModel(err.to_string()))?;

    let Some(domain) = model.electromagnetic.as_ref() else {
        return Err(FeaRunError::InvalidModel(
            "electromagnetic solve requires model.electromagnetic".to_string(),
        ));
    };
    if !domain.enabled {
        return Err(FeaRunError::InvalidModel(
            "electromagnetic solve requires enabled model.electromagnetic".to_string(),
        ));
    }
    if !domain.reference_frequency_hz.is_finite() || domain.reference_frequency_hz <= 0.0 {
        return Err(FeaRunError::InvalidModel(
            "electromagnetic solve requires finite positive reference_frequency_hz".to_string(),
        ));
    }
    if !domain.applied_current_a.is_finite() || domain.applied_current_a <= 0.0 {
        return Err(FeaRunError::InvalidModel(
            "electromagnetic solve requires finite positive applied_current_a".to_string(),
        ));
    }

    let mut summary = assemble_linear_system(model, options.prep_context, None, None);
    let node_count = summary.dof_count.max(8);
    let conductivity_mean = {
        let mut count = 0usize;
        let mut sum = 0.0_f64;
        for material in &model.materials {
            if let Some(electrical) = material.electrical.as_ref() {
                sum += electrical.conductivity_s_per_m.max(1.0e-9);
                count = count.saturating_add(1);
            }
        }
        if count == 0 {
            1.0
        } else {
            sum / count as f64
        }
    };

    let mu0 = 4.0e-7 * std::f64::consts::PI;
    let rel_permeability = 1.0_f64;
    let reluctivity = 1.0 / (mu0 * rel_permeability);
    let omega = 2.0 * std::f64::consts::PI * domain.reference_frequency_hz;
    let h = 1.0 / (node_count - 1) as f64;
    let coupling = reluctivity / (h * h);
    let mass_term = (omega * conductivity_mean).max(1.0e-9);

    let mut constrained = vec![false; node_count];
    constrained[0] = true;
    constrained[node_count - 1] = true;
    let mut rhs = vec![0.0_f64; node_count];
    for (i, rhs_i) in rhs.iter_mut().enumerate() {
        if constrained[i] {
            continue;
        }
        let x = i as f64 * h;
        *rhs_i = domain.applied_current_a * (std::f64::consts::PI * x).sin().abs();
    }

    summary.dof_count = node_count;
    summary.constrained_dof_count = constrained.iter().filter(|v| **v).count();
    summary.operator = OperatorSystem {
        dof_count: node_count,
        constrained: constrained.clone(),
        stiffness_diag: vec![2.0 * coupling + mass_term; node_count],
        stiffness_upper: vec![coupling; node_count - 1],
        mass_diag: vec![1.0; node_count],
        damping_diag: vec![0.0; node_count],
        rhs,
    };

    let backend_kind = if backend == ComputeBackend::Gpu {
        LinearAlgebraBackendKind::RuntimeTensor
    } else {
        LinearAlgebraBackendKind::CpuReference
    };
    let cpu_backend = CpuReferenceBackend;
    let solve = solve_linear_system(
        &summary,
        SpdPreconditionerKind::Jacobi,
        backend_kind,
        &cpu_backend,
    );

    let vector_potential = solve.solution;
    let mut flux_density = vec![0.0_f64; node_count];
    for i in 1..(node_count - 1) {
        flux_density[i] = ((vector_potential[i + 1] - vector_potential[i - 1]) / (2.0 * h)).abs();
    }
    if node_count > 1 {
        flux_density[0] = flux_density[1];
        flux_density[node_count - 1] = flux_density[node_count - 2];
    }

    let max_residual_norm = solve.residual_norm;
    let solve_quality = if options.residual_target <= 0.0 || !max_residual_norm.is_finite() {
        0.0
    } else {
        (options.residual_target / (max_residual_norm + options.residual_target)).clamp(0.0, 1.0)
    };
    let energy_proxy = flux_density.iter().map(|v| v * v).sum::<f64>();

    let mut diagnostics = solve.diagnostics;
    diagnostics.push(FeaDiagnostic {
        code: "FEA_EM_STATIC".to_string(),
        severity: if max_residual_norm <= options.residual_target {
            FeaDiagnosticSeverity::Info
        } else {
            FeaDiagnosticSeverity::Warning
        },
        message: format!(
            "enabled={} reference_frequency_hz={} applied_current_a={} conductivity_mean_s_per_m={} reluctivity={} max_residual_norm={} solve_quality={} placeholder_quality={} energy_proxy={}",
            domain.enabled,
            domain.reference_frequency_hz,
            domain.applied_current_a,
            conductivity_mean,
            reluctivity,
            max_residual_norm,
            solve_quality,
            solve_quality,
            energy_proxy,
        ),
    });

    let run = FeaRunResult {
        backend,
        solver_backend: solve.solver_backend,
        solver_device_apply_k_ratio: if solve.device_apply_k_attempt_count == 0 {
            0.0
        } else {
            solve.device_apply_k_count as f64 / solve.device_apply_k_attempt_count as f64
        },
        solver_method: solve.solver_method,
        preconditioner: solve.preconditioner,
        solver_host_sync_count: solve.host_sync_count,
        diagnostics,
        displacement_field: AnalysisField::host_f64(
            "field_em_vector_potential",
            vec![node_count],
            vector_potential.clone(),
        ),
        von_mises_field: AnalysisField::host_f64(
            "field_em_flux_density",
            vec![node_count],
            flux_density.clone(),
        ),
    };

    Ok(FeaElectromagneticRunResult {
        run,
        reference_frequency_hz: domain.reference_frequency_hz,
        applied_current_a: domain.applied_current_a,
        vector_potential_field: AnalysisField::host_f64(
            "field_em_vector_potential",
            vec![node_count],
            vector_potential,
        ),
        flux_density_field: AnalysisField::host_f64(
            "field_em_flux_density",
            vec![node_count],
            flux_density,
        ),
        max_residual_norm,
        solve_quality,
    })
}
