//! Rejected endpoints earn no numerical or accelerator qualification evidence.
use super::*;

pub(super) fn check(
    spec: &FixtureSpec,
    model: &AnalysisModel,
    solver_mesh_artifact_path: Option<&str>,
    failures: &mut Vec<String>,
) -> Option<String> {
    let expected = spec.expect_run_error.expect("rejection fixture contract");
    let run = |mode: Option<GpuMode>| {
        use runmat_analysis_core::{CoupledPhysicsFamily, PhysicsFamily};
        let family = fixture_procedure_step(model).family;
        if matches!(
            family,
            PhysicsFamily::Fluid
                | PhysicsFamily::Coupled(
                    CoupledPhysicsFamily::ConjugateHeatTransfer
                        | CoupledPhysicsFamily::FluidStructureInteraction
                )
        ) {
            // Test each removed public endpoint directly. Unsupported transient default
            // solver selection must not prevent this boundary from being exercised.
            let backend = if mode.is_some() {
                ComputeBackend::Gpu
            } else {
                ComputeBackend::Cpu
            };
            let invoke = || {
                boxed_fixture_run_result(match family {
                    PhysicsFamily::Coupled(CoupledPhysicsFamily::ConjugateHeatTransfer) => {
                        analysis_run_cht_op(model, OperationContext::new(None, None))
                    }
                    PhysicsFamily::Coupled(CoupledPhysicsFamily::FluidStructureInteraction) => {
                        analysis_run_fsi_op(model, OperationContext::new(None, None))
                    }
                    _ => analysis_run_cfd_with_options_op(
                        model,
                        backend,
                        AnalysisCfdRunOptions::default(),
                        OperationContext::new(None, None),
                    ),
                })
            };
            return match mode {
                Some(GpuMode::WithProvider) => with_harness_provider(invoke),
                _ => invoke(),
            };
        }
        match mode {
            Some(mode) => run_fixture_gpu(spec, model, mode, solver_mesh_artifact_path),
            None => run_fixture_cpu(spec, model, solver_mesh_artifact_path),
        }
    };
    let mut results = vec![("cpu", run(None))];
    if let Some(mode) = spec.gpu_mode {
        results.push(("gpu", run(Some(mode))));
    }
    let mut error_code = None;
    for (placement, result) in results {
        // GPU placement falls back to the CPU reference when the accelerator path cannot run
        // the solve, so every placement reports the CPU reference's rejection.
        match result {
            Ok(_) => failures.push(format!(
                "expected {placement} run error code {expected}, but run succeeded"
            )),
            Err(error) => {
                error_code = Some(error.error_code.clone());
                if error.error_code != expected {
                    failures.push(format!(
                        "{placement} run error mismatch: expected {expected}, got {}: {}",
                        error.error_code, error.message
                    ));
                }
            }
        }
    }
    error_code
}

#[test]
fn removed_cfd_endpoint_rejects_all_declared_placements_without_numerical_credit() {
    assert_removed_endpoint_placements(
        |id| id.starts_with("cfd_steady_") || id.starts_with("cfd_transient_"),
        6,
        "RM.FEA.RUN_CFD.AUTHORED_STUDY_REQUIRED",
    );
}

// The coupled-flow physics these placements once claimed is qualified on the typed path:
// runmat-analysis-fea physics::conjugate_heat::tests and physics::fluid_structure::tests
// (series conduction, contact-resistance jump, plug-flow energy closure, lubrication-theory
// channel, interface energy; contact interfaces rejected as coupling interfaces).
#[test]
fn removed_coupled_flow_endpoints_reject_all_declared_placements_without_numerical_credit() {
    assert_removed_endpoint_placements(
        |id| id.starts_with("cht_coupled_") && !id.ends_with("_invalid_cfd_domain"),
        5,
        "RM.FEA.RUN_CHT.AUTHORED_STUDY_REQUIRED",
    );
    assert_removed_endpoint_placements(
        |id| id.starts_with("fsi_coupled_") && !id.ends_with("_invalid_cfd_domain"),
        5,
        "RM.FEA.RUN_FSI.AUTHORED_STUDY_REQUIRED",
    );
}

fn assert_removed_endpoint_placements(
    select: impl Fn(&str) -> bool,
    placements: usize,
    code: &'static str,
) {
    let _admission = crate::fea_admission::guard();
    let specs = super::super::manifest::manifest_specs()
        .into_iter()
        .filter(|spec| select(spec.id))
        .collect::<Vec<_>>();
    assert_eq!(
        specs.len(),
        placements,
        "preserve all original endpoint placements"
    );
    for spec in specs {
        assert_eq!(spec.expect_run_error, Some(code));
        assert!(spec.expected_numerically_acceptable.is_none());
        assert!(spec.expected_solver_backend.is_none());
        let record = run_fixture(&spec, None);
        assert!(
            record.failures.is_empty(),
            "{}: {:?}",
            spec.id,
            record.failures
        );
        assert!(
            record.validate_ok,
            "{}: {:?}",
            spec.id, record.validate_error_code
        );
        assert!(!record.run_ok);
        assert_eq!(record.run_error_code.as_deref(), spec.expect_run_error);
        assert!(record.numerically_acceptable.is_none());
        assert!(record.published_primary_field_storage.is_none());
        assert!(record.gpu_fallback_events.is_empty());
        assert!(record.threshold_assertions.is_empty());
    }
}
