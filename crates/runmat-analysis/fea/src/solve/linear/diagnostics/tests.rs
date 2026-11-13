use super::*;

#[test]
fn successful_profile_bounds_each_supported_placement_outcome() {
    let bound = crate::solve::linear::successful_diagnostics_profile().unwrap();
    let mut largest_actual_count = 0;
    for strategy in all_strategies() {
        for messages in [
            vec![SuccessfulDiagnostic::CpuMethod(strategy)],
            vec![
                SuccessfulDiagnostic::AdmissionFallback,
                SuccessfulDiagnostic::CpuMethod(strategy),
            ],
            vec![SuccessfulDiagnostic::AcceleratorMethod(
                SpdPreconditionerKind::Jacobi,
            )],
        ] {
            let actual: Vec<_> = messages.iter().map(|value| value.diagnostic()).collect();
            largest_actual_count = largest_actual_count.max(actual.len());
            assert!(actual.len() <= crate::solve::linear::successful_diagnostic_record_count());
            let profile = profile_serialized(&actual, PROFILE_LIMITS).unwrap();
            assert!(profile.encoded_bytes_upper_bound <= bound.encoded_bytes_upper_bound);
            assert!(
                profile.materialization_resident_bytes_estimate
                    <= bound.materialization_resident_bytes_estimate
            );
            let borrowed: Vec<_> = messages.iter().map(|value| value.wire()).collect();
            assert_eq!(
                profile,
                profile_serialized(&borrowed, PROFILE_LIMITS).unwrap()
            );
            assert_eq!(
                serde_json::to_value(actual).unwrap(),
                serde_json::to_value(borrowed).unwrap()
            );
        }
    }
    assert_eq!(
        crate::solve::linear::successful_diagnostic_record_count(),
        largest_actual_count
    );
}

#[test]
fn extracted_templates_preserve_existing_diagnostic_content() {
    for (message, code, severity, expected) in [
        (SuccessfulDiagnostic::CpuMethod(SpdStrategy::Pcg(runmat_sparse_solvers::policy::PreconditionerChoice::Jacobi)),
         "FEA_SOLVER_METHOD", FeaDiagnosticSeverity::Info,
         "solver=pcg preconditioner=jacobi"),
        (SuccessfulDiagnostic::CpuMethod(SpdStrategy::Cholesky),
         "FEA_SOLVER_METHOD", FeaDiagnosticSeverity::Info,
         "solver=sparse_cholesky ordering=nested_dissection"),
        (SuccessfulDiagnostic::AcceleratorMethod(SpdPreconditionerKind::Ilu0),
         "FEA_SOLVER_METHOD", FeaDiagnosticSeverity::Info,
         "solver=pcg preconditioner=ilu0 matrix_free=false backend=accelerator_native_sparse"),
        (SuccessfulDiagnostic::AdmissionFallback,
         "FEA_ACCELERATOR_ADMISSION_FALLBACK", FeaDiagnosticSeverity::Warning,
         "native sparse execution was rejected before provider allocation; using the cpu_reference solver"),
    ] {
        assert_eq!(message.diagnostic(), FeaDiagnostic {
            code: code.into(), severity, message: expected.into(),
        });
    }
}
