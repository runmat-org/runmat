//! Static pipeline admission is distinct from sparse-provider operation conformance.
use super::*;

fn assert_gpu_rejected_without_publication() {
    let before = storage::list_run_metadata().unwrap().len();
    let error = runmat_analysis_fea::accelerator_admission::with_accelerator_admission_policy(
        runmat_analysis_core::AcceleratorAdmissionPolicy::Required,
        || {
            analysis_run_linear_static_with_options(
                &sample_model(),
                ComputeBackend::Gpu,
                Default::default(),
                OperationContext::new(None, None),
            )
        },
    )
    .expect_err("static GPU pipeline must remain closed before accelerator-bound admission");
    assert_eq!(error.error_code, ACCELERATOR_REQUIRED_ERROR_CODE);
    assert!(!error.retryable);
    assert!(error
        .message
        .contains("accelerator-bound native sparse admission"));
    assert_eq!(storage::list_run_metadata().unwrap().len(), before);
}

#[test]
fn required_static_gpu_without_provider_rejects_instead_of_falling_back() {
    let _guard = analysis_test_guard();
    let _provider = runmat_accelerate_api::ThreadProviderGuard::set(None);
    assert_gpu_rejected_without_publication();
}

#[test]
fn static_gpu_with_sparse_provider_rejects_without_allocation_or_publication() {
    let _guard = analysis_test_guard();
    let provider: &'static runmat_accelerate::simple_provider::InProcessProvider = Box::leak(
        Box::new(runmat_accelerate::simple_provider::InProcessProvider::new()),
    );
    let _provider = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    assert_eq!(provider.placement_resources().live_bytes, 0);
    assert_gpu_rejected_without_publication();
    assert_eq!(provider.placement_resources().live_bytes, 0);
}

#[test]
fn static_gpu_with_upload_only_provider_rejects_before_upload() {
    let _guard = analysis_test_guard();
    struct UploadOnly;
    static UPLOADS: AtomicU64 = AtomicU64::new(0);
    impl AccelProvider for UploadOnly {
        fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
            UPLOADS.fetch_add(1, Ordering::Relaxed);
            Err(anyhow::anyhow!("unadmitted upload"))
        }
        fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
            Box::pin(async { Err(anyhow::anyhow!("unadmitted download")) })
        }
        fn device_info(&self) -> String {
            "upload-only-contract-fixture".into()
        }
    }
    static PROVIDER: UploadOnly = UploadOnly;
    let _provider = runmat_accelerate_api::ThreadProviderGuard::set(Some(&PROVIDER));
    UPLOADS.store(0, Ordering::Relaxed);
    assert_gpu_rejected_without_publication();
    assert_eq!(UPLOADS.load(Ordering::Relaxed), 0);
}

#[test]
fn sparse_provider_operator_contract_solves_spd_and_releases_resources() {
    // In-process provider conformance only: neither actual GPU hardware evidence nor admission
    // of the complete structural static pipeline is implied by this operation-level test.
    let _guard = analysis_test_guard();
    let provider: &'static runmat_accelerate::simple_provider::InProcessProvider = Box::leak(
        Box::new(runmat_accelerate::simple_provider::InProcessProvider::new()),
    );
    let _provider = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    use runmat_analysis_fea::algebra::{
        AssembledRealSystem, CanonicalRealOperators, MatrixProperties, OperatorRole,
        RealSparseMatrix,
    };
    let dimension = 32;
    let stiffness = RealSparseMatrix::diagonal(
        vec![2.0; dimension],
        OperatorRole::Stiffness,
        MatrixProperties::SYMMETRIC_POSITIVE_DEFINITE,
    )
    .unwrap();
    let mass = RealSparseMatrix::diagonal(
        vec![1.0; dimension],
        OperatorRole::Mass,
        MatrixProperties::SYMMETRIC_POSITIVE_DEFINITE,
    )
    .unwrap();
    let damping = RealSparseMatrix::diagonal(
        vec![0.0; dimension],
        OperatorRole::Damping,
        MatrixProperties::SYMMETRIC_POSITIVE_SEMIDEFINITE,
    )
    .unwrap();
    let system = AssembledRealSystem::try_new(
        CanonicalRealOperators::try_new(stiffness, mass, damping).unwrap(),
        vec![false; dimension],
        vec![1.0; dimension],
    )
    .unwrap();
    let result = runmat_analysis_fea::solve::linear::solve_operator_system(
        &system,
        runmat_analysis_fea::solve::preconditioner::SpdPreconditionerKind::Jacobi,
        LinearAlgebraBackendKind::Accelerator,
        &runmat_analysis_fea::solve::backend::cpu_reference::CpuReferenceBackend,
        None,
    )
    .expect("valid static admission fixture must solve");
    assert!(result.converged);
    assert!(result
        .solution
        .iter()
        .all(|value| (value - 0.5).abs() < 1e-10));
    let evidence = result.accelerator_evidence.expect("provider evidence");
    evidence.validate().unwrap();
    assert!(evidence.satisfies_tolerance());
    assert_eq!(evidence.provider().host_synchronizations, 0);
    assert_eq!(evidence.provider().device_readbacks, 0);
    assert_eq!(evidence.terminal_solution_readbacks(), 1);
    assert!(evidence.provider().resident_matrix_bytes > 0);
    assert!(evidence.provider().resident_preconditioner_bytes > 0);
    assert!(evidence.provider().scratch_bytes > 0);
    assert!(!evidence.provider().used_host_fallback);
    assert_eq!(provider.placement_resources().live_bytes, 0);
}

#[test]
fn cpu_static_run_keeps_authoritative_host_fields() {
    let _guard = analysis_test_guard();
    let envelope = analysis_run_linear_static_with_options(
        &sample_model(),
        ComputeBackend::Cpu,
        Default::default(),
        OperationContext::new(None, None),
    )
    .unwrap();
    assert_eq!(
        envelope.data.provenance.solver_backend,
        LinearAlgebraBackendKind::CpuReference
    );
    for id in [
        FEA_FIELD_STRUCTURAL_DISPLACEMENT,
        FEA_FIELD_STRUCTURAL_BEAM_BENDING_STRESS,
    ] {
        assert!(matches!(
            envelope.data.run.field(id).unwrap().values,
            AnalysisFieldValues::HostF64(_)
        ));
    }
    assert!(envelope
        .data
        .run
        .field(FEA_FIELD_STRUCTURAL_VON_MISES)
        .is_none());
}

#[test]
fn cpu_balanced_and_strict_quality_policies_accept_clean_and_reject_conflicted_materials() {
    let _guard = analysis_test_guard();
    for policy in [QualityPolicy::Balanced, QualityPolicy::Strict] {
        for conflict in [false, true] {
            let model = if conflict {
                sample_model_with_material_assignment_mismatch()
            } else {
                sample_model()
            };
            let envelope = analysis_run_linear_static_with_options(
                &model,
                ComputeBackend::Cpu,
                AnalysisRunOptions {
                    quality_policy: policy,
                    deterministic_mode: true,
                    precision_mode: PrecisionMode::Fp64,
                    preconditioner_mode: PreconditionerMode::Auto,
                    solver_mesh_artifact_path: None,
                },
                OperationContext::new(None, None),
            )
            .unwrap();
            assert_eq!(envelope.data.solver_convergence, QualityGate::Pass);
            assert_eq!(envelope.data.numerically_acceptable, !conflict);
            assert_eq!(
                envelope.data.result_quality,
                if conflict {
                    QualityGate::Warn
                } else {
                    QualityGate::Pass
                }
            );
            assert_eq!(
                envelope.data.run_status,
                if conflict {
                    RunStatus::Degraded
                } else {
                    RunStatus::Completed
                }
            );
            assert_eq!(
                envelope
                    .data
                    .quality_reasons
                    .iter()
                    .any(|reason| reason.code == QualityReasonCode::MaterialAssignmentConflict),
                conflict
            );
            if !conflict {
                assert!(envelope.data.quality_reasons.is_empty());
            }
        }
    }
}
