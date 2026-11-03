use super::*;
use crate::execution::MagnetostaticWorkload;
use crate::physics::{electrical::tests::mesh, magnetic::test_support::authored_model};
use crate::results::field::canonical_field_partitions;
use runmat_meshing_core::ElementOrder;

#[test]
fn magnetic_recovery_profiles_cover_actual_catalog_identity_and_storage_both_orders() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let mesh = mesh(order, 2);
        let workload = MagnetostaticWorkload {
            model: authored_model(0.0),
            step_id: runmat_analysis_core::ProcedureStepId("potential".into()),
            relative_residual_target: 1e-10,
        };
        let result = workload.execute(&mesh).unwrap();
        let request = result.native_field_request(&workload, &mesh).unwrap();
        let fields = result
            .recover_native_fields(&workload, &mesh, &request)
            .unwrap();
        let (bounds, workspace) = magnetic_field_recovery_shape(&request).unwrap();
        let mut actual = vec![fields.manifest().unwrap().canonical_bytes().unwrap().len() as u64];
        for field in fields.fields() {
            actual.extend(
                canonical_field_partitions(field)
                    .unwrap()
                    .iter()
                    .map(|p| p.canonical_bytes().unwrap().len() as u64),
            );
        }
        assert_eq!(actual.len(), bounds.len());
        assert_eq!(
            magnetic_field_recovery_partition_count(&request).unwrap(),
            bounds.len()
        );
        assert!(actual
            .iter()
            .zip(&bounds)
            .all(|(actual, bound)| actual <= bound));
        assert!(workspace.0 > actual.iter().sum::<u64>());
        let bytes = bounds.iter().sum::<u64>();
        let output = PreparedOutputResourceFacts {
            encoded_payload_bytes_upper_bound: ByteCount(bytes),
            transport_closure_bytes_upper_bound: ByteCount(bytes + 1024),
            construction_resident_bytes_admission_budget: ByteCount(2 * bytes + 2048),
        };
        let plan = build_magnetic_field_recovery_resource_plan(
            &request,
            magnetic_recovery_source_validation_workspace(&workload, &mesh).unwrap(),
            PreparedPhaseResourceFacts {
                inputs: PreparedInputResourceFacts {
                    encoded_payload_bytes: ByteCount(8),
                    transport_closure_bytes: ByteCount(16),
                    decoded_resident_bytes_admission_budget: ByteCount(1024),
                },
                output: output.clone(),
                output_set: PreparedOutputSetResourceFacts {
                    encoded_bytes_upper_bound: ByteCount(1024),
                },
            },
            crate::planning::resource::LocalExecutionPlan::serial(),
        )
        .unwrap();
        assert_eq!(
            plan.input
                .demand
                .output_facts(crate::execution::AnalysisNumericalObjectKind::Field),
            Some(&output)
        );
        assert!(plan
            .input
            .demand
            .output_facts(crate::execution::AnalysisNumericalObjectKind::SolverResult)
            .is_none());
    }
}

#[test]
fn magnetic_recovery_shape_rejects_missing_or_relabelled_native_inventory() {
    let mesh = mesh(ElementOrder::Tet4, 1);
    let workload = MagnetostaticWorkload {
        model: authored_model(0.0),
        step_id: runmat_analysis_core::ProcedureStepId("potential".into()),
        relative_residual_target: 1e-10,
    };
    let result = workload.execute(&mesh).unwrap();
    let original = result.native_field_request(&workload, &mesh).unwrap();
    let mut plan = original.plan.clone();
    plan.requirements.pop();
    let request = crate::execution::FieldRecoveryRequest::new(
        original.producer,
        original.source,
        original.topology,
        plan,
    )
    .unwrap();
    assert!(magnetic_field_recovery_shape(&request).is_err());
    let mut plan = original.plan.clone();
    for requirement in &mut plan.requirements[1..] {
        requirement.cost.source_entity_count = 16;
        requirement.cost.output_entity_count = 16;
        requirement.cost.arithmetic_operation_count = 48;
    }
    let request = crate::execution::FieldRecoveryRequest::new(
        original.producer,
        original.source,
        original.topology,
        plan,
    )
    .unwrap();
    assert!(magnetic_field_recovery_shape(&request).is_err());
    let request = crate::execution::FieldRecoveryRequest::new(
        original.producer,
        original.source,
        crate::execution::FieldTopologyIdentity::VolumeMesh {
            mesh_digest: mesh.canonical_digest,
        },
        original.plan.clone(),
    )
    .unwrap();
    assert!(magnetic_field_recovery_shape(&request).is_err());
    let mut plan = original.plan.clone();
    plan.requirements.swap(1, 2);
    let request = crate::execution::FieldRecoveryRequest::new(
        original.producer,
        original.source,
        original.topology,
        plan,
    )
    .unwrap();
    assert!(magnetic_field_recovery_shape(&request).is_err());
}

/// Calibration probe (ignored): magnetostatic count reservation components against the exact
/// symbolic factor of the derived saddle, for an `RM_DIAG_N` slab (Tet10), then optionally the
/// solve's process peak RSS. Run alone (`--exact --ignored`).
#[test]
#[ignore]
fn diagnostic_magnetostatic_reservation_at_scale() {
    let n: usize = std::env::var("RM_DIAG_N").ok().and_then(|v| v.parse().ok()).unwrap_or(4);
    let mesh = mesh(ElementOrder::Tet10, n);
    let workload = MagnetostaticWorkload {
        model: authored_model(1.0),
        step_id: runmat_analysis_core::ProcedureStepId("potential".into()),
        relative_residual_target: 1e-10,
    };
    let r = crate::physics::magnetic::resources::require_canonical_ceiling(
        &workload.model,
        &mesh,
        u64::MAX >> 12,
    )
    .unwrap();
    println!("{r:?}");
    let input = crate::physics::magnetic::compile_canonical_input(
        &workload.model,
        &mesh,
        workload.step_id.as_str(),
        workload.relative_residual_target,
    )
    .unwrap();
    let derived = crate::physics::magnetic::MagneticSystemPlan::derive_with_spaces(
        &input.projection,
        &input.grounded_faces,
    )
    .unwrap();
    let saddle = derived.plan.saddle_pattern();
    let offsets = saddle.row_offsets().iter().map(|&v| v as usize).collect::<Vec<_>>();
    let columns = saddle.column_indices().iter().map(|&v| v as usize).collect::<Vec<_>>();
    let estimate = runmat_sparse_solvers::direct::SymbolicFactorization::analyze_pattern::<f64>(
        saddle.rows(),
        &offsets,
        &columns,
        &runmat_sparse_solvers::direct::DirectOptions::new(
            runmat_sparse_solvers::direct::FactorKind::BunchKaufman,
        ),
        runmat_sparse_solvers::control::SolveControl::NONE,
    )
    .unwrap()
    .estimate();
    println!(
        "nodes {} elements {} saddle_rows {} saddle_nnz {} physical_nnz {} exact_factor_peak {}",
        mesh.topology.nodes.len(),
        mesh.topology.volume_elements.len(),
        saddle.rows(),
        saddle.column_indices().len(),
        derived.plan.physical_pattern().column_indices().len(),
        estimate.peak_bytes()
    );
    drop((input, derived, offsets, columns));
    if std::env::var("RM_DIAG_SOLVE").is_ok() {
        let peak = || {
            let mut usage = std::mem::MaybeUninit::<libc::rusage>::zeroed();
            unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) };
            unsafe { usage.assume_init() }.ru_maxrss
        };
        let before = peak();
        let result = workload.execute(&mesh);
        println!(
            "rss_peak_before_solve {before} rss_peak_after_solve {} ok {}",
            peak(),
            result.is_ok()
        );
    }
}
