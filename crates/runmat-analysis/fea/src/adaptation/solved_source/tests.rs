//! Actual original static solves feed transfer, recovery, observation and shared accuracy.
use runmat_analysis_core::{AnalysisFieldLayout, ObservationQuantity, FieldLocation};
use super::*;
use crate::adaptation::{StructuralAdaptationDecisionStatus, StructuralRecoveryEstimatorOptions};
use crate::{ComputeBackend, LinearStaticSolveOptions};
use runmat_analysis_core::*;
use runmat_meshing_core::{fixtures::canonical_tetrahedron_solver_mesh, ElementOrder};
mod fixture;
use fixture::*;

// witness: FEA-SHARED-ACCURACY-001/dimensioned_targets_and_contributions @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel; coverage: partial;
//   boundary: Original static results retain typed convergence and scalar observations for both
//   element orders.
#[test]
fn original_both_order_static_results_retain_typed_convergence_and_scalar_observations() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let mesh = canonical_tetrahedron_solver_mesh(order);
        let model = model(&mesh);
        let run = solve(&model, &mesh);
        let convergence = run.structural_static_convergence.as_ref().unwrap();
        assert!(convergence.converged);
        assert_eq!(convergence.relative_residual, Some(0.0));
        let id = ObservationId("tip_displacement".into());
        let input = source(&model, &mesh, &run, &id);
        let evidence = StructuralAdaptationSolvedSource::derive(input).unwrap();
        assert!((evidence.target_quantity.value - 0.02).abs() < 1e-14);
        let cold_run: FeaRunResult =
            serde_json::from_slice(&serde_json::to_vec(&run).unwrap()).unwrap();
        let cold: StructuralAdaptationSolvedSource =
            serde_json::from_slice(&serde_json::to_vec(&evidence).unwrap()).unwrap();
        cold.validate_against(source(&model, &mesh, &cold_run, &id))
            .unwrap();
        // Drop typed convergence while leaving human diagnostic text and physical balance intact.
        let mut missing = serde_json::to_value(&run).unwrap();
        missing
            .as_object_mut()
            .unwrap()
            .remove("structural_static_convergence");
        assert!(serde_json::from_value::<FeaRunResult>(missing).is_err());
        let mut different_report = run.clone();
        different_report.diagnostics.clear();
        let alternate =
            StructuralAdaptationSolvedSource::derive(source(&model, &mesh, &different_report, &id))
                .unwrap();
        assert_eq!(
            alternate.solver.result_identity,
            evidence.solver.result_identity
        );
        assert_ne!(
            alternate.original_content_digest,
            evidence.original_content_digest
        );
        let mut forged = evidence.clone();
        forged.solver.iteration_count += 1;
        assert!(forged.validate_against(input).is_err());
        let mut changed = run.clone();
        let field = changed
            .fields
            .iter_mut()
            .find(|f| f.field_id == crate::contracts::FEA_FIELD_STRUCTURAL_DISPLACEMENT)
            .unwrap();
        *field = AnalysisField::host_f64(
            &field.field_id, field.layout.clone(),
            field.shape.clone(),
            vec![0.0; field.as_host_f64().unwrap().len()],
        );
        assert!(evidence
            .validate_against(source(&model, &mesh, &changed, &id))
            .is_err());
        assert!(
            StructuralAdaptationSolvedSource::derive(source(&model, &mesh, &changed, &id)).is_err()
        );
        let mut invalid = serde_json::to_value(&run).unwrap();
        invalid["structural_static_convergence"]["relative_residual"] = serde_json::Value::Null;
        assert!(serde_json::from_value::<FeaRunResult>(invalid).is_err());
    }
}

#[test]
fn actual_h_refinement_resolves_interior_dofs_and_governs_original_evidence() {
    let (coarse, fine, map, lineage) = crate::adaptation::iteration::tests::adaptation_fixture();
    let model = model(&coarse);
    let original = solve(&model, &coarse);
    let refined = solve(&model, &fine);
    assert!(
        refined
            .structural_static_convergence
            .as_ref()
            .unwrap()
            .iterations
            > 0
    );
    // Independent affine reference: E=100, nu=.25, u=(.02x,0,0); sigma=(2.4,.8,.8,0,0,0).
    for run in [&original, &refined] {
        for tensor in run
            .field(crate::contracts::FEA_FIELD_STRUCTURAL_INTEGRATION_POINT_STRESS)
            .unwrap()
            .as_host_f64()
            .unwrap()
            .chunks_exact(6)
        {
            for (actual, reference) in tensor.iter().zip([2.4, 0.8, 0.8, 0.0, 0.0, 0.0]) {
                assert!((actual - reference).abs() < 1e-12);
            }
        }
    }
    for (node, u) in fine.topology.nodes.iter().zip(
        refined
            .field(crate::contracts::FEA_FIELD_STRUCTURAL_DISPLACEMENT)
            .unwrap()
            .as_host_f64()
            .unwrap()
            .chunks_exact(3),
    ) {
        assert!((u[0] - 0.02 * node.coordinates_m.canonical_value()[0]).abs() < 1e-14);
        assert!(u[1].abs() < 1e-14 && u[2].abs() < 1e-14);
    }
    let id = ObservationId("tip_displacement".into());
    let input = StructuralAdaptationSolvedIterationInput {
        source: source(&model, &coarse, &original, &id),
        target: source(&model, &fine, &refined, &id),
        transfer_map: &map,
        lineage: &lineage,
        estimator_options: StructuralRecoveryEstimatorOptions::default(),
        previous: None,
    };
    let plan = crate::adaptation::iteration::tests::governed::plan();
    let iteration =
        build_structural_adaptation_iteration_from_solved_results(input, &plan).unwrap();
    assert_eq!(
        iteration.iteration.decision.status,
        StructuralAdaptationDecisionStatus::Converged
    );
    assert_eq!(
        iteration.iteration.source_solver_result,
        iteration.source.solver
    );
    assert_eq!(iteration.iteration.solver_result, iteration.target.solver);
    assert!(
        iteration.iteration.transfer_errors[0]
            .relative_l2_error
            .unwrap()
            < 1e-14
    );
    let cold: StructuralAdaptationSolvedIteration =
        serde_json::from_slice(&serde_json::to_vec(&iteration).unwrap()).unwrap();
    cold.validate_against(input, &plan).unwrap();
    let mut tampered = cold.clone();
    tampered.iteration.target_quantity.value += 0.001;
    assert!(tampered.validate_against(input, &plan).is_err());
    let mut changed = model.clone();
    changed.sequence.steps[0].observations.clear();
    assert!(
        StructuralAdaptationSolvedSource::derive(source(&changed, &fine, &refined, &id)).is_err()
    );
    let mut missing = refined.clone();
    missing.observations.clear();
    assert!(
        StructuralAdaptationSolvedSource::derive(source(&model, &fine, &missing, &id)).is_err()
    );
    let previous = StructuralAdaptationSolvedIterationInput {
        previous: Some(&cold),
        ..input
    };
    assert!(build_structural_adaptation_iteration_from_solved_results(previous, &plan).is_err());
}

#[test]
fn original_result_materialization_is_admitted_before_encoding() {
    let mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    let model = model(&mesh);
    let run = solve(&model, &mesh);
    let id = ObservationId("tip_displacement".into());
    let mut input = source(&model, &mesh, &run, &id);
    let required = profile_serialized(
        &(&model, mesh.canonical_digest, &run),
        input.limits.canonical,
    )
    .unwrap()
    .materialization_resident_bytes_estimate
        + profile_serialized(&run.fields, input.limits.canonical)
            .unwrap()
            .materialization_resident_bytes_estimate;
    input.limits.maximum_materialization_bytes = required - 1;
    assert_eq!(
        StructuralAdaptationSolvedSource::derive(input).unwrap_err(),
        Error::ResourceLimit
    );
    input.limits.maximum_materialization_bytes = required;
    StructuralAdaptationSolvedSource::derive(input).unwrap();
    let token = crate::progress::FeaCancellationToken::new();
    token.cancel();
    let _guard =
        crate::progress::replace_fea_progress_context(crate::progress::FeaProgressContext {
            cancellation: token,
            ..Default::default()
        });
    assert_eq!(
        StructuralAdaptationSolvedSource::derive(input).unwrap_err(),
        Error::Cancelled
    );
}

#[test]
fn shared_field_transfer_observes_cancelled_owner_before_copying_payloads() {
    let mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    let field = AnalysisField::host_f64(
        crate::contracts::FEA_FIELD_STRUCTURAL_DISPLACEMENT, AnalysisFieldLayout::vector3(ObservationQuantity::Displacement, FieldLocation::Node),
        vec![mesh.topology.nodes.len(), 3],
        vec![0.0; mesh.topology.nodes.len() * 3],
    );
    let token = crate::progress::FeaCancellationToken::new();
    token.cancel();
    let _guard =
        crate::progress::replace_fea_progress_context(crate::progress::FeaProgressContext {
            cancellation: token,
            ..Default::default()
        });
    assert_eq!(
        crate::adaptation::transfer_solver_field(&mesh, &mesh, None, "nodes", &field).unwrap_err(),
        crate::adaptation::SolverFieldTransferError::Cancelled
    );
}

#[test]
fn source_materialization_is_admitted_before_constraint_lowering() {
    let mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    let mut model = model(&mesh);
    let run = solve(&model, &mesh);
    let id = ObservationId("tip_displacement".into());
    let AnalysisConstraint::Structural(constraint) = &mut model.constraints[0] else {
        panic!("fixture constraint")
    };
    let StructuralConstraintKind::Prescribed(prescribed) = &mut constraint.kind else {
        panic!("fixture prescribed DOFs")
    };
    // A target that would fail during lowering distinguishes early budget admission.
    prescribed.target = RegionTarget::point("node:999999");
    let mut input = source(&model, &mesh, &run, &id);
    input.limits.maximum_materialization_bytes = 0;
    assert_eq!(
        StructuralAdaptationSolvedSource::derive(input).unwrap_err(),
        Error::ResourceLimit
    );
}
