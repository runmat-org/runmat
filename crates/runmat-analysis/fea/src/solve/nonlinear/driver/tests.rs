use super::block::NonlinearLoadBlockResult;
use super::*;
use crate::{
    assembly::{dofs::StructuralDofKind, nonlinear_solid::tests::fixture},
    constraints::ResolvedConstraintMatrix,
    solve::nonlinear::nonlinear_solver_content_digest,
};
use runmat_analysis_core::{
    AnalysisConstraint, CoordinateFrame, Displacement, PrescribedStructuralDofs, PrescribedValue,
    RegionTarget, StructuralConstraint, StructuralConstraintKind, StructuralTranslationDof,
    TranslationDofValue,
};
use runmat_execution::Digest;
use runmat_meshing_core::{ElementOrder, SolverMeshArtifact};

fn uniaxial(order: ElementOrder) -> (AnalysisModel, AssemblySummary, NonlinearSolveOptions) {
    let (mut model, mesh, mut summary) = fixture(order, true);
    model.sequence.steps = vec![runmat_analysis_core::ProcedureStep::structural_nonlinear(
        "j2",
    )];
    let mut fixed = Vec::new();
    summary.physical_load = vec![0.0; summary.dof_count];
    for (node_index, node) in mesh.topology.nodes.iter().enumerate() {
        let translations = [
            StructuralTranslationDof::Ux,
            StructuralTranslationDof::Uy,
            StructuralTranslationDof::Uz,
        ]
        .into_iter()
        .enumerate()
        .filter(|(axis, _)| node.coordinates_m.canonical_value()[*axis] == 0.0)
        .map(|(_, dof)| TranslationDofValue {
            dof,
            value: PrescribedValue::Constant(Displacement::new(0.0)),
        })
        .collect::<Vec<_>>();
        if !translations.is_empty() {
            model
                .constraints
                .push(AnalysisConstraint::Structural(StructuralConstraint {
                    constraint_id: format!("pin_{}", node.node_id),
                    frame: CoordinateFrame::Global,
                    kind: StructuralConstraintKind::Prescribed(PrescribedStructuralDofs {
                        target: RegionTarget::point(format!("node:{}", node.node_id)),
                        translations,
                        rotations: Vec::new(),
                    }),
                }));
        }
        for (axis, kind) in [
            StructuralDofKind::Ux,
            StructuralDofKind::Uy,
            StructuralDofKind::Uz,
        ]
        .into_iter()
        .enumerate()
        {
            if node.coordinates_m.canonical_value()[axis] == 0.0 {
                fixed.push(
                    summary
                        .structural_dof_layout
                        .index(node_index, kind)
                        .unwrap(),
                );
            }
        }
        let x = node.coordinates_m.canonical_value();
        // Independent inclined-face traction integral: sigma_xx=5.6, area_x=1/2.
        // Tri3 weights area/3; affine Tri6 has zero corner weights, area/3 midside.
        if (x.iter().sum::<f64>() - 1.0).abs() < 1e-14 {
            let weight = match order {
                ElementOrder::Tet4 => 1.0 / 6.0,
                ElementOrder::Tet10 if x.iter().filter(|v| **v == 0.5).count() == 2 => 1.0 / 6.0,
                _ => 0.0,
            };
            summary.physical_load[summary
                .structural_dof_layout
                .index(node_index, StructuralDofKind::Ux)
                .unwrap()] = 5.6 * weight;
        }
    }
    fixed.sort_unstable();
    summary.constraint_matrix = Some(ResolvedConstraintMatrix {
        full_dof_count: summary.dof_count,
        row_offsets: (0..=fixed.len()).collect(),
        column_indices: fixed.clone(),
        coefficients: vec![1.0; fixed.len()],
        right_hand_side: vec![0.0; fixed.len()],
        row_identities: fixed
            .iter()
            .map(|row| Digest::sha256(row.to_le_bytes()))
            .collect(),
        constraint_ids: fixed.iter().map(|row| format!("anchor_{row}")).collect(),
    });
    let options = options(mesh);
    (model, summary, options)
}

fn options(mesh: SolverMeshArtifact) -> NonlinearSolveOptions {
    NonlinearSolveOptions {
        increment_count: 4,
        tolerance: 1e-9,
        linear_tolerance: 1e-12,
        solver_mesh: Some(mesh),
        ..NonlinearSolveOptions::default()
    }
}

#[test]
fn internal_driver_publishes_accepted_physical_force_stress_and_plastic_history() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let (model, summary, options) = uniaxial(order);
        let result =
            solve_nonlinear_system(&model, &summary, &options, ComputeBackend::Cpu).unwrap();
        assert_eq!(result.load_factors.last(), Some(&1.0));
        assert_eq!(result.converged_increments, result.total_increments);
        assert_eq!(result.accepted_state.load_factor(), 1.0);
        assert_eq!(
            result.accepted_state.displacement(),
            result.displacement_snapshots.last().unwrap()
        );
        let mesh = runmat_meshing_core::fixtures::canonical_tetrahedron_solver_mesh(order);
        let plan = crate::topology::CanonicalSolidIntegrationPlan::derive(
            &mesh,
            crate::topology::CanonicalSolidIntegrationRule::Keast15,
        )
        .unwrap();
        assert_eq!(
            result.accepted_state.integration_identity(),
            plan.identity()
        );
        let history = result.accepted_state.point_history().collect::<Vec<_>>();
        assert_eq!(history.len(), plan.total_point_count());
        assert!(history
            .iter()
            .all(|(_, _, state)| (state.equivalent_plastic_strain - 0.02).abs() < 2e-9));
        let encoded = serde_json::to_vec(&result.accepted_state).unwrap();
        let cold: crate::solve::nonlinear::AcceptedSolidState =
            serde_json::from_slice(&encoded).unwrap();
        assert_eq!(cold, result.accepted_state);
        assert!(result.residual_norms.iter().all(|r| *r <= 1e-9));
        assert!(result.contact_pressure_snapshots.iter().all(Vec::is_empty));
        assert!(result.contact_gap_snapshots.iter().all(Vec::is_empty));
        assert!((result.stress_snapshots.last().unwrap()[0] - 5.6).abs() < 2e-9);
        for strain in result
            .total_strain_snapshots
            .last()
            .unwrap()
            .chunks_exact(6)
        {
            for (actual, expected) in strain.iter().zip([
                5.6 / 1200.0 + 0.02,
                -0.25 * 5.6 / 1200.0 - 0.01,
                -0.25 * 5.6 / 1200.0 - 0.01,
                0.0,
                0.0,
                0.0,
            ]) {
                assert!((actual - expected).abs() < 2e-9);
            }
        }
        assert!((result.von_mises_snapshots.last().unwrap()[0] - 5.6).abs() < 2e-9);
        assert!(
            (result.equivalent_plastic_strain_snapshots.last().unwrap()[0] - 0.02).abs() < 2e-9
        );
        let ep = result.plastic_strain_snapshots.last().unwrap();
        for (actual, expected) in ep.iter().zip([0.02, -0.01, -0.01, 0.0, 0.0, 0.0]) {
            assert!((actual - expected).abs() < 2e-9);
        }
        let force = result.internal_force_snapshots.last().unwrap();
        let reduced = summary
            .constraint_matrix
            .as_ref()
            .unwrap()
            .affine_reduction()
            .unwrap();
        let residual = force
            .iter()
            .zip(&summary.physical_load)
            .map(|(a, b)| a - b)
            .collect::<Vec<_>>();
        assert!(reduced
            .transpose_apply(&residual)
            .unwrap()
            .iter()
            .all(|r| r.abs() < 2e-9));
        let limits = crate::planning::resource::NonlinearHistoryLimits {
            encoded_bytes: crate::planning::resource::ByteCount(1 << 26),
            collection_items: crate::planning::resource::Count(1 << 26),
        };
        let original = nonlinear_solver_content_digest(&result, limits).unwrap();
        let mut changed_history = result.clone();
        changed_history.accepted_state.constitutive.points[0].equivalent_plastic_strain += 0.01;
        assert_ne!(
            nonlinear_solver_content_digest(&changed_history, limits).unwrap(),
            original
        );
        let mut changed = result;
        changed.internal_force_snapshots.last_mut().unwrap()[0] += 1.0;
        assert_ne!(
            nonlinear_solver_content_digest(&changed, limits).unwrap(),
            original
        );
    }
}

#[test]
fn driver_honors_authored_boundaries_and_global_accepted_cap() {
    let (model, summary, mut options) = uniaxial(ElementOrder::Tet4);
    options.increment_count = 2;
    let schedule = LoadIncrementSchedule::canonical(&[], &[0.3], 0.0, 1.0).unwrap();
    let result = solve_nonlinear_system_at_boundaries(
        &model,
        &summary,
        &options,
        NonlinearExecution {
            backend: ComputeBackend::Cpu,
            contact: None,
            solver_admission: None,
        },
        0.0,
        1.0,
        &schedule,
    )
    .unwrap();
    assert!(result.load_factors.contains(&0.3));
    options.max_accepted_increments = 1;
    assert!(matches!(
        solve_nonlinear_system(&model, &summary, &options, ComputeBackend::Cpu),
        Err(FeaRunError::Solve(_))
    ));
    assert!(solve_nonlinear_system_at_boundaries(
        &model,
        &summary,
        &options,
        NonlinearExecution {
            backend: ComputeBackend::Cpu,
            contact: None,
            solver_admission: None
        },
        0.3,
        1.0,
        &schedule
    )
    .is_err());
}

#[test]
fn driver_does_not_fallback_for_unqualified_backend_or_missing_physical_mesh() {
    let (model, summary, mut options) = uniaxial(ElementOrder::Tet4);
    assert!(solve_nonlinear_system(&model, &summary, &options, ComputeBackend::Gpu).is_err());
    options.solver_mesh = None;
    assert!(solve_nonlinear_system(&model, &summary, &options, ComputeBackend::Cpu).is_err());
}

mod blocks;
