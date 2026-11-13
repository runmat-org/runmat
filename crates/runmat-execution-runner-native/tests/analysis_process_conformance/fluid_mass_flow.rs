//! Typed signed mass flow through the scheduled public study and cold publication path.
use runmat_analysis_core::*;
use runmat_analysis_fea::physics::cfd::canonical::qualification::{mass_flow::uniform, reference};

pub(super) async fn run() {
    for (speed, density, rotated) in [(0.001, 1.0, false), (-0.002, 3.0, true)] {
        let fixture = uniform(2, speed, density, rotated);
        let cold = super::fluid_study_fixture::cold_fixture(&fixture, "signed-mass-flow").await;
        assert!(cold
            .product
            .workload()
            .model
            .excitations
            .iter()
            .all(|a| matches!(a, ExcitationAction::MassFlow(_))));
        let original = cold.result.result.original();
        let velocity = if rotated {
            reference::rotate([speed, 0.0, 0.0])
        } else {
            [speed, 0.0, 0.0]
        };
        for u in original.velocity_m_per_s.canonical_values() {
            let error = u
                .iter()
                .zip(velocity)
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();
            assert!(error < 1e-8, "independent uniform velocity error {error:e}");
        }
        assert!(original
            .pressure_fluctuation_pa
            .canonical_values()
            .iter()
            .all(|p| p.abs() < 1e-7));
        assert_eq!(
            original.pressure_reference_pa,
            runmat_types::physical::Quantity::new(vec![1e12; fixture.topology.cells.len()])
        );
        for id in ["mass-inlet", "mass-outlet"] {
            let balance = original
                .residual
                .boundary_balances
                .iter()
                .find(|b| b.authority_id == id)
                .unwrap();
            let outward = density * speed * if id == "mass-inlet" { -1.0 } else { 1.0 };
            assert!(
                (balance.mass_flow_rate_kg_per_s - runmat_types::physical::Quantity::new(outward))
                    .abs()
                    < runmat_types::physical::Quantity::new(1e-14)
            );
            assert!((balance.area_m2.canonical_value() - 1.0).abs() < 1e-14);
        }
        eprintln!("scheduled signed mass flow speed={speed}, density={density}, rotated={rotated}: original fields, full state, receipts and cold publication preserved");
    }
    transient().await;
}

async fn transient() {
    let fixture =
        runmat_analysis_fea::physics::cfd::canonical::qualification::mass_flow::accelerating(
            2, 0.001, 0.01, 1.0,
        );
    let (mut workload, _) = super::fluid_transient::fixture();
    workload.model.constraints = fixture
        .constraints
        .iter()
        .cloned()
        .map(AnalysisConstraint::Fluid)
        .collect();
    workload.model.boundaries = fixture.operators.clone();
    workload.model.signals = fixture.signals.clone();
    workload.model.excitations = fixture.actions.clone();
    workload.model.excitation_cases = vec![ExcitationCase {
        case_id: ExcitationCaseId("mass-histories".into()),
        action_ids: workload
            .model
            .excitations
            .iter()
            .map(|a| a.excitation().excitation_id.clone())
            .collect(),
        activation: ExcitationCaseActivation::Always,
    }];
    let initial = workload.initial_checkpoint(&fixture.mesh).unwrap();
    let (full, _) =
        super::fluid_transient::run_block(&workload, &fixture.mesh, initial.clone(), 3, 8192, 0)
            .await;
    let (first, _) =
        super::fluid_transient::run_block(&workload, &fixture.mesh, initial, 1, 1024, 1).await;
    let (split, _) =
        super::fluid_transient::run_block(&workload, &fixture.mesh, first, 3, 8192, 2).await;
    assert_eq!(split.next_interval(), 3);
    assert_eq!(split.next_output_ordinal(), 3);
    assert_eq!(
        split.intervals().collect::<Vec<_>>(),
        full.intervals().collect::<Vec<_>>()
    );
    assert_eq!(
        split.retained_results().last(),
        full.retained_results().last()
    );
    for (ordinal, result) in split.retained_results() {
        let ordinal = *ordinal;
        let time = 0.3 * ((ordinal + 1) as f64 / 3.0);
        let speed = 0.001 + 0.01 * time;
        for u in result.velocity_m_per_s.canonical_values() {
            assert!((u[0] - speed).abs() < 1e-8 && u[1].abs() < 1e-8 && u[2].abs() < 1e-8);
        }
        assert!(result
            .pressure_reference_pa
            .canonical_values()
            .iter()
            .all(|p| *p == 1e12));
        for (cell, fluctuation) in fixture
            .topology
            .cells
            .iter()
            .zip(result.pressure_fluctuation_pa.canonical_values())
        {
            assert!((fluctuation + 0.01 * (cell.centroid_m[0] - 1.0)).abs() < 2e-7);
        }
        for id in ["mass-inlet", "mass-outlet"] {
            let report = result
                .residual
                .boundary_balances
                .iter()
                .find(|b| b.authority_id == id)
                .unwrap();
            let outward = speed * if id == "mass-inlet" { -1.0 } else { 1.0 };
            assert!(
                (report.mass_flow_rate_kg_per_s - runmat_types::physical::Quantity::new(outward))
                    .abs()
                    < runmat_types::physical::Quantity::new(1e-14)
            );
        }
    }
    eprintln!("scheduled mass-flow histories PASS: three terminated producer sessions, cold continuation, exact split history and independent affine fluctuation pressure");
}
