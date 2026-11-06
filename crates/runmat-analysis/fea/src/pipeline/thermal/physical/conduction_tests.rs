//! The conduction multigrid path against the proving factorization on one thermal problem.
use std::sync::Arc;

use runmat_analysis_core::{HeatTransferCoefficient, ThermalInterfaceTransfer};
use runmat_simulation_usage::{OperationCount, UsageContext, UsageLedger};

use super::interface_tests::{mesh_with, workload, COLD, HOT};
use crate::{
    contracts::{FeaRunError, SolverPreconditioner},
    execution::ThermalWorkload,
    planning::resource::{
        ThermalHistoryResourceFacts, ThermalOperatorResourceFacts, ThermalSolvePath,
    },
};

fn joint() -> ThermalInterfaceTransfer {
    ThermalInterfaceTransfer::Conductance(HeatTransferCoefficient::new(500.0))
}

/// The factor and conduction multigrid numerical reservations of `input` on `mesh`.
fn reservations(input: &ThermalWorkload, mesh: &runmat_meshing_core::SolverMeshArtifact) -> (u64, u64) {
    let step = &input.model.sequence.steps[0];
    let schedule =
        crate::execution::ThermalSolveSchedule::derive(&input.model, step, input.procedure)
            .unwrap();
    let history =
        ThermalHistoryResourceFacts::derive(mesh, &schedule, &input.model.interfaces).unwrap();
    let facts = ThermalOperatorResourceFacts::derive(mesh, &input.model.interfaces).unwrap();
    let factor = facts.numerical_reservation_bytes(&history).unwrap().0;
    let multigrid = facts
        .on_path(ThermalSolvePath::ConductionMultigrid)
        .numerical_reservation_bytes(&history)
        .unwrap()
        .0;
    (factor, multigrid)
}

/// The same workload with a resident budget just below the proving factor's reservation.
fn below_factor(mesh: &runmat_meshing_core::SolverMeshArtifact) -> ThermalWorkload {
    let mut input = workload(mesh, joint());
    let (factor, multigrid) = reservations(&input, mesh);
    assert!(multigrid < factor, "multigrid {multigrid} vs factor {factor}");
    input.model.sequence.steps[0]
        .accuracy
        .resources
        .set_resident_byte_budget(factor - 1);
    input
}

// witness: FEA-THERMAL-SOLVER-001/conduction_multigrid @ thermal_steady/tet4/numerical_kernel;
//   coverage: partial;
//   boundary: On a two-layer joint box with a contact conductance, a budget below the proving
//   factor selects the conduction multigrid (AMG-PCG, no factorization charged), and its
//   temperature matches the factor solve and the series-resistance solution at every node.
#[test]
#[ignore = "needs a mesh whose factor exceeds the planning-derivation estimate; see fea-work/q-thermal-scale/HANDBACK-1.md"]
fn conduction_multigrid_matches_the_proving_factorization() {
    let mesh = mesh_with([5, 14, 5]);
    let direct = workload(&mesh, joint()).execute(&mesh).unwrap();
    assert_ne!(direct.run.preconditioner, SolverPreconditioner::Amg);
    let input = below_factor(&mesh);
    let ledger = Arc::new(
        UsageLedger::new(
            OperationCount::from_operations(u64::MAX / 4),
            OperationCount::from_operations(u64::MAX / 4),
        )
        .unwrap(),
    );
    let iterative = UsageContext::new(Arc::clone(&ledger))
        .run_sync(|| input.execute(&mesh))
        .unwrap();
    assert_eq!(iterative.run.preconditioner, SolverPreconditioner::Amg);
    let usage = ledger.snapshot().unwrap().completed;
    assert_eq!(usage.factorization, OperationCount::ZERO);
    assert!(usage.solve > OperationCount::ZERO);
    let direct_t = direct.temperature_snapshots[0].as_host_f64().unwrap();
    let iterative_t = iterative.temperature_snapshots[0].as_host_f64().unwrap();
    let largest = direct_t
        .iter()
        .zip(iterative_t)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    assert!(largest < 1e-6 * (HOT - COLD), "largest difference {largest} K");
    for balance in &iterative.energy_balances {
        assert!(balance.relative_residual <= input.residual_target);
    }
}

// witness: FEA-THERMAL-SOLVER-001/conduction_multigrid_usage @ thermal_steady/tet4/numerical_kernel;
//   coverage: partial;
//   boundary: A usage allowance that runs out inside the conduction multigrid solve refuses the
//   run with the typed usage error.
#[test]
#[ignore = "needs a mesh whose factor exceeds the planning-derivation estimate; see fea-work/q-thermal-scale/HANDBACK-1.md"]
fn conduction_multigrid_accounting_refusal_is_typed_usage() {
    let mesh = mesh_with([4, 10, 4]);
    let input = below_factor(&mesh);
    let measured = Arc::new(
        UsageLedger::new(
            OperationCount::from_operations(u64::MAX / 4),
            OperationCount::from_operations(u64::MAX / 4),
        )
        .unwrap(),
    );
    UsageContext::new(Arc::clone(&measured))
        .run_sync(|| input.execute(&mesh))
        .unwrap();
    let usage = measured.snapshot().unwrap().completed;
    let total = [usage.meshing, usage.assembly, usage.factorization, usage.solve, usage.other]
        .iter()
        .map(|count| count.operations())
        .sum::<u64>();
    let allowance = OperationCount::from_operations(total - usage.solve.operations() / 2);
    let ledger = Arc::new(UsageLedger::new(allowance, allowance).unwrap());
    let error = UsageContext::new(ledger)
        .run_sync(|| input.execute(&mesh))
        .unwrap_err();
    assert!(matches!(error, FeaRunError::Usage(_)), "{error:?}");
}

// witness: FEA-THERMAL-SOLVER-001/conduction_multigrid_anchor @ thermal_steady/tet4/numerical_kernel;
//   coverage: partial;
//   boundary: A body with no prescribed temperature and no convection is refused on the
//   conduction multigrid path with an error naming the undetermined temperature.
#[test]
#[ignore = "needs a mesh whose factor exceeds the planning-derivation estimate; see fea-work/q-thermal-scale/HANDBACK-1.md"]
fn conduction_multigrid_refuses_an_unanchored_body() {
    let mesh = mesh_with([3, 8, 3]);
    let mut input = below_factor(&mesh);
    // A zero conductance leaves the upper body floating once its cold face is released.
    input.model.interfaces[0].kind =
        runmat_analysis_core::AnalysisInterfaceKind::ConjugateHeatTransfer(
            runmat_analysis_core::ConjugateHeatTransferInterfaceModel {
                transfer: ThermalInterfaceTransfer::Conductance(HeatTransferCoefficient::new(0.0)),
            },
        );
    input
        .model
        .constraints
        .retain(|constraint| match constraint {
            runmat_analysis_core::AnalysisConstraint::Thermal(thermal) => {
                thermal.constraint_id != "upper_y_max"
            }
            _ => true,
        });
    let error = input.execute(&mesh).unwrap_err();
    assert!(
        error.to_string().contains("undetermined"),
        "{error}"
    );
}
