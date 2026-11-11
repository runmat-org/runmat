//! The conduction multigrid path against the proving factorization on one thermal problem.
use std::sync::Arc;

use runmat_analysis_core::{HeatTransferCoefficient, ThermalInterfaceTransfer};
use runmat_simulation_usage::{OperationCount, UsageContext, UsageLedger};

use super::interface_tests::{mesh_with, workload, COLD, HOT};
use crate::{
    contracts::{FeaRunError, SolverPreconditioner},
    execution::ThermalWorkload,
    planning::resource::{
        ThermalHistoryResourceFacts, ThermalOperatorResourceFacts, ThermalPreassemblyResourceFacts,
        ThermalSolvePath,
    },
};

fn joint() -> ThermalInterfaceTransfer {
    ThermalInterfaceTransfer::Conductance(HeatTransferCoefficient::new(500.0))
}

/// The preassembly facts the run derives for `input` on `mesh` at its own budget.
fn preassembly(
    input: &ThermalWorkload,
    mesh: &runmat_meshing_core::SolverMeshArtifact,
) -> ThermalPreassemblyResourceFacts {
    let identity = crate::admission::admit_nonstructural(
        runmat_analysis_core::PhysicsFamily::Thermal,
        input.procedure.procedure(),
        Some(mesh),
        runmat_analysis_core::SourceImplementation::CanonicalVolumeProjection,
        runmat_analysis_core::BoundaryTransferImplementation::CanonicalBoundaryProjection,
    )
    .unwrap();
    ThermalPreassemblyResourceFacts::derive(
        &[],
        &input.model,
        &input.model.sequence.steps[0],
        mesh,
        &input.solve_schedule().unwrap(),
        &identity,
    )
    .unwrap()
}

/// The whole preassembly reservations of `input` on `mesh` on the proving factor and on the
/// conduction multigrid path. Both paths share every component besides the numerical reservation.
fn reservations(
    input: &ThermalWorkload,
    mesh: &runmat_meshing_core::SolverMeshArtifact,
) -> (u64, u64) {
    let schedule = input.solve_schedule().unwrap();
    let history =
        ThermalHistoryResourceFacts::derive(mesh, &schedule, &input.model.interfaces).unwrap();
    let facts = ThermalOperatorResourceFacts::derive(mesh, &input.model.interfaces).unwrap();
    let factor_numerical = facts.numerical_reservation_bytes(&history).unwrap().0;
    let multigrid_numerical = facts
        .on_path(ThermalSolvePath::ConductionMultigrid)
        .numerical_reservation_bytes(&history)
        .unwrap()
        .0;
    let preassembly = preassembly(input, mesh);
    assert_eq!(
        preassembly.operators.path(),
        ThermalSolvePath::ProvingFactorization
    );
    let factor = preassembly.reservation_bytes().unwrap().0;
    (factor, factor - factor_numerical + multigrid_numerical)
}

/// The same workload with a resident budget just below the proving factor's preassembly
/// reservation, which the conduction multigrid's reservation fits.
fn below_factor(mesh: &runmat_meshing_core::SolverMeshArtifact) -> ThermalWorkload {
    below_factor_of(workload(mesh, joint()), mesh)
}

/// `input` with a resident budget just below its proving factor's preassembly reservation.
fn below_factor_of(
    mut input: ThermalWorkload,
    mesh: &runmat_meshing_core::SolverMeshArtifact,
) -> ThermalWorkload {
    let (factor, multigrid) = reservations(&input, mesh);
    assert!(
        multigrid < factor,
        "multigrid {multigrid} vs factor {factor}"
    );
    input.model.sequence.steps[0]
        .accuracy
        .resources
        .set_resident_byte_budget(factor - 1);
    input
}

// witness: FEA-THERMAL-SOLVE-001/conduction_multigrid @ thermal_steady/tet4/numerical_kernel;
//   coverage: partial;
//   boundary: On a two-layer joint box with a contact conductance, a budget below the proving
//   factor selects the conduction multigrid (AMG-PCG, no proving factor), and its
//   temperature matches the factor solve and the series-resistance solution at every node.
#[test]
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
    // The multigrid setup is charged as preconditioner factorization work; PCG iterations as
    // solve work.
    let usage = ledger.snapshot().unwrap().completed;
    assert!(usage.solve > OperationCount::ZERO);
    let direct_t = direct.temperature_snapshots[0].as_host_f64().unwrap();
    let iterative_t = iterative.temperature_snapshots[0].as_host_f64().unwrap();
    let largest = direct_t
        .iter()
        .zip(iterative_t)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    assert!(
        largest < 1e-6 * (HOT - COLD),
        "largest difference {largest} K"
    );
    for balance in &iterative.energy_balances {
        assert!(balance.relative_residual <= input.residual_target);
    }
}

// witness: FEA-THERMAL-SOLVE-001/conduction_multigrid_usage @ thermal_steady/tet4/numerical_kernel;
//   coverage: partial;
//   boundary: A usage allowance that runs out inside the conduction multigrid solve refuses the
//   run with the typed usage error.
#[test]
fn conduction_multigrid_accounting_refusal_is_typed_usage() {
    let mesh = mesh_with([5, 14, 5]);
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
    let total = [
        usage.meshing,
        usage.assembly,
        usage.factorization,
        usage.solve,
        usage.other,
    ]
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

// witness: FEA-THERMAL-SOLVE-001/conduction_multigrid_anchor @ thermal_steady/tet4/numerical_kernel;
//   coverage: partial;
//   boundary: A body with no prescribed temperature and no convection is refused on the
//   conduction multigrid path with an error naming the undetermined temperature.
#[test]
fn conduction_multigrid_refuses_an_unanchored_body() {
    let mesh = mesh_with([5, 14, 5]);
    let mut input = workload(&mesh, joint());
    // Replacing every prescribed temperature by a heat flux leaves the joined bodies with no
    // anchor: the fixture has no convection, so the steady temperature is determined only up to
    // a constant.
    input.model.constraints.retain(|constraint| {
        !matches!(
            constraint,
            runmat_analysis_core::AnalysisConstraint::Thermal(_)
        )
    });
    input
        .model
        .boundaries
        .push(runmat_analysis_core::AnalysisBoundary {
            boundary_id: "heated_floor".into(),
            target: super::interface_tests::boundary(&mesh, "lower_y_min"),
            operator: runmat_analysis_core::BoundaryOperator::ThermalHeatFlux {
                value: runmat_analysis_core::PrescribedValue::Constant(
                    runmat_analysis_core::HeatFlux::new(1.0),
                ),
            },
        });
    let input = below_factor_of(input, &mesh);
    let error = input.execute(&mesh).unwrap_err();
    assert!(error.to_string().contains("undetermined"), "{error}");
}

// witness: FEA-THERMAL-SOLVE-001/support_derivation_memory_interface @ thermal_steady/tet4/numerical_kernel;
//   coverage: partial;
//   boundary: With a finite contact conductance the split projection and the interface cliques
//   stay within the planning law.
#[test]
fn split_interface_support_derivation_fits_the_planning_law() {
    let mesh = mesh_with([5, 14, 5]);
    let input = workload(&mesh, joint());
    crate::planning::resource::thermal_operator_tests::derivation_peak_fits_the_planning_law(
        &mesh,
        &input.model.interfaces,
    );
}

/// Requested-heap peak of a whole run of `input`, on a one-thread pool so every worker
/// allocation lands on the measured thread.
fn run_peak(input: &ThermalWorkload, mesh: &runmat_meshing_core::SolverMeshArtifact) -> usize {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    pool.install(|| {
        // The whole run: preassembly, the solve, then the publication phase (field and
        // partition identities, the manifest and its canonical header encoding).
        let (published, _, peak) = crate::topology::measure_test_allocations(|| {
            let result = input.execute(mesh).unwrap();
            let manifest =
                crate::execution::ThermalSolverResultManifest::from_result(&result).unwrap();
            let header = manifest.canonical_bytes().unwrap();
            (result, manifest, header)
        });
        drop(published);
        peak
    })
}

// witness: FEA-THERMAL-SOLVE-001/steady_reservation @ thermal_steady/tet4/numerical_kernel;
//   coverage: partial;
//   boundary: The measured requested-heap peak of a whole steady linear run (preassembly, solve,
//   then manifest construction and header encoding), on both solve paths, fits the preassembly
//   reservation the run admits.
#[test]
fn steady_run_peak_fits_the_preassembly_reservation() {
    for cells in [[5, 14, 5], [8, 22, 8]] {
        let mesh = mesh_with(cells);
        for input in [workload(&mesh, joint()), below_factor(&mesh)] {
            let facts = preassembly(&input, &mesh);
            let total = facts.reservation_bytes().unwrap().0;
            let operators = &facts.operators;
            let history = &facts.history;
            let workspace = operators.workspace().reservation_bytes().unwrap().0;
            let projection = operators.workspace().projection_estimated_bytes.0;
            let csr = operators.csr_payload_bytes_upper_bound().0;
            let solver = operators.solver_resident_bytes().unwrap().0;
            let planning = operators.planning_resident_bytes().0;
            let numerical = operators.numerical_reservation_bytes(history).unwrap().0;
            let rows = operators.support().rows() as u64;
            let peak = run_peak(&input, &mesh) as u64;
            println!(
                "steady run cells={cells:?} path={:?} rows={rows} peak={peak} ({} B/row) total={total} ({} B/row) numerical={numerical} csr={csr} ({} B/row) solver={solver} workspace={workspace} ({} B/row) projection_term={projection} ({} B/row) planning={planning} ({} B/row)",
                operators.path(),
                peak / rows,
                total / rows,
                csr / rows,
                workspace / rows,
                projection / rows,
                planning / rows,
            );
            println!(
                "  per row: history {} sources {} observations {} realization {} metadata {} model {} properties {} field_construction {} catalog {} manifest_metadata {} retained {}",
                history.reservation_bytes().unwrap().0 / rows,
                facts.sources.reservation_bytes().unwrap().0 / rows,
                facts.observations.reservation_bytes().unwrap().0 / rows,
                facts.realization.reservation_bytes().unwrap().0 / rows,
                facts.metadata.reservation_bytes().unwrap().0 / rows,
                facts.model_materialization_bytes.0 / rows,
                facts.property_compilation_bytes.0 / rows,
                facts.field_construction.reservation_bytes().0 / rows,
                facts.manifest_catalog.reservation_bytes().unwrap().0 / rows,
                facts.manifest_metadata.reservation_bytes().0 / rows,
                facts.manifest_retained.reservation_bytes().unwrap().0 / rows,
            );
            let (prepared, _, prepare_peak) = crate::topology::measure_test_allocations(|| {
                crate::properties::PreparedProperties::prepare(&input.model, Some(&mesh), u64::MAX)
                    .unwrap()
            });
            let estimate = prepared.estimated_resident_bytes;
            let selected_metadata = prepared.selected_metadata_bytes;
            let (properties, retained, materialize_peak) =
                crate::topology::measure_test_allocations(|| prepared.materialize().unwrap());
            drop(properties);
            println!(
                "  properties per row: estimate {} selected_metadata {} prepare_peak {} materialize_peak {} retained {}",
                estimate / rows,
                selected_metadata / rows,
                prepare_peak as u64 / rows,
                materialize_peak as u64 / rows,
                retained as u64 / rows,
            );
            assert!(peak <= total, "peak {peak} exceeds the reservation {total}");
        }
    }
}
