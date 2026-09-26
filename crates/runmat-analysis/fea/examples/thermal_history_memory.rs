//! Measure connected insulated thermal history execution against its partial reservation.
//! Mesh/model construction and the reporting preflight are outside the measurement; execution's
//! own preflight, solve, recovery and observations are inside. Rust heap growth is not process RSS
//! or native allocation coverage. Run each case in a fresh process, e.g. `-- tet10 257 history`.

use std::{env, error::Error, time::Instant};

#[path = "../support/peak_allocator.rs"]
mod peak_allocator;

use runmat_analysis_core::*;
use runmat_analysis_fea::{
    execution::ThermalWorkload,
    fixtures::{fixture_model, FixtureId},
    planning::resource::{
        ThermalHistoryResourceFacts, ThermalObservationResourceFacts, ThermalOperatorResourceFacts,
    },
    ComputeBackend, ThermalProcedureControl,
};
use runmat_meshing_core::fixtures::{
    connected_quadratic_tetrahedron_solver_mesh, connected_tetrahedron_solver_mesh,
};

fn main() -> Result<(), Box<dyn Error>> {
    let args = env::args().skip(1).collect::<Vec<_>>();
    if args.len() != 3 || !matches!(args[2].as_str(), "history" | "fields") {
        return Err(
            "usage: thermal_history_memory <tet4|tet10> <2..8193 samples> <fields|history>".into(),
        );
    }
    let mesh = match args[0].as_str() {
        "tet4" => connected_tetrahedron_solver_mesh(),
        "tet10" => connected_quadratic_tetrahedron_solver_mesh(),
        _ => return Err("element order must be tet4 or tet10".into()),
    };
    let samples: usize = args[1].parse()?;
    if !(2..=8193).contains(&samples) {
        return Err("samples must be in 2..8193".into());
    }
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    model.thermal = Some(ThermalDomain {
        reference_temperature: Temperature::new(293.15),
    });
    let mut step = ProcedureStep::thermal_transient("insulated");
    step.accuracy.resources.maximum_samples = samples;
    if args[2] == "history" {
        step.observations
            .push(ObservationRequest::History(ObservationSpec {
                observation_id: ObservationId("temperature".into()),
                quantity: ObservationQuantity::Temperature,
                value_shape: ObservationValueShape::Scalar,
                target: ObservationTarget::WholeModel,
                source_location: FieldLocation::Node,
                output_location: FieldLocation::Node,
                recovery: RecoveryPolicy::Native,
                component: ObservationComponent::All,
                frame: CoordinateFrame::Global,
                sign_convention: SignConvention::Native,
                complex_representation: ComplexRepresentation::Real,
                sampling: SamplingPolicy::TimeSeconds(
                    (0..samples)
                        .map(|i| i as f64 / (samples - 1) as f64)
                        .collect(),
                ),
                reduction: None,
                storage: ObservationStoragePolicy {
                    precision: SignalPrecision::Real64,
                    compression: CompressionPolicy::Lossless,
                    retention: RetentionPolicy::Canonical,
                    partial_access: PartialAccessPolicy::ManifestAndChunks,
                },
                purpose: ObservationPurpose::Publication,
                cost: ObservationCostPolicy::Automatic,
            }));
    }
    model.sequence = AnalysisSequence::from_steps("thermal_memory", vec![step]);
    let workload = ThermalWorkload {
        step_id: ProcedureStepId("insulated".into()),
        model,
        backend: ComputeBackend::Cpu,
        formulation_requirement: FormulationRequirement::Experimental,
        procedure: ThermalProcedureControl::BackwardEuler {
            snapshot_count: samples,
        },
        residual_target: 1e-4,
    };
    let history = ThermalHistoryResourceFacts::derive(&mesh, &workload.solve_schedule()?)?;
    let operators = ThermalOperatorResourceFacts::derive(&mesh)?;
    let observations =
        ThermalObservationResourceFacts::derive(&mesh, &workload.model.sequence.steps[0])?;
    let reservation = operators
        .numerical_reservation_bytes(&history)?
        .0
        .checked_add(observations.reservation_bytes()?.0)
        .ok_or("reservation overflow")?;
    // Do not retain the diagnostic planner's support during measured execution.
    drop(operators);
    let baseline = peak_allocator::begin_measurement();
    let start = Instant::now();
    let result = workload.execute(&mesh);
    let elapsed = start.elapsed();
    let growth = peak_allocator::end_measurement().saturating_sub(baseline);
    let result = result?;
    result.validate_complete()?;
    let retained_samples: usize = result
        .run
        .observation_histories
        .iter()
        .map(|history| history.samples().len())
        .sum();
    if retained_samples != if args[2] == "history" { samples } else { 0 } {
        return Err("observation sample cardinality failed".into());
    }
    if result
        .temperature_snapshots
        .iter()
        .flat_map(|field| field.as_host_f64().expect("CPU host field"))
        .any(|temperature| !temperature.is_finite())
    {
        return Err("nonfinite insulated temperature".into());
    }
    let max_error = result
        .temperature_snapshots
        .iter()
        .flat_map(|field| field.as_host_f64().expect("CPU host field"))
        .map(|temperature| (temperature - 293.15).abs())
        .fold(0.0_f64, f64::max);
    println!("order={} nodes={} elements={} samples={} mode={} reservation_bytes={} rust_heap_growth_bytes={} process_peak_rss_bytes={:?} max_temperature_error_k={max_error:e} elapsed_s={:.6}",
        args[0], mesh.topology.nodes.len(), mesh.topology.volume_elements.len(), samples, args[2],
        reservation, growth, peak_allocator::process_peak_rss_bytes(), elapsed.as_secs_f64());
    if max_error > 1e-8 || result.temperature_snapshots.len() != samples {
        return Err("insulated reference or snapshot cardinality failed".into());
    }
    if growth > reservation {
        return Err("measured Rust heap growth exceeds partial thermal reservation".into());
    }
    Ok(())
}
