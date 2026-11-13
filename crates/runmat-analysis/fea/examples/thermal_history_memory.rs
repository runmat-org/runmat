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
    planning::resource::{ThermalOperatorResourceFacts, ThermalPreassemblyResourceFacts},
    ComputeBackend, ThermalProcedureControl,
};
use runmat_meshing_core::fixtures::{
    connected_quadratic_tetrahedron_solver_mesh, connected_tetrahedron_solver_mesh,
};
use runmat_meshing_core::CanonicalMeshingContract;

fn main() -> Result<(), Box<dyn Error>> {
    let args = env::args().skip(1).collect::<Vec<_>>();
    if !(3..=4).contains(&args.len()) || !matches!(args[2].as_str(), "history" | "fields") {
        return Err(
            "usage: thermal_history_memory <tet4|tet10> <2..8193 samples> <fields|history> [0..128 heat sources]".into(),
        );
    }
    let mesh = match args[0].as_str() {
        "tet4" => connected_tetrahedron_solver_mesh(),
        "tet10" => connected_quadratic_tetrahedron_solver_mesh(),
        _ => return Err("element order must be tet4 or tet10".into()),
    };
    // Independently measure the shared validator, without including fixture construction or
    // its resource-profile walk in the peak. This remains Rust-heap-only evidence.
    let validation_reservation = mesh.validation_resident_bytes_estimate()?;
    let baseline = peak_allocator::begin_measurement();
    let validation = mesh.validate_canonical();
    let validation_growth = peak_allocator::end_measurement().saturating_sub(baseline);
    validation?;
    if validation_growth > validation_reservation {
        return Err("mesh validation Rust heap exceeds its estimated reservation".into());
    }
    println!("mesh_validation_reservation_bytes={validation_reservation} mesh_validation_rust_heap_growth_bytes={validation_growth}");
    let planning_reservation =
        ThermalOperatorResourceFacts::derivation_resident_bytes_estimate(&mesh)?.0;
    let baseline = peak_allocator::begin_measurement();
    let planning = ThermalOperatorResourceFacts::derive(&mesh, &[]);
    let planning_growth = peak_allocator::end_measurement().saturating_sub(baseline);
    drop(planning?);
    if planning_growth > planning_reservation {
        return Err("thermal support construction Rust heap exceeds estimate".into());
    }
    println!("operator_planning_reservation_bytes={planning_reservation} operator_planning_rust_heap_growth_bytes={planning_growth}");
    let samples: usize = args[1].parse()?;
    if !(2..=8193).contains(&samples) {
        return Err("samples must be in 2..8193".into());
    }
    let source_count: usize = args
        .get(3)
        .map(|value| value.parse())
        .transpose()?
        .unwrap_or(0);
    if source_count > 128 {
        return Err("heat source count must be in 0..128".into());
    }
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    model.thermal = Some(ThermalDomain {
        reference_temperature: Temperature::new(293.15),
    });
    let material = model.materials.first().ok_or("missing fixture material")?;
    let volumetric_capacity = material
        .mass_density_kg_per_m3
        .map(|quantity| quantity.canonical_value())
        .ok_or("missing density")?
        * material
            .thermal
            .as_ref()
            .ok_or("missing thermal material")?
            .specific_heat_j_per_kgk
            .canonical_value();
    // Keep the total heating fixed at 1 K/s, split equally among genuine typed sources.
    // The zero-source case remains the original constant-temperature reference.
    let total_heat_generation = if source_count == 0 {
        0.0
    } else {
        volumetric_capacity
    };
    add_heat_sources(&mut model, source_count, total_heat_generation);
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
        initial_fields: Vec::new(),
        step_id: ProcedureStepId("insulated".into()),
        model,
        backend: ComputeBackend::Cpu,
        procedure: ThermalProcedureControl::BackwardEuler {
            snapshot_count: samples,
        },
        residual_target: 1e-4,
    };
    let schedule = workload.solve_schedule()?;
    let facts = ThermalPreassemblyResourceFacts::derive(
        &[],
        &workload.model,
        &workload.model.sequence.steps[0],
        &mesh,
        &schedule,
        &runmat_analysis_fea::admission::admit_nonstructural(
            runmat_analysis_core::PhysicsFamily::Thermal,
            workload.procedure.procedure(),
            Some(&mesh),
            runmat_analysis_core::SourceImplementation::CanonicalVolumeProjection,
            runmat_analysis_core::BoundaryTransferImplementation::CanonicalBoundaryProjection,
        )?,
    )?;
    let expected_evidence = samples
        .checked_mul(source_count)
        .ok_or("evidence count overflow")?;
    if facts.sources.retained_records.0 != u64::try_from(expected_evidence)? {
        return Err("planned source evidence cardinality failed".into());
    }
    let reservation = facts.reservation_bytes()?.0;
    // Do not retain the diagnostic planner's support during measured execution.
    drop(facts);
    drop(schedule);
    let baseline = peak_allocator::begin_measurement();
    let start = Instant::now();
    let result = workload.execute(&mesh);
    let elapsed = start.elapsed();
    let growth = peak_allocator::end_measurement().saturating_sub(baseline);
    let result = result?;
    result.validate_complete()?;
    // Separately measure identity construction above the already retained result baseline.
    // This is not a bound on total producer residency or canonical result serialization.
    let identity_baseline = peak_allocator::begin_measurement();
    let identity = result.source_result_identity();
    let identity_growth = peak_allocator::end_measurement().saturating_sub(identity_baseline);
    identity?;
    println!("source_identity_rust_heap_growth_bytes={identity_growth}");
    if result.run.spatial_projection_evidence.len() != expected_evidence
        || result
            .run
            .spatial_projection_evidence
            .iter()
            .any(|evidence| !evidence.accepted)
    {
        return Err("actual source evidence cardinality or conservation failed".into());
    }
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
        .zip(&result.time_points_s)
        .flat_map(|(field, time)| {
            let expected = 293.15 + total_heat_generation * time / volumetric_capacity;
            field
                .as_host_f64()
                .expect("CPU host field")
                .iter()
                .map(move |temperature| (temperature - expected).abs())
        })
        .fold(0.0_f64, f64::max);
    // Nonzero heating performs iterative solves: relative equation convergence is not an
    // absolute temperature guarantee. Require a millikelvin reference (0.1% of the 1 K rise),
    // retaining the original tighter constant-temperature check when no sources are present.
    let reference_tolerance_k = if source_count == 0 { 1e-8 } else { 1e-3 };
    println!("order={} nodes={} elements={} samples={} mode={} sources={source_count} total_heat_generation_w_per_m3={total_heat_generation} source_evidence_records={expected_evidence} reservation_bytes={} rust_heap_growth_bytes={} process_peak_rss_bytes={:?} max_temperature_error_k={max_error:e} reference_tolerance_k={reference_tolerance_k:e} elapsed_s={:.6}",
        args[0], mesh.topology.nodes.len(), mesh.topology.volume_elements.len(), samples, args[2],
        reservation, growth, peak_allocator::process_peak_rss_bytes(), elapsed.as_secs_f64());
    if max_error > reference_tolerance_k || result.temperature_snapshots.len() != samples {
        return Err("insulated reference or snapshot cardinality failed".into());
    }
    if growth > reservation {
        return Err("measured Rust heap growth exceeds partial thermal reservation".into());
    }
    Ok(())
}

fn add_heat_sources(model: &mut AnalysisModel, count: usize, total_w_per_m3: f64) {
    model.signals.clear();
    model.excitations.clear();
    model.excitation_cases.clear();
    for index in 0..count {
        let signal_id = SignalId(format!("heat_{index}.signal"));
        let excitation_id = ExcitationId(format!("heat_{index}"));
        model
            .signals
            .push(PhysicalSignal::HeatGenerationDensity(Signal {
                signal_id: signal_id.clone(),
                shape: SignalShape::Scalar,
                frame: None,
                representation: SignalRepresentation::Constant(SignalValue::Scalar(
                    HeatGenerationDensity::new(total_w_per_m3 / count as f64),
                )),
                precision: SignalPrecision::Real64,
                evidence: SignalApproximationEvidence {
                    max_absolute_error: Some(0.0),
                    max_relative_error: Some(0.0),
                    derivative_max_absolute_error: None,
                    sample_count: 1,
                },
                provenance: SignalProvenance::Authored,
            }));
        model
            .excitations
            .push(ExcitationAction::HeatGeneration(AnalysisExcitation {
                excitation_id: excitation_id.clone(),
                target: ExcitationTarget::WholeModel,
                frame: CoordinateFrame::Global,
                direction: None,
                distribution: ExcitationDistribution::IntensityPerVolume,
                kinematics: ExcitationKinematics::FixedFrame,
                signal_id,
                active_step_ids: vec!["insulated".into()],
                combination_group: None,
                projection: ProjectionRequirement {
                    preserve_resultant: true,
                    preserve_moment: false,
                    relative_tolerance: 1e-12,
                },
            }));
        model.excitation_cases.push(ExcitationCase {
            case_id: ExcitationCaseId(format!("heat_{index}.case")),
            action_ids: vec![excitation_id],
            activation: ExcitationCaseActivation::Always,
        });
    }
}
