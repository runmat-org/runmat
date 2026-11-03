use super::*;
use crate::fixtures::{fixture_model, FixtureId};
use crate::topology::test_meshes::entity;
use runmat_analysis_core::*;
use runmat_meshing_core::{MeshRegion, PersistentEntityKind};

const TOTAL_W: f64 = 2.0;

/// Unit slab split at x = 0.25 into a "die" (0.25 m³) and a "board"; both ends held at the
/// reference temperature, `TOTAL_W` generated in the die as a total heat rate.
fn heated_die() -> (AnalysisModel, SolverMeshArtifact) {
    let mut mesh = runmat_meshing_core::fixtures::slab_mesh(4);
    let regions = [
        entity(PersistentEntityKind::Region, "die"),
        entity(PersistentEntityKind::Region, "board"),
    ];
    let mut elements = [Vec::new(), Vec::new()];
    let topology = &mut *mesh.topology;
    for element in &mut topology.volume_elements {
        let x = element.node_ids[..4]
            .iter()
            .map(|id| {
                topology.nodes[*id as usize - 1]
                    .coordinates_m
                    .canonical_value()[0]
            })
            .sum::<f64>()
            / 4.0;
        let side = usize::from(x > 0.25);
        element.region_id = regions[side].clone();
        elements[side].push(element.element_id);
    }
    mesh.topology.regions = regions
        .into_iter()
        .zip(elements)
        .map(|(region_id, element_ids)| MeshRegion {
            region_id,
            element_ids,
        })
        .collect();
    mesh.seal_canonical_digest().unwrap();

    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    model.thermal = Some(ThermalDomain {
        reference_temperature: Temperature::new(293.15),
    });
    model.sequence = AnalysisSequence::from_steps(
        "thermal_sequence",
        vec![ProcedureStep::thermal_steady("heat")],
    );
    let mut die = model.material_assignments[0].clone();
    die.target = PropertyRegionTarget::solid(runmat_meshing_core::fixtures::fixture_region("die"));
    let mut board = model.material_assignments[0].clone();
    board.target =
        PropertyRegionTarget::solid(runmat_meshing_core::fixtures::fixture_region("board"));
    model.material_assignments = vec![die, board];
    let mut action = model.excitations[0].excitation().clone();
    action.target = ExcitationTarget::Region(RegionTarget::volume(
        runmat_meshing_core::fixtures::fixture_region("die"),
    ));
    action.direction = None;
    action.distribution = ExcitationDistribution::PreserveResultant;
    action.active_step_ids = Vec::new();
    action.projection.preserve_moment = false;
    model.signals = vec![PhysicalSignal::HeatRate(Signal {
        signal_id: action.signal_id.clone(),
        shape: SignalShape::Scalar,
        frame: None,
        representation: SignalRepresentation::Constant(SignalValue::Scalar(Power::new(TOTAL_W))),
        precision: SignalPrecision::Real64,
        evidence: SignalApproximationEvidence {
            max_absolute_error: Some(0.0),
            max_relative_error: Some(0.0),
            derivative_max_absolute_error: None,
            sample_count: 1,
        },
        provenance: SignalProvenance::Authored,
    })];
    model.excitation_cases = vec![ExcitationCase {
        case_id: ExcitationCaseId("heat-case".into()),
        action_ids: vec![action.excitation_id.clone()],
        activation: ExcitationCaseActivation::Always,
    }];
    model.excitations = vec![ExcitationAction::HeatGeneration(action)];
    model.boundaries.clear();
    model.constraints.clear();
    for (id, face) in [("left", "slab_left"), ("right", "slab_right")] {
        model
            .constraints
            .push(AnalysisConstraint::Thermal(ThermalConstraint {
                constraint_id: id.into(),
                kind: ThermalConstraintKind::Prescribed {
                    target: RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face(
                        face,
                    )),
                    value: PrescribedValue::Constant(Temperature::new(293.15)),
                },
            }));
    }
    (model, mesh)
}

// witness: FEA-THERMAL-SOURCES-001/heat_generation @ thermal_steady/tet4/numerical_kernel;
//   coverage: partial; boundary: a total heat rate on a solid region compiles to P / V over the
//   region's meshed volume, and the steady energy balance generates and removes exactly P.
#[test]
fn total_heat_rate_is_p_over_the_meshed_region_volume_and_balances() {
    let (mut model, mesh) = heated_die();
    distribute_heat_rate_sources(&mut model, &mesh).unwrap();
    let PhysicalSignal::HeatGenerationDensity(signal) = &model.signals[0] else {
        panic!("compiled to a density: {:?}", model.signals[0]);
    };
    let SignalRepresentation::Constant(SignalValue::Scalar(density)) = &signal.representation
    else {
        panic!("constant density");
    };
    // The die is a quarter of the unit slab.
    assert!((density.canonical_value() - TOTAL_W / 0.25).abs() < 1e-9);
    assert_eq!(
        model.excitations[0].excitation().distribution,
        ExcitationDistribution::IntensityPerVolume
    );
    runmat_analysis_core::validate_model(&model).unwrap();
    let result = crate::run_thermal_with_options(
        &model,
        crate::ComputeBackend::Cpu,
        crate::ThermalSolveOptions {
            procedure: crate::ThermalProcedureControl::Steady,
            solver_mesh: Some(mesh),
            ..crate::ThermalSolveOptions::default()
        },
    )
    .unwrap();
    let balance = &result.energy_balances[0];
    balance.validate().unwrap();
    eprintln!(
        "heat rate: applied {} W, through held faces {} W",
        balance.applied_power_w.canonical_value(),
        balance.prescribed_boundary_power_w.canonical_value()
    );
    assert!((balance.applied_power_w.canonical_value() - TOTAL_W).abs() < 1e-9 * TOTAL_W);
    assert!(
        (balance.prescribed_boundary_power_w.canonical_value().abs() - TOTAL_W).abs()
            < 1e-6 * TOTAL_W
    );
}

#[test]
fn a_total_heat_rate_needs_a_scalar_constant_or_tabulated_signal_on_a_meshed_region() {
    let (mut model, mesh) = heated_die();
    if let ExcitationAction::HeatGeneration(action) = &mut model.excitations[0] {
        action.target = ExcitationTarget::Region(RegionTarget::volume(
            runmat_meshing_core::fixtures::fixture_region("absent"),
        ));
    }
    assert!(distribute_heat_rate_sources(&mut model, &mesh).is_err());
}
