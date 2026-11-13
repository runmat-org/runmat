use super::*;

pub(super) fn mesh(quadratic: bool, cells: usize) -> SolverMeshArtifact {
    if quadratic {
        crate::topology::test_meshes::quadratic_slab_mesh(cells)
    } else {
        crate::topology::test_meshes::slab_mesh(cells)
    }
}

pub(super) fn base() -> ThermalWorkload {
    let mut input = workload();
    input.residual_target = RESIDUAL;
    input.model.constraints.clear();
    input.model.boundaries.clear();
    input.model.signals.clear();
    input.model.excitations.clear();
    input.model.excitation_cases.clear();
    input.model.initial_conditions.clear();
    input.model.material_assignments[0].target =
        PropertyRegionTarget::solid(runmat_meshing_core::fixtures::fixture_solid("slab"));
    input.model.materials[0].mass_density_kg_per_m3 =
        (Some(1.0)).map(runmat_types::physical::MassDensity::new);
    let thermal = input.model.materials[0].thermal.as_mut().unwrap();
    thermal.conductivity_w_per_mk = runmat_types::physical::ThermalConductivity::new(1.0);
    thermal.specific_heat_j_per_kgk = runmat_types::physical::SpecificHeat::new(1.0);
    input.model.thermal.as_mut().unwrap().reference_temperature = Temperature::new(REFERENCE);
    input
}

pub(super) fn ends(input: &mut ThermalWorkload) {
    for face in ["slab_left", "slab_right"] {
        input
            .model
            .constraints
            .push(AnalysisConstraint::Thermal(ThermalConstraint {
                constraint_id: face.into(),
                kind: ThermalConstraintKind::Prescribed {
                    target: RegionTarget::boundary(
                        runmat_meshing_core::fixtures::fixture_boundary_selection(&(face)),
                    ),
                    value: PrescribedValue::Constant(Temperature::new(REFERENCE)),
                },
            }));
    }
}

pub(super) fn insulate_all(input: &mut ThermalWorkload) {
    // Natural insulation is still an explicit authored model condition. These disjoint
    // face groups cover every exterior triangle of the canonical cube on both orders.
    for face in ["slab_left", "slab_right", "slab_insulated"] {
        input.model.boundaries.push(AnalysisBoundary {
            boundary_id: format!("zero-flux-{face}"),
            target: RegionTarget::boundary(
                runmat_meshing_core::fixtures::fixture_boundary_selection(&(face)),
            ),
            operator: BoundaryOperator::ThermalHeatFlux {
                value: PrescribedValue::Constant(HeatFlux::new(0.0)),
            },
        });
    }
}

pub(super) fn transient(input: &mut ThermalWorkload, intervals: usize) {
    let mut step = ProcedureStep::thermal_transient("heat_a");
    step.independent_domain = IndependentDomain::TimeSeconds {
        origin: 0.0,
        start: 0.0,
        end: FINAL_TIME,
        interval_count: intervals,
    };
    input.model.sequence = AnalysisSequence::from_steps("cooling", vec![step]);
    input.procedure = ThermalProcedureControl::BackwardEuler {
        snapshot_count: intervals + 1,
    };
}

pub(super) fn generation(
    input: &mut ThermalWorkload,
    representation: SignalRepresentation<HeatGenerationDensity>,
) {
    let sample_count = match &representation {
        SignalRepresentation::Constant(_) => 1,
        SignalRepresentation::Tabulated(table) => table.knots.len(),
        _ => panic!("continuum fixture admits only constant or explicit ramp inputs"),
    };
    let id = SignalId("generation".into());
    input.model.signals = vec![PhysicalSignal::HeatGenerationDensity(Signal {
        signal_id: id.clone(),
        shape: SignalShape::Scalar,
        frame: None,
        representation,
        precision: SignalPrecision::Real64,
        evidence: SignalApproximationEvidence {
            max_absolute_error: Some(0.0),
            max_relative_error: Some(0.0),
            derivative_max_absolute_error: None,
            sample_count,
        },
        provenance: SignalProvenance::Authored,
    })];
    input.model.excitations = vec![ExcitationAction::HeatGeneration(AnalysisExcitation {
        excitation_id: ExcitationId("generation".into()),
        target: ExcitationTarget::WholeModel,
        frame: CoordinateFrame::Global,
        direction: None,
        distribution: ExcitationDistribution::IntensityPerVolume,
        kinematics: ExcitationKinematics::FixedFrame,
        signal_id: id,
        active_step_ids: vec![input.step_id.0.clone()],
        combination_group: None,
        projection: ProjectionRequirement {
            preserve_resultant: true,
            preserve_moment: false,
            relative_tolerance: 1e-12,
        },
    })];
    input.model.excitation_cases = vec![ExcitationCase {
        case_id: ExcitationCaseId("generation".into()),
        action_ids: vec![ExcitationId("generation".into())],
        activation: ExcitationCaseActivation::Always,
    }];
}
