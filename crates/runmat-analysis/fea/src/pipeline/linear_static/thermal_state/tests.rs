use super::*;
use crate::{
    fixtures::{fixture_model, FixtureId},
    ThermalProcedureControl, ThermalSolveOptions,
};
use runmat_analysis_core::*;
use runmat_meshing_core::{fixtures::canonical_tetrahedron_solver_mesh, ElementOrder};

#[test]
fn retained_thermal_state_drives_restrained_expansion_and_source_linked_results() {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    model.excitations.clear();
    model.excitation_cases.clear();
    model.signals.clear();
    model.constraints.clear();
    model.thermal = Some(ThermalDomain {
        reference_temperature: Temperature::new(293.15),
    });
    model.thermo_mechanical = Some(ThermoMechanicalDomain {
        reference_temperature: Temperature::new(293.15),
    });
    model.materials[0]
        .thermal
        .as_mut()
        .unwrap()
        .expansion_coefficient_per_k = runmat_types::physical::ThermalExpansion::new(1.0e-5);
    let producer = ProcedureStep::thermal_steady("heat");
    let mut consumer = ProcedureStep::coupled(
        "expand",
        CoupledPhysicsFamily::Thermomechanical,
        AnalysisProcedure::Static,
    );
    consumer.state_inputs.push(StateInput {
        input_id: "temperature".into(),
        required_kind: StateKind::Thermal,
        source: StateSource::Upstream {
            producer_step: producer.step_id.clone(),
            state_id: producer.state_outputs[0].state_id.clone(),
        },
        use_as: StateUse::Prestress,
        conflict: InitialStateConflictPolicy::Reject,
    });
    model.sequence = AnalysisSequence::from_steps(
        "thermal_expansion",
        vec![producer.clone(), consumer.clone()],
    );
    model.sequence.dependencies[0].reason = SequenceDependencyReason::StateTransfer;
    model.sequence.transitions.push(StateTransition {
        transition_id: TransitionId("temperature_reuse".into()),
        producer_step: producer.step_id,
        consumer_step: consumer.step_id,
        state_id: producer.state_outputs[0].state_id.clone(),
        method: StateTransitionMethod::Reuse,
        topology: TopologyTransitionPolicy::RequireIdentical,
        reference: ReferenceStatePolicy::Preserve,
    });
    let mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    let projection = crate::topology::AnalysisProjection::derive(&mesh).unwrap();
    let face = projection
        .boundary_faces
        .iter()
        .find(|face| {
            face.node_indices
                .iter()
                .all(|&node| mesh.topology.nodes[node].coordinates_m.canonical_value()[0] == 0.0)
        })
        .unwrap();
    let target = RegionTarget::boundary(face.provenance[0].analysis_selection_id().unwrap());
    model
        .constraints
        .push(AnalysisConstraint::Thermal(ThermalConstraint {
            constraint_id: "heat_wall".into(),
            kind: ThermalConstraintKind::Prescribed {
                target: target.clone(),
                value: PrescribedValue::Constant(Temperature::new(303.15)),
            },
        }));
    model
        .constraints
        .push(AnalysisConstraint::Structural(StructuralConstraint {
            constraint_id: "fixed_wall".into(),
            frame: CoordinateFrame::Global,
            kind: StructuralConstraintKind::Fixed { target },
        }));
    let thermal = crate::run_thermal_with_options(
        &model,
        ComputeBackend::Cpu,
        ThermalSolveOptions {
            procedure: ThermalProcedureControl::Steady,
            solver_mesh: Some(mesh.clone()),
            ..ThermalSolveOptions::default()
        },
    )
    .unwrap();
    let source = CanonicalSemanticState::from_thermal_result(&model, &mesh, &thermal).unwrap();
    let retained = source.clone();
    let options = LinearStaticSolveOptions {
        solver_mesh: Some(mesh.clone()),
        ..LinearStaticSolveOptions::default()
    };
    let result =
        run_linear_static_from_thermal_state(&model, ComputeBackend::Cpu, options.clone(), &source)
            .unwrap();
    assert_eq!(source, retained);
    result.run.validate_solver_formulation().unwrap();
    assert_eq!(result.thermal_transition.source_state, source.identity);
    let field = |id: &str| {
        result
            .run
            .fields
            .iter()
            .find(|field| field.field_id == id)
            .unwrap()
            .as_host_f64()
            .unwrap()
    };
    let mechanical = model.materials[0].require_mechanical().unwrap();
    let nu = mechanical.poisson_ratio.canonical_value();
    let eigenstrain = 1.0e-4;
    let expected_extension = eigenstrain * (1.0 + nu) / (1.0 - nu);
    let displacement = field(crate::FEA_FIELD_STRUCTURAL_DISPLACEMENT);
    for (node, value) in mesh.topology.nodes.iter().zip(displacement.chunks_exact(3)) {
        assert!(
            (value[0] - expected_extension * node.coordinates_m.canonical_value()[0]).abs()
                < 1.0e-12
        );
        assert!(value[1].abs() < 1.0e-12 && value[2].abs() < 1.0e-12);
    }
    let expected_transverse_stress =
        -mechanical.youngs_modulus_pa.canonical_value() * eigenstrain / (1.0 - nu);
    let stress = field(crate::FEA_FIELD_STRUCTURAL_STRESS);
    assert!(stress[0].abs() < 1.0e-4);
    for value in &stress[1..3] {
        assert!((value / expected_transverse_stress - 1.0).abs() < 1.0e-10);
    }
    let expected_energy =
        mechanical.youngs_modulus_pa.canonical_value() * eigenstrain.powi(2) / (1.0 - nu) / 6.0;
    assert!(
        (field(crate::FEA_FIELD_STRUCTURAL_TOTAL_STRAIN_ENERGY)[0] / expected_energy - 1.0).abs()
            < 1.0e-10
    );
    let repeated =
        run_linear_static_from_thermal_state(&model, ComputeBackend::Cpu, options.clone(), &source)
            .unwrap();
    assert_eq!(
        result.solver_result_identity,
        repeated.solver_result_identity
    );
    assert_eq!(
        result.run.identity.realization().coupling,
        CouplingImplementation::OneWaySolvedThermoelasticState
    );
    let temperature_field = result
        .run
        .fields
        .iter()
        .find(|field| field.field_id == crate::fea_thermo_mechanical_temperature_field_id(0))
        .unwrap();
    // One integration-point entity per point: one tetrahedron with four points.
    assert_eq!(temperature_field.shape, vec![4]);
    assert!(temperature_field
        .as_host_f64()
        .unwrap()
        .iter()
        .all(|temperature| (temperature - 303.15).abs() < 1.0e-10));
    let mut missing_thermal = model.clone();
    missing_thermal.materials[0].thermal = None;
    assert!(run_linear_static_from_thermal_state(
        &missing_thermal,
        ComputeBackend::Cpu,
        options.clone(),
        &source,
    )
    .is_err());
    let options = LinearStaticSolveOptions {
        solver_mesh: Some(mesh.clone()),
        ..LinearStaticSolveOptions::default()
    };
    assert!(
        crate::run_linear_static_with_options(&model, ComputeBackend::Cpu, options.clone())
            .is_err()
    );
    let reaction = field(crate::FEA_FIELD_STRUCTURAL_REACTION_FORCE);
    for axis in 0..3 {
        assert!(
            reaction
                .chunks_exact(3)
                .map(|value| value[axis])
                .sum::<f64>()
                .abs()
                < 1.0e-4
        );
    }
    model.materials[0]
        .thermal
        .as_mut()
        .unwrap()
        .expansion_coefficient_per_k = runmat_types::physical::Quantity::new(
        model.materials[0]
            .thermal
            .as_mut()
            .unwrap()
            .expansion_coefficient_per_k
            .canonical_value()
            * (2.0),
    );
    let doubled =
        run_linear_static_from_thermal_state(&model, ComputeBackend::Cpu, options.clone(), &source)
            .unwrap();
    assert_ne!(
        result.solver_result_identity,
        doubled.solver_result_identity
    );
    let doubled_energy = doubled
        .run
        .fields
        .iter()
        .find(|field| field.field_id == crate::FEA_FIELD_STRUCTURAL_TOTAL_STRAIN_ENERGY)
        .unwrap()
        .as_host_f64()
        .unwrap()[0];
    assert!((doubled_energy / expected_energy - 4.0).abs() < 1.0e-10);
    let mut conflicting = model.clone();
    crate::fixtures::append_fixture_temperature_field(
        &mut conflicting,
        "conflicting_temperature",
        320.0,
        ExcitationTarget::WholeModel,
    );
    let error = run_linear_static_from_thermal_state(
        &conflicting,
        ComputeBackend::Cpu,
        options.clone(),
        &source,
    )
    .err()
    .unwrap();
    assert!(
        error
            .to_string()
            .contains("conflicts with an active prescribed temperature"),
        "{error}"
    );
    if let AnalysisConstraint::Thermal(constraint) = &mut model.constraints[0] {
        if let ThermalConstraintKind::Prescribed { value, .. } = &mut constraint.kind {
            *value = PrescribedValue::Constant(Temperature::new(313.15));
        }
    }
    let hotter = crate::run_thermal_with_options(
        &model,
        ComputeBackend::Cpu,
        ThermalSolveOptions {
            procedure: ThermalProcedureControl::Steady,
            solver_mesh: Some(mesh.clone()),
            ..ThermalSolveOptions::default()
        },
    )
    .unwrap();
    let hotter_source =
        CanonicalSemanticState::from_thermal_result(&model, &mesh, &hotter).unwrap();
    let hotter_result =
        run_linear_static_from_thermal_state(&model, ComputeBackend::Cpu, options, &hotter_source)
            .unwrap();
    assert_ne!(source.identity, hotter_source.identity);
    assert_ne!(
        doubled.solver_result_identity,
        hotter_result.solver_result_identity
    );
    assert_eq!(
        hotter_result.thermal_transition.source_state,
        hotter_source.identity
    );
}
