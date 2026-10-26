//! Physical solid qualification of the solved-temperature transfer path.
//!
//! The retained tetrahedron cases are continuum patches, not beam benchmarks. The
//! `continuum` child separately verifies affine Tet4/Tet10 mesh refinement against an
//! independent nonuniform-temperature thermoelastic solution.
//! Temperatures come from a thermal solve; no structural answer is injected into the state.

mod continuum;
mod downstream;
mod modal;
mod prepared;

use super::*;
use crate::{
    fixtures::{fixture_model, FixtureId},
    ThermalProcedureControl, ThermalSolveOptions,
};
use runmat_analysis_core::*;
use runmat_meshing_core::{
    fixtures::canonical_tetrahedron_solver_mesh, CanonicalMeshingContract, ElementOrder,
    SolverMeshArtifact,
};

const REFERENCE: f64 = 293.15;
const DELTA: f64 = 10.0;
const ALPHA: f64 = 1.0e-5;

mod prestress;

fn model_and_mesh(order: ElementOrder, restrained: bool) -> (AnalysisModel, SolverMeshArtifact) {
    let mesh = canonical_tetrahedron_solver_mesh(order);
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    model.excitations.clear();
    model.excitation_cases.clear();
    model.signals.clear();
    model.constraints.clear();
    model.boundaries.clear();
    model.thermal = Some(ThermalDomain {
        reference_temperature: Temperature::new(REFERENCE),
    });
    model.thermo_mechanical = Some(ThermoMechanicalDomain {
        reference_temperature: Temperature::new(REFERENCE),
    });
    let thermal = model.materials[0].thermal.as_mut().unwrap();
    thermal.reference_temperature_k = runmat_types::physical::Temperature::new(REFERENCE);
    thermal.expansion_coefficient_per_k = runmat_types::physical::ThermalExpansion::new(ALPHA);
    let producer = ProcedureStep::thermal_steady("qualification_heat");
    let mut consumer = ProcedureStep::coupled(
        "qualification_expansion",
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
        "solved_temperature_patch",
        vec![producer.clone(), consumer.clone()],
    );
    model.sequence.dependencies[0].reason = SequenceDependencyReason::StateTransfer;
    model.sequence.transitions.push(StateTransition {
        transition_id: TransitionId("temperature_patch_reuse".into()),
        producer_step: producer.step_id,
        consumer_step: consumer.step_id,
        state_id: producer.state_outputs[0].state_id.clone(),
        method: StateTransitionMethod::Reuse,
        topology: TopologyTransitionPolicy::RequireIdentical,
        reference: ReferenceStatePolicy::Preserve,
    });
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
    model
        .constraints
        .push(AnalysisConstraint::Thermal(ThermalConstraint {
            constraint_id: "isothermal_wall".into(),
            kind: ThermalConstraintKind::Prescribed {
                target: RegionTarget::boundary(face.provenance[0].analysis_selection_id().unwrap()),
                value: PrescribedValue::Constant(Temperature::new(REFERENCE + DELTA)),
            },
        }));
    for node in &mesh.topology.nodes {
        // Exactly six zero components remove rigid modes without constraining dilation:
        // origin xyz, x-axis corner yz, y-axis corner z. All other DOFs are solved.
        let axes: &[usize] =
            if restrained || node.coordinates_m.canonical_value() == [0.0, 0.0, 0.0] {
                &[0, 1, 2]
            } else if node.coordinates_m.canonical_value() == [1.0, 0.0, 0.0] {
                &[1, 2]
            } else if node.coordinates_m.canonical_value() == [0.0, 1.0, 0.0] {
                &[2]
            } else {
                &[]
            };
        if axes.is_empty() {
            continue;
        }
        model
            .constraints
            .push(AnalysisConstraint::Structural(StructuralConstraint {
                constraint_id: format!("qualification_node_{}", node.node_id),
                frame: CoordinateFrame::Global,
                kind: StructuralConstraintKind::Prescribed(PrescribedStructuralDofs {
                    target: RegionTarget::point(format!("node:{}", node.node_id)),
                    translations: axes
                        .iter()
                        .map(|&axis| TranslationDofValue {
                            dof: [
                                StructuralTranslationDof::Ux,
                                StructuralTranslationDof::Uy,
                                StructuralTranslationDof::Uz,
                            ][axis],
                            value: PrescribedValue::Constant(Displacement::new(0.0)),
                        })
                        .collect(),
                    rotations: Vec::new(),
                }),
            }));
    }
    (model, mesh)
}

fn solve_temperature(model: &AnalysisModel, mesh: &SolverMeshArtifact) -> CanonicalSemanticState {
    solve_temperature_profile(model, mesh, 0.0)
}

fn solve_temperature_profile(
    model: &AnalysisModel,
    mesh: &SolverMeshArtifact,
    gradient: f64,
) -> CanonicalSemanticState {
    let thermal = crate::run_thermal_with_options(
        model,
        ComputeBackend::Cpu,
        ThermalSolveOptions {
            procedure: ThermalProcedureControl::Steady,
            solver_mesh: Some(mesh.clone()),
            ..ThermalSolveOptions::default()
        },
    )
    .unwrap();
    let source = CanonicalSemanticState::from_thermal_result(model, mesh, &thermal).unwrap();
    let temperatures = source
        .payload
        .thermal()
        .unwrap()
        .temperature
        .as_host_f64()
        .unwrap();
    assert_eq!(temperatures.len(), mesh.topology.nodes.len());
    for (node, value) in mesh.topology.nodes.iter().zip(temperatures) {
        assert!(
            (value - REFERENCE - DELTA - gradient * node.coordinates_m.canonical_value()[0]).abs()
                < 1.0e-9
        );
    }
    source
}

#[test]
fn solved_linear_temperature_drives_compatible_quadratic_tet10_deformation() {
    let (mut model, mesh) = model_and_mesh(ElementOrder::Tet10, false);
    let gradient = 20.0;
    let projection = crate::topology::AnalysisProjection::derive(&mesh).unwrap();
    let inclined = projection
        .boundary_faces
        .iter()
        .find(|face| {
            face.node_indices.iter().all(|&index| {
                (mesh.topology.nodes[index]
                    .coordinates_m
                    .canonical_value()
                    .iter()
                    .sum::<f64>()
                    - 1.0)
                    .abs()
                    < 1.0e-12
            })
        })
        .unwrap();
    let conductivity = model.materials[0]
        .require_thermal()
        .unwrap()
        .conductivity_w_per_mk
        .canonical_value();
    // T=Tref+DELTA+b*x satisfies Laplace's equation. Its inward applied heat flux
    // on x+y+z=1 is k*b/sqrt(3); y=0,z=0 are naturally insulated, x=0 is prescribed.
    model.boundaries.push(AnalysisBoundary {
        boundary_id: "exact_gradient_flux".into(),
        target: RegionTarget::boundary(inclined.provenance[0].analysis_selection_id().unwrap()),
        operator: BoundaryOperator::ThermalHeatFlux {
            value: PrescribedValue::Constant(HeatFlux::new(
                conductivity * gradient / 3.0_f64.sqrt(),
            )),
        },
    });
    let source = solve_temperature_profile(&model, &mesh, gradient);
    let retained = source.clone();
    let result =
        run_linear_static_from_thermal_state(&model, ComputeBackend::Cpu, options(&mesh), &source)
            .unwrap();
    assert_eq!(source, retained);
    assert_eq!(result.thermal_transition.source_state, source.identity);
    let displacement = field(&result.run, crate::FEA_FIELD_STRUCTURAL_DISPLACEMENT);
    assert_eq!(displacement.len(), mesh.topology.nodes.len() * 3);
    // Independent compatible continuum solution: sym(grad u)=alpha*(DELTA+b*x)*I.
    // It obeys exactly the six rigid-mode pins and has curvature d²u_x/dy²=-alpha*b.
    // All remaining components, including midside displacements, are solved unknowns.
    for (node, actual) in mesh.topology.nodes.iter().zip(displacement.chunks_exact(3)) {
        let [x, y, z] = node.coordinates_m.canonical_value();
        let expected = [
            ALPHA * (DELTA * x + 0.5 * gradient * (x * x - y * y - z * z)),
            ALPHA * (DELTA * y + gradient * x * y),
            ALPHA * (DELTA * z + gradient * x * z),
        ];
        for axis in 0..3 {
            assert!(
                (actual[axis] - expected[axis]).abs() < 1.0e-11,
                "gradient node {} axis {axis}: {} != {}",
                node.node_id,
                actual[axis],
                expected[axis]
            );
        }
    }
    let stress_scale = model.materials[0]
        .require_mechanical()
        .unwrap()
        .youngs_modulus_pa
        .canonical_value()
        * ALPHA
        * (DELTA + gradient);
    assert!(field(&result.run, crate::FEA_FIELD_STRUCTURAL_STRESS)
        .iter()
        .all(|value| value.abs() < stress_scale * 1.0e-7));
    assert!(
        field(&result.run, crate::FEA_FIELD_STRUCTURAL_REACTION_FORCE)
            .iter()
            .all(|value| value.abs() < stress_scale * 1.0e-7)
    );
    assert!(field(&result.run, crate::FEA_FIELD_STRUCTURAL_TOTAL_STRAIN_ENERGY)[0].abs() < 1.0e-12);
    assert!(result
        .linearization
        .geometric_stiffness()
        .values()
        .iter()
        .all(|value| value.abs() < stress_scale * 1.0e-7));
}

fn options(mesh: &SolverMeshArtifact) -> LinearStaticSolveOptions {
    LinearStaticSolveOptions {
        solver_mesh: Some(mesh.clone()),
        ..LinearStaticSolveOptions::default()
    }
}

fn field<'a>(run: &'a FeaRunResult, id: &str) -> &'a [f64] {
    run.fields
        .iter()
        .find(|field| field.field_id == id)
        .unwrap()
        .as_host_f64()
        .unwrap()
}

#[test]
fn solved_uniform_temperature_produces_free_tet4_and_tet10_dilation() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let (model, mesh) = model_and_mesh(order, false);
        let source = solve_temperature(&model, &mesh);
        let retained = source.clone();
        let result = run_linear_static_from_thermal_state(
            &model,
            ComputeBackend::Cpu,
            options(&mesh),
            &source,
        )
        .unwrap();
        assert_eq!(source, retained);
        assert_eq!(result.thermal_transition.source_state, source.identity);
        assert_eq!(
            result.run.identity.realization().coupling,
            CouplingImplementation::OneWaySolvedThermoelasticState
        );
        let displacement = field(&result.run, crate::FEA_FIELD_STRUCTURAL_DISPLACEMENT);
        assert_eq!(displacement.len(), mesh.topology.nodes.len() * 3);
        for (node, actual) in mesh.topology.nodes.iter().zip(displacement.chunks_exact(3)) {
            for (axis, actual_value) in actual.iter().enumerate() {
                let expected = ALPHA * DELTA * node.coordinates_m.canonical_value()[axis];
                assert!(
                    (actual_value - expected).abs() < 1.0e-11,
                    "{order:?} node {} axis {axis}: {} != {expected}",
                    node.node_id,
                    actual_value
                );
            }
        }
        // epsilon(u)=alpha*deltaT*I, so elastic strain, stress, energy and reactions vanish.
        let stress_scale = model.materials[0]
            .require_mechanical()
            .unwrap()
            .youngs_modulus_pa
            .canonical_value()
            * ALPHA
            * DELTA;
        assert!(result
            .linearization
            .geometric_stiffness()
            .values()
            .iter()
            .all(|value| value.abs() < stress_scale * 1.0e-7));
        for id in [
            crate::FEA_FIELD_STRUCTURAL_STRESS,
            crate::FEA_FIELD_STRUCTURAL_REACTION_FORCE,
        ] {
            assert!(
                field(&result.run, id)
                    .iter()
                    .all(|value| value.abs() < stress_scale * 1.0e-7),
                "{order:?}: nonzero {id} in free dilation"
            );
        }
        assert!(
            field(&result.run, crate::FEA_FIELD_STRUCTURAL_TOTAL_STRAIN_ENERGY)[0].abs() < 1.0e-12
        );
    }
}

#[test]
fn solved_uniform_temperature_produces_restrained_hydrostatic_compression() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let (model, mesh) = model_and_mesh(order, true);
        let source = solve_temperature(&model, &mesh);
        let retained = source.clone();
        let result = run_linear_static_from_thermal_state(
            &model,
            ComputeBackend::Cpu,
            options(&mesh),
            &source,
        )
        .unwrap();
        assert_eq!(source, retained);
        assert!(field(&result.run, crate::FEA_FIELD_STRUCTURAL_DISPLACEMENT)
            .iter()
            .all(|value| value.abs() < 1.0e-14));
        let mechanical = model.materials[0].require_mechanical().unwrap();
        let strain = ALPHA * DELTA;
        let pressure = mechanical.youngs_modulus_pa.canonical_value() * strain
            / (1.0 - 2.0 * mechanical.poisson_ratio.canonical_value());
        // Independent geometric virtual work for v=(x,0,0): integral sigma_xx dV=-p/6.
        let geometric = result.linearization.geometric_stiffness();
        let moment: f64 = mesh
            .topology
            .nodes
            .iter()
            .enumerate()
            .map(|(i, a)| {
                mesh.topology
                    .nodes
                    .iter()
                    .enumerate()
                    .map(|(j, b)| {
                        a.coordinates_m.canonical_value()[0]
                            * geometric.get(3 * i, 3 * j)
                            * b.coordinates_m.canonical_value()[0]
                    })
                    .sum::<f64>()
            })
            .sum();
        assert!((moment / (-pressure / 6.0) - 1.0).abs() < 1.0e-9);
        let stress = field(&result.run, crate::FEA_FIELD_STRUCTURAL_STRESS);
        assert_eq!(stress.len(), 6);
        for value in &stress[..3] {
            assert!(
                (value / -pressure - 1.0).abs() < 1.0e-9,
                "{order:?}: {value}"
            );
        }
        assert!(stress[3..]
            .iter()
            .all(|value| value.abs() < pressure * 1.0e-10));
        // W=1/2 epsilon_el:sigma times unit-simplex volume 1/6.
        let energy = 1.5 * pressure * strain / 6.0;
        assert!(
            (field(&result.run, crate::FEA_FIELD_STRUCTURAL_TOTAL_STRAIN_ENERGY)[0] / energy - 1.0)
                .abs()
                < 1.0e-9
        );
        let reactions = field(&result.run, crate::FEA_FIELD_STRUCTURAL_REACTION_FORCE);
        assert_eq!(reactions.len(), mesh.topology.nodes.len() * 3);
        assert!(reactions
            .iter()
            .any(|value| value.abs() > pressure * 1.0e-3));
        for axis in 0..3 {
            assert!(
                reactions
                    .chunks_exact(3)
                    .map(|value| value[axis])
                    .sum::<f64>()
                    .abs()
                    < pressure * 1.0e-9
            );
            // Independent virtual-work moment: integral(sigma_aa)dV = -p/6.
            let moment = mesh
                .topology
                .nodes
                .iter()
                .zip(reactions.chunks_exact(3))
                .map(|(node, reaction)| node.coordinates_m.canonical_value()[axis] * reaction[axis])
                .sum::<f64>();
            assert!(
                (moment / (-pressure / 6.0) - 1.0).abs() < 1.0e-9,
                "{order:?} reaction moment {axis}: {moment}"
            );
        }
    }
}

#[test]
fn solved_temperature_transfer_rejects_valid_other_mesh_and_missing_authored_link() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let (model, mesh) = model_and_mesh(order, false);
        let source = solve_temperature(&model, &mesh);
        let retained = source.clone();
        // Same node IDs/count/order, but different physical geometry and a valid new digest.
        let mut other_mesh = mesh.clone();
        for node in &mut other_mesh.topology.nodes {
            {
                let quantity_change = 2.0;
                let quantity_index = 2;
                let quantity_slot = &mut node.coordinates_m;
                let mut quantity_coordinates = quantity_slot.canonical_value();
                quantity_coordinates[quantity_index] *= quantity_change;
                *quantity_slot = runmat_types::physical::Quantity::<
                    runmat_types::physical::kind::Position,
                    _,
                >::new(quantity_coordinates);
            };
        }
        other_mesh.seal_canonical_digest().unwrap();
        other_mesh.validate_canonical().unwrap();
        let consumer = &model.sequence.steps[1].step_id;
        // A separately solved, internally valid state with identical temperatures is not
        // interchangeable with the declared producer in this sequence.
        let mut other_model = model.clone();
        other_model.sequence.sequence_id = "another_valid_thermal_sequence".into();
        let other_source = solve_temperature(&other_model, &mesh);
        other_source.validate().unwrap();
        assert_ne!(other_source.identity, source.identity);
        assert!(crate::execution::BoundThermalStateTransfer::bind(
            &model,
            consumer,
            &other_source,
            &mesh,
        )
        .is_err());
        assert!(crate::execution::BoundThermalStateTransfer::bind(
            &model,
            consumer,
            &source,
            &other_mesh,
        )
        .is_err());
        let mut unlinked = model.clone();
        unlinked.sequence.transitions.clear();
        assert!(crate::execution::BoundThermalStateTransfer::bind(
            &unlinked, consumer, &source, &mesh,
        )
        .is_err());
        let mut different_reference = model.clone();
        different_reference
            .thermo_mechanical
            .as_mut()
            .unwrap()
            .reference_temperature = Temperature::new(REFERENCE + 1.0);
        assert!(crate::execution::BoundThermalStateTransfer::bind(
            &different_reference,
            consumer,
            &source,
            &mesh,
        )
        .is_err());
        assert_eq!(source, retained);
    }
}
