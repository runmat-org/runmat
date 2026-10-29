//! Independent conjugate references: exact series conduction through a solid plate and a fluid
//! layer (linear elements reproduce it at every node), an exact contact-resistance jump, an
//! exact Raviart–Thomas lift of uniform flow and energy closure of heated plug flow.
use super::*;
use crate::{
    fixtures::{fixture_model, FixtureId},
    properties::compile_property_map,
    topology::{FiniteVolumeCellSelection, FiniteVolumeProjectionPolicy},
};
use runmat_analysis_core::*;
use runmat_meshing_core::{
    fixtures::layered_box_mesh, PersistentEntityId, PersistentEntityKind, SolverMeshArtifact,
};

const PLATE: f64 = 0.2;
const HEIGHT: f64 = 1.0;
const K_SOLID: f64 = 40.0;
const K_FLUID: f64 = 0.6;
const RHO_C_FLUID: f64 = 1000.0 * 4000.0;
const FLUX: f64 = 5000.0;
const TOP: f64 = 300.0;

fn material(id: &str, k: f64, density: f64, specific_heat: f64, fluid: bool) -> MaterialModel {
    let mut m = fixture_model(FixtureId::CantileverLinearStatic).materials[0].clone();
    m.material_id = id.into();
    m.name = id.into();
    m.mass_density_kg_per_m3 = Some(runmat_types::physical::Quantity::new(density));
    m.elasticity = None;
    m.damping = None;
    m.acoustic = None;
    m.electrical = None;
    m.plastic = None;
    m.thermal = Some(MaterialThermalModel {
        reference_temperature_k: runmat_types::physical::Quantity::new(293.15),
        modulus_temp_coeff_per_k: runmat_types::physical::Quantity::new(0.0),
        conductivity_w_per_mk: runmat_types::physical::Quantity::new(k),
        specific_heat_j_per_kgk: runmat_types::physical::Quantity::new(specific_heat),
        expansion_coefficient_per_k: runmat_types::physical::Quantity::new(0.0),
    });
    m.fluid = fluid.then_some(MaterialFluidModel::IncompressibleNewtonian {
        dynamic_viscosity: DynamicViscosity::new(1e-3),
    });
    m
}

fn bind(mesh: &SolverMeshArtifact, target: &mut RegionTarget) {
    let entities = mesh
        .topology
        .boundary_faces
        .iter()
        .flat_map(|f| f.provenance.iter())
        .filter(|e| target.matches_persistent_entity(e))
        .cloned()
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();
    target.bind_resolved_entities(entities).unwrap();
}

fn boundary(mesh: &SolverMeshArtifact, name: &str) -> RegionTarget {
    let mut target = RegionTarget::boundary(
        runmat_meshing_core::fixtures::fixture_boundary_selection(&(name)),
    );
    bind(mesh, &mut target);
    target
}

struct Case {
    model: AnalysisModel,
    mesh: SolverMeshArtifact,
}

fn case(cells: [usize; 3], transfer: ThermalInterfaceTransfer) -> Case {
    let mesh = layered_box_mesh(
        cells,
        [2.0, HEIGHT, 0.5],
        &[(PLATE, "plate"), (HEIGHT + 1.0, "channel")],
    );
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    model.structural = None;
    model.thermo_mechanical = None;
    model.electro_thermal = None;
    model.electromagnetic = None;
    model.constraints.clear();
    model.boundaries.clear();
    model.signals.clear();
    model.excitations.clear();
    model.excitation_cases.clear();
    model.initial_conditions.clear();
    model.connectors.clear();
    model.response_spectra.clear();
    model.random_excitations.clear();
    model.frame = CoordinateFrame::Global;
    model.thermal = Some(ThermalDomain {
        reference_temperature: Temperature::new(293.15),
    });
    model.materials = vec![
        material("aluminum", K_SOLID, 2700.0, 900.0, false),
        material("water", K_FLUID, 1000.0, 4000.0, true),
    ];
    let mut solid = model.material_assignments[0].clone();
    solid.target =
        PropertyRegionTarget::solid(runmat_meshing_core::fixtures::fixture_solid("plate"));
    solid.expected_material_id = "aluminum".into();
    solid.assigned_material_id = "aluminum".into();
    let mut fluid = solid.clone();
    fluid.target =
        PropertyRegionTarget::cell(runmat_meshing_core::fixtures::fixture_region("channel"));
    fluid.expected_material_id = "water".into();
    fluid.assigned_material_id = "water".into();
    model.material_assignments = vec![solid, fluid];
    model.cfd = Some(CfdDomain {
        density: MassDensity::new(1000.0),
        dynamic_viscosity: DynamicViscosity::new(1e-3),
        turbulence_intensity: TurbulenceIntensity::new(0.0),
        rheology: None,
        porous_zones: Vec::new(),
    });
    model.sequence = AnalysisSequence::from_steps(
        "conjugate",
        vec![ProcedureStep::coupled(
            "operating_point",
            CoupledPhysicsFamily::ConjugateHeatTransfer,
            AnalysisProcedure::Static,
        )],
    );
    model.boundaries.push(AnalysisBoundary {
        boundary_id: "heater".into(),
        target: boundary(&mesh, "plate_y_min"),
        operator: BoundaryOperator::ThermalHeatFlux {
            value: PrescribedValue::Constant(HeatFlux::new(FLUX)),
        },
    });
    model.interfaces = vec![AnalysisInterface {
        interface_id: "wetted_wall".into(),
        primary: boundary(&mesh, "interface_channel_plate"),
        secondary: boundary(&mesh, "interface_channel_plate"),
        kind: AnalysisInterfaceKind::ConjugateHeatTransfer(ConjugateHeatTransferInterfaceModel {
            transfer,
        }),
    }];
    Case { model, mesh }
}

fn prescribe(case: &mut Case, name: &str, value: f64) {
    let target = boundary(&case.mesh, name);
    case.model
        .constraints
        .push(AnalysisConstraint::Thermal(ThermalConstraint {
            constraint_id: format!("temperature_{name}"),
            kind: ThermalConstraintKind::Prescribed {
                target,
                value: PrescribedValue::Constant(Temperature::new(value)),
            },
        }));
}

struct Prepared {
    projection: AnalysisProjection,
    topology: FiniteVolumeTopology,
    properties: CompiledPropertyMap,
}

fn prepare(case: &Case) -> Prepared {
    let projection = AnalysisProjection::derive(&case.mesh).unwrap();
    let topology = FiniteVolumeTopology::derive_selected(
        &projection,
        FiniteVolumeProjectionPolicy::default(),
        FiniteVolumeCellSelection::regions(vec![PersistentEntityId::new(
            PersistentEntityKind::Region,
            "channel".into(),
            vec!["root".into()],
        )])
        .unwrap(),
    )
    .unwrap();
    let properties = compile_property_map(&case.model, Some(&case.mesh)).unwrap();
    Prepared {
        projection,
        topology,
        properties,
    }
}

fn input<'a>(case: &'a Case, p: &'a Prepared, flux: &'a [f64]) -> ConjugateHeatInput<'a> {
    ConjugateHeatInput {
        model: &case.model,
        step: &case.model.sequence.steps[0],
        projection: &p.projection,
        properties: &p.properties,
        fluid_topology: &p.topology,
        face_volume_flux_m3_per_s: flux,
        sample: ExcitationSamplePoint::Static,
        policy: ConjugateHeatPolicy::default(),
    }
}

/// Series conduction: T = TOP + q (H − y) / k_f in the fluid and continues with slope q / k_s
/// through the plate, plus the contact jump q R at the interface.
fn exact(y: f64, resistance: f64) -> f64 {
    let interface = TOP + FLUX * (HEIGHT - PLATE) / K_FLUID;
    if y >= PLATE {
        TOP + FLUX * (HEIGHT - y) / K_FLUID
    } else {
        interface + FLUX * resistance + FLUX * (PLATE - y) / K_SOLID
    }
}

#[test]
fn series_conduction_through_solid_and_fluid_is_reproduced_at_every_node() {
    let mut c = case([4, 10, 2], ThermalInterfaceTransfer::Perfect);
    prescribe(&mut c, "channel_y_max", TOP);
    let p = prepare(&c);
    let flux = vec![0.0; p.topology.faces.len()];
    let problem = ConjugateHeatProblem::prepare(input(&c, &p, &flux)).unwrap();
    let result = problem.solve().unwrap();
    for (node, t) in p.projection.nodes.iter().zip(&result.temperature_k) {
        let reference = exact(node.coordinates_m.canonical_value()[1], 0.0);
        assert!(
            (t - reference).abs() < 1e-7 * reference,
            "node at y={} carries {t}, exact {reference}",
            node.coordinates_m.canonical_value()[1]
        );
    }
    let heated_area = 2.0 * 0.5;
    let [interface] = result.evidence.interfaces.as_slice() else {
        panic!("one declared interface");
    };
    assert!((interface.heat_into_second_side_w - FLUX * heated_area).abs() < 1e-6 * FLUX);
    assert!((interface.heat_into_first_side_w + FLUX * heated_area).abs() < 1e-6 * FLUX);
    assert!(interface.relative_imbalance < 1e-8);
    let energy = &result.evidence.energy;
    assert!((energy.applied_heat_w - FLUX * heated_area).abs() < 1e-9 * FLUX);
    assert!((energy.prescribed_temperature_heat_w - FLUX * heated_area).abs() < 1e-6 * FLUX);
    assert!(energy.relative_imbalance < 1e-9);
    problem.recover(&result).unwrap();
    let mut tampered = result.clone();
    tampered.temperature_k[0] += 1e-3;
    assert!(problem.recover(&tampered).is_err());
}

#[test]
fn contact_resistance_carries_the_exact_temperature_jump_and_conserves_heat() {
    let resistance = 2e-4;
    let mut c = case(
        [3, 10, 2],
        ThermalInterfaceTransfer::Resistance(ThermalContactResistance::new(resistance)),
    );
    prescribe(&mut c, "channel_y_max", TOP);
    let p = prepare(&c);
    let flux = vec![0.0; p.topology.faces.len()];
    let problem = ConjugateHeatProblem::prepare(input(&c, &p, &flux)).unwrap();
    let result = problem.solve().unwrap();
    // Published node values are the first (solid) side; the fluid side is retained separately.
    for (node, t) in p.projection.nodes.iter().zip(&result.temperature_k) {
        let y = node.coordinates_m.canonical_value()[1];
        let reference = if (y - PLATE).abs() < 1e-12 {
            exact(PLATE - 1e-15, resistance)
        } else {
            exact(y, resistance)
        };
        assert!(
            (t - reference).abs() < 1e-7 * reference,
            "y={y}: {t} vs {reference}"
        );
    }
    assert!(!result.interface_second_side_temperature_k.is_empty());
    for (_, t) in &result.interface_second_side_temperature_k {
        assert!((t - exact(PLATE, 0.0)).abs() < 1e-7 * t);
    }
    let [interface] = result.evidence.interfaces.as_slice() else {
        panic!("one declared interface");
    };
    assert!((interface.maximum_temperature_jump_k - FLUX * resistance).abs() < 1e-7);
    assert!(interface.law_relative_residual < 1e-8);
    assert!(interface.relative_imbalance < 1e-8);
    problem.recover(&result).unwrap();
}

#[test]
fn raviart_thomas_lift_reproduces_uniform_flow_exactly() {
    let c = case([3, 6, 2], ThermalInterfaceTransfer::Perfect);
    let p = prepare(&c);
    let u = [0.01, 0.0, 0.0];
    let flux = p
        .topology
        .faces
        .iter()
        .map(|f| assembly::dot(u, f.owner_area_vector_m2))
        .collect::<Vec<_>>();
    let lifted = velocity::lift(&p.projection, &p.topology, &flux).unwrap();
    for (element, v) in p.projection.elements.iter().zip(&lifted) {
        if let Some(v) = v {
            for corner in &element.node_indices {
                let at = v.at(p.projection.nodes[*corner].coordinates_m.canonical_value());
                for k in 0..3 {
                    assert!((at[k] - u[k]).abs() < 1e-14, "{at:?}");
                }
            }
        }
    }
    assert_eq!(
        lifted.iter().filter(|v| v.is_some()).count(),
        p.topology.cells.len()
    );
}

#[test]
fn heated_plug_flow_carries_the_applied_heat_out_of_the_outlet() {
    let mut c = case([8, 10, 2], ThermalInterfaceTransfer::Perfect);
    prescribe(&mut c, "channel_x_min", TOP);
    let p = prepare(&c);
    let speed = 1e-4;
    let flux = p
        .topology
        .faces
        .iter()
        .map(|f| speed * f.owner_area_vector_m2[0])
        .collect::<Vec<_>>();
    let problem = ConjugateHeatProblem::prepare(input(&c, &p, &flux)).unwrap();
    let result = problem.solve().unwrap();
    let energy = &result.evidence.energy;
    let applied = FLUX * 2.0 * 0.5;
    assert!((energy.applied_heat_w - applied).abs() < 1e-9 * applied);
    // Conservative transport: enthalpy leaving plus the heat drawn at the inlet temperature
    // equals the applied heat to the solver tolerance.
    assert!(energy.relative_imbalance < 1e-8, "{energy:?}");
    assert!(energy.enthalpy_outflow_w > 0.5 * applied, "{energy:?}");
    let [interface] = result.evidence.interfaces.as_slice() else {
        panic!("one declared interface");
    };
    assert!(interface.heat_into_second_side_w > 0.0);
    assert!(interface.relative_imbalance < 1e-8, "{interface:?}");
    // Bulk outlet temperature rise matches the enthalpy carried by the plug flow.
    let outlet_area = (HEIGHT - PLATE) * 0.5;
    let rise = energy.enthalpy_outflow_w / (RHO_C_FLUID * speed * outlet_area);
    assert!(rise > 0.0 && rise.is_finite());
    problem.recover(&result).unwrap();
}

#[test]
fn undeclared_fluid_solid_faces_and_missing_inlet_temperature_are_rejected() {
    let mut c = case([2, 5, 1], ThermalInterfaceTransfer::Perfect);
    prescribe(&mut c, "channel_y_max", TOP);
    let p = prepare(&c);
    let flux = vec![0.0; p.topology.faces.len()];
    let mut undeclared = Case {
        model: c.model.clone(),
        mesh: c.mesh.clone(),
    };
    undeclared.model.interfaces.clear();
    assert!(ConjugateHeatProblem::prepare(input(&undeclared, &p, &flux)).is_err());
    let speed = 1e-4;
    let moving = p
        .topology
        .faces
        .iter()
        .map(|f| speed * f.owner_area_vector_m2[0])
        .collect::<Vec<_>>();
    // Inflow at x = 0 with no inlet temperature.
    assert!(ConjugateHeatProblem::prepare(input(&c, &p, &moving)).is_err());
}

#[test]
fn warm_up_stores_the_applied_heat_and_settles_on_the_steady_state() {
    let mut c = case([3, 10, 1], ThermalInterfaceTransfer::Perfect);
    prescribe(&mut c, "channel_y_max", TOP);
    let p = prepare(&c);
    let flux = vec![0.0; p.topology.faces.len()];
    let problem = ConjugateHeatProblem::prepare(input(&c, &p, &flux)).unwrap();
    let steady = problem.solve().unwrap();
    let mut state = problem.uniform_state(TOP);
    // Diffusion time of the fluid layer h^2 rho c / k ~ 5e6 s; march well beyond it.
    let dt = 2.0e6;
    for _ in 0..30 {
        let next = problem.step_backward_euler(&state, dt).unwrap();
        let e = &next.evidence.energy;
        assert!(e.relative_imbalance < 1e-8, "{e:?}");
        assert!(e.stored_heat_rate_w >= -1e-9 * e.applied_heat_w);
        problem.recover_step(&state, &next, dt).unwrap();
        state = next;
    }
    for (a, b) in state.temperature_k.iter().zip(&steady.temperature_k) {
        assert!(
            (a - b).abs() < 1e-3 * (b - TOP).abs().max(1.0),
            "{a} vs {b}"
        );
    }
}

// Moved from the retired runtime fixture `cht_coupled_invalid_interface_mapping`: a structural
// contact interface cannot stand in for the conjugate heat transfer interface on the wetted wall.
#[test]
fn structural_contact_interface_is_not_a_conjugate_interface() {
    let mut c = case([2, 5, 1], ThermalInterfaceTransfer::Perfect);
    prescribe(&mut c, "channel_y_max", TOP);
    let p = prepare(&c);
    let flux = vec![0.0; p.topology.faces.len()];
    assert!(ConjugateHeatProblem::prepare(input(&c, &p, &flux)).is_ok());
    c.model.interfaces[0].kind = AnalysisInterfaceKind::Contact(ContactInterfaceModel {
        friction: ContactFriction::Frictionless,
    });
    assert!(ConjugateHeatProblem::prepare(input(&c, &p, &flux)).is_err());
}
