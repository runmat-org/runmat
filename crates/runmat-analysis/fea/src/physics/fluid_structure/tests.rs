//! A compliant channel: laminar flow over an elastic slab whose top face is the wetted wall.
//! Fixed below and on rollers at its sides, the slab compresses in uniaxial strain, so the
//! channel widens by C p with C = t / M (M the constrained modulus). Lubrication theory then
//! gives the coupled inlet pressure (h0 + C p_in)^4 = h0^4 + 48 µ Q C L, against the rigid
//! 12 µ Q L / h0^3. The ratio of the two is the independent reference.
use super::*;
use crate::{
    fixtures::{fixture_model, FixtureId},
    properties::{compile_property_map, CompiledPropertyMap},
};
use runmat_analysis_core::*;
use runmat_meshing_core::{fixtures::layered_box_mesh, PersistentEntityId, PersistentEntityKind};

const LENGTH: f64 = 2.0;
const DEPTH: f64 = 0.05;
const PLATE: f64 = 0.1;
const HEIGHT: f64 = 0.3;
const SPEED: f64 = 0.01;
const VISCOSITY: f64 = 1.0;
const POISSON: f64 = 0.3;

fn target(mesh: &SolverMeshArtifact, name: &str) -> RegionTarget {
    let mut target = RegionTarget::boundary(
        runmat_meshing_core::fixtures::fixture_boundary_selection(&(name)),
    );
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
    target
}

pub(super) struct Case {
    model: AnalysisModel,
    mesh: SolverMeshArtifact,
    constraints: Vec<FluidConstraint>,
    boundaries: Vec<AnalysisBoundary>,
    material: CfdDomain,
}

pub(super) fn case(youngs_modulus: f64) -> Case {
    let mesh = layered_box_mesh(
        [20, 9, 1],
        [LENGTH, HEIGHT, DEPTH],
        &[(PLATE, "plate"), (HEIGHT + 1.0, "channel")],
    );
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    model.thermal = None;
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
    let mut solid = model.materials[0].clone();
    solid.material_id = "elastomer".into();
    solid.name = "elastomer".into();
    solid.mass_density_kg_per_m3 = Some(runmat_types::physical::Quantity::new(1000.0));
    solid.elasticity = Some(MaterialElasticityModel::IsotropicLinear(
        MaterialMechanicalModel {
            youngs_modulus_pa: runmat_types::physical::Quantity::new(youngs_modulus),
            poisson_ratio: runmat_types::physical::Quantity::new(POISSON),
        },
    ));
    let mut fluid = solid.clone();
    fluid.material_id = "oil".into();
    fluid.name = "oil".into();
    fluid.elasticity = None;
    fluid.thermal = None;
    fluid.fluid = Some(MaterialFluidModel::IncompressibleNewtonian {
        dynamic_viscosity: DynamicViscosity::new(VISCOSITY),
    });
    model.materials = vec![solid, fluid];
    let mut plate = model.material_assignments[0].clone();
    plate.target =
        PropertyRegionTarget::solid(runmat_meshing_core::fixtures::fixture_solid("plate"));
    plate.expected_material_id = "elastomer".into();
    plate.assigned_material_id = "elastomer".into();
    let mut channel = plate.clone();
    channel.target =
        PropertyRegionTarget::cell(runmat_meshing_core::fixtures::fixture_region("channel"));
    channel.expected_material_id = "oil".into();
    channel.assigned_material_id = "oil".into();
    model.material_assignments = vec![plate, channel];
    model.sequence = AnalysisSequence::from_steps(
        "compliant",
        vec![ProcedureStep::coupled(
            "equilibrium",
            CoupledPhysicsFamily::FluidStructureInteraction,
            AnalysisProcedure::Static,
        )],
    );
    let mut structural = |id: &str, kind: StructuralConstraintKind| {
        model
            .constraints
            .push(AnalysisConstraint::Structural(StructuralConstraint {
                constraint_id: id.into(),
                frame: CoordinateFrame::Global,
                kind,
            }))
    };
    structural(
        "base",
        StructuralConstraintKind::Fixed {
            target: target(&mesh, "plate_y_min"),
        },
    );
    for side in ["plate_x_min", "plate_x_max", "plate_z_min", "plate_z_max"] {
        structural(
            side,
            StructuralConstraintKind::Symmetry {
                target: target(&mesh, side),
            },
        );
    }
    model.interfaces = vec![AnalysisInterface {
        interface_id: "wetted_wall".into(),
        primary: target(&mesh, "interface_channel_plate"),
        secondary: target(&mesh, "interface_channel_plate"),
        kind: AnalysisInterfaceKind::FluidStructure(FluidStructureInterfaceModel {
            kinematics: FluidStructureKinematics::NoSlip,
        }),
    }];
    let fluid_constraint = |id: &str, kind| FluidConstraint {
        constraint_id: id.into(),
        kind,
    };
    let constraints = vec![
        fluid_constraint(
            "inlet",
            FluidConstraintKind::PrescribedVelocity {
                target: target(&mesh, "channel_x_min"),
                value: PrescribedValue::Constant([SPEED, 0.0, 0.0].map(Velocity::new)),
            },
        ),
        fluid_constraint(
            "lid",
            FluidConstraintKind::NoSlipWall {
                target: target(&mesh, "channel_y_max"),
            },
        ),
        fluid_constraint(
            "back",
            FluidConstraintKind::Symmetry {
                target: target(&mesh, "channel_z_min"),
            },
        ),
        fluid_constraint(
            "front",
            FluidConstraintKind::Symmetry {
                target: target(&mesh, "channel_z_max"),
            },
        ),
        fluid_constraint(
            "wetted_wall",
            FluidConstraintKind::NoSlipWall {
                target: target(&mesh, "interface_channel_plate"),
            },
        ),
    ];
    let boundaries = vec![AnalysisBoundary {
        boundary_id: "outlet".into(),
        target: target(&mesh, "channel_x_max"),
        operator: BoundaryOperator::FluidOutletPressure {
            value: PrescribedValue::Constant(Pressure::new(0.0)),
        },
    }];
    Case {
        model,
        mesh,
        constraints,
        boundaries,
        material: CfdDomain {
            density: MassDensity::new(1.0),
            dynamic_viscosity: DynamicViscosity::new(VISCOSITY),
            turbulence_intensity: TurbulenceIntensity::new(0.0),
            rheology: None,
            porous_zones: Vec::new(),
        },
    }
}

pub(super) struct Prepared {
    projection: AnalysisProjection,
    properties: CompiledPropertyMap,
    selection: FiniteVolumeCellSelection,
}

pub(super) fn prepared(c: &Case) -> Prepared {
    Prepared {
        projection: AnalysisProjection::derive(&c.mesh).unwrap(),
        properties: compile_property_map(&c.model, Some(&c.mesh)).unwrap(),
        selection: FiniteVolumeCellSelection::regions(vec![PersistentEntityId::new(
            PersistentEntityKind::Region,
            "channel".into(),
            vec!["root".into()],
        )])
        .unwrap(),
    }
}

pub(super) fn problem<'a>(
    c: &'a Case,
    p: &'a Prepared,
    tolerance: f64,
) -> FluidStructureProblem<'a> {
    try_problem(c, p, tolerance).unwrap()
}

fn try_problem<'a>(
    c: &'a Case,
    p: &'a Prepared,
    tolerance: f64,
) -> Result<FluidStructureProblem<'a>, FluidStructureError> {
    FluidStructureProblem::prepare(FluidStructureInput {
        model: &c.model,
        mesh: &c.mesh,
        projection: &p.projection,
        properties: &p.properties,
        fluid_selection: &p.selection,
        geometry_policy: FiniteVolumeProjectionPolicy::default(),
        fluid: FluidAuthority {
            material: &c.material,
            constraints: &c.constraints,
            boundary_operators: &c.boundaries,
            signals: &[],
            excitations: &[],
            policy: CanonicalFluidPolicy {
                pressure_velocity_coupling: Default::default(),
                fixed_point_acceleration: None,
                maximum_outer_iterations: 3000,
                // Tight outer and linear targets on the coupled reference raise the
                // conservative work bound past the interactive default.
                maximum_operator_work: None,
                // The deformed-passage pressure system stagnates near 1e-9; the coupled
                // reference needs only the physical target below the coupling tolerance.
                linear_relative_tolerance: 1e-8,
                physical_relative_tolerance: 1e-7,
                ..CanonicalFluidPolicy::default()
            },
        },
        policy: FluidStructurePolicy {
            coupling_relative_tolerance: tolerance,
            ..FluidStructurePolicy::default()
        },
    })
}

/// (result, inlet pressure, inflow per unit depth). The inlet prescribes velocity on a face that
/// stretches with the wall, so the delivered flow rate is measured, not assumed.
fn run(c: &Case) -> (FluidStructureResult, f64, f64) {
    let p = prepared(c);
    let problem = problem(c, &p, 1e-6);
    let result = problem.solve().unwrap();
    problem.recover(&result).unwrap();
    // Inlet pressure: mean of the cells touching the inlet face.
    let mut sum = 0.0;
    let mut count = 0.0;
    let mut inflow = 0.0;
    for (i, face) in problem.reference_fluid().faces.iter().enumerate() {
        if face.neighbor.is_none() && face.centroid_m[0] < 1e-12 {
            sum += result.flow.pressure_pa.canonical_values()[face.owner];
            count += 1.0;
            inflow -= result.flow.face_volume_flux_m3_per_s.canonical_values()[i];
        }
    }
    let mut tampered = result.clone();
    tampered.structural_response_m[0][1] += 1e-3;
    assert!(problem.recover(&tampered).is_err());
    (result, sum / count, inflow / DEPTH)
}

#[test]
fn compliant_channel_widens_and_drops_pressure_as_lubrication_theory_predicts() {
    // Soft enough that the lubrication reference departs clearly from the rigid channel.
    let youngs_modulus = 10.0;
    let (rigid, rigid_inlet, rigid_q) = run(&case(1e9));
    let (compliant, compliant_inlet, compliant_q) = run(&case(youngs_modulus));
    assert!(rigid.evidence.largest_interface_displacement_m < 1e-9);
    let constrained = youngs_modulus * (1.0 - POISSON) / ((1.0 + POISSON) * (1.0 - 2.0 * POISSON));
    let c = PLATE / constrained;
    let h0 = HEIGHT - PLATE;
    assert!((rigid_q - SPEED * h0).abs() < 1e-6 * SPEED * h0);
    // Rigid: dp/dx = −12 µ q / h³. Compliant: h = h0 + C p gives (h0 + C p_in)⁴ = h0⁴ + 48 µ q C L.
    let rigid_reference = 12.0 * VISCOSITY * rigid_q * LENGTH / h0.powi(3);
    let coupled_reference =
        ((h0.powi(4) + 48.0 * VISCOSITY * compliant_q * c * LENGTH).powf(0.25) - h0) / c;
    let measured = compliant_inlet / rigid_inlet;
    let expected = coupled_reference / rigid_reference;
    assert!(
        expected < 0.95 && compliant_q > 1.1 * rigid_q,
        "the test must exercise real coupling: {expected}"
    );
    assert!(
        (measured - expected).abs() < 0.05 * expected,
        "compliant/rigid inlet pressure {measured} vs lubrication {expected}"
    );
    let e = &compliant.evidence;
    assert!(e.interface_displacement_relative_residual <= 1e-6);
    assert!(e.work_energy_relative_residual < 1e-5, "{e:?}");
    assert!(e.structural_equilibrium_relative_residual < 1e-8);
    assert!(e.mesh_motion_relative_residual < 1e-8);
    assert!(compliant.coupling_iterations > 1);
    // The fluid pushes the wall down into the slab.
    assert!(e.interface_force_n[1] < 0.0);
}

#[test]
fn impulsively_started_flow_over_a_compliant_wall_conserves_interface_energy() {
    // Stiff enough that the wall's swept volume rate stays a few percent of the inflow, so
    // the pressure outlet never has to admit backflow during the impulsive start.
    let c = case(1.0e6);
    let p = prepared(&c);
    let problem = problem(&c, &p, 1e-6);
    let times = (0..=12).map(|i| 0.01 * i as f64).collect::<Vec<_>>();
    let result = problem.solve_transient(&times).unwrap();
    assert_eq!(result.steps.len(), 12);
    for step in &result.steps {
        assert!(
            step.interface_displacement_relative_residual <= 1e-6,
            "{step:?}"
        );
        assert!(step.structural_dynamic_relative_residual < 1e-5, "{step:?}");
    }
    // Average acceleration: interface work equals stored strain plus kinetic energy.
    assert!(
        result.energy_relative_residual < 1e-4,
        "{}",
        result.energy_relative_residual
    );
    let last = result.steps.last().unwrap();
    assert!(last.interface_work_j > 0.0);
    assert!(last.interface_force_n[1] < 0.0);
    problem.recover_transient(&result).unwrap();
    let mut tampered = result.clone();
    tampered.structure.displacement[0] += 1e-6;
    assert!(problem.recover_transient(&tampered).is_err());
}

// Moved from the retired runtime fixture `fsi_coupled_invalid_interface_mapping`: a structural
// contact interface cannot stand in for the wetted fluid–structure interface.
#[test]
fn structural_contact_interface_is_not_a_fluid_structure_interface() {
    let mut c = case(1e9);
    c.model.interfaces[0].kind = AnalysisInterfaceKind::Contact(ContactInterfaceModel {
        friction: ContactFriction::Frictionless,
    });
    let p = prepared(&c);
    assert!(matches!(
        try_problem(&c, &p, 1e-6),
        Err(FluidStructureError::Unsupported(_))
    ));
}

/// The compliant channel closed above by a second solid, the lid, instead of an exterior wall.
/// The lid's face on the channel carries the authored fluid wall `lid_wall` unless `wall` is
/// false; the plate's face keeps the declared interface `wetted_wall`.
fn lidded_case(youngs_modulus: f64, wall: bool) -> Case {
    const LID: f64 = 2.0 * HEIGHT / 9.0;
    let mut c = case(youngs_modulus);
    c.mesh = layered_box_mesh(
        [20, 11, 1],
        [LENGTH, HEIGHT + LID, DEPTH],
        &[(PLATE, "plate"), (HEIGHT, "channel"), (HEIGHT + 1.0, "lid")],
    );
    let mesh = &c.mesh;
    let mut lid = c.model.material_assignments[0].clone();
    lid.target = PropertyRegionTarget::solid(runmat_meshing_core::fixtures::fixture_solid("lid"));
    c.model.material_assignments.push(lid);
    // Rebind every target on the new mesh.
    for constraint in &mut c.model.constraints {
        if let AnalysisConstraint::Structural(s) = constraint {
            match &mut s.kind {
                StructuralConstraintKind::Fixed { target: bound }
                | StructuralConstraintKind::Symmetry { target: bound } => {
                    *bound = target(mesh, &bound.region_id.clone());
                }
                _ => unreachable!(),
            }
        }
    }
    c.model
        .constraints
        .push(AnalysisConstraint::Structural(StructuralConstraint {
            constraint_id: "lid_fixed".into(),
            frame: CoordinateFrame::Global,
            kind: StructuralConstraintKind::Fixed {
                target: target(mesh, "lid_y_max"),
            },
        }));
    c.model.interfaces[0].primary = target(mesh, "interface_channel_plate");
    c.model.interfaces[0].secondary = target(mesh, "interface_channel_plate");
    let lid_wall = FluidConstraint {
        constraint_id: "lid_wall".into(),
        kind: FluidConstraintKind::NoSlipWall {
            target: target(mesh, "interface_channel_lid"),
        },
    };
    if wall {
        c.model
            .constraints
            .push(AnalysisConstraint::Fluid(lid_wall.clone()));
    }
    // The exterior lid becomes the authored wall on the lid solid.
    c.constraints.retain(|c| c.constraint_id != "lid");
    for constraint in &mut c.constraints {
        let bound = match &mut constraint.kind {
            FluidConstraintKind::PrescribedVelocity { target, .. }
            | FluidConstraintKind::NoSlipWall { target }
            | FluidConstraintKind::Symmetry { target } => target,
            _ => unreachable!(),
        };
        *bound = target(mesh, &bound.region_id.clone());
    }
    c.constraints.push(lid_wall);
    for boundary in &mut c.boundaries {
        boundary.target = target(mesh, &boundary.target.region_id.clone());
    }
    c
}

fn lid_nodes(c: &Case) -> Vec<usize> {
    c.mesh
        .topology
        .nodes
        .iter()
        .enumerate()
        .filter(|(_, n)| n.coordinates_m.canonical_value()[1] >= HEIGHT - 1e-12)
        .map(|(i, _)| i)
        .collect()
}

// G14: a fluid–solid face with a fluid wall condition and no fluid–structure interface is a
// rigid, uncoupled wall: the fluid sees no slip, the wall stays in place and no fluid load
// enters the solid. The plate's declared interface still couples, so the channel matches the
// one closed by an exterior wall.
#[test]
fn a_fluid_wall_on_a_solid_without_an_interface_is_a_rigid_uncoupled_wall() {
    let youngs_modulus = 10.0;
    let lidded = lidded_case(youngs_modulus, true);
    let p = prepared(&lidded);
    let problem = problem(&lidded, &p, 1e-6);
    let lid = lid_nodes(&lidded);
    assert!(!problem.rigid_wall_nodes.is_empty());
    assert!(problem
        .rigid_wall_nodes
        .iter()
        .all(|n| lid.binary_search(n).is_ok()));
    assert!(problem
        .interface_nodes
        .iter()
        .all(|n| lid.binary_search(n).is_err()));
    let (result, inlet, _) = run(&lidded);
    // No load into the lid: its structural response is exactly zero, and its wall stays put.
    for (ordinal, &node) in problem.structure.nodes.iter().enumerate() {
        if lid.binary_search(&node).is_ok() {
            assert_eq!(result.structural_response_m[ordinal], [0.0; 3]);
            assert_eq!(result.displacement_m[node], [0.0; 3]);
        }
    }
    // The plate's interface still couples, and the channel matches the exterior-lid channel.
    assert!(result.coupling_iterations > 1);
    assert!(result.evidence.interface_force_n[1] < 0.0);
    let (_, open_inlet, _) = run(&case(youngs_modulus));
    assert!(
        (inlet - open_inlet).abs() < 1e-4 * open_inlet.abs(),
        "lidded {inlet} vs exterior-lid {open_inlet}"
    );
}

// G14: a fluid–solid face with neither an interface nor a fluid wall condition is the shared
// admission error (`runmat check` reports the same code on the CAD's shared faces).
#[test]
fn a_fluid_solid_face_with_neither_an_interface_nor_a_wall_is_an_admission_error() {
    let lidded = lidded_case(1e9, false);
    let p = prepared(&lidded);
    let Err(FluidStructureError::Admission(issue)) = try_problem(&lidded, &p, 1e-6) else {
        panic!("an undeclared fluid–solid face must fail admission");
    };
    assert_eq!(issue.code, crate::admission::FLUID_SOLID_FACE_CODE);
    assert!(issue.message.contains("interface_channel_lid"), "{issue}");
}

// A face selected by the interface and an authored wall is ambiguous.
#[test]
fn a_fluid_solid_face_with_an_interface_and_a_wall_is_an_admission_error() {
    let mut c = case(1e9);
    c.model.constraints.push(AnalysisConstraint::Fluid(FluidConstraint {
        constraint_id: "extra_wall".into(),
        kind: FluidConstraintKind::NoSlipWall {
            target: target(&c.mesh, "interface_channel_plate"),
        },
    }));
    let p = prepared(&c);
    assert!(matches!(
        try_problem(&c, &p, 1e-6),
        Err(FluidStructureError::Admission(_))
    ));
}
