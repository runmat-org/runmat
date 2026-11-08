//! Independent planar two-fluid scattering: no discrete solve supplies the reference.
use super::*;
use crate::topology::test_meshes::entity;
use crate::topology::test_meshes::quadratic_slab_mesh;
use runmat_meshing_core::{MeshRegion, PersistentEntityKind};

const Z1: f64 = 6.0;
const Z2: f64 = 20.0;
const FREQUENCY: f64 = 0.2;

fn multiply(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
    [a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]]
}

// Exact uniform-simplex moments integrate exp(-ikx), not its centroid value.
fn mean_wave(xs: [f64; 4], k: f64) -> [f64; 2] {
    let mut moments = [0.0; 32];
    moments[0] = 1.0;
    for x in xs {
        for n in 1..moments.len() {
            moments[n] += x * moments[n - 1];
        }
    }
    let mut coefficient = [1.0, 0.0];
    let mut mean = [0.0; 2];
    for (n, moment) in moments.into_iter().enumerate() {
        if n > 0 {
            coefficient = [
                coefficient[1] * k / n as f64,
                -coefficient[0] * k / n as f64,
            ];
        }
        let integral = 6.0 * moment / ((n + 1) * (n + 2) * (n + 3)) as f64;
        for i in 0..2 {
            mean[i] += coefficient[i] * integral;
        }
    }
    mean
}

// Independent recovered-field surface integral; no assembled B quadratic form.
fn outward_recovered_power(result: &CanonicalAcousticResponse, target: &str) -> f64 {
    let mut power = 0.0;
    for face in result
        .projection
        .boundary_faces_for_target(&RegionTarget::boundary(
            runmat_meshing_core::fixtures::fixture_boundary_selection(&(target)),
        ))
    {
        let velocity =
            &result.element_velocity_m_per_s.canonical_values()[face.adjacent_element_indices[0]];
        result
            .projection
            .visit_boundary_quadrature(face, |point| {
                let mut pressure = [0.0; 2];
                for (&index, &shape) in face.node_indices.iter().zip(point.shape_values()) {
                    pressure[0] += shape * result.pressure_pa.canonical_values()[index].real;
                    pressure[1] += shape * result.pressure_pa.canonical_values()[index].imaginary;
                }
                for (i, area) in point.area_vector_weight_m2.iter().enumerate() {
                    power += 0.5
                        * (pressure[0] * velocity.real[i] + pressure[1] * velocity.imaginary[i])
                        * area;
                }
            })
            .unwrap();
    }
    power
}

fn phase(angle: f64) -> [f64; 2] {
    [angle.cos(), angle.sin()]
}

fn scale(value: [f64; 2], factor: f64) -> [f64; 2] {
    value.map(|x| x * factor)
}

fn direction(rotated: bool) -> [f64; 3] {
    if rotated {
        [
            std::f64::consts::FRAC_1_SQRT_2,
            std::f64::consts::FRAC_1_SQRT_2,
            0.0,
        ]
    } else {
        [1.0, 0.0, 0.0]
    }
}

fn axial(position: [f64; 3], direction: [f64; 3]) -> f64 {
    position
        .into_iter()
        .zip(direction)
        .map(|(x, d)| x * d)
        .sum()
}

fn wave_numbers() -> (f64, f64) {
    (
        std::f64::consts::TAU * FREQUENCY / 2.0,
        std::f64::consts::TAU * FREQUENCY / 4.0,
    )
}

fn reflection() -> f64 {
    (Z2 - Z1) / (Z2 + Z1)
}

fn transmitted(amplitude: [f64; 2]) -> [f64; 2] {
    let (k1, k2) = wave_numbers();
    scale(
        multiply(amplitude, phase((k2 - k1) * 0.5)),
        2.0 * Z2 / (Z2 + Z1),
    )
}

pub(super) fn exact_pressure(x: f64, amplitude: [f64; 2]) -> [f64; 2] {
    let (k1, k2) = wave_numbers();
    if x <= 0.5 {
        let incident = multiply(amplitude, phase(-k1 * x));
        let reflected = scale(multiply(amplitude, phase(k1 * (x - 1.0))), reflection());
        std::array::from_fn(|i| incident[i] + reflected[i])
    } else {
        multiply(transmitted(amplitude), phase(-k2 * x))
    }
}

pub(super) fn exact_mean_velocity(xs: [f64; 4], left: bool, amplitude: [f64; 2]) -> [f64; 2] {
    let (k1, k2) = wave_numbers();
    if left {
        let incident = multiply(amplitude, mean_wave(xs, k1));
        let reflected = scale(
            multiply(multiply(amplitude, phase(-k1)), mean_wave(xs, -k1)),
            reflection(),
        );
        std::array::from_fn(|i| (incident[i] - reflected[i]) / Z1)
    } else {
        scale(
            multiply(transmitted(amplitude), mean_wave(xs, k2)),
            1.0 / Z2,
        )
    }
}

#[test]
fn analytic_interface_obeys_pressure_velocity_and_power_continuity() {
    assert_eq!(reflection(), 7.0 / 13.0);
    assert_eq!(2.0 * Z2 / (Z2 + Z1), 20.0 / 13.0);
    for amplitude in [[1.0, 0.4], [-0.4, 1.0]] {
        let pressure_left = exact_pressure(0.5, amplitude);
        let (_, k2) = wave_numbers();
        let pressure_right = multiply(transmitted(amplitude), phase(-k2 * 0.5));
        let velocity_left = exact_mean_velocity([0.5; 4], true, amplitude);
        let velocity_right = exact_mean_velocity([0.5; 4], false, amplitude);
        for i in 0..2 {
            assert!((pressure_left[i] - pressure_right[i]).abs() < 1e-14);
            assert!((velocity_left[i] - velocity_right[i]).abs() < 1e-14);
        }
        let incoming = amplitude.into_iter().map(|x| x * x).sum::<f64>() / (2.0 * Z1);
        let reflected = incoming * reflection().powi(2);
        let outgoing = transmitted(amplitude)
            .into_iter()
            .map(|x| x * x)
            .sum::<f64>()
            / (2.0 * Z2);
        assert!((incoming - reflected - outgoing).abs() < 1e-15);
    }
}

pub(super) fn two_medium(
    cells: usize,
    quadratic: bool,
    amplitude: [f64; 2],
    rotated: bool,
) -> (AnalysisModel, SolverMeshArtifact) {
    let mut mesh = if quadratic {
        quadratic_slab_mesh(cells)
    } else {
        slab_mesh(cells)
    };
    if rotated {
        let a = std::f64::consts::FRAC_1_SQRT_2;
        for node in &mut mesh.topology.nodes {
            let [x, y, z] = node.coordinates_m.canonical_value();
            node.coordinates_m = runmat_types::physical::Quantity::<
                runmat_types::physical::kind::Position,
                _,
            >::new([a * (x - y), a * (x + y), z]);
        }
    }
    let mut model = model();
    model.sequence.steps[0].independent_domain = IndependentDomain::FrequencyHertz(vec![FREQUENCY]);
    let inlet = exact_pressure(0.0, amplitude);
    model.signals.push(PhysicalSignal::Pressure(complex_signal(
        Pressure::new(inlet[0]),
        Pressure::new(inlet[1]),
    )));
    model
        .constraints
        .push(AnalysisConstraint::Acoustic(AcousticConstraint {
            constraint_id: "total_inlet_pressure".into(),
            kind: AcousticConstraintKind::PrescribedPressure {
                target: RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face(
                    "slab_left",
                )),
                value: PrescribedValue::Signal(SignalId("drive".into())),
            },
        }));
    model.boundaries.push(AnalysisBoundary {
        boundary_id: "matched_transmitted_medium".into(),
        target: RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face("slab_right")),
        operator: BoundaryOperator::AcousticImpedance {
            impedance: SpecificAcousticImpedance::new(Z2),
        },
    });
    let direction = direction(rotated);
    let regions = [
        entity(PersistentEntityKind::Region, "incident_medium"),
        entity(PersistentEntityKind::Region, "transmitted_medium"),
    ];
    let mut elements = [Vec::new(), Vec::new()];
    let topo = &mut *mesh.topology;
    for element in &mut topo.volume_elements {
        let mean = element.node_ids[..4]
            .iter()
            .map(|id| {
                axial(
                    topo.nodes[*id as usize - 1].coordinates_m.canonical_value(),
                    direction,
                )
            })
            .sum::<f64>()
            / 4.0;
        let side = usize::from(mean > 0.5);
        element.region_id = regions[side].clone();
        elements[side].push(element.element_id);
    }
    assert_eq!(elements[0].len(), elements[1].len());
    mesh.topology.regions = regions
        .into_iter()
        .zip(elements)
        .map(|(region_id, element_ids)| MeshRegion {
            region_id,
            element_ids,
        })
        .collect();
    mesh.seal_canonical_digest().unwrap();
    mesh.validate().unwrap();
    model.materials[0].mass_density_kg_per_m3 =
        (Some(3.0)).map(runmat_types::physical::MassDensity::new);
    model.materials[0]
        .acoustic
        .as_mut()
        .unwrap()
        .speed_of_sound_m_per_s = runmat_types::physical::SpeedOfSound::new(2.0);
    model.material_assignments[0].target = PropertyRegionTarget::solid(
        runmat_meshing_core::fixtures::fixture_region("incident_medium"),
    );
    let mut right = model.materials[0].clone();
    right.material_id = "transmitted_fluid".into();
    right.mass_density_kg_per_m3 = (Some(5.0)).map(runmat_types::physical::MassDensity::new);
    right.acoustic.as_mut().unwrap().speed_of_sound_m_per_s =
        runmat_types::physical::SpeedOfSound::new(4.0);
    let mut assignment = model.material_assignments[0].clone();
    assignment.target = PropertyRegionTarget::solid(runmat_meshing_core::fixtures::fixture_region(
        "transmitted_medium",
    ));
    assignment.assigned_material_id = right.material_id.clone();
    assignment.expected_material_id = right.material_id.clone();
    model.materials.push(right);
    model.material_assignments.push(assignment);
    (model, mesh)
}

// witness: FEA-ACOUSTIC-OPERATOR-001/regions @ acoustic_harmonic/tet4|tet10/numerical_kernel;
//   check: two-fluid-interface; boundary: Aligned two-medium slab, phase rotation and spatial refinement; not arbitrary
//   heterogeneous or curved-interface qualification.
#[test]
fn two_fluid_interface_pressure_mean_velocity_and_power_refine() {
    let mut failures = Vec::new();
    let mut inventory = std::collections::BTreeSet::new();
    for quadratic in [false, true] {
        for rotated in [false, true] {
            let direction = direction(rotated);
            for (phase, amplitude) in [[1.0, 0.4], [-0.4, 1.0]].into_iter().enumerate() {
                let norm = amplitude.into_iter().map(|x| x * x).sum::<f64>();
                let expected_power = norm / (2.0 * Z1) * (1.0 - reflection().powi(2));
                let transmitted_power = transmitted(amplitude)
                    .into_iter()
                    .map(|x| x * x)
                    .sum::<f64>()
                    / (2.0 * Z2);
                assert!((expected_power - transmitted_power).abs() < 1e-15);
                let mut pressure_errors = Vec::new();
                let mut velocity_errors = Vec::new();
                for cells in [2, 4, 8] {
                    assert!(inventory.insert((quadratic, rotated, phase, cells)));
                    let (model, mesh) = two_medium(cells, quadratic, amplitude, rotated);
                    let result = solve_canonical_harmonic(
                        &model,
                        &mesh,
                        "acoustic",
                        Frequency::new(FREQUENCY),
                        AcousticHarmonicPolicy {
                            relative_tolerance: 1e-11,
                            ..Default::default()
                        },
                        1e-8,
                    )
                    .unwrap();
                    let pressure = (result
                        .projection
                        .nodes
                        .iter()
                        .zip(result.pressure_pa.canonical_values())
                        .map(|(node, p)| {
                            let reference = exact_pressure(
                                axial(node.coordinates_m.canonical_value(), direction),
                                amplitude,
                            );
                            (p.real - reference[0]).powi(2) + (p.imaginary - reference[1]).powi(2)
                        })
                        .sum::<f64>()
                        / result.pressure_pa.canonical_values().len() as f64)
                        .sqrt();
                    let velocity = (result
                        .projection
                        .elements
                        .iter()
                        .zip(result.element_velocity_m_per_s.canonical_values())
                        .map(|(element, v)| {
                            let xs = std::array::from_fn(|i| {
                                axial(
                                    result.projection.nodes[element.node_indices[i]]
                                        .coordinates_m
                                        .canonical_value(),
                                    direction,
                                )
                            });
                            let left = xs.into_iter().sum::<f64>() / 4.0 < 0.5;
                            let reference = exact_mean_velocity(xs, left, amplitude);
                            (0..3)
                                .map(|i| {
                                    (v.real[i] - direction[i] * reference[0]).powi(2)
                                        + (v.imaginary[i] - direction[i] * reference[1]).powi(2)
                                })
                                .sum::<f64>()
                        })
                        .sum::<f64>()
                        / result.projection.elements.len() as f64)
                        .sqrt();
                    pressure_errors.push(pressure);
                    velocity_errors.push(velocity);
                    let inlet = outward_recovered_power(&result, "slab_left");
                    let outlet = outward_recovered_power(&result, "slab_right");
                    eprintln!("two-fluid quadratic={quadratic} rotated={rotated} phase={phase} cells={cells} pressure={pressure} velocity={velocity} inlet={inlet} outlet={outlet} expected_power={expected_power}");
                    if !(result.relative_residual < 1e-10 && result.power_relative_error < 1e-8) {
                        failures.push(format!("algebraic closure q={quadratic} rotated={rotated} phase={phase} cells={cells}"));
                    }
                    if cells == 8
                        && !(pressure < 0.004 * norm.sqrt()
                            && velocity < 0.01
                            && inlet < 0.0
                            && outlet > 0.0
                            && result.mean_impedance_loss_w.canonical_value() > 0.0
                            && (inlet + expected_power).abs() < 0.005
                            && (outlet - expected_power).abs() < 0.005
                            && (result.mean_impedance_loss_w.canonical_value() - expected_power)
                                .abs()
                                < 0.005)
                    {
                        failures.push(format!("independent finest field/power gate q={quadratic} rotated={rotated} phase={phase}"));
                    }
                }
                if !(pressure_errors
                    .windows(2)
                    .all(|p| p[0] / p[1] > if quadratic { 6.0 } else { 3.0 })
                    && velocity_errors
                        .windows(2)
                        .all(|v| v[0] / v[1] > if quadratic { 3.0 } else { 1.5 }))
                {
                    failures.push(format!("independent refinement gate q={quadratic} rotated={rotated} phase={phase}: pressure={pressure_errors:?} velocity={velocity_errors:?}"));
                }
            }
        }
    }
    let expected = [false, true]
        .into_iter()
        .flat_map(|quadratic| {
            [false, true].into_iter().flat_map(move |rotated| {
                (0..2).flat_map(move |phase| {
                    [2, 4, 8]
                        .into_iter()
                        .map(move |cells| (quadratic, rotated, phase, cells))
                })
            })
        })
        .collect();
    assert_eq!(inventory, expected, "two-medium solve inventory differs");
    assert!(
        failures.is_empty(),
        "two-fluid numerical gate failures: {}",
        failures.join("\n")
    );
}
