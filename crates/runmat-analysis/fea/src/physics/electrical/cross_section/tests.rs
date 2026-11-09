mod references;

use super::*;
use references::*;
use std::f64::consts::FRAC_PI_2;

const UM: f64 = 1e-6;
const MM: f64 = 1e-3;

fn dielectric(name: &str, thickness: f64, er: f64) -> StackLayer {
    StackLayer {
        name: name.into(),
        thickness_m: thickness,
        kind: StackLayerKind::Dielectric {
            relative_permittivity: er,
        },
    }
}

fn copper(name: &str, thickness: f64, role: CopperRole) -> StackLayer {
    StackLayer {
        name: name.into(),
        thickness_m: thickness,
        kind: StackLayerKind::Copper {
            role,
            fill_relative_permittivity: None,
        },
    }
}

fn mask(name: &str, thickness: f64, er: f64) -> StackLayer {
    StackLayer {
        name: name.into(),
        thickness_m: thickness,
        kind: StackLayerKind::SolderMask {
            relative_permittivity: er,
        },
    }
}

fn trace(name: &str, layer: &str, center: f64, width: f64) -> Conductor {
    Conductor {
        name: name.into(),
        layer: layer.into(),
        role: ConductorRole::Signal,
        center_m: center,
        width_m: width,
        etch_angle_rad: FRAC_PI_2,
        base: None,
    }
}

/// Surface microstrip: trace of width w and thickness t on h of εr over a plane.
fn microstrip(w: f64, h: f64, t: f64, er: f64) -> CrossSection {
    CrossSection {
        layers: vec![
            copper("F.Cu", t, CopperRole::Signal),
            dielectric("core", h, er),
            copper("GND", 35.0 * UM, CopperRole::ReferencePlane),
        ],
        conductors: vec![trace("line", "F.Cu", 0.0, w)],
        mesh: CrossSectionMeshControls::default(),
        enclosure: None,
    }
}

/// Edge-coupled pair (w, gap s) with one dielectric layer of εr between each plane and the
/// centred signal layer of thickness t; `b` is the plane separation.
fn stripline(widths: &[(f64, f64)], b: f64, t: f64, er: f64) -> CrossSection {
    let h = 0.5 * (b - t);
    CrossSection {
        layers: vec![
            copper("L1", 35.0 * UM, CopperRole::ReferencePlane),
            dielectric("upper", h, er),
            copper("L2", t, CopperRole::Signal),
            dielectric("lower", h, er),
            copper("L3", 35.0 * UM, CopperRole::ReferencePlane),
        ],
        conductors: widths
            .iter()
            .enumerate()
            .map(|(i, &(center, width))| trace(&format!("s{i}"), "L2", center, width))
            .collect(),
        mesh: CrossSectionMeshControls::default(),
        enclosure: None,
    }
}

fn relative(computed: f64, reference: f64) -> f64 {
    computed / reference - 1.0
}

// witness: FEA-EM-LINE-001/line_parameters @ quasi_static_tem/tri3/numerical_kernel;
//   check: microstrip-hammerstad-jensen; boundary: Zero-thickness surface microstrip,
//   w/h = 0.5 to 4 and εr 2.2 to 9.8 on the default mesh: Z0 and εeff against Hammerstad and
//   Jensen within 0.5 %; not thick or masked traces.
#[test]
fn microstrip_matches_hammerstad_jensen() {
    for (u, er) in [(0.5, 4.4), (1.0, 4.4), (2.0, 4.4), (4.0, 2.2), (1.0, 9.8)] {
        let h = 0.2 * MM;
        let line = solve_cross_section(&microstrip(u * h, h, 0.0, er)).unwrap();
        let (z0, eeff) = hammerstad_jensen(u, er);
        let (ez, ee) = (
            relative(line.impedance_ohm[0], z0),
            relative(line.effective_permittivity[0], eeff),
        );
        eprintln!(
            "HJ u={u} er={er}: Z0 {:.4} vs {z0:.4} ({:+.3} %), eeff {:.5} vs {eeff:.5} ({:+.3} %), nodes {}",
            line.impedance_ohm[0],
            100.0 * ez,
            line.effective_permittivity[0],
            100.0 * ee,
            line.mesh.nodes
        );
        assert!(ez.abs() < 5e-3, "u={u} er={er}: Z0 {ez}");
        assert!(ee.abs() < 5e-3, "u={u} er={er}: eeff {ee}");
        let c = lines::speed_of_light();
        assert!((line.delay_s_per_m[0] * c / eeff.sqrt() - 1.0).abs() < 5e-3);
    }
}

// witness: FEA-EM-LINE-001/line_parameters @ quasi_static_tem/tri3/numerical_kernel;
//   check: stripline-cohn; boundary: Zero-thickness centred stripline, w/b = 0.2 to 1:
//   Z0 against Cohn's exact conformal-mapping form within 0.3 %.
// witness: FEA-EM-LINE-001/pair_parameters @ quasi_static_tem/tri3/numerical_kernel;
//   check: stripline-cohn; boundary: Zero-thickness edge-coupled stripline pairs: even- and
//   odd-mode impedances against Cohn's exact forms within 0.3 %, Zdiff = 2 Zodd and
//   Zcommon = Zeven / 2.
#[test]
fn stripline_matches_cohn_exact_forms() {
    let b = 0.5 * MM;
    let er = 3.0;
    for wb in [0.2, 0.5, 1.0] {
        let line = solve_cross_section(&stripline(&[(0.0, wb * b)], b, 0.0, er)).unwrap();
        let reference = cohn_stripline(wb * b, b, er);
        let error = relative(line.impedance_ohm[0], reference);
        eprintln!(
            "Cohn w/b={wb}: Z0 {:.4} vs {reference:.4} ({:+.3} %), nodes {}",
            line.impedance_ohm[0],
            100.0 * error,
            line.mesh.nodes
        );
        assert!(error.abs() < 3e-3, "w/b={wb}: {error}");
        // Homogeneous dielectric: εeff = εr exactly (same mesh, same nodes).
        assert!((line.effective_permittivity[0] - er).abs() < 1e-9);
    }
    for (wb, sb) in [(0.5, 0.2), (0.5, 0.5), (0.3, 1.0)] {
        let (w, s) = (wb * b, sb * b);
        let pitch = 0.5 * (w + s);
        let line =
            solve_cross_section(&stripline(&[(-pitch, w), (pitch, w)], b, 0.0, er)).unwrap();
        let pair = line.pair.unwrap();
        let (even, odd) = cohn_coupled_stripline(w, s, b, er);
        let (ee, eo) = (
            relative(pair.even_mode_impedance_ohm, even),
            relative(pair.odd_mode_impedance_ohm, odd),
        );
        eprintln!(
            "Cohn pair w/b={wb} s/b={sb}: Zeven {:.4} vs {even:.4} ({:+.3} %), Zodd {:.4} vs {odd:.4} ({:+.3} %)",
            pair.even_mode_impedance_ohm,
            100.0 * ee,
            pair.odd_mode_impedance_ohm,
            100.0 * eo
        );
        assert!(ee.abs() < 3e-3 && eo.abs() < 3e-3, "w/b={wb} s/b={sb}: {ee} {eo}");
        assert!((pair.differential_impedance_ohm - 2.0 * pair.odd_mode_impedance_ohm).abs() < 1e-9);
        assert!((pair.common_mode_impedance_ohm - 0.5 * pair.even_mode_impedance_ohm).abs() < 1e-9);
        // Homogeneous dielectric: no far-end coupling, and both modes travel at c/sqrt(εr).
        assert!(pair.far_end_coupling.abs() < 1e-9, "{}", pair.far_end_coupling);
        assert!((pair.odd_mode_effective_permittivity - er).abs() < 1e-9);
        assert!((pair.even_mode_effective_permittivity - er).abs() < 1e-9);
        assert!(pair.near_end_coupling > 0.0);
    }
}

// witness: FEA-EM-LINE-001/line_parameters @ quasi_static_tem/tri3/numerical_kernel;
//   check: trace-impedance-ipc-2141a; boundary: Thick-copper surface microstrip and
//   symmetric stripline inside the IPC-2141A stated ranges: Z0 within the 5 % tolerance set
//   for these empirical forms.
#[test]
fn microstrip_and_stripline_agree_with_ipc_2141a_inside_their_ranges() {
    let t = 35.0 * UM;
    for (w, h, er) in [
        (0.35 * MM, 0.2 * MM, 4.1),
        (0.2 * MM, 0.15 * MM, 4.3),
        (0.15 * MM, 0.25 * MM, 3.5),
    ] {
        let line = solve_cross_section(&microstrip(w, h, t, er)).unwrap();
        let reference = ipc_microstrip(w, h, t, er);
        let error = relative(line.impedance_ohm[0], reference);
        eprintln!(
            "IPC microstrip w={w} h={h} er={er}: Z0 {:.3} vs {reference:.3} ({:+.2} %)",
            line.impedance_ohm[0],
            100.0 * error
        );
        assert!(error.abs() < 0.05, "{error}");
    }
    for (w, b, er) in [(0.15 * MM, 0.6 * MM, 4.2), (0.1 * MM, 0.5 * MM, 3.6), (0.2 * MM, 0.8 * MM, 4.4)] {
        let line = solve_cross_section(&stripline(&[(0.0, w)], b, t, er)).unwrap();
        let reference = ipc_stripline(w, b, t, er);
        let error = relative(line.impedance_ohm[0], reference);
        eprintln!(
            "IPC stripline w={w} b={b} er={er}: Z0 {:.3} vs {reference:.3} ({:+.2} %)",
            line.impedance_ohm[0],
            100.0 * error
        );
        assert!(error.abs() < 0.05, "{error}");
    }
}

// witness: FEA-EM-LINE-001/mesh_convergence @ quasi_static_tem/tri3/numerical_kernel;
//   check: trace-impedance-mesh-convergence; boundary: A 35 µm-thick microstrip at
//   w/h = 1.75: halving the edge size three times changes Z0 by successively smaller
//   amounts, the last below 0.1 %, and doubling the enclosure changes Z0 by under 0.05 %.
#[test]
fn microstrip_impedance_converges_with_the_mesh_and_enclosure() {
    let base = microstrip(0.35 * MM, 0.2 * MM, 35.0 * UM, 4.1);
    let mut previous: Option<f64> = None;
    let mut changes = Vec::new();
    for level in 0..4 {
        let mut section = base.clone();
        section.mesh.edge_size_m = Some(35.0 * UM / (5.0 * 2f64.powi(level)));
        let line = solve_cross_section(&section).unwrap();
        let z0 = line.impedance_ohm[0];
        if let Some(previous) = previous {
            changes.push(relative(z0, previous).abs());
        }
        eprintln!(
            "convergence level {level}: edge {:.3e} m, nodes {}, Z0 {z0:.5}",
            line.mesh.edge_size_m, line.mesh.nodes
        );
        previous = Some(z0);
    }
    eprintln!("successive changes {changes:?}");
    assert!(changes.windows(2).all(|pair| pair[1] < pair[0]), "{changes:?}");
    assert!(*changes.last().unwrap() < 1e-3, "{changes:?}");
    let default = solve_cross_section(&base).unwrap().impedance_ohm[0];
    let mut wide = base.clone();
    let enclosure = geometry::Geometry::build(&base).unwrap();
    wide.enclosure = Some(Enclosure {
        half_width_m: enclosure.x_high - enclosure.x_low,
        air_m: 2.0 * geometry::ENCLOSURE_HEIGHTS * enclosure.height,
    });
    let doubled = solve_cross_section(&wide).unwrap().impedance_ohm[0];
    eprintln!("enclosure: default {default:.5}, doubled {doubled:.5}");
    assert!(relative(doubled, default).abs() < 5e-4, "{default} {doubled}");
}

// witness: FEA-EM-LINE-001/pair_parameters @ quasi_static_tem/tri3/numerical_kernel;
//   boundary: Surface microstrip pair: inhomogeneous dielectric gives distinct odd and even
//   εeff, negative far-end coupling (inductive coupling dominates), Zdiff below twice the
//   single-ended Z0, and wider spacing moves Zdiff toward 2 Z0.
#[test]
fn microstrip_pair_couples_as_an_inhomogeneous_line() {
    let (w, h, t, er) = (0.15 * MM, 0.1 * MM, 35.0 * UM, 4.1);
    let single = solve_cross_section(&microstrip(w, h, t, er)).unwrap().impedance_ohm[0];
    let mut previous = 0.0;
    for gap in [0.1 * MM, 0.2 * MM, 0.4 * MM] {
        let mut section = microstrip(w, h, t, er);
        let pitch = 0.5 * (w + gap);
        section.conductors = vec![trace("p", "F.Cu", -pitch, w), trace("n", "F.Cu", pitch, w)];
        let line = solve_cross_section(&section).unwrap();
        let pair = line.pair.unwrap();
        eprintln!("microstrip pair gap {gap}: {pair:?}");
        assert!(pair.differential_impedance_ohm < 2.0 * single);
        assert!(pair.differential_impedance_ohm > previous);
        previous = pair.differential_impedance_ohm;
        assert!(pair.odd_mode_effective_permittivity < pair.even_mode_effective_permittivity);
        assert!(pair.far_end_coupling < 0.0);
        assert!(pair.near_end_coupling > 0.0 && pair.near_end_coupling < 0.25);
        // Reciprocity of the per-length matrices.
        assert_eq!(line.capacitance_f_per_m[0][1], line.capacitance_f_per_m[1][0]);
        assert!(line.capacitance_f_per_m[0][1] < 0.0);
        // Symmetric pair: both lines see the same Z0 and εeff.
        assert!(relative(line.impedance_ohm[0], line.impedance_ohm[1]).abs() < 1e-6);
    }
}

// witness: FEA-EM-LINE-001/cross_section_inputs @ quasi_static_tem/tri3/numerical_kernel;
//   boundary: Solder mask, etch angle and coplanar ground each move Z0 the physical way
//   (mask and coplanar ground lower it, a sloped sidewall raises it), the stackup's inner
//   copper fill follows the dielectric above, and malformed stackups are rejected.
#[test]
fn stackup_inputs_change_impedance_the_physical_way() {
    let (w, h, t, er) = (0.35 * MM, 0.2 * MM, 35.0 * UM, 4.1);
    let bare = solve_cross_section(&microstrip(w, h, t, er)).unwrap().impedance_ohm[0];
    let mut masked = microstrip(w, h, t, er);
    masked.layers.insert(0, mask("F.Mask", 20.0 * UM, 3.8));
    let masked = solve_cross_section(&masked).unwrap().impedance_ohm[0];
    let mut etched = microstrip(w, h, t, er);
    etched.conductors[0].etch_angle_rad = 45f64.to_radians();
    let etched = solve_cross_section(&etched).unwrap().impedance_ohm[0];
    let mut coplanar = microstrip(w, h, t, er);
    for (name, center) in [("gl", -(w + 0.2 * MM)), ("gr", w + 0.2 * MM)] {
        let mut ground = trace(name, "F.Cu", center, w);
        ground.role = ConductorRole::Ground;
        coplanar.conductors.push(ground);
    }
    let coplanar = solve_cross_section(&coplanar).unwrap().impedance_ohm[0];
    eprintln!("bare {bare:.3}, masked {masked:.3}, etched 45° {etched:.3}, coplanar {coplanar:.3}");
    assert!(masked < bare && masked > 0.9 * bare, "{masked} {bare}");
    assert!(etched > bare, "{etched} {bare}");
    assert!(coplanar < bare, "{coplanar} {bare}");
}

#[test]
fn malformed_cross_sections_are_rejected() {
    let good = microstrip(0.3 * MM, 0.2 * MM, 35.0 * UM, 4.1);
    let reject = |section: CrossSection, needle: &str| {
        let error = solve_cross_section(&section).unwrap_err().to_string();
        assert!(error.contains(needle), "{error}");
    };
    let mut overlap = good.clone();
    overlap.conductors.push(trace("b", "F.Cu", 0.1 * MM, 0.3 * MM));
    reject(overlap, "overlap");
    let mut no_return = good.clone();
    no_return.layers.pop();
    reject(no_return, "return conductor");
    let mut steep = good.clone();
    steep.conductors[0].etch_angle_rad = 0.05;
    reject(steep, "no far face");
    let mut missing = good.clone();
    missing.conductors[0].layer = "In7.Cu".into();
    reject(missing, "not in the stackup");
    let mut plane = good.clone();
    plane.conductors[0].layer = "GND".into();
    reject(plane, "signal copper layer");
    let mut low = good.clone();
    low.layers[1] = dielectric("core", 0.2 * MM, 0.5);
    reject(low, "at least 1");
    let mut growth = good;
    growth.mesh.growth = 1.0;
    reject(growth, "growth");
}

/// The sections of docs/simulation/examples/trace-impedance.fea: 1 oz top copper under
/// 20 µm of mask on 0.2 mm of Dk 4.1 prepreg over the In1.Cu plane, 60° etch.
fn example_section(traces: &[(&str, f64, f64)]) -> CrossSection {
    CrossSection {
        layers: vec![
            mask("F.Mask", 20.0 * UM, 3.8),
            copper("F.Cu", 35.0 * UM, CopperRole::Signal),
            dielectric("dielectric 1", 0.2 * MM, 4.1),
            copper("In1.Cu", 35.0 * UM, CopperRole::ReferencePlane),
        ],
        conductors: traces
            .iter()
            .map(|&(name, center, width)| Conductor {
                etch_angle_rad: 60f64.to_radians(),
                ..trace(name, "F.Cu", center, width)
            })
            .collect(),
        mesh: CrossSectionMeshControls::default(),
        enclosure: None,
    }
}

#[test]
fn site_example_sections_meet_their_targets() {
    let start = std::time::Instant::now();
    let single = solve_cross_section(&example_section(&[("line", 0.0, 0.36 * MM)])).unwrap();
    let single_time = start.elapsed();
    let start = std::time::Instant::now();
    let pair = solve_cross_section(&example_section(&[
        ("p", -0.2 * MM, 0.23 * MM),
        ("n", 0.2 * MM, 0.23 * MM),
    ]))
    .unwrap();
    let pair_time = start.elapsed();
    let parameters = pair.pair.unwrap();
    eprintln!(
        "example: Z0 {:.2} Ω, εeff {:.4}, delay {:.2} ps/mm, nodes {}, {single_time:?}; pair Zdiff {:.2} Ω, Zcommon {:.2} Ω, Kb {:.4}, Kf {:.5}, nodes {}, {pair_time:?}",
        single.impedance_ohm[0],
        single.effective_permittivity[0],
        single.delay_s_per_m[0] * 1e9,
        single.mesh.nodes,
        parameters.differential_impedance_ohm,
        parameters.common_mode_impedance_ohm,
        parameters.near_end_coupling,
        parameters.far_end_coupling,
        pair.mesh.nodes,
    );
    eprintln!("example pair C {:?} L {:?}", pair.capacitance_f_per_m, pair.inductance_h_per_m);
    assert!((single.impedance_ohm[0] - 50.0).abs() < 2.5);
    assert!((parameters.differential_impedance_ohm - 100.0).abs() < 5.0);
}

/// A checked `cross_section` definition maps field for field onto the solver section.
#[test]
fn authored_definition_maps_onto_the_solver_section() {
    use runmat_geometry_core::{
        CrossSectionDefinition, CrossSectionLayer, CrossSectionLayerRole, CrossSectionTrace,
        CrossSectionTraceRole,
    };
    let layer = |name: &str, thickness_m: f64, role| CrossSectionLayer {
        name: name.into(),
        thickness_m,
        role,
    };
    let definition = CrossSectionDefinition {
        layers: vec![
            layer(
                "F.Mask",
                20.0 * UM,
                CrossSectionLayerRole::SolderMask { permittivity: 3.8 },
            ),
            layer(
                "F.Cu",
                35.0 * UM,
                CrossSectionLayerRole::Signal {
                    fill_permittivity: None,
                },
            ),
            layer(
                "dielectric 1",
                0.2 * MM,
                CrossSectionLayerRole::Dielectric { permittivity: 4.1 },
            ),
            layer("In1.Cu", 35.0 * UM, CrossSectionLayerRole::Plane),
        ],
        stackup: None,
        traces: vec![CrossSectionTrace {
            name: "line".into(),
            role: CrossSectionTraceRole::Signal,
            layer: "F.Cu".into(),
            center_m: 0.0,
            width_m: 0.36 * MM,
            etch_angle_rad: 60f64.to_radians(),
            base: None,
        }],
        enclosure: None,
    };
    definition.validate().unwrap();
    assert_eq!(
        CrossSection::from_authored(&definition, None),
        example_section(&[("line", 0.0, 0.36 * MM)])
    );
    let controls = runmat_meshing_core::CrossSectionMeshSettings {
        edge_size_m: Some(1.0 * UM),
        growth: Some(1.2),
        max_size_m: None,
    };
    let section = CrossSection::from_authored(&definition, Some(&controls));
    assert_eq!(section.mesh.edge_size_m, Some(1.0 * UM));
    assert_eq!(section.mesh.growth, 1.2);
}

#[test]
#[ignore = "calibration sweep for the default mesh controls; prints a table"]
fn calibrate_mesh_defaults() {
    let b = 0.5 * MM;
    let h = 0.2 * MM;
    for growth in [1.15, 1.1, 1.05] {
        for divisions in [200.0, 1000.0, 5000.0] {
            let mut strip = stripline(&[(0.0, 0.2 * b)], b, 0.0, 3.0);
            strip.mesh.growth = growth;
            strip.mesh.edge_size_m = Some(0.2 * b / divisions);
            let start = std::time::Instant::now();
            let s = solve_cross_section(&strip).unwrap();
            let es = relative(s.impedance_ohm[0], cohn_stripline(0.2 * b, b, 3.0));
            let mut micro = microstrip(0.5 * h, h, 0.0, 4.4);
            micro.mesh.growth = growth;
            micro.mesh.edge_size_m = Some(0.5 * h / divisions);
            let m = solve_cross_section(&micro).unwrap();
            let em = relative(m.impedance_ohm[0], hammerstad_jensen(0.5, 4.4).0);
            let mut thick = microstrip(0.35 * MM, h, 35.0 * UM, 4.1);
            thick.mesh.growth = growth;
            thick.mesh.edge_size_m = Some(35.0 * UM / divisions);
            let t = solve_cross_section(&thick).unwrap();
            eprintln!(
                "CAL g={growth} div={divisions}: cohn {:+.3} % ({} nodes), hj {:+.3} % ({} nodes), thick Z0 {:.4} ({} nodes), {:?}",
                100.0 * es, s.mesh.nodes, 100.0 * em, m.mesh.nodes, t.impedance_ohm[0], t.mesh.nodes, start.elapsed()
            );
        }
    }
}

/// The published mesh is exactly the mesh the potential solves ran on, as one canonical
/// surface artifact whose geometry identity is the section digest.
#[test]
fn solved_section_publishes_its_mesh_as_one_canonical_artifact() {
    use runmat_meshing_core::{CanonicalMeshingContract, MeshDimension, StableDigest};
    let section = example_section(&[("p", -0.2 * MM, 0.23 * MM), ("n", 0.2 * MM, 0.23 * MM)]);
    let tolerance = runmat_meshing_core::GeometryTolerancePolicy {
        source_tolerance_m: runmat_types::physical::Length::new(0.0),
        absolute_floor_m: runmat_types::physical::Length::new(1.0e-10),
        model_relative_term: 1.0e-9,
        requested_deviation_m: runmat_types::physical::Length::new(1.0e-5),
        maximum_healing_displacement_m: runmat_types::physical::Length::new(1.0e-6),
    };
    let request = runmat_meshing_core::resolve_meshing_request(
        tolerance,
        runmat_meshing_core::MeshingRequestSettings::default(),
    )
    .unwrap();
    let digest = StableDigest::from_bytes([7; 32]);
    let solved = solve_cross_section_with_mesh(&section, digest, request).unwrap();
    let reference = solve_cross_section(&section).unwrap();
    assert_eq!(solved.parameters, reference);
    let mesh = &solved.mesh;
    mesh.validate_canonical().unwrap();
    assert_eq!(mesh.resolved_request.dimension, MeshDimension::Surface);
    assert_eq!(mesh.topology.nodes.len(), reference.mesh.nodes);
    assert_eq!(mesh.topology.boundary_faces.len(), reference.mesh.elements);
    assert_eq!(mesh.geometry, cross_section_geometry(digest));
    assert!(mesh.topology.nodes.iter().all(|node| node.coordinates_m.canonical_value()[2] == 0.0));
    let platform = runmat_meshing_core::PlatformBuildIdentity {
        capability_cohort: "test-native".into(),
        target_triple: "test-target".into(),
        build_digest: StableDigest::from_bytes([9; 32]),
        exact_kernel_abi: None,
    };
    let evidence = cross_section_meshing_evidence(
        mesh,
        runmat_meshing_core::MeshingExecutionCeilings::PORTABLE_DEFAULT,
        platform,
        solved.meshing_time_ms,
    )
    .unwrap();
    evidence.validate(mesh).unwrap();
}

/// Every authored line-parameter output becomes one observation result, valid under the
/// step's plan identity; scalar outputs feed assessments as scalar evidence.
#[test]
fn line_parameter_outputs_publish_one_observation_result_each() {
    use runmat_analysis_core::{
        AnalysisProcedure, AnalysisSequence, AuthoredOutput, AuthoredOutputExpression,
        AuthoredOutputPolicy, IndependentDomain, ObservationQuantity, PhysicsFamily,
        ProcedureStep, SolutionSpace,
    };
    use runmat_meshing_core::StableDigest;
    let section = example_section(&[("p", -0.2 * MM, 0.23 * MM), ("n", 0.2 * MM, 0.23 * MM)]);
    let request = runmat_meshing_core::resolve_meshing_request(
        runmat_meshing_core::GeometryTolerancePolicy {
            source_tolerance_m: runmat_types::physical::Length::new(0.0),
            absolute_floor_m: runmat_types::physical::Length::new(1.0e-10),
            model_relative_term: 1.0e-9,
            requested_deviation_m: runmat_types::physical::Length::new(1.0e-5),
            maximum_healing_displacement_m: runmat_types::physical::Length::new(1.0e-6),
        },
        runmat_meshing_core::MeshingRequestSettings::default(),
    )
    .unwrap();
    let solved =
        solve_cross_section_with_mesh(&section, StableDigest::from_bytes([7; 32]), request)
            .unwrap();
    let step = ProcedureStep::new(
        "lines",
        PhysicsFamily::Electromagnetic,
        AnalysisProcedure::LineParameters,
        SolutionSpace::Direct,
        IndependentDomain::Static,
        vec![runmat_analysis_core::StateOutput {
            state_id: runmat_analysis_core::StateId("primary".into()),
            kind: runmat_analysis_core::StateKind::ElectromagneticStatic,
            completeness: runmat_analysis_core::StateCompleteness::Complete,
            persistence: runmat_analysis_core::StatePersistence::UntilDependentsComplete,
        }],
    );
    let mut sequence = AnalysisSequence::from_steps("lines", vec![step]);
    let output = |id: &str, quantity, conductor: Option<&str>| AuthoredOutput {
        output_id: id.into(),
        expression: AuthoredOutputExpression::LineParameter {
            step_id: "lines".into(),
            quantity,
            conductor: conductor.map(Into::into),
            signal_conductors: 2,
        },
        policy: AuthoredOutputPolicy::default(),
    };
    let outputs = vec![
        output("z_p", ObservationQuantity::LineImpedance, Some("p")),
        output("zdiff", ObservationQuantity::DifferentialImpedance, None),
        output("c", ObservationQuantity::LineCapacitance, None),
    ];
    runmat_analysis_core::attach_authored_outputs(&outputs, &mut sequence).unwrap();
    let step = &sequence.steps[0];
    let results = crate::results::observation::evaluate_line_parameter_observations(
        step,
        &solved.mesh,
        &solved.parameters,
    )
    .unwrap();
    crate::results::observation::validate_line_parameter_observations(step, &solved.mesh, &results)
        .unwrap();
    assert_eq!(results.len(), 3);
    let value = |index: usize| results[index].field.as_host_f64().unwrap().to_vec();
    assert_eq!(value(0), vec![solved.parameters.impedance_ohm[0]]);
    assert_eq!(
        value(1),
        vec![solved.parameters.pair.unwrap().differential_impedance_ohm]
    );
    assert_eq!(value(2).len(), 4);
    let scalar = crate::results::assessment::scalar_observation_evidence(&results[1]).unwrap();
    assert!((scalar.value - 100.0).abs() < 5.0);
    // A result from another mesh does not satisfy this plan.
    let mut moved = solved.mesh.clone();
    moved.canonical_digest = StableDigest::from_bytes([3; 32]);
    assert!(crate::results::observation::validate_line_parameter_observations(
        step, &moved, &results
    )
    .is_err());
}
