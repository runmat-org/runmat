//! A part whose profile comes from `.m` functions, run in a native RunMat session: the disc cam
//! (tests/fixtures/disc_cam) sizes its base circle and its station radii from its motion law.
#![cfg(feature = "m-values")]

use runmat_design_tools::inspect::m_values::call_with_m_values;
use runmat_design_tools::Files;
use serde_json::json;

fn files() -> Files {
    [
        ("disc_cam.part", include_str!("fixtures/disc_cam/disc_cam.part")),
        ("camBaseRadius.m", include_str!("fixtures/disc_cam/camBaseRadius.m")),
        ("camRadius.m", include_str!("fixtures/disc_cam/camRadius.m")),
        ("cycloidal.m", include_str!("fixtures/disc_cam/cycloidal.m")),
        ("runmat.toml", include_str!("fixtures/disc_cam/runmat.toml")),
    ]
    .into_iter()
    .map(|(p, t)| (format!("/cam/{p}"), t.to_string()))
    .collect()
}

/// The cam's motion law, written again here: the area inside its pitch curve,
/// ½∮r(θ)² dθ, integrated finely.
fn motion_law_area(lift: f64) -> f64 {
    let (rise, dwell, pa) = (150f64.to_radians(), 30f64.to_radians(), 30f64.to_radians());
    let base = (2.0 * lift / rise) / pa.tan() - lift / 2.0;
    let cyc = |u: f64| u - (2.0 * std::f64::consts::PI * u).sin() / (2.0 * std::f64::consts::PI);
    let r = |t: f64| {
        base + if t < rise {
            cyc(t / rise) * lift
        } else if t < rise + dwell {
            lift
        } else if t < 2.0 * rise + dwell {
            (1.0 - cyc((t - rise - dwell) / rise)) * lift
        } else {
            0.0
        }
    };
    let n = 20_000;
    let dt = 2.0 * std::f64::consts::PI / n as f64;
    (0..n).map(|i| 0.5 * r(i as f64 * dt).powi(2) * dt).sum()
}

#[test]
fn the_disc_cam_regenerates_from_its_motion_law() {
    let files = files();
    let mut volumes = Vec::new();
    for lift in [8.0, 10.0] {
        let out = call_with_m_values(
            "design.inspect",
            json!({ "path": "/cam/disc_cam.part", "sections": ["mass"], "set": { "lift": format!("{lift} mm") } }),
            &files,
        )
        .unwrap_or_else(|e| panic!("{e:?}"));
        let v = out["sections"]["mass"]["volume_mm3"].as_f64().unwrap_or_else(|| panic!("{out}"));
        // The spline through 12 stations follows the motion law's pitch curve to within 5% in area
        // (it is 3.2% over at 8 mm of lift); a wrong base radius or lift is off by far more.
        let law = motion_law_area(lift * 1e-3) * 1e6 * 10.0;
        assert!((v - law).abs() < 0.05 * law, "lift {lift} mm: volume {v} mm³, motion-law prism {law} mm³");
        volumes.push(v);
    }
    assert!(volumes[1] > volumes[0]);
}

/// The io circuit with one component's value from `rcResistor.m` (PCB-38).
fn circuit_files() -> Files {
    const CALL: &str = "rcResistor(fc = 160 Hz, c = 100 nF) as ohm";
    let projects = concat!(env!("CARGO_MANIFEST_DIR"), "/../runmat-circuit/tests/projects/io");
    let read = |name: &str| std::fs::read_to_string(format!("{projects}/{name}")).expect(name);
    let circuit = read("io.circuit").replacen(
        "component RS = { part = \"R0402-10K\" }",
        &format!("component RS = {{ part = \"R0402-10K\", value = {CALL} }}"),
        1,
    );
    assert!(circuit.contains(CALL), "the fixture has component RS");
    [
        ("io.circuit", circuit),
        ("io.circuit.layout", read("io.circuit.layout")),
        ("io.circuit.lock", read("io.circuit.lock")),
        ("rcResistor.m", "function r = rcResistor(fc, c)\n    r = 1 / (2 * pi * fc * c);\nend\n".to_owned()),
    ]
    .into_iter()
    .map(|(p, t)| (format!("/p/{p}"), t))
    .collect()
}

#[test]
fn the_circuit_checks_run_a_components_m_value_natively() {
    use runmat_circuit::runtime::values::PENDING_VALUES;
    use runmat_design_tools::ci::check_circuit;
    use runmat_design_tools::circuit_values::circuit_options;
    use runmat_design_tools::inspect::m_values::with_m_values;
    use runmat_design_syntax::m_calls::TableResult;
    let files = circuit_files();
    let pending = |c: &runmat_design_tools::ci::CircuitCheck| c.findings.iter().any(|f| f.message.contains(PENDING_VALUES));
    // Without a runtime the value waits for one.
    let bare = check_circuit("/p/io.circuit", &files, false).expect("check");
    assert!(pending(&bare), "{:?}", bare.findings);
    // With one, the call runs and the checks see the value.
    let ran = with_m_values(&files, || check_circuit("/p/io.circuit", &files, false)).expect("check");
    assert!(!pending(&ran), "{:?}", ran.findings);
    assert!(ran.findings.iter().all(|f| f.check != "compile" || f.severity != "error"), "{:?}", ran.findings);
    let options = with_m_values(&files, || circuit_options("/p/io.circuit", &files["/p/io.circuit"], &files));
    let r = 1.0 / (2.0 * std::f64::consts::PI * 160.0 * 100e-9);
    let values: Vec<&TableResult> = options.values.values().collect();
    assert_eq!(values.len(), 1);
    match values[0] {
        TableResult::Values(v) => assert!((v[0] - r).abs() < 1e-6 * r, "{v:?}"),
        other => panic!("{other:?}"),
    }
}

/// `runmat-design check` on a circuit whose component value calls `.m` code: the call runs,
/// so the check goes on past the compile (ERC and design review run) with no value waiting.
#[test]
fn runmat_design_check_runs_a_circuits_m_values() {
    let dir = std::env::temp_dir().join(format!("runmat-design-circuit-m-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    for (path, text) in circuit_files() {
        std::fs::write(dir.join(path.trim_start_matches("/p/")), text).unwrap();
    }
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_runmat-design"))
        .args(["check", "--json", "--no-simulate"])
        .arg(dir.join("io.circuit"))
        .output()
        .unwrap();
    let report = String::from_utf8_lossy(&out.stdout);
    assert!(!report.contains(runmat_circuit::runtime::values::PENDING_VALUES), "{report}");
    let reports: serde_json::Value = serde_json::from_str(&report).unwrap_or_else(|e| panic!("{e}: {report}"));
    let ran = reports[0]["ran"].as_array().cloned().unwrap_or_default();
    assert!(ran.iter().any(|r| r == "erc"), "the checks past the compile ran: {ran:?}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// The RC filter fixture (tests/fixtures/rc_filter): R1's value comes from rcResistor.m, and
/// the simulated corner lands where the function sized it.
#[test]
fn the_rc_filter_corner_lands_where_its_m_function_sized_the_resistor() {
    let files: Files = [
        ("lowpass.circuit", include_str!("fixtures/rc_filter/lowpass.circuit")),
        ("rcResistor.m", include_str!("fixtures/rc_filter/rcResistor.m")),
        ("runmat.toml", include_str!("fixtures/rc_filter/runmat.toml")),
    ]
    .into_iter()
    .map(|(p, t)| (format!("/rc/{p}"), t.to_string()))
    .collect();
    let check = call_with_m_values("design.check", json!({ "path": "/rc/lowpass.circuit" }), &files).expect("check");
    assert_eq!(check["errors"], json!(0), "{check}");
    let out = call_with_m_values("design.simulate", json!({ "path": "/rc/lowpass.circuit", "simulation": "corner" }), &files).expect("simulate");
    let f3db = out["measures"].as_array().and_then(|m| m.iter().find(|m| m["name"] == "f3db")).cloned().unwrap_or_default();
    // 1/(2π·R·C) with R from rcResistor.m is the 160 Hz it was sized for; the part's own
    // 10 kΩ would put the corner at 159.15 Hz.
    let corner = f3db["value"].as_f64().unwrap_or_else(|| panic!("{out}"));
    assert!((corner - 160.0).abs() < 0.2, "corner {corner} Hz");
    assert_eq!(f3db["pass"], json!(true));
}
