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
