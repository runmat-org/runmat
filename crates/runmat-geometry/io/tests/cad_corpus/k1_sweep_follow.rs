//! Sweeps whose profile follows the path, on the OCCT kernel: pipe runs that turn through a
//! corner or an arc into the profile's own plane. A section centred on the path sweeps
//! A × (path length) exactly, mitred corners and arcs included (Pappus), so volume is the
//! closed-form oracle; the history names every side by its path curve and both caps.

use crate::k1_ops::{close, end_cap, face_of, new_tool, rect, start_cap, volume};
use crate::kernel_under_test::KernelUnderTest;
use runmat_cad::regen::kernel::*;
use runmat_geometry_kernel::frame::Frame;
use runmat_geometry_kernel::history::{HistorySource, KernelTag};
use runmat_geometry_kernel::ids::SketchEntityId;
use runmat_geometry_kernel::names::Role;

const A: f64 = 0.01;

fn side_along(curve: &str, path_curve: u32) -> HistorySource {
    HistorySource::Generated {
        role: Role::Side,
        tag: KernelTag::Curve {
            profile: 0,
            curve: SketchEntityId::new(curve),
        },
        ordinal: Some(path_curve),
        seed: None,
        instance: None,
    }
}

fn follow(path: Vec<[f64; 3]>, path_curves: Vec<u32>) -> KernelOp {
    new_tool(ToolShape::Sweep {
        profile: rect(Frame::WORLD, -A, -A, A, A),
        path,
        path_curves,
        follow_path: true,
    })
}

fn centroid(b: &KernelBody, face: u32) -> [f64; 3] {
    b.topology.faces[face as usize].geom.centroid
}

fn near(p: [f64; 3], q: [f64; 3]) -> bool {
    (0..3).all(|i| (p[i] - q[i]).abs() < 1e-7)
}

#[test]
fn k1_follow_sweep_turns_a_corner_into_the_profile_plane() {
    let k = KernelUnderTest::new();
    // Up 0.2 m, then 0.3 m along x: the second segment lies in the profile's plane.
    let out = k
        .execute(
            &follow(
                vec![[0.0, 0.0, 0.0], [0.0, 0.0, 0.2], [0.3, 0.0, 0.2]],
                vec![],
            ),
            &[],
        )
        .expect("an L-shaped run");
    assert_eq!(out.bodies.len(), 1);
    let body = &out.bodies[0];
    let (v, _) = volume(&k, body);
    let want = 4.0 * A * A * 0.5;
    assert!(close(v, want, 1e-6), "{v} vs {want}");
    for c in ["c1", "c2", "c3", "c4"] {
        face_of(&out, 0, &side_along(c, 0));
        face_of(&out, 0, &side_along(c, 1));
    }
    assert!(near(
        centroid(body, face_of(&out, 0, &start_cap())),
        [0.0, 0.0, 0.0]
    ));
    assert!(near(
        centroid(body, face_of(&out, 0, &end_cap())),
        [0.3, 0.0, 0.2]
    ));
    assert_eq!(body.topology.faces.len(), 10);
}

#[test]
fn k1_follow_sweep_through_line_arc_line() {
    let k = KernelUnderTest::new();
    // Up 0.2 m, a quarter bend of radius 0.05 m, then 0.35 m along x.
    let r = 0.05;
    let mut path = vec![[0.0, 0.0, 0.0], [0.0, 0.0, 0.2]];
    let mut curves = vec![0];
    let n = 8;
    for i in 1..=n {
        let t = std::f64::consts::FRAC_PI_2 * i as f64 / n as f64;
        path.push([r - r * t.cos(), 0.0, 0.2 + r * t.sin()]);
        curves.push(1);
    }
    path.push([r + 0.35, 0.0, 0.2 + r]);
    curves.push(2);
    let out = k.execute(&follow(path, curves), &[]).expect("a bent run");
    assert_eq!(out.bodies.len(), 1);
    let body = &out.bodies[0];
    let (v, _) = volume(&k, body);
    let want = 4.0 * A * A * (0.2 + std::f64::consts::FRAC_PI_2 * r + 0.35);
    assert!(close(v, want, 1e-6), "{v} vs {want}");
    for c in ["c1", "c2", "c3", "c4"] {
        for pc in 0..3 {
            face_of(&out, 0, &side_along(c, pc));
        }
    }
    assert!(near(
        centroid(body, face_of(&out, 0, &end_cap())),
        [r + 0.35, 0.0, 0.2 + r]
    ));
}

#[test]
fn k1_follow_sweep_along_a_straight_path_is_the_prism() {
    let k = KernelUnderTest::new();
    let out = k
        .execute(&follow(vec![[0.0, 0.0, 0.0], [0.0, 0.0, 0.2]], vec![]), &[])
        .expect("a straight run");
    let (v, _) = volume(&k, &out.bodies[0]);
    assert!(close(v, 4.0 * A * A * 0.2, 1e-9));
    for c in ["c1", "c2", "c3", "c4"] {
        face_of(
            &out,
            0,
            &HistorySource::Generated {
                role: Role::Side,
                tag: KernelTag::Curve {
                    profile: 0,
                    curve: SketchEntityId::new(c),
                },
                ordinal: None,
                seed: None,
                instance: None,
            },
        );
    }
}
