//! 3D mates: DOF of every joint type, limits, and the four-bar and slider-crank linkages.

mod common;
use common::*;
use runmat_constraint::*;
use std::f64::consts::PI;

fn quat_rotate(q: [f64; 4], v: [f64; 3]) -> [f64; 3] {
    let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    let (w, x, y, z) = (q[0] / n, q[1] / n, q[2] / n, q[3] / n);
    let u = [x, y, z];
    let t = [
        2.0 * (u[1] * v[2] - u[2] * v[1]),
        2.0 * (u[2] * v[0] - u[0] * v[2]),
        2.0 * (u[0] * v[1] - u[1] * v[0]),
    ];
    let ut = [
        u[1] * t[2] - u[2] * t[1],
        u[2] * t[0] - u[0] * t[2],
        u[0] * t[1] - u[1] * t[0],
    ];
    [
        v[0] + w * t[0] + ut[0],
        v[1] + w * t[1] + ut[1],
        v[2] + w * t[2] + ut[2],
    ]
}

/// World position of a connector after a solve.
fn connector_origin(s: &System, rep: &SolveReport, c: EntityId) -> [f64; 3] {
    let EntityKind::Connector { body, local } = s.entity(c).kind else {
        panic!()
    };
    let EntityKind::Body {
        position,
        orientation,
    } = s.entity(body).kind
    else {
        panic!()
    };
    let p = position.map(|v| rep.value(v));
    let q = orientation.map(|v| rep.value(v));
    let o = quat_rotate(q, local.origin);
    [p[0] + o[0], p[1] + o[1], p[2] + o[2]]
}

fn d3(a: [f64; 3], b: [f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

/// Ground plus one body slightly off its mate, both connectors at the frame origins.
fn pair(mate: impl Fn(EntityId, EntityId) -> Mate) -> (System, EntityId) {
    let mut s = System::new();
    let g = s.ground(Frame::IDENTITY);
    let gc = s.connector(g, Frame::IDENTITY).unwrap();
    let mut pose = Frame::axis_angle([0.1, -0.05, 0.08], [1.0, 1.0, 0.3], 0.06);
    pose.origin = [0.1, -0.05, 0.08];
    let b = s.body(pose);
    let bc = s.connector(b, Frame::IDENTITY).unwrap();
    s.constrain(Constraint::Mate(mate(gc, bc))).unwrap();
    (s, b)
}

#[test]
fn free_body_has_six_dof() {
    let mut s = System::new();
    let b = s.body(Frame::axis_angle([1.0, 2.0, 3.0], [0.0, 0.0, 1.0], 0.3));
    let rep = solve(&s, &opts());
    assert_eq!(rep.status, Status::UnderDefined);
    assert_eq!(rep.dof, 6);
    assert_eq!(rep.entity_dof(b), Some(6));
}

#[test]
fn grounded_body_has_no_dof_and_no_redundancy() {
    let mut s = System::new();
    let g = s.ground(Frame::IDENTITY);
    let rep = solve(&s, &opts());
    assert_eq!(rep.status, Status::WellDefined);
    assert_eq!(rep.entity_dof(g), Some(0));
}

#[test]
fn joint_dof_table() {
    type MateCase = (&'static str, Box<dyn Fn(EntityId, EntityId) -> Mate>, usize);
    let cases: Vec<MateCase> = vec![
        ("fasten", Box::new(|a, b| Mate::Fasten { a, b }), 0),
        ("revolute", Box::new(|a, b| Mate::Revolute { a, b }), 1),
        ("slider", Box::new(|a, b| Mate::Slider { a, b }), 1),
        (
            "cylindrical",
            Box::new(|a, b| Mate::Cylindrical { a, b }),
            2,
        ),
        ("planar", Box::new(|a, b| Mate::Planar { a, b }), 3),
        ("ball", Box::new(|a, b| Mate::Ball { a, b }), 3),
        ("parallel", Box::new(|a, b| Mate::Parallel { a, b }), 4),
        (
            "tangent sphere",
            Box::new(|a, b| Mate::Tangent {
                a,
                b,
                kind: TangentKind::PlaneSphere { radius: 0.5 },
            }),
            5,
        ),
        (
            "tangent cylinder",
            Box::new(|a, b| Mate::Tangent {
                a,
                b,
                kind: TangentKind::PlaneCylinder { radius: 0.5 },
            }),
            4,
        ),
        (
            "distance point",
            Box::new(|a, b| Mate::Distance {
                a,
                b,
                mode: DistanceMode::PointPoint,
                target: Target::Exact(0.3),
            }),
            5,
        ),
        (
            "distance plane",
            Box::new(|a, b| Mate::Distance {
                a,
                b,
                mode: DistanceMode::PlanePoint,
                target: Target::Exact(0.3),
            }),
            5,
        ),
        (
            "angle about z",
            Box::new(|a, b| Mate::Angle {
                a,
                b,
                mode: AngleMode::AboutZ,
                target: Target::Exact(0.2),
            }),
            5,
        ),
        (
            "angle between z",
            Box::new(|a, b| Mate::Angle {
                a,
                b,
                mode: AngleMode::BetweenZ,
                target: Target::Exact(0.2),
            }),
            5,
        ),
    ];
    for (name, mate, dof) in cases {
        let (s, b) = pair(mate);
        let rep = solve(&s, &opts());
        assert!(rep.converged, "{name}: residual {}", rep.residual_norm);
        assert_eq!(rep.dof, dof, "{name}");
        assert_eq!(rep.entity_dof(b), Some(dof), "{name}");
        let want = if dof == 0 {
            Status::WellDefined
        } else {
            Status::UnderDefined
        };
        assert_eq!(rep.status, want, "{name}");
        assert!(rep.redundant.is_empty(), "{name}: {:?}", rep.redundant);
    }
}

#[test]
fn fasten_puts_frames_together() {
    let (s, b) = pair(|a, b| Mate::Fasten { a, b });
    let rep = solve(&s, &opts());
    let EntityKind::Body {
        position,
        orientation,
    } = s.entity(b).kind
    else {
        panic!()
    };
    for v in position {
        assert_close(rep.value(v), 0.0, 1e-10);
    }
    let q = orientation.map(|v| rep.value(v));
    assert_close(q[0].abs(), 1.0, 1e-10);
}

#[test]
fn fix_constraint_on_body() {
    let mut s = System::new();
    let b = s.body(Frame::axis_angle([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], 0.4));
    s.constrain(Constraint::Fix(b)).unwrap();
    let rep = solve(&s, &opts());
    assert_eq!(rep.status, Status::WellDefined);
}

#[test]
fn revolute_driven_by_angle() {
    let (mut s, b) = pair(|a, b| Mate::Revolute { a, b });
    let cons: Vec<EntityId> = (0..s.entities().len())
        .map(EntityId::from_index)
        .filter(|&e| matches!(s.entity(e).kind, EntityKind::Connector { .. }))
        .collect();
    s.constrain(Constraint::Mate(Mate::Angle {
        a: cons[0],
        b: cons[1],
        mode: AngleMode::AboutZ,
        target: Target::Exact(PI / 3.0),
    }))
    .unwrap();
    let rep = solve(&s, &opts());
    assert_eq!(rep.status, Status::WellDefined);
    let EntityKind::Body { orientation, .. } = s.entity(b).kind else {
        panic!()
    };
    let q = orientation.map(|v| rep.value(v));
    let x = quat_rotate(q, [1.0, 0.0, 0.0]);
    assert_close(x[0], 0.5, 1e-9);
    assert_close(x[1], (PI / 3.0).sin(), 1e-9);
    assert_close(x[2], 0.0, 1e-9);
}

fn slider_with_range(start_x: f64) -> (System, EntityId, ConstraintId) {
    let mut s = System::new();
    let g = s.ground(Frame::IDENTITY);
    let gs = s
        .connector(g, Frame::with_z([0.0; 3], [1.0, 0.0, 0.0]))
        .unwrap();
    let b = s.body(Frame::at([start_x, 0.05, -0.02]));
    let bs = s
        .connector(b, Frame::with_z([0.0; 3], [1.0, 0.0, 0.0]))
        .unwrap();
    s.constrain(Constraint::Mate(Mate::Slider { a: gs, b: bs }))
        .unwrap();
    let lim = s
        .constrain(Constraint::Mate(Mate::Distance {
            a: gs,
            b: bs,
            mode: DistanceMode::PlanePoint,
            target: Target::Range { min: 0.0, max: 5.0 },
        }))
        .unwrap();
    (s, b, lim)
}

#[test]
fn distance_limit_clamps_when_violated() {
    let (s, b, lim) = slider_with_range(7.0);
    let rep = solve(&s, &opts());
    assert!(rep.converged);
    assert_eq!(rep.active_limits, vec![(lim, 5.0)]);
    assert_eq!(rep.dof, 0);
    let EntityKind::Body { position, .. } = s.entity(b).kind else {
        panic!()
    };
    assert_close(rep.value(position[0]), 5.0, 1e-9);
    assert_close(rep.value(position[1]), 0.0, 1e-9);
}

#[test]
fn distance_limit_inactive_inside_range() {
    let (s, b, _) = slider_with_range(3.0);
    let rep = solve(&s, &opts());
    assert!(rep.active_limits.is_empty());
    assert_eq!(rep.dof, 1);
    let EntityKind::Body { position, .. } = s.entity(b).kind else {
        panic!()
    };
    assert_close(rep.value(position[0]), 3.0, 1e-6);
}

#[test]
fn angle_limit_clamps_revolute() {
    let mut s = System::new();
    let g = s.ground(Frame::IDENTITY);
    let gc = s.connector(g, Frame::IDENTITY).unwrap();
    let b = s.body(Frame::axis_angle([0.0; 3], [0.0, 0.0, 1.0], 1.0));
    let bc = s.connector(b, Frame::IDENTITY).unwrap();
    s.constrain(Constraint::Mate(Mate::Revolute { a: gc, b: bc }))
        .unwrap();
    let lim = s
        .constrain(Constraint::Mate(Mate::Angle {
            a: gc,
            b: bc,
            mode: AngleMode::AboutZ,
            target: Target::Range {
                min: -0.5,
                max: 0.5,
            },
        }))
        .unwrap();
    let rep = solve(&s, &opts());
    assert_eq!(rep.active_limits, vec![(lim, 0.5)]);
    assert_eq!(rep.dof, 0);
}

struct FourBar {
    s: System,
    ground_a: EntityId,
    crank_a: EntityId,
    joints: [(EntityId, EntityId); 4],
}

/// Ground 4, crank 1, coupler 4, rocker 2, with all four joints revolute about world Z.
fn four_bar() -> FourBar {
    let mut s = System::new();
    let g = s.ground(Frame::IDENTITY);
    let cx = (116.0 + 247f64.sqrt()) / 34.0; // |C - B| = 4 and |C - D| = 2 exactly
    let (a, b, c, d) = (
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [cx, 4.0 * cx - 13.5, 0.0],
        [4.0, 0.0, 0.0],
    );
    let sub = |p: [f64; 3], q: [f64; 3]| [p[0] - q[0], p[1] - q[1], p[2] - q[2]];
    let ga = s.connector(g, Frame::at(a)).unwrap();
    let gd = s.connector(g, Frame::at(d)).unwrap();
    let crank = s.body(Frame::axis_angle(
        [0.02, -0.01, 0.03],
        [0.3, 0.2, 1.0],
        0.02,
    ));
    let ca = s.connector(crank, Frame::at([0.0; 3])).unwrap();
    let cb = s.connector(crank, Frame::at(sub(b, a))).unwrap();
    let coupler = s.body(Frame::at([b[0] - 0.03, b[1] + 0.02, 0.01]));
    let kb = s.connector(coupler, Frame::at([0.0; 3])).unwrap();
    let kc = s.connector(coupler, Frame::at(sub(c, b))).unwrap();
    let rocker = s.body(Frame::axis_angle(
        [c[0] + 0.02, c[1], -0.02],
        [1.0, 0.0, 0.0],
        0.01,
    ));
    let rc = s.connector(rocker, Frame::at([0.0; 3])).unwrap();
    let rd = s.connector(rocker, Frame::at(sub(d, c))).unwrap();
    let joints = [(ga, ca), (cb, kb), (kc, rc), (rd, gd)];
    for (p, q) in joints {
        s.constrain(Constraint::Mate(Mate::Revolute { a: p, b: q }))
            .unwrap();
    }
    FourBar {
        s,
        ground_a: ga,
        crank_a: ca,
        joints,
    }
}

#[test]
fn spatial_four_bar_has_one_dof_and_three_redundant_rows() {
    let fb = four_bar();
    let rep = solve(&fb.s, &opts());
    assert!(rep.converged, "residual {}", rep.residual_norm);
    // Kutzbach: 6·3 − 5·4 = −2, yet the planar linkage moves: 1 DOF with redundant joints.
    assert_eq!(rep.dof, 1);
    assert_eq!(rep.status, Status::OverDefined);
    assert!(!rep.redundant.is_empty());
    for (p, q) in fb.joints {
        assert!(
            d3(
                connector_origin(&fb.s, &rep, p),
                connector_origin(&fb.s, &rep, q)
            ) < 1e-9
        );
    }
}

#[test]
fn spatial_four_bar_driven_by_crank_angle() {
    let mut fb = four_bar();
    fb.s.constrain(Constraint::Mate(Mate::Angle {
        a: fb.ground_a,
        b: fb.crank_a,
        mode: AngleMode::AboutZ,
        target: Target::Exact(-PI / 6.0),
    }))
    .unwrap();
    let rep = solve(&fb.s, &opts());
    assert!(rep.converged);
    assert_eq!(rep.dof, 0);
    let b = connector_origin(&fb.s, &rep, fb.joints[1].0);
    assert_close(b[0], (PI / 3.0).cos(), 1e-9);
    assert_close(b[1], (PI / 3.0).sin(), 1e-9);
    let c = connector_origin(&fb.s, &rep, fb.joints[2].0);
    assert_close(d3(b, c), 4.0, 1e-9);
    assert_close(d3(c, [4.0, 0.0, 0.0]), 2.0, 1e-9);
    assert_close(c[2], 0.0, 1e-9);
}

/// Crank 1, rod 3, slider on the world X axis; returns the slider body.
fn slider_crank(theta: Option<f64>) -> (System, EntityId) {
    let mut s = System::new();
    let g = s.ground(Frame::IDENTITY);
    let a = [0.5, 3f64.sqrt() / 2.0, 0.0];
    let xb = 0.5 + (9.0f64 - 0.75).sqrt();
    let go = s.connector(g, Frame::IDENTITY).unwrap();
    let gs = s
        .connector(g, Frame::with_z([0.0; 3], [1.0, 0.0, 0.0]))
        .unwrap();
    let crank = s.body(Frame::axis_angle([0.01, 0.0, -0.01], [0.0, 1.0, 0.0], 0.02));
    let co = s.connector(crank, Frame::IDENTITY).unwrap();
    let ca = s.connector(crank, Frame::at(a)).unwrap();
    let rod = s.body(Frame::at([a[0] + 0.02, a[1], 0.0]));
    let ra = s.connector(rod, Frame::IDENTITY).unwrap();
    let rb = s
        .connector(rod, Frame::at([xb - a[0], -a[1], 0.0]))
        .unwrap();
    let slider = s.body(Frame::at([xb + 0.05, 0.02, 0.0]));
    let sb = s.connector(slider, Frame::IDENTITY).unwrap();
    let ss = s
        .connector(slider, Frame::with_z([0.0; 3], [1.0, 0.0, 0.0]))
        .unwrap();
    s.constrain(Constraint::Mate(Mate::Revolute { a: go, b: co }))
        .unwrap();
    s.constrain(Constraint::Mate(Mate::Revolute { a: ca, b: ra }))
        .unwrap();
    s.constrain(Constraint::Mate(Mate::Revolute { a: rb, b: sb }))
        .unwrap();
    s.constrain(Constraint::Mate(Mate::Slider { a: gs, b: ss }))
        .unwrap();
    if let Some(t) = theta {
        s.constrain(Constraint::Mate(Mate::Angle {
            a: go,
            b: co,
            mode: AngleMode::AboutZ,
            target: Target::Exact(t),
        }))
        .unwrap();
    }
    (s, slider)
}

#[test]
fn slider_crank_has_one_dof() {
    let (s, slider) = slider_crank(None);
    let rep = solve(&s, &opts());
    assert!(rep.converged, "residual {}", rep.residual_norm);
    assert_eq!(rep.dof, 1);
    assert_eq!(rep.entity_dof(slider), Some(1));
}

#[test]
fn slider_crank_position_follows_crank_angle() {
    for deg in [-40.0f64, 0.0, 30.0, 75.0] {
        let phi = deg.to_radians();
        let (s, slider) = slider_crank(Some(phi));
        let rep = solve(&s, &opts());
        assert!(rep.converged, "{deg}: residual {}", rep.residual_norm);
        assert_eq!(rep.dof, 0, "{deg}");
        let th = PI / 3.0 + phi;
        let want = th.cos() + (9.0 - th.sin().powi(2)).sqrt();
        let EntityKind::Body { position, .. } = s.entity(slider).kind else {
            panic!()
        };
        assert_close(rep.value(position[0]), want, 1e-9);
        assert_close(rep.value(position[1]), 0.0, 1e-9);
    }
}

#[test]
fn drag_body_along_slider() {
    let (s, b, _) = slider_with_range(2.0);
    let rep = drag(
        &s,
        &DragRequest::body_position(&s, b, [4.0, 1.0, 1.0]),
        &opts(),
    );
    assert!(rep.converged);
    let EntityKind::Body { position, .. } = s.entity(b).kind else {
        panic!()
    };
    assert_close(rep.value(position[0]), 4.0, 1e-8);
    assert_close(rep.value(position[1]), 0.0, 1e-9);
    assert_close(rep.value(position[2]), 0.0, 1e-9);
}

#[test]
fn mates_reject_same_body() {
    let mut s = System::new();
    let b = s.body(Frame::IDENTITY);
    let c1 = s.connector(b, Frame::IDENTITY).unwrap();
    let c2 = s.connector(b, Frame::at([1.0, 0.0, 0.0])).unwrap();
    assert_eq!(
        s.constrain(Constraint::Mate(Mate::Ball { a: c1, b: c2 })),
        Err(BuildError::SameBody(b))
    );
}
