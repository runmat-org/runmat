//! Validation of the dynamics against closed-form results for a compound (physical) pendulum:
//! a rigid body hinged at a point `d` from its centre of mass, `I` its moment of inertia about
//! the hinge axis through the pivot.
//!
//! - Small amplitude: `T₀ = 2π √(I / (m g d))`.
//! - Any amplitude `θ₀`: `T = 4 √(I / (m g d)) K(sin(θ₀/2))`, `K` the complete elliptic
//!   integral of the first kind (e.g. Landau & Lifshitz, *Mechanics*, §11).
//!
//! Tolerances: the period within 0.1 % of the closed form (stated by the study requirement; the
//! runs agree far closer), the total energy within 1e-6 of `m g d` over ten periods, and the
//! hinge reaction at release within 1e-6 relative of Newton's second law.

use super::model::{BodyMass, Model, Sample};
use super::STANDARD_GRAVITY as G;
use crate::assembly::motion::kinematics::{run_study, MotionContext};
use crate::assembly::resolve::{resolve, ResolveContext, ResolveRequest};
use crate::assembly::source::{Aabb, ConfiguredSummary, DefinitionSummary, MemorySource};
use crate::ids::FilePath;
use crate::units::Units;
use crate::value::NoDynamicValues;
use runmat_constraint as rc;
use std::f64::consts::PI;

/// `K(k)` by the arithmetic–geometric mean: `K = π / (2 AGM(1, √(1 − k²)))`.
fn elliptic_k(k: f64) -> f64 {
    let (mut a, mut b) = (1.0_f64, (1.0 - k * k).sqrt());
    for _ in 0..40 {
        let (an, bn) = ((a + b) / 2.0, (a * b).sqrt());
        a = an;
        b = bn;
    }
    PI / (2.0 * a)
}

fn box_inertia(m: f64, size: [f64; 3]) -> [[f64; 3]; 3] {
    let [a, b, c] = size;
    [
        [m * (b * b + c * c) / 12.0, 0.0, 0.0],
        [0.0, m * (a * a + c * c) / 12.0, 0.0],
        [0.0, 0.0, m * (a * a + b * b) / 12.0],
    ]
}

/// Mean time between upward zero crossings of `angle` (linear interpolation between samples).
fn period(times: &[f64], angle: &[f64]) -> f64 {
    let mut crossings = Vec::new();
    for k in 1..angle.len() {
        if angle[k - 1] < 0.0 && angle[k] >= 0.0 {
            let s = -angle[k - 1] / (angle[k] - angle[k - 1]);
            crossings.push(times[k - 1] + s * (times[k] - times[k - 1]));
        }
    }
    assert!(crossings.len() >= 2, "fewer than two crossings");
    (crossings[crossings.len() - 1] - crossings[0]) / (crossings.len() - 1) as f64
}

struct Pendulum {
    mass: f64,
    d: f64,
    inertia: [[f64; 3]; 3],
}

impl Pendulum {
    fn new() -> Self {
        let mass = 2.0;
        Pendulum {
            mass,
            d: 0.3,
            inertia: box_inertia(mass, [0.05, 0.6, 0.02]),
        }
    }
    /// `I` about the pivot axis.
    fn pivot_inertia(&self) -> f64 {
        self.inertia[2][2] + self.mass * self.d * self.d
    }
    fn exact_period(&self, theta0: f64) -> f64 {
        4.0 * (self.pivot_inertia() / (self.mass * G * self.d)).sqrt()
            * elliptic_k((theta0 / 2.0).sin())
    }
}

/// Runs the pendulum released at rest from `theta0` (about Z, gravity −Y) for `seconds`,
/// sampled every millisecond: times, angles, energies and samples.
fn swing(p: &Pendulum, theta0: f64, seconds: f64) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<Sample>) {
    let mut s = rc::System::new();
    let ground = s.ground(rc::Frame::IDENTITY);
    // The body frame sits at the pivot; the centre of mass hangs `d` below it before the turn.
    let body = s.body(rc::Frame::axis_angle([0.0; 3], [0.0, 0.0, 1.0], theta0));
    let a = s.connector(ground, rc::Frame::IDENTITY).unwrap();
    let b = s.connector(body, rc::Frame::IDENTITY).unwrap();
    s.constrain(rc::Constraint::Mate(rc::Mate::Revolute { a, b }))
        .unwrap();
    let mass = BodyMass {
        mass: p.mass,
        center: [0.0, -p.d, 0.0],
        inertia: p.inertia,
    };
    let model = Model::new(&s, &[(body, mass)], [0.0, -G, 0.0], Vec::new()).unwrap();
    let q0: Vec<f64> = s.vars().iter().map(|v| v.value).collect();
    let dt = 1e-3;
    let frames = (seconds / dt) as usize + 1;
    let (mut t, mut angle, mut energy, mut samples) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    model
        .simulate(&q0, &[0.0; 6], frames, dt, dt, |sample| {
            let c = model.center(&sample.q, 0);
            t.push(sample.time);
            angle.push(c[0].atan2(-c[1]));
            energy.push(model.energy(&sample.q, &sample.u));
            if samples.is_empty() {
                samples.push(sample.clone());
            }
        })
        .unwrap();
    // The release reaction, by Newton's second law at rest: F = m (α ẑ × r_c) − m g.
    let alpha = -p.mass * G * p.d * theta0.sin() / p.pivot_inertia();
    let rc_ = [p.d * theta0.sin(), -p.d * theta0.cos(), 0.0];
    let expected = [
        p.mass * (-alpha * rc_[1]),
        p.mass * (alpha * rc_[0]) + p.mass * G,
        0.0,
    ];
    let rows: Vec<usize> = (0..samples[0].owners.len()).collect();
    let (force, _) = model.reaction_on(&samples[0], &rows, 0);
    let scale = p.mass * G;
    for k in 0..3 {
        assert!(
            (force[k] - expected[k]).abs() < 1e-6 * scale,
            "reaction {force:?} vs {expected:?}"
        );
    }
    (t, angle, energy, samples)
}

#[test]
fn small_swings_keep_the_linear_period() {
    let p = Pendulum::new();
    let theta0: f64 = 0.01;
    let t0 = 2.0 * PI * (p.pivot_inertia() / (p.mass * G * p.d)).sqrt();
    let (t, angle, energy, _) = swing(&p, theta0, 10.5 * t0);
    let measured = period(&t, &angle);
    let error = (measured - t0).abs() / t0;
    assert!(
        error < 1e-3,
        "period {measured} s vs 2π√(I/mgd) = {t0} s ({error:e})"
    );
    // Closer still to the exact period (the amplitude correction is θ₀²/16 ≈ 6e-6).
    let exact = p.exact_period(theta0);
    assert!(
        (measured - exact).abs() / exact < 1e-5,
        "{measured} vs {exact}"
    );
    let e0 = energy[0];
    let drift = energy.iter().fold(0.0_f64, |m, e| m.max((e - e0).abs()));
    assert!(drift < 1e-6 * p.mass * G * p.d, "energy drift {drift:e} J");
}

#[test]
fn large_swings_keep_the_elliptic_period() {
    let p = Pendulum::new();
    let theta0 = 2.0; // about 115 degrees
    let exact = p.exact_period(theta0);
    let (t, angle, energy, _) = swing(&p, theta0, 10.5 * exact);
    let measured = period(&t, &angle);
    let error = (measured - exact).abs() / exact;
    assert!(
        error < 1e-3,
        "period {measured} s vs 4√(I/mgd) K(sin θ₀/2) = {exact} s ({error:e})"
    );
    assert!(
        error < 1e-5,
        "the integration is far closer than the stated tolerance: {error:e}"
    );
    // The linear formula is well off at this amplitude: the run is not a small-angle model.
    let linear = 2.0 * PI * (p.pivot_inertia() / (p.mass * G * p.d)).sqrt();
    assert!((measured - linear) / linear > 0.2);
    let e0 = energy[0];
    let drift = energy.iter().fold(0.0_f64, |m, e| m.max((e - e0).abs()));
    assert!(drift < 1e-6 * p.mass * G * p.d, "energy drift {drift:e} J");
}

#[test]
fn a_torsion_spring_sets_the_period_without_gravity() {
    // A body on a hinge, centre of mass on the axis, spring k: T = 2π √(I/k).
    let mut s = rc::System::new();
    let ground = s.ground(rc::Frame::IDENTITY);
    let body = s.body(rc::Frame::axis_angle([0.0; 3], [0.0, 0.0, 1.0], 0.5));
    let a = s.connector(ground, rc::Frame::IDENTITY).unwrap();
    let b = s.connector(body, rc::Frame::IDENTITY).unwrap();
    s.constrain(rc::Constraint::Mate(rc::Mate::Revolute { a, b }))
        .unwrap();
    let inertia = box_inertia(1.5, [0.2, 0.1, 0.05]);
    let k = 3.0;
    let spring = super::model::MateLoad {
        a,
        b,
        measure: super::model::MateMeasure::Angle,
        stiffness: k,
        damping: 0.0,
        free: 0.0,
        constant: 0.0,
    };
    let model = Model::new(
        &s,
        &[(
            body,
            BodyMass {
                mass: 1.5,
                center: [0.0; 3],
                inertia,
            },
        )],
        [0.0; 3],
        vec![spring],
    )
    .unwrap();
    let q0: Vec<f64> = s.vars().iter().map(|v| v.value).collect();
    let expected = 2.0 * PI * (inertia[2][2] / k).sqrt();
    let (mut t, mut angle) = (Vec::new(), Vec::new());
    let dt = 1e-3;
    model
        .simulate(
            &q0,
            &[0.0; 6],
            (4.5 * expected / dt) as usize,
            dt,
            dt,
            |sample| {
                t.push(sample.time);
                let pose = model.pose(&sample.q, 0);
                angle.push(pose.rotation[1][0].atan2(pose.rotation[0][0]));
            },
        )
        .unwrap();
    let measured = period(&t, &angle);
    assert!(
        (measured - expected).abs() / expected < 1e-4,
        "{measured} vs {expected}"
    );
}

/// The arm: 0.4 m long along X from its origin, its centre of mass 0.2 m out.
const ARM: (f64, f64) = (1.2, 0.2);

/// Runs study `name` of the hinge assembly `text` (a base and an arm, both parts), with the
/// arm's mass properties as [`ARM`] and a heavy base; the run and the resolved assembly.
fn hinge_run(
    text: &str,
    name: &str,
) -> (
    crate::assembly::motion::kinematics::MotionRun,
    crate::assembly::resolve::ResolvedAssembly,
) {
    let scope = crate::modules::ModuleScope::from_files(
        [
            ("hinge.assembly", text),
            ("block.part", "part block {\n}\n"),
            ("arm.part", "part arm {\n}\n"),
        ]
        .iter()
        .copied(),
    );
    let parsed = crate::syntax::parse_assembly_in("hinge.assembly", text, &scope);
    let doc = parsed
        .document
        .unwrap_or_else(|| panic!("{:?}", parsed.diagnostics));
    let (m, d) = ARM;
    let inertia = box_inertia(m, [0.4, 0.04, 0.02]);
    // The definitions the occurrences name, as resolution reads them (a first pass with no parts).
    let mut probe = MemorySource::default();
    probe.add_assembly(doc.clone());
    let probe_ctx = ResolveContext {
        source: &probe,
        units: &Units,
        dynamic: &NoDynamicValues,
        lock: None,
        catalog: None,
        tables: None,
        gauge: None,
    };
    let named = resolve(
        &FilePath::new("hinge.assembly"),
        &ResolveRequest::default(),
        &probe_ctx,
    );
    let mut src = MemorySource::default();
    for n in named.nodes.iter().filter(|n| n.parent.is_some()) {
        let arm = n.segment.name == "arm";
        src.add_part(
            DefinitionSummary::new(n.definition.clone()),
            ConfiguredSummary {
                mass: Some(if arm { m } else { 10.0 }),
                center_of_mass: Some(if arm { [d, 0.0, 0.0] } else { [0.0; 3] }),
                inertia: Some(if arm {
                    inertia
                } else {
                    box_inertia(10.0, [0.1; 3])
                }),
                // The arm's box centre, the tracked point, 30 mm out from the hinge.
                bbox: arm.then_some(Aabb {
                    min: [0.0, -0.02, -0.01],
                    max: [0.06, 0.02, 0.01],
                }),
                ..Default::default()
            },
        );
    }
    src.add_assembly(doc.clone());
    let ctx = ResolveContext {
        source: &src,
        units: &Units,
        dynamic: &NoDynamicValues,
        lock: None,
        catalog: None,
        tables: None,
        gauge: None,
    };
    let request = ResolveRequest::default();
    let base = resolve(&FilePath::new("hinge.assembly"), &request, &ctx);
    let run = run_study(
        &doc,
        &base,
        name,
        &MotionContext {
            resolve: &ctx,
            request: &request,
            frame_bytes: 8 << 20,
        },
    );
    assert!(
        run.problems.is_empty() && run.stopped.is_none(),
        "{:?} {:?}",
        run.problems,
        run.stopped
    );
    (run, base)
}

const HINGE: &str = "runmat 0.7\nstd 1\n\nassembly hinge {\n    use part block as base\n    use part arm as arm {\n        fixed = false\n    }\n    connector pivot = { occurrence = [{ name = base }], anchor = { anchor = origin } }\n    connector root = { occurrence = [{ name = arm }], anchor = { anchor = origin } }\n    mate hinge = { kind = { mate = revolute }, a = pivot, b = root }\n    motion swing = { kind = dynamic, duration = 3 s, rate = 200, gravity = [0 m/s^2, -9.80665 m/s^2, 0 m/s^2] }\n    motion wound = { kind = dynamic, duration = 1 s, gravity = [0 m/s^2, -9.80665 m/s^2, 0 m/s^2], springs = [{ mate = hinge, stiffness = 2 N*m/rad, damping = 0.01 N*m*s/rad, free = 30 deg }], motors = [{ mate = hinge, torque = 0.3 N*m }] }\n}\n";

/// The same check end to end: an assembly with a hinged arm whose part carries its mass,
/// centre of mass and inertia, studied with `kind = dynamic`.
#[test]
fn an_assembly_study_swings_the_arm_at_the_pendulum_period() {
    let (run, base) = hinge_run(HINGE, "swing");
    let (m, d) = ARM;
    let inertia = box_inertia(m, [0.4, 0.04, 0.02]);
    // Released along +X with gravity −Y: a quarter turn from hanging (θ₀ = 90°).
    let arm = (0..base.nodes.len() as u32)
        .find(|&n| base.path(n) == "arm")
        .unwrap();
    let angle: Vec<f64> = run
        .frames
        .iter()
        .map(|f| {
            let c = f[arm as usize].apply([d, 0.0, 0.0]);
            c[0].atan2(-c[1])
        })
        .collect();
    let i_pivot = inertia[2][2] + m * d * d;
    let exact = 4.0 * (i_pivot / (m * G * d)).sqrt() * elliptic_k((PI / 4.0).sin());
    let measured = period(&run.times, &angle);
    assert!(
        (measured - exact).abs() / exact < 1e-3,
        "{measured} vs {exact}"
    );
    // The hinge reaction is recorded on the arm at every frame.
    let hinge = run.reactions.iter().find(|r| r.mate == "hinge").unwrap();
    assert_eq!(hinge.frames.len(), run.times.len());
    assert_eq!(hinge.frames[0][0].occurrence, "arm");
    assert!(!run.moving.is_empty() && run.energy.len() == run.times.len());
}

/// d'Alembert's principle at every frame (Haug 1989, §11.2): the loads the mates put on the arm,
/// its weight and its inertia load (`−m a` and `−(J α + ω × J ω)` about the centre of mass)
/// sum to zero, forces and moments about the centre of mass, within 1e-6 of the largest term.
/// The free swing, and a wound spring with a damper and a motor on the hinge (their torque is
/// part of what the mate passes to the arm).
#[test]
fn the_mate_loads_weight_and_inertia_of_the_arm_balance_at_every_frame() {
    let norm = |v: [f64; 3]| v.iter().map(|x| x * x).sum::<f64>().sqrt();
    let add = |a: [f64; 3], b: [f64; 3]| [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
    let cross = |a: [f64; 3], b: [f64; 3]| {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    };
    for name in ["swing", "wound"] {
        let (run, base) = hinge_run(HINGE, name);
        let g = run.gravity.unwrap();
        let body = run.bodies.iter().find(|b| b.occurrence == "arm").unwrap();
        assert_eq!(body.frames.len(), run.times.len());
        let arm = (0..base.nodes.len())
            .find(|&n| base.path(n as u32) == "arm")
            .unwrap();
        let frames: Vec<usize> = (0..run.times.len())
            .step_by((run.times.len() / 7).max(1))
            .collect();
        assert!(frames.len() >= 5, "{name}: {} frames", run.times.len());
        for k in frames {
            let state = &body.frames[k];
            let rotation = run.frames[k][arm].rotation;
            let (inertia_force, inertia_moment) =
                state.inertia_load(body.mass, &body.inertia, &rotation);
            let weight = g.map(|c| c * body.mass);
            let mut force = add(weight, inertia_force);
            let mut moment = inertia_moment;
            let mut scale_f = norm(weight).max(norm(inertia_force));
            let mut scale_m = norm(inertia_moment);
            for mate in &run.reactions {
                for l in mate.frames[k].iter().filter(|l| l.occurrence == "arm") {
                    let about = add(
                        l.moment,
                        cross([0, 1, 2].map(|a| l.point[a] - state.center[a]), l.force),
                    );
                    force = add(force, l.force);
                    moment = add(moment, about);
                    scale_f = scale_f.max(norm(l.force));
                    scale_m = scale_m.max(norm(about));
                }
            }
            assert!(
                norm(force) <= 1e-6 * scale_f,
                "{name} frame {k}: force residual {force:?} of {scale_f}"
            );
            assert!(
                norm(moment) <= 1e-6 * scale_m,
                "{name} frame {k}: moment residual {moment:?} of {scale_m}"
            );
            if k > 0 {
                assert!(
                    norm(state.angular_velocity) > 0.0,
                    "{name} frame {k}: the arm turns"
                );
            }
        }
    }
}

/// A study that doesn't set its rate takes frames close enough that the arm turns at most 5°
/// between them at its fastest, evenly spaced to the study's end; a slow one keeps 30 a second.
#[test]
fn a_fast_swing_gets_frames_close_enough_for_a_smooth_trail() {
    let fast = HINGE.replace("duration = 3 s, rate = 200,", "duration = 1 s,");
    let (run, base) = hinge_run(&fast, "swing");
    let arm = (0..base.nodes.len())
        .find(|&n| base.path(n as u32) == "arm")
        .unwrap();
    let n = run.times.len();
    assert!(n > 31, "faster than 30 frames a second: {n}");
    assert!(
        (run.times[n - 1] - 1.0).abs() < 1e-12,
        "ends at the study's end: {:?}",
        run.times.last()
    );
    let dt = run.times[1] - run.times[0];
    assert!(
        run.times
            .windows(2)
            .all(|w| ((w[1] - w[0]) - dt).abs() < 1e-12),
        "evenly spaced"
    );
    for k in 1..n {
        let turn = super::super::kinematics::rotation_between(
            &run.frames[k - 1][arm],
            &run.frames[k][arm],
        );
        assert!(
            turn <= super::FRAME_TURN * 1.05,
            "frame {k} turns {} deg",
            turn.to_degrees()
        );
    }
    // Slow: a stiff damper barely lets it move; 30 frames a second hold.
    let slow = HINGE.replace(
        "springs = [{ mate = hinge, stiffness = 2 N*m/rad, damping = 0.01 N*m*s/rad, free = 30 deg }], motors = [{ mate = hinge, torque = 0.3 N*m }]",
        "springs = [{ mate = hinge, damping = 100 N*m*s/rad }]",
    );
    let (run, _) = hinge_run(&slow, "wound");
    assert_eq!(run.times.len(), 31, "{:?}", run.times.len());
}

const SWEEP: &str = "runmat 0.7\nstd 1\n\nassembly hinge {\n    use part block as base\n    use part arm as arm {\n        fixed = false\n    }\n    connector pivot = { occurrence = [{ name = base }], anchor = { anchor = origin } }\n    connector root = { occurrence = [{ name = arm }], anchor = { anchor = origin } }\n    mate hinge = { kind = { mate = revolute }, a = pivot, b = root }\n    motion sweep = { duration = 3 s, drives = [{ mate = hinge, from = 0 deg, to = 180 deg }], track = arm }\n    motion rise = { duration = 2 s, law = cycloidal, drives = [{ mate = hinge, from = 0 deg, to = 90 deg }], track = arm }\n}\n";

/// A kinematic sweep at a constant 60°/s carries the arm's box centre, 30 mm from the hinge, at
/// exactly ω r and ω² r at every frame, the first and last included: the rates come from the
/// velocity and acceleration equations, not from differences of the frames.
#[test]
fn a_constant_sweep_reads_omega_r_at_every_frame() {
    let (run, _) = hinge_run(SWEEP, "sweep");
    let t = run.track.as_ref().expect("the arm is tracked");
    let (w, r) = (PI / 3.0, 0.03);
    assert!(t.velocity.len() > 30);
    for (k, ((v, a), x)) in t
        .velocity
        .iter()
        .zip(&t.acceleration)
        .zip(&t.position)
        .enumerate()
    {
        let speed = v.iter().map(|c| c * c).sum::<f64>().sqrt();
        let acc = a.iter().map(|c| c * c).sum::<f64>().sqrt();
        assert!((speed - w * r).abs() < 1e-9, "speed {speed} at frame {k}");
        assert!(
            (acc - w * w * r).abs() < 1e-7,
            "acceleration {acc} at frame {k}"
        );
        assert!(
            (t.angular_speed[k] - w).abs() < 1e-9,
            "ω {} at frame {k}",
            t.angular_speed[k]
        );
        // The velocity is tangent to the circle and turns the way the drive does.
        let radial = x[0] * v[0] + x[1] * v[1] + x[2] * v[2];
        assert!(radial.abs() < 1e-9, "radial speed {radial} at frame {k}");
        let along = x[0] * v[1] - x[1] * v[0];
        assert!(
            (along - w * r * r).abs() < 1e-9,
            "turn {along} at frame {k}"
        );
        // The acceleration points at the hinge.
        let inward = -(x[0] * a[0] + x[1] * a[1] + x[2] * a[2]) / r;
        assert!(
            (inward - w * w * r).abs() < 1e-7,
            "centripetal {inward} at frame {k}"
        );
    }
}

/// A cycloidal rise starts and ends at rest, and in between its rates are the drive law's own:
/// `θ̇ = Δθ (1 − cos 2πτ) / T` and the arm's tangential acceleration `r θ̈`.
#[test]
fn a_cycloidal_rise_follows_its_law_exactly() {
    let (run, _) = hinge_run(SWEEP, "rise");
    let t = run.track.as_ref().expect("the arm is tracked");
    let (span, duration, r) = (PI / 2.0, 2.0, 0.03);
    for (k, (&time, v)) in run.times.iter().zip(&t.velocity).enumerate() {
        let tau = time / duration;
        let rate = span * (1.0 - (2.0 * PI * tau).cos()) / duration;
        let speed = v.iter().map(|c| c * c).sum::<f64>().sqrt();
        assert!(
            (speed - rate * r).abs() < 1e-9,
            "speed {speed} at frame {k}, want {}",
            rate * r
        );
        let accel = span * 2.0 * PI * (2.0 * PI * tau).sin() / (duration * duration);
        let a = t.acceleration[k];
        let want = ((accel * r).powi(2) + (rate * rate * r).powi(2)).sqrt();
        let got = a.iter().map(|c| c * c).sum::<f64>().sqrt();
        assert!(
            (got - want).abs() < 1e-7,
            "acceleration {got} at frame {k}, want {want}"
        );
    }
    assert!(
        t.velocity[0].iter().all(|c| c.abs() < 1e-12),
        "at rest at the start"
    );
}
