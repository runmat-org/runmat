//! The equations of motion of a mated mechanism and their time integration.
//!
//! Coordinates are maximal: each free body has its frame's position `r` and Euler parameters
//! `p = (e₀, e)` (a unit quaternion), the variables the mate solver already carries, so the mate
//! constraints `Φ(r, p) = 0` and their exact Jacobian come from `runmat-constraint`
//! ([`rc::constraint_rows`]). Velocities are each body's centre-of-mass velocity `v` (world) and
//! angular velocity `ω'` (body axes), with `ṗ = ½ Gᵀ ω'`, `G = [−e, e₀I − ẽ]`, and
//! `ṙ = v + A c̃' ω'` for a centre of mass at `c'` in the body frame (Nikravesh, *Computer-Aided
//! Analysis of Mechanical Systems*, 1988, ch. 6 and 11; Haug, *Computer-Aided Kinematics and
//! Dynamics of Mechanical Systems*, 1989, ch. 6 and 11). The velocity constraint is `B u = 0`
//! with `B = [Φ_r, Φ_r A c̃' + ½ Φ_p Gᵀ]` per body.
//!
//! The Newton–Euler equations with the constraint reactions,
//!
//! ```text
//! m v̇ = f + B_vᵀ λ,    J' ω̇' = n' − ω̃' J' ω' + B_ωᵀ λ,    B u̇ = γ − 2α Φ̇ − β² Φ,
//! ```
//!
//! are the index-1 form of the index-3 DAE, with Baumgarte's stabilisation of the acceleration
//! constraint (Baumgarte, *Comput. Methods Appl. Mech. Eng.* 1, 1972). `λ` comes from the Schur
//! complement `(B M⁻¹ Bᵀ) λ = γ_s − B M⁻¹ F`; `γ = −(B u)_q q̇` is the derivative of `B u` along
//! the motion with `u` held, taken by a central difference of the exact Jacobian. Each fixed
//! step is classical Runge–Kutta (4th order); after it, the positions are projected back onto
//! `Φ = 0` (minimum-norm Newton steps) and the velocities onto `B u = 0` (mass-weighted), the
//! coordinate projection of Nikravesh 1988 §11.

use super::dense::{
    add, cross, dot, inverse3, mat3_t_vec, mat3_vec, mul, mul_t, scale, solve_spd, sub,
};
use crate::regen::kernel::Transform;
use runmat_constraint as rc;

/// Baumgarte's α and β (1/s): critically damped constraint error between projections.
const BAUMGARTE: f64 = 10.0;
/// Position projection: most Newton steps, and the constraint error accepted (m or rad).
const PROJECTION_STEPS: usize = 12;
const PROJECTION_TOLERANCE: f64 = 1e-10;

/// A body's mass properties in its own frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BodyMass {
    pub mass: f64,
    /// The centre of mass, body frame (m).
    pub center: [f64; 3],
    /// The inertia tensor about the centre of mass, body axes (kg·m²).
    pub inertia: [[f64; 3]; 3],
}

/// A free body of the mechanism.
#[derive(Clone, Debug)]
pub struct DynBody {
    pub entity: rc::EntityId,
    pub mass: BodyMass,
    inv_inertia: [[f64; 3]; 3],
    position: [usize; 3],
    orientation: [usize; 4],
}

/// What a mate's own value measures.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MateMeasure {
    /// The angle of `b`'s X axis from `a`'s about `a`'s Z axis.
    Angle,
    /// The distance of `b`'s origin from `a`'s along `a`'s Z axis.
    Length,
}

/// A load acting along a mate's own value: `−k (x − x₀) − c ẋ + constant`, a torque about the
/// mate axis for an angle, a force along it for a length.
#[derive(Clone, Debug, PartialEq)]
pub struct MateLoad {
    pub a: rc::EntityId,
    pub b: rc::EntityId,
    pub measure: MateMeasure,
    pub stiffness: f64,
    pub damping: f64,
    pub free: f64,
    pub constant: f64,
}

/// A driven mate of a kinematic study: the measure of its value, the constraints that hold the
/// value (which the drive replaces), and the value's rate and acceleration from the drive law.
#[derive(Clone, Debug, PartialEq)]
pub struct Driven {
    pub mate: MateLoad,
    pub rows: Vec<rc::ConstraintId>,
    pub rate: f64,
    pub acceleration: f64,
}

/// A body's motion at an instant, world axes: its centre of mass's position, velocity and
/// acceleration, its angular velocity and angular acceleration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BodyRate {
    pub center: [f64; 3],
    pub velocity: [f64; 3],
    pub acceleration: [f64; 3],
    pub omega: [f64; 3],
    pub alpha: [f64; 3],
}

impl BodyRate {
    /// The velocity and acceleration of the body's point now at world `x`.
    pub fn at(&self, x: [f64; 3]) -> ([f64; 3], [f64; 3]) {
        let r = sub(x, self.center);
        let w = self.omega;
        (
            add(self.velocity, cross(w, r)),
            add(
                add(self.acceleration, cross(self.alpha, r)),
                cross(w, cross(w, r)),
            ),
        )
    }
}

/// The mechanism: its constraint system, free bodies and loads.
pub struct Model<'a> {
    pub sys: &'a rc::System,
    pub bodies: Vec<DynBody>,
    pub gravity: [f64; 3],
    pub loads: Vec<MateLoad>,
    /// Rows of [`rc::constraint_rows`] that are quaternion normalisations (their velocity rows
    /// vanish identically).
    norm_rows: Vec<bool>,
}

/// An integrated state: the time, every variable's value and the velocities.
#[derive(Clone, Debug)]
pub struct State {
    pub time: f64,
    pub q: Vec<f64>,
    pub u: Vec<f64>,
}

/// `u̇`, the reaction multipliers, and the constraint each multiplier belongs to.
type Accelerations = (Vec<f64>, Vec<f64>, Vec<rc::ConstraintId>);

/// A solved instant: every variable's value, the velocities, their rates, and the reaction
/// multiplier of every constraint row.
#[derive(Clone, Debug)]
pub struct Sample {
    pub time: f64,
    pub q: Vec<f64>,
    pub u: Vec<f64>,
    /// `u̇` from the same solve as `lambda`: each body's centre-of-mass acceleration (world) and
    /// angular acceleration (body axes), so the instant's loads and its inertia balance exactly.
    pub udot: Vec<f64>,
    pub lambda: Vec<f64>,
    /// The constraint each multiplier belongs to.
    pub owners: Vec<rc::ConstraintId>,
}

fn quat(q: &[f64], idx: &[usize; 4]) -> [f64; 4] {
    idx.map(|i| q[i])
}

/// `A(p)`, body to world.
fn rotation(p: [f64; 4]) -> [[f64; 3]; 3] {
    Transform::from_quaternion(p, [0.0; 3]).rotation
}

/// `ṗ = ½ Gᵀ ω'` for Euler parameters `p = (e₀, e)` and body angular velocity `ω'`.
fn p_dot(p: [f64; 4], w: [f64; 3]) -> [f64; 4] {
    let [e0, e1, e2, e3] = p;
    let e = [e1, e2, e3];
    let ew = cross(e, w);
    [
        -0.5 * dot(e, w),
        0.5 * (e0 * w[0] + ew[0]),
        0.5 * (e0 * w[1] + ew[1]),
        0.5 * (e0 * w[2] + ew[2]),
    ]
}

impl<'a> Model<'a> {
    /// The model of `sys` with `bodies` free (each with its mass properties). Every free
    /// variable of the system must belong to one of them.
    pub fn new(
        sys: &'a rc::System,
        bodies: &[(rc::EntityId, BodyMass)],
        gravity: [f64; 3],
        loads: Vec<MateLoad>,
    ) -> Result<Model<'a>, String> {
        let mut owned = vec![false; sys.var_count()];
        let mut out = Vec::new();
        for &(entity, mass) in bodies {
            let vars = sys.entity_vars(entity);
            if vars.len() != 7 {
                return Err("a moving component is not a rigid body".into());
            }
            if mass.mass.is_nan() || mass.mass <= 0.0 {
                return Err("a moving component has no mass".into());
            }
            let inv_inertia =
                inverse3(&mass.inertia).ok_or("a moving component has no rotational inertia")?;
            for v in &vars {
                owned[v.index()] = true;
            }
            out.push(DynBody {
                entity,
                mass,
                inv_inertia,
                position: [vars[0].index(), vars[1].index(), vars[2].index()],
                orientation: [vars[3].index(), vars[4].index(), vars[5].index(), vars[6].index()],
            });
        }
        if let Some(i) = sys
            .vars()
            .iter()
            .enumerate()
            .position(|(i, v)| v.flag == rc::VarFlag::Free && !owned[i])
        {
            let _ = i;
            return Err(
                "a mate adds its own unknowns (a path or cam parameter); a dynamic study integrates rigid bodies only"
                    .into(),
            );
        }
        let values: Vec<f64> = sys.vars().iter().map(|v| v.value).collect();
        let rows = rc::constraint_rows(sys, &values);
        let norm_rows = rows
            .owners
            .iter()
            .map(|c| {
                matches!(
                    sys.constraint(*c).constraint,
                    rc::ConstraintKindRecord::Internal {
                        kind: rc::InternalKind::QuaternionNorm,
                        ..
                    }
                )
            })
            .collect();
        Ok(Model {
            sys,
            bodies: out,
            gravity,
            loads,
            norm_rows,
        })
    }

    fn dofs(&self) -> usize {
        6 * self.bodies.len()
    }

    /// Body `i`'s world pose at `q`.
    pub fn pose(&self, q: &[f64], i: usize) -> Transform {
        let b = &self.bodies[i];
        Transform::from_quaternion(quat(q, &b.orientation), b.position.map(|k| q[k]))
    }

    /// Body `i`'s centre of mass, world.
    pub fn center(&self, q: &[f64], i: usize) -> [f64; 3] {
        self.pose(q, i).apply(self.bodies[i].mass.center)
    }

    /// A connector's world frame at `q` (on a free body or on ground).
    pub fn connector(&self, q: &[f64], c: rc::EntityId) -> Transform {
        let rc::EntityKind::Connector { body, local } = &self.sys.entity(c).kind else {
            return Transform::translation([0.0; 3]);
        };
        let vars = self.sys.entity_vars(*body);
        let body_pose = Transform::from_quaternion(
            [vars[3], vars[4], vars[5], vars[6]].map(|v| q[v.index()]),
            [vars[0], vars[1], vars[2]].map(|v| q[v.index()]),
        );
        body_pose.compose(&Transform::from_quaternion(local.rotation, local.origin))
    }

    fn body_of_entity(&self, e: rc::EntityId) -> Option<usize> {
        self.bodies.iter().position(|b| b.entity == e)
    }

    /// The world velocity of a point fixed in body `i` now at world `x`, and the body's world
    /// angular velocity.
    fn point_velocity(&self, q: &[f64], u: &[f64], i: usize, x: [f64; 3]) -> ([f64; 3], [f64; 3]) {
        let a = rotation(quat(q, &self.bodies[i].orientation));
        let w = mat3_vec(&a, [u[6 * i + 3], u[6 * i + 4], u[6 * i + 5]]);
        let v = [u[6 * i], u[6 * i + 1], u[6 * i + 2]];
        let rc_ = self.center(q, i);
        (add(v, cross(w, sub(x, rc_))), w)
    }

    /// `q̇` (every variable; fixed ones don't move) from the velocities.
    pub fn q_dot(&self, q: &[f64], u: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; q.len()];
        for (i, b) in self.bodies.iter().enumerate() {
            let p = quat(q, &b.orientation);
            let a = rotation(p);
            let w = [u[6 * i + 3], u[6 * i + 4], u[6 * i + 5]];
            // ṙ = v + A (c' × ω') ... = v − A (ω' × c').
            let rdot = sub([u[6 * i], u[6 * i + 1], u[6 * i + 2]], mat3_vec(&a, cross(w, b.mass.center)));
            for k in 0..3 {
                out[b.position[k]] = rdot[k];
            }
            let pd = p_dot(p, w);
            for k in 0..4 {
                out[b.orientation[k]] = pd[k];
            }
        }
        out
    }

    /// `Φ(q)`, `B(q)` (rows × 6n, row-major) and each row's owner, normalisation rows dropped.
    fn velocity_rows(&self, q: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<rc::ConstraintId>) {
        let rows = rc::constraint_rows(self.sys, q);
        let n = self.dofs();
        let mut phi = Vec::new();
        let mut b = Vec::new();
        let mut owners = Vec::new();
        for r in 0..rows.rows() {
            if self.norm_rows.get(r).copied().unwrap_or(false) {
                continue;
            }
            let mut row = vec![0.0; n];
            for (i, body) in self.bodies.iter().enumerate() {
                let p = quat(q, &body.orientation);
                let a = rotation(p);
                let jr = body.position.map(|k| rows.at(r, k));
                let jp = body.orientation.map(|k| rows.at(r, k));
                row[6 * i..6 * i + 3].copy_from_slice(&jr);
                // Columns for ω': Φ_r ∂ṙ/∂ω' + Φ_p ∂ṗ/∂ω', each column the image of a unit ω'.
                for k in 0..3 {
                    let mut e = [0.0; 3];
                    e[k] = 1.0;
                    let dr = scale(mat3_vec(&a, cross(e, body.mass.center)), -1.0);
                    let dp = p_dot(p, e);
                    row[6 * i + 3 + k] = dot(jr, dr) + jp.iter().zip(dp).map(|(x, y)| x * y).sum::<f64>();
                }
            }
            phi.push(rows.residual[r]);
            b.extend(row);
            owners.push(rows.owners[r]);
        }
        (phi, b, owners)
    }

    /// `γ = −(B u)_q q̇`, by a central difference along the motion.
    fn gamma(&self, q: &[f64], u: &[f64], m: usize) -> Vec<f64> {
        let qd = self.q_dot(q, u);
        let size = qd.iter().fold(0.0_f64, |s, v| s.max(v.abs()));
        if size == 0.0 {
            return vec![0.0; m];
        }
        let h = 1e-6 / size;
        let at = |sign: f64| {
            let shifted: Vec<f64> = q.iter().zip(&qd).map(|(x, d)| x + sign * h * d).collect();
            let (_, b, _) = self.velocity_rows(&shifted);
            mul(&b, u, m, self.dofs())
        };
        let (up, down) = (at(1.0), at(-1.0));
        up.iter().zip(&down).map(|(a, b)| -(a - b) / (2.0 * h)).collect()
    }

    fn inv_mass(&self, f: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; f.len()];
        for (i, b) in self.bodies.iter().enumerate() {
            for k in 0..3 {
                out[6 * i + k] = f[6 * i + k] / b.mass.mass;
            }
            let w = mat3_vec(&b.inv_inertia, [f[6 * i + 3], f[6 * i + 4], f[6 * i + 5]]);
            out[6 * i + 3..6 * i + 6].copy_from_slice(&w);
        }
        out
    }

    /// Adds a world force `f` at world point `x` (and a world torque `t`) on body `i` to the
    /// generalized forces.
    fn apply(&self, q: &[f64], gen: &mut [f64], i: usize, x: [f64; 3], f: [f64; 3], t: [f64; 3]) {
        let a = rotation(quat(q, &self.bodies[i].orientation));
        let moment = add(t, cross(sub(x, self.center(q, i)), f));
        let local = mat3_t_vec(&a, moment);
        for k in 0..3 {
            gen[6 * i + k] += f[k];
            gen[6 * i + 3 + k] += local[k];
        }
    }

    /// The mate's value and its rate.
    pub fn measure(&self, q: &[f64], u: &[f64], load: &MateLoad) -> (f64, f64) {
        let (fa, fb) = (self.connector(q, load.a), self.connector(q, load.b));
        let col = |t: &Transform, k: usize| [t.rotation[0][k], t.rotation[1][k], t.rotation[2][k]];
        let (xa, za, xb) = (col(&fa, 0), col(&fa, 2), col(&fb, 0));
        let motion = |c: rc::EntityId, at: [f64; 3]| {
            let rc::EntityKind::Connector { body, .. } = &self.sys.entity(c).kind else {
                return ([0.0; 3], [0.0; 3]);
            };
            match self.body_of_entity(*body) {
                Some(i) => self.point_velocity(q, u, i, at),
                None => ([0.0; 3], [0.0; 3]),
            }
        };
        let (va, wa) = motion(load.a, fb.translation);
        let (vb, wb) = motion(load.b, fb.translation);
        match load.measure {
            MateMeasure::Angle => (
                dot(cross(xa, xb), za).atan2(dot(xa, xb)),
                dot(sub(wb, wa), za),
            ),
            MateMeasure::Length => (
                dot(sub(fb.translation, fa.translation), za),
                dot(sub(vb, va), za),
            ),
        }
    }

    /// Generalized applied forces: gravity, the gyroscopic term and the mate loads.
    fn forces(&self, q: &[f64], u: &[f64]) -> Vec<f64> {
        let mut gen = vec![0.0; self.dofs()];
        for (i, b) in self.bodies.iter().enumerate() {
            let w = [u[6 * i + 3], u[6 * i + 4], u[6 * i + 5]];
            let gyro = cross(w, mat3_vec(&b.mass.inertia, w));
            for k in 0..3 {
                gen[6 * i + k] += b.mass.mass * self.gravity[k];
                gen[6 * i + 3 + k] -= gyro[k];
            }
        }
        for load in &self.loads {
            for i in 0..self.bodies.len() {
                if let Some((at, f, t)) = self.mate_load_on(q, u, load, i) {
                    self.apply(q, &mut gen, i, at, f, t);
                }
            }
        }
        gen
    }

    /// `u̇` and the reaction multipliers at `(q, u)`.
    fn accelerations(&self, q: &[f64], u: &[f64]) -> Result<Accelerations, String> {
        let n = self.dofs();
        let (phi, b, owners) = self.velocity_rows(q);
        let m = phi.len();
        let f = self.forces(q, u);
        let minv_f = self.inv_mass(&f);
        let bu = mul(&b, u, m, n);
        let gamma = self.gamma(q, u, m);
        let rhs: Vec<f64> = (0..m)
            .map(|r| {
                let stab = gamma[r] - 2.0 * BAUMGARTE * bu[r] - BAUMGARTE * BAUMGARTE * phi[r];
                stab - (0..n).map(|c| b[r * n + c] * minv_f[c]).sum::<f64>()
            })
            .collect();
        // B M⁻¹ Bᵀ.
        let minv_bt: Vec<Vec<f64>> = (0..m).map(|r| self.inv_mass(&b[r * n..(r + 1) * n])).collect();
        let mut s = vec![0.0; m * m];
        for r in 0..m {
            for c in 0..m {
                s[r * m + c] = (0..n).map(|k| b[r * n + k] * minv_bt[c][k]).sum();
            }
        }
        let lambda = solve_spd(&s, &rhs, m).ok_or("the mechanism's equations of motion are singular")?;
        let reaction = mul_t(&b, &lambda, m, n);
        let total: Vec<f64> = f.iter().zip(&reaction).map(|(a, r)| a + r).collect();
        Ok((self.inv_mass(&total), lambda, owners))
    }

    /// Renormalises every body's Euler parameters.
    fn normalize(&self, q: &mut [f64]) {
        for b in &self.bodies {
            let p = quat(q, &b.orientation);
            let n = p.iter().map(|v| v * v).sum::<f64>().sqrt();
            for k in 0..4 {
                q[b.orientation[k]] = p[k] / n;
            }
        }
    }

    /// Projects `q` onto `Φ = 0` by minimum-norm Newton steps over the free bodies' variables.
    fn project_positions(&self, q: &mut [f64]) -> Result<(), String> {
        let cols: Vec<usize> = self
            .bodies
            .iter()
            .flat_map(|b| b.position.into_iter().chain(b.orientation))
            .collect();
        for _ in 0..PROJECTION_STEPS {
            let rows = rc::constraint_rows(self.sys, q);
            let m = rows.rows();
            let err = rows.residual.iter().fold(0.0_f64, |s, v| s.max(v.abs()));
            if err <= PROJECTION_TOLERANCE {
                return Ok(());
            }
            let n = cols.len();
            let j: Vec<f64> = (0..m).flat_map(|r| cols.iter().map(move |&c| (r, c))).map(|(r, c)| rows.at(r, c)).collect();
            let mut s = vec![0.0; m * m];
            for r in 0..m {
                for c in 0..m {
                    s[r * m + c] = (0..n).map(|k| j[r * n + k] * j[c * n + k]).sum();
                }
            }
            let y = solve_spd(&s, &rows.residual, m).ok_or("the position projection is singular")?;
            let dx = mul_t(&j, &y, m, n);
            for (k, &c) in cols.iter().enumerate() {
                q[c] -= dx[k];
            }
            self.normalize(q);
        }
        let rows = rc::constraint_rows(self.sys, q);
        let err = rows.residual.iter().fold(0.0_f64, |s, v| s.max(v.abs()));
        if err <= 1e3 * PROJECTION_TOLERANCE {
            Ok(())
        } else {
            Err(format!("the mates no longer hold (error {err:.2e}): the mechanism locked or left its assembly"))
        }
    }

    /// Projects `u` onto `B u = 0`, mass-weighted.
    fn project_velocities(&self, q: &[f64], u: &mut [f64]) -> Result<(), String> {
        let n = self.dofs();
        let (_, b, _) = self.velocity_rows(q);
        let m = b.len() / n.max(1);
        let bu = mul(&b, u, m, n);
        let minv_bt: Vec<Vec<f64>> = (0..m).map(|r| self.inv_mass(&b[r * n..(r + 1) * n])).collect();
        let mut s = vec![0.0; m * m];
        for r in 0..m {
            for c in 0..m {
                s[r * m + c] = (0..n).map(|k| b[r * n + k] * minv_bt[c][k]).sum();
            }
        }
        let y = solve_spd(&s, &bu, m).ok_or("the velocity projection is singular")?;
        for (row, yr) in minv_bt.iter().zip(&y) {
            for (uk, mk) in u.iter_mut().zip(row) {
                *uk -= mk * yr;
            }
        }
        Ok(())
    }

    /// Kinetic plus gravitational plus spring energy (J): conserved without dampers or motors.
    pub fn energy(&self, q: &[f64], u: &[f64]) -> f64 {
        let mut e = 0.0;
        for (i, b) in self.bodies.iter().enumerate() {
            let v = [u[6 * i], u[6 * i + 1], u[6 * i + 2]];
            let w = [u[6 * i + 3], u[6 * i + 4], u[6 * i + 5]];
            e += 0.5 * b.mass.mass * dot(v, v) + 0.5 * dot(w, mat3_vec(&b.mass.inertia, w));
            e -= b.mass.mass * dot(self.gravity, self.center(q, i));
        }
        for load in &self.loads {
            let (x, _) = self.measure(q, u, load);
            e += 0.5 * load.stiffness * (x - load.free).powi(2);
        }
        e
    }

    /// The state's derivative.
    fn derivative(&self, q: &[f64], u: &[f64]) -> Result<(Vec<f64>, Vec<f64>), String> {
        let (ud, _, _) = self.accelerations(q, u)?;
        Ok((self.q_dot(q, u), ud))
    }

    /// One classical Runge–Kutta step of `h`, then the projections.
    fn step(&self, q: &mut [f64], u: &mut [f64], h: f64) -> Result<(), String> {
        let shift = |x: &[f64], d: &[f64], s: f64| -> Vec<f64> { x.iter().zip(d).map(|(a, b)| a + s * b).collect() };
        let (k1q, k1u) = self.derivative(q, u)?;
        let (k2q, k2u) = self.derivative(&shift(q, &k1q, h / 2.0), &shift(u, &k1u, h / 2.0))?;
        let (k3q, k3u) = self.derivative(&shift(q, &k2q, h / 2.0), &shift(u, &k2u, h / 2.0))?;
        let (k4q, k4u) = self.derivative(&shift(q, &k3q, h), &shift(u, &k3u, h))?;
        for k in 0..q.len() {
            q[k] += h / 6.0 * (k1q[k] + 2.0 * k2q[k] + 2.0 * k3q[k] + k4q[k]);
        }
        for k in 0..u.len() {
            u[k] += h / 6.0 * (k1u[k] + 2.0 * k2u[k] + 2.0 * k3u[k] + k4u[k]);
        }
        if q.iter().chain(u.iter()).any(|v| !v.is_finite()) {
            return Err("the motion diverged".into());
        }
        self.normalize(q);
        self.project_positions(q)?;
        self.project_velocities(q, u)
    }

    /// The instant `(q, u)` with its reactions.
    pub fn sample(&self, time: f64, q: &[f64], u: &[f64]) -> Result<Sample, String> {
        let (udot, lambda, owners) = self.accelerations(q, u)?;
        Ok(Sample {
            time,
            q: q.to_vec(),
            u: u.to_vec(),
            udot,
            lambda,
            owners,
        })
    }

    /// Integrates from `(q, u)` at rest or moving over `steps` equal steps of `h`, and returns
    /// the state `(t, q, u)` after every step (the first at time 0). On a failure the states so
    /// far stand and the error says when and why.
    pub fn integrate(&self, q0: &[f64], u0: &[f64], steps: usize, h: f64) -> (Vec<State>, Option<(f64, String)>) {
        let mut q = q0.to_vec();
        let mut u = u0.to_vec();
        self.normalize(&mut q);
        if let Err(e) = self.project_positions(&mut q).and_then(|_| self.project_velocities(&q, &mut u)) {
            return (Vec::new(), Some((0.0, e)));
        }
        let mut out = Vec::with_capacity(steps + 1);
        out.push(State { time: 0.0, q: q.clone(), u: u.clone() });
        for s in 0..steps {
            if let Err(e) = self.step(&mut q, &mut u, h) {
                return (out, Some((s as f64 * h, e)));
            }
            out.push(State { time: (s + 1) as f64 * h, q: q.clone(), u: u.clone() });
        }
        (out, None)
    }

    /// Integrates from `(q, u)` at rest or moving, reporting `frames` samples `frame_dt` apart
    /// (the first at time 0), each frame interval split into equal steps no longer than
    /// `max_step`. On a failure the samples so far stand and the error says when and why.
    pub fn simulate(
        &self,
        q0: &[f64],
        u0: &[f64],
        frames: usize,
        frame_dt: f64,
        max_step: f64,
        mut on_frame: impl FnMut(&Sample),
    ) -> Result<(), (f64, String)> {
        let per = (frame_dt / max_step).ceil().max(1.0) as usize;
        let (states, failed) = self.integrate(q0, u0, per * frames.saturating_sub(1), frame_dt / per as f64);
        for (k, s) in states.iter().step_by(per).enumerate() {
            let t = k as f64 * frame_dt;
            on_frame(&self.sample(t, &s.q, &s.u).map_err(|e| (t, e))?);
        }
        failed.map_or(Ok(()), Err)
    }

    /// Body `i`'s angular velocity at `u` (body axes, rad/s).
    pub fn body_rate(&self, u: &[f64], i: usize) -> [f64; 3] {
        [u[6 * i + 3], u[6 * i + 4], u[6 * i + 5]]
    }

    /// Body `i`'s centre-of-mass velocity at `u` (world, m/s).
    pub fn center_velocity(&self, u: &[f64], i: usize) -> [f64; 3] {
        [u[6 * i], u[6 * i + 1], u[6 * i + 2]]
    }

    /// The load `load` puts on body `i` at `(q, u)` (none when neither of its connectors is on
    /// that body): a world force acting at a world point and a world torque, the terms
    /// [`Model::forces`] applies.
    pub fn mate_load_on(&self, q: &[f64], u: &[f64], load: &MateLoad, i: usize) -> Option<([f64; 3], [f64; 3], [f64; 3])> {
        let (x, xd) = self.measure(q, u, load);
        let value = -load.stiffness * (x - load.free) - load.damping * xd + load.constant;
        let (fa, fb) = (self.connector(q, load.a), self.connector(q, load.b));
        let za = [fa.rotation[0][2], fa.rotation[1][2], fa.rotation[2][2]];
        let (f, t) = match load.measure {
            MateMeasure::Angle => ([0.0; 3], scale(za, value)),
            MateMeasure::Length => (scale(za, value), [0.0; 3]),
        };
        let mut out: Option<([f64; 3], [f64; 3], [f64; 3])> = None;
        for (c, sign) in [(load.b, 1.0), (load.a, -1.0)] {
            let rc::EntityKind::Connector { body, .. } = &self.sys.entity(c).kind else { continue };
            if self.body_of_entity(*body) == Some(i) {
                let (pf, pt) = out.map_or(([0.0; 3], [0.0; 3]), |o| (o.1, o.2));
                out = Some((fb.translation, add(pf, scale(f, sign)), add(pt, scale(t, sign))));
            }
        }
        out
    }

    /// The reaction a set of constraint rows puts on body `i`: a world force at its centre of
    /// mass and a world moment about it.
    pub fn reaction_on(&self, sample: &Sample, rows: &[usize], i: usize) -> ([f64; 3], [f64; 3]) {
        let q = &sample.q;
        let (_, b, _) = self.velocity_rows(q);
        let n = self.dofs();
        let mut gen = [0.0; 6];
        for &r in rows {
            for (k, g) in gen.iter_mut().enumerate() {
                *g += b[r * n + 6 * i + k] * sample.lambda[r];
            }
        }
        let a = rotation(quat(q, &self.bodies[i].orientation));
        ([gen[0], gen[1], gen[2]], mat3_vec(&a, [gen[3], gen[4], gen[5]]))
    }

    /// The model of a kinematic study's pose: every free body of `sys` weighs alike, with its
    /// centre at its frame's origin. A driven mechanism's velocities follow from its constraints
    /// alone ([`Model::driven_rates`]), so mass doesn't enter them.
    pub fn kinematic(sys: &'a rc::System, bodies: &[rc::EntityId]) -> Result<Model<'a>, String> {
        let unit = BodyMass {
            mass: 1.0,
            center: [0.0; 3],
            inertia: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        };
        let bodies: Vec<_> = bodies.iter().map(|&e| (e, unit)).collect();
        Self::new(sys, &bodies, [0.0; 3], Vec::new())
    }

    /// The velocities `u` and their rates `u̇` of the solved pose `q` of a kinematic study, from
    /// the constraint Jacobian (Haug 1989, ch. 3): the velocity equations `B u = 0` of every
    /// mate row except the driven mates' own value rows, which the drives replace with the rate
    /// of each mate's value, `ẋ = ṡ`; the acceleration equations `B u̇ = γ` with
    /// `ẍ = s̈ − (∂ẋ/∂q) q̇`. `ṡ` and `s̈` come from the drive law, so nothing here differences
    /// the solved poses; `γ` and the drive's own `(∂ẋ/∂q) q̇` take the same central difference
    /// of the exact Jacobian along the motion as the dynamic study does.
    pub fn driven_rates(
        &self,
        q: &[f64],
        drives: &[Driven],
    ) -> Result<(Vec<f64>, Vec<f64>), String> {
        let n = self.dofs();
        let (_, b, owners) = self.velocity_rows(q);
        let keep: Vec<usize> = (0..owners.len())
            .filter(|&r| !drives.iter().any(|d| d.rows.contains(&owners[r])))
            .collect();
        // `ẋ` is linear in `u`: its row is its value at each unit velocity.
        let drive_row = |q: &[f64], d: &Driven| -> Vec<f64> {
            (0..n)
                .map(|j| {
                    let mut e = vec![0.0; n];
                    e[j] = 1.0;
                    self.measure(q, &e, &d.mate).1
                })
                .collect()
        };
        let mut a: Vec<f64> = keep
            .iter()
            .flat_map(|&r| b[r * n..(r + 1) * n].iter().copied())
            .collect();
        for d in drives {
            a.extend(drive_row(q, d));
        }
        let m = keep.len() + drives.len();
        let solve = |rhs: &[f64]| -> Result<Vec<f64>, String> {
            // The minimum-norm solution `Aᵀ (A Aᵀ)⁻¹ rhs`; a driven mechanism has one.
            let mut s = vec![0.0; m * m];
            for r in 0..m {
                for c in 0..m {
                    s[r * m + c] = (0..n).map(|k| a[r * n + k] * a[c * n + k]).sum();
                }
            }
            let y =
                solve_spd(&s, rhs, m).ok_or("the mechanism's velocity equations are singular")?;
            Ok(mul_t(&a, &y, m, n))
        };
        let mut rhs = vec![0.0; m];
        for (k, d) in drives.iter().enumerate() {
            rhs[keep.len() + k] = d.rate;
        }
        let u = solve(&rhs)?;
        let gamma = self.gamma(q, &u, owners.len());
        let qd = self.q_dot(q, &u);
        let size = qd.iter().fold(0.0_f64, |s, v| s.max(v.abs()));
        let mut rhs: Vec<f64> = keep.iter().map(|&r| gamma[r]).collect();
        for d in drives {
            let curvature = if size == 0.0 {
                0.0
            } else {
                let h = 1e-6 / size;
                let at = |sign: f64| {
                    let shifted: Vec<f64> =
                        q.iter().zip(&qd).map(|(x, v)| x + sign * h * v).collect();
                    self.measure(&shifted, &u, &d.mate).1
                };
                (at(1.0) - at(-1.0)) / (2.0 * h)
            };
            rhs.push(d.acceleration - curvature);
        }
        let udot = solve(&rhs)?;
        if u.iter().chain(&udot).any(|v| !v.is_finite()) {
            return Err("the mechanism's velocities aren't finite".into());
        }
        Ok((u, udot))
    }

    /// Body `i`'s motion at `(q, u, u̇)`, world axes.
    pub fn body_motion(&self, q: &[f64], u: &[f64], udot: &[f64], i: usize) -> BodyRate {
        let a = rotation(quat(q, &self.bodies[i].orientation));
        let turn = |k: usize, x: &[f64]| mat3_vec(&a, [x[6 * k + 3], x[6 * k + 4], x[6 * k + 5]]);
        BodyRate {
            center: self.center(q, i),
            velocity: self.center_velocity(u, i),
            acceleration: [udot[6 * i], udot[6 * i + 1], udot[6 * i + 2]],
            omega: turn(i, u),
            // d/dt (A ω') = A ω̇' (ω × ω vanishes).
            alpha: turn(i, udot),
        }
    }

    /// Index of the free body with entity `e`.
    pub fn body_index(&self, e: rc::EntityId) -> Option<usize> {
        self.body_of_entity(e)
    }
}
