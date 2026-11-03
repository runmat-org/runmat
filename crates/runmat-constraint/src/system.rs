//! The constraint system: variables, entities and constraints.
//!
//! A [`System`] is built incrementally. Entities own variables (a point owns `x` and `y`, a
//! circle owns its radius, a rigid body owns a position and a quaternion). Constraints refer to
//! entities and are validated when they are added, so a solve never meets a malformed system.

use crate::ids::{ConstraintId, EntityId, VarId};
use std::fmt;

/// How the solver treats a variable.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VarFlag {
    /// An unknown the solver may move.
    Free,
    /// A constant held at its current value (grounded geometry).
    Fixed,
    /// A constant whose value comes from outside the sketch (an input parameter or a value
    /// driven by another solve). The solver treats it like [`VarFlag::Fixed`]; the distinction
    /// is kept so the CAD layer can report it differently.
    Driven,
}

/// What a variable means inside its entity. Used for diagnostics and warm-start keys.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum VarRole {
    X,
    Y,
    Z,
    Radius,
    SemiMajor,
    SemiMinor,
    Rotation,
    QuatW,
    QuatX,
    QuatY,
    QuatZ,
    /// A curve parameter introduced by a point-on-spline constraint.
    CurveParameter,
    /// A variable created through [`System::add_var`].
    User,
}

/// A scalar unknown.
#[derive(Clone, Debug)]
pub struct Variable {
    pub value: f64,
    pub flag: VarFlag,
    pub role: VarRole,
    /// The entity that owns this variable, if any.
    pub owner: Option<EntityId>,
}

/// A rigid frame: an origin and a unit quaternion `[w, x, y, z]`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Frame {
    pub origin: [f64; 3],
    pub rotation: [f64; 4],
}

impl Frame {
    pub const IDENTITY: Frame = Frame { origin: [0.0; 3], rotation: [1.0, 0.0, 0.0, 0.0] };

    pub fn at(origin: [f64; 3]) -> Frame {
        Frame { origin, rotation: [1.0, 0.0, 0.0, 0.0] }
    }

    /// A frame rotated by `angle` radians about the unit `axis`.
    pub fn axis_angle(origin: [f64; 3], axis: [f64; 3], angle: f64) -> Frame {
        let n = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
        let (s, c) = (0.5 * angle).sin_cos();
        let k = if n > 0.0 { s / n } else { 0.0 };
        Frame { origin, rotation: [c, axis[0] * k, axis[1] * k, axis[2] * k] }
    }

    /// A frame whose local Z axis points along `z` (the X axis is chosen deterministically).
    pub fn with_z(origin: [f64; 3], z: [f64; 3]) -> Frame {
        let n = (z[0] * z[0] + z[1] * z[1] + z[2] * z[2]).sqrt();
        let z = [z[0] / n, z[1] / n, z[2] / n];
        // Rotation taking +Z to z.
        let d = z[2];
        if d > 1.0 - 1e-12 {
            return Frame::at(origin);
        }
        if d < -1.0 + 1e-12 {
            return Frame::axis_angle(origin, [1.0, 0.0, 0.0], std::f64::consts::PI);
        }
        let axis = [-z[1], z[0], 0.0];
        Frame::axis_angle(origin, axis, d.acos())
    }
}

/// Which end of an open curve.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CurveEnd {
    Start,
    End,
}

/// The kinds of entity.
#[derive(Clone, Debug)]
pub enum EntityKind {
    /// A 2D point.
    Point2 { x: VarId, y: VarId },
    /// A segment between two points.
    Line { start: EntityId, end: EntityId },
    /// A full circle.
    Circle { center: EntityId, radius: VarId },
    /// A counter-clockwise arc from `start` to `end` about `center`. An internal constraint
    /// keeps `|start - center| = |end - center|`; the radius is that distance.
    Arc { center: EntityId, start: EntityId, end: EntityId },
    /// A full ellipse: centre, semi-axes and the rotation (radians) of the major axis.
    Ellipse { center: EntityId, semi_major: VarId, semi_minor: VarId, rotation: VarId },
    /// A non-rational B-spline given by control points and a clamped knot vector.
    BSpline { degree: usize, control_points: Vec<EntityId>, knots: Vec<f64> },
    /// A rigid body with 6 DOF: position and a quaternion `[w, x, y, z]` (normalised by an
    /// internal constraint).
    Body { position: [VarId; 3], orientation: [VarId; 4] },
    /// A mate connector: a frame fixed in a body, supplied by the CAD layer.
    Connector { body: EntityId, local: Frame },
}

/// An entity and its flags.
#[derive(Clone, Debug)]
pub struct Entity {
    pub kind: EntityKind,
    /// Construction geometry: solved like any other entity, flagged for the CAD layer so it
    /// does not become part of a profile.
    pub construction: bool,
}

/// A dimension target for mate distance and angle constraints.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Target {
    /// The measured value must equal this.
    Exact(f64),
    /// The measured value must lie in `[min, max]`. Bounds are clamped: when the unconstrained
    /// solve violates a bound, the bound becomes an equality and the solve repeats.
    Range { min: f64, max: f64 },
}

/// How a mate distance is measured.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DistanceMode {
    /// Between the two connector origins.
    PointPoint,
    /// From connector A's XY plane to connector B's origin, along A's Z axis (signed).
    PlanePoint,
}

/// How a mate angle is measured.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AngleMode {
    /// The signed angle from A's X axis to B's X axis, measured about A's Z axis. Use it to
    /// drive a revolute joint.
    AboutZ,
    /// The unsigned angle between the two Z axes, in `[0, π]`.
    BetweenZ,
}

/// Geometry pairs for the tangent mate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TangentKind {
    /// A sphere of this radius centred at B's origin touches A's XY plane.
    PlaneSphere { radius: f64 },
    /// A cylinder of this radius along B's Z axis touches A's XY plane.
    PlaneCylinder { radius: f64 },
}

/// 3D mates between two connectors.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Mate {
    /// Both frames coincide (0 DOF left between the bodies).
    Fasten { a: EntityId, b: EntityId },
    /// Origins coincide and Z axes align (1 DOF: rotation about Z).
    Revolute { a: EntityId, b: EntityId },
    /// Z axes align, B's origin is on A's Z axis, rotation about Z locked (1 DOF: translation).
    Slider { a: EntityId, b: EntityId },
    /// Z axes align and B's origin is on A's Z axis (2 DOF).
    Cylindrical { a: EntityId, b: EntityId },
    /// Z axes align and B's origin lies in A's XY plane (3 DOF).
    Planar { a: EntityId, b: EntityId },
    /// Origins coincide (3 DOF).
    Ball { a: EntityId, b: EntityId },
    /// Z axes parallel (4 DOF).
    Parallel { a: EntityId, b: EntityId },
    /// Tangency between A's XY plane and a sphere or cylinder on B.
    Tangent { a: EntityId, b: EntityId, kind: TangentKind },
    Distance { a: EntityId, b: EntityId, mode: DistanceMode, target: Target },
    Angle { a: EntityId, b: EntityId, mode: AngleMode, target: Target },
}

/// Constraints. 2D variants take point and curve entities; [`Constraint::Mate`] takes
/// connectors. Angles are radians; lengths are in the caller's unit.
#[derive(Clone, Debug, PartialEq)]
pub enum Constraint {
    /// Two points coincide.
    Coincident(EntityId, EntityId),
    /// A line is horizontal.
    Horizontal(EntityId),
    /// Two points share a y coordinate.
    HorizontalPoints(EntityId, EntityId),
    /// A line is vertical.
    Vertical(EntityId),
    /// Two points share an x coordinate.
    VerticalPoints(EntityId, EntityId),
    Parallel(EntityId, EntityId),
    Perpendicular(EntityId, EntityId),
    /// Line–circle, line–arc, or circle/arc–circle/arc tangency. The side (and internal vs
    /// external contact) is kept from the starting geometry.
    Tangent(EntityId, EntityId),
    /// G1 continuity where two open curves (line, arc or spline) meet at the given ends.
    /// Coincidence of the ends is a separate [`Constraint::Coincident`].
    TangentAtEnd { a: EntityId, a_end: CurveEnd, b: EntityId, b_end: CurveEnd },
    /// G2 continuity between a spline end and an arc: tangent directions and curvature match.
    /// Coincidence is separate. Needs spline degree ≥ 2.
    CurvatureContinuity { spline: EntityId, end: CurveEnd, arc: EntityId },
    EqualLength(EntityId, EntityId),
    EqualRadius(EntityId, EntityId),
    /// Circles/arcs share a centre.
    Concentric(EntityId, EntityId),
    /// Circles/arcs share a centre and a radius.
    Coradial(EntityId, EntityId),
    Midpoint { point: EntityId, line: EntityId },
    /// `a` and `b` are mirror images in the line `axis`.
    Symmetric { a: EntityId, b: EntityId, axis: EntityId },
    Collinear(EntityId, EntityId),
    /// Holds an entity at its current values (points, lines, circles, arcs, ellipses,
    /// splines, bodies).
    Fix(EntityId),
    /// A point lies on a line (infinite), circle, arc (its circle), ellipse or spline.
    PointOnCurve { point: EntityId, curve: EntityId },
    /// Point–point distance, or unsigned point–line distance when one side is a line.
    Distance { a: EntityId, b: EntityId, value: f64 },
    /// The signed angle from line `a`'s direction to line `b`'s, radians.
    Angle { a: EntityId, b: EntityId, value: f64 },
    Radius { curve: EntityId, value: f64 },
    Diameter { curve: EntityId, value: f64 },
    Length { line: EntityId, value: f64 },
    /// A raw variable equals a value (for example an ellipse semi-axis).
    VarValue { var: VarId, value: f64 },
    Mate(Mate),
}

impl Constraint {
    /// Whether the constraint has a numeric value a reference (driven) dimension can measure.
    pub fn is_dimension(&self) -> bool {
        matches!(
            self,
            Constraint::Distance { .. }
                | Constraint::Angle { .. }
                | Constraint::Radius { .. }
                | Constraint::Diameter { .. }
                | Constraint::Length { .. }
                | Constraint::VarValue { .. }
                | Constraint::Mate(Mate::Distance { target: Target::Exact(_), .. })
                | Constraint::Mate(Mate::Angle { target: Target::Exact(_), .. })
        )
    }
}

/// Why something was internal-added.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InternalKind {
    /// Keeps an arc's two ends at the same radius.
    ArcRadius,
    /// Keeps a body's quaternion at unit length.
    QuaternionNorm,
}

/// A stored constraint.
#[derive(Clone, Debug)]
pub struct ConstraintRecord {
    pub constraint: ConstraintKindRecord,
    /// A reference ("driven") dimension: measured and reported, never enforced.
    pub reference: bool,
    /// Variables owned by this constraint (the curve parameter of a point-on-spline).
    pub aux: Vec<VarId>,
    /// Fix targets captured when the constraint was added.
    pub(crate) fix_targets: Vec<f64>,
}

/// User constraint or an internal one.
#[derive(Clone, Debug, PartialEq)]
pub enum ConstraintKindRecord {
    User(Constraint),
    Internal { kind: InternalKind, entity: EntityId },
}

/// A constraint the system refused.
#[derive(Clone, Debug, PartialEq)]
pub enum BuildError {
    UnknownEntity(EntityId),
    UnknownVar(VarId),
    /// The entity has the wrong kind for this argument.
    WrongKind { entity: EntityId, expected: &'static str },
    /// Both arguments are the same entity where two are needed.
    SameEntity(EntityId),
    /// A B-spline needs `degree ≥ 1`, at least `degree + 1` control points, and at most
    /// [`MAX_SPLINE_CONTROL_POINTS`] of them.
    BadSpline(&'static str),
    /// A value is not finite, or a dimension is negative where it must not be.
    BadValue(&'static str),
    /// The connector belongs to the same body on both sides of a mate.
    SameBody(EntityId),
    /// A reference dimension needs a dimensional constraint.
    NotADimension,
}

impl fmt::Display for BuildError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BuildError::UnknownEntity(e) => write!(f, "unknown entity {e}"),
            BuildError::UnknownVar(v) => write!(f, "unknown variable {v}"),
            BuildError::WrongKind { entity, expected } => {
                write!(f, "entity {entity} is not {expected}")
            }
            BuildError::SameEntity(e) => write!(f, "constraint uses {e} twice"),
            BuildError::BadSpline(why) => write!(f, "invalid B-spline: {why}"),
            BuildError::BadValue(why) => write!(f, "invalid value: {why}"),
            BuildError::SameBody(e) => write!(f, "mate connects body {e} to itself"),
            BuildError::NotADimension => write!(f, "only dimensions can be reference constraints"),
        }
    }
}

impl std::error::Error for BuildError {}

/// The largest spline the solver accepts (keeps per-constraint derivative width bounded).
pub const MAX_SPLINE_CONTROL_POINTS: usize = 30;

/// A constraint system.
#[derive(Clone, Debug, Default)]
pub struct System {
    pub(crate) vars: Vec<Variable>,
    pub(crate) entities: Vec<Entity>,
    pub(crate) constraints: Vec<ConstraintRecord>,
}

impl System {
    pub fn new() -> Self {
        Self::default()
    }

    // ---------------------------------------------------------------- access

    pub fn vars(&self) -> &[Variable] {
        &self.vars
    }
    pub fn var(&self, v: VarId) -> &Variable {
        &self.vars[v.index()]
    }
    pub fn value(&self, v: VarId) -> f64 {
        self.vars[v.index()].value
    }
    pub fn entities(&self) -> &[Entity] {
        &self.entities
    }
    pub fn entity(&self, e: EntityId) -> &Entity {
        &self.entities[e.index()]
    }
    pub fn constraints(&self) -> &[ConstraintRecord] {
        &self.constraints
    }
    pub fn constraint(&self, c: ConstraintId) -> &ConstraintRecord {
        &self.constraints[c.index()]
    }
    pub fn var_count(&self) -> usize {
        self.vars.len()
    }

    /// Sets a variable's value (for example to apply a solve or a parameter change).
    pub fn set_value(&mut self, v: VarId, value: f64) {
        self.vars[v.index()].value = value;
    }
    pub fn set_flag(&mut self, v: VarId, flag: VarFlag) {
        self.vars[v.index()].flag = flag;
    }

    /// Writes all values of a finished solve back into the system.
    pub fn apply(&mut self, values: &[f64]) {
        for (var, &v) in self.vars.iter_mut().zip(values) {
            var.value = v;
        }
    }

    /// Sets every variable of `e` (and of the entities it is built from) to `flag`.
    pub fn set_entity_flag(&mut self, e: EntityId, flag: VarFlag) {
        for v in self.entity_vars(e) {
            self.vars[v.index()].flag = flag;
        }
    }

    pub fn set_construction(&mut self, e: EntityId, construction: bool) {
        self.entities[e.index()].construction = construction;
    }

    /// The 2D coordinates of a point.
    pub fn point(&self, p: EntityId) -> [f64; 2] {
        match &self.entities[p.index()].kind {
            EntityKind::Point2 { x, y } => [self.value(*x), self.value(*y)],
            _ => panic!("{p} is not a point"),
        }
    }

    /// Every variable an entity is defined by, in a fixed order (a line's are its endpoints'
    /// coordinates; a connector's are its body's).
    pub fn entity_vars(&self, e: EntityId) -> Vec<VarId> {
        let mut out = Vec::new();
        self.collect_vars(e, &mut out);
        out
    }

    fn collect_vars(&self, e: EntityId, out: &mut Vec<VarId>) {
        match &self.entities[e.index()].kind {
            EntityKind::Point2 { x, y } => out.extend([*x, *y]),
            EntityKind::Line { start, end } => {
                self.collect_vars(*start, out);
                self.collect_vars(*end, out);
            }
            EntityKind::Circle { center, radius } => {
                self.collect_vars(*center, out);
                out.push(*radius);
            }
            EntityKind::Arc { center, start, end } => {
                self.collect_vars(*center, out);
                self.collect_vars(*start, out);
                self.collect_vars(*end, out);
            }
            EntityKind::Ellipse { center, semi_major, semi_minor, rotation } => {
                self.collect_vars(*center, out);
                out.extend([*semi_major, *semi_minor, *rotation]);
            }
            EntityKind::BSpline { control_points, .. } => {
                for p in control_points {
                    self.collect_vars(*p, out);
                }
            }
            EntityKind::Body { position, orientation } => {
                out.extend(position.iter().copied());
                out.extend(orientation.iter().copied());
            }
            EntityKind::Connector { body, .. } => self.collect_vars(*body, out),
        }
    }

    // ---------------------------------------------------------------- building

    /// Adds a free-standing variable (generic core use).
    pub fn add_var(&mut self, value: f64, flag: VarFlag) -> VarId {
        self.push_var(value, flag, VarRole::User, None)
    }

    fn push_var(&mut self, value: f64, flag: VarFlag, role: VarRole, owner: Option<EntityId>) -> VarId {
        let id = VarId(self.vars.len() as u32);
        self.vars.push(Variable { value, flag, role, owner });
        id
    }

    fn next_entity(&self) -> EntityId {
        EntityId(self.entities.len() as u32)
    }

    fn push_entity(&mut self, kind: EntityKind) -> EntityId {
        let id = self.next_entity();
        self.entities.push(Entity { kind, construction: false });
        id
    }

    /// A free 2D point.
    pub fn point2(&mut self, x: f64, y: f64) -> EntityId {
        let e = self.next_entity();
        let vx = self.push_var(x, VarFlag::Free, VarRole::X, Some(e));
        let vy = self.push_var(y, VarFlag::Free, VarRole::Y, Some(e));
        self.push_entity(EntityKind::Point2 { x: vx, y: vy })
    }

    pub fn line(&mut self, start: EntityId, end: EntityId) -> Result<EntityId, BuildError> {
        self.expect_point(start)?;
        self.expect_point(end)?;
        if start == end {
            return Err(BuildError::SameEntity(start));
        }
        Ok(self.push_entity(EntityKind::Line { start, end }))
    }

    /// A line with two new endpoints.
    pub fn line_xy(&mut self, a: [f64; 2], b: [f64; 2]) -> EntityId {
        let pa = self.point2(a[0], a[1]);
        let pb = self.point2(b[0], b[1]);
        self.line(pa, pb).expect("fresh points")
    }

    pub fn circle(&mut self, center: EntityId, radius: f64) -> Result<EntityId, BuildError> {
        self.expect_point(center)?;
        finite(radius)?;
        let e = self.next_entity();
        let r = self.push_var(radius, VarFlag::Free, VarRole::Radius, Some(e));
        Ok(self.push_entity(EntityKind::Circle { center, radius: r }))
    }

    /// An arc; adds the internal equal-radius constraint.
    pub fn arc(&mut self, center: EntityId, start: EntityId, end: EntityId) -> Result<EntityId, BuildError> {
        self.expect_point(center)?;
        self.expect_point(start)?;
        self.expect_point(end)?;
        if center == start || center == end || start == end {
            return Err(BuildError::SameEntity(center));
        }
        let e = self.push_entity(EntityKind::Arc { center, start, end });
        self.constraints.push(ConstraintRecord {
            constraint: ConstraintKindRecord::Internal { kind: InternalKind::ArcRadius, entity: e },
            reference: false,
            aux: Vec::new(),
            fix_targets: Vec::new(),
        });
        Ok(e)
    }

    pub fn ellipse(
        &mut self,
        center: EntityId,
        semi_major: f64,
        semi_minor: f64,
        rotation: f64,
    ) -> Result<EntityId, BuildError> {
        self.expect_point(center)?;
        finite(semi_major)?;
        finite(semi_minor)?;
        finite(rotation)?;
        if semi_major <= 0.0 || semi_minor <= 0.0 {
            return Err(BuildError::BadValue("ellipse semi-axes must be positive"));
        }
        let e = self.next_entity();
        let a = self.push_var(semi_major, VarFlag::Free, VarRole::SemiMajor, Some(e));
        let b = self.push_var(semi_minor, VarFlag::Free, VarRole::SemiMinor, Some(e));
        let r = self.push_var(rotation, VarFlag::Free, VarRole::Rotation, Some(e));
        Ok(self.push_entity(EntityKind::Ellipse { center, semi_major: a, semi_minor: b, rotation: r }))
    }

    /// A clamped B-spline with a uniform knot vector on `[0, 1]`.
    pub fn bspline(&mut self, degree: usize, control_points: &[EntityId]) -> Result<EntityId, BuildError> {
        let n = control_points.len();
        if degree == 0 || n < degree + 1 {
            return Err(BuildError::BadSpline("need degree >= 1 and at least degree + 1 control points"));
        }
        let interior = n - degree - 1;
        let mut knots = vec![0.0; degree + 1];
        for i in 1..=interior {
            knots.push(i as f64 / (interior + 1) as f64);
        }
        knots.extend(std::iter::repeat(1.0).take(degree + 1));
        self.bspline_with_knots(degree, control_points, knots)
    }

    /// A B-spline with explicit knots. Knots must be non-decreasing, clamped
    /// (`degree + 1` equal values at each end) and number `control_points + degree + 1`.
    pub fn bspline_with_knots(
        &mut self,
        degree: usize,
        control_points: &[EntityId],
        knots: Vec<f64>,
    ) -> Result<EntityId, BuildError> {
        let n = control_points.len();
        if degree == 0 || n < degree + 1 {
            return Err(BuildError::BadSpline("need degree >= 1 and at least degree + 1 control points"));
        }
        if n > MAX_SPLINE_CONTROL_POINTS {
            return Err(BuildError::BadSpline("too many control points"));
        }
        if knots.len() != n + degree + 1 {
            return Err(BuildError::BadSpline("knot count must be control points + degree + 1"));
        }
        if knots.windows(2).any(|w| !(w[1] >= w[0])) || knots.iter().any(|k| !k.is_finite()) {
            return Err(BuildError::BadSpline("knots must be finite and non-decreasing"));
        }
        let (k0, k1) = (knots[0], knots[knots.len() - 1]);
        if knots[..=degree].iter().any(|&k| k != k0) || knots[n..].iter().any(|&k| k != k1) || k1 <= k0 {
            return Err(BuildError::BadSpline("knots must be clamped"));
        }
        for p in control_points {
            self.expect_point(*p)?;
        }
        Ok(self.push_entity(EntityKind::BSpline { degree, control_points: control_points.to_vec(), knots }))
    }

    /// A rigid body at `pose` (6 DOF); adds the internal quaternion-norm constraint.
    pub fn body(&mut self, pose: Frame) -> EntityId {
        let e = self.next_entity();
        let roles = [VarRole::X, VarRole::Y, VarRole::Z];
        let position = [0, 1, 2].map(|i| self.push_var(pose.origin[i], VarFlag::Free, roles[i], Some(e)));
        let q = normalize4(pose.rotation);
        let qroles = [VarRole::QuatW, VarRole::QuatX, VarRole::QuatY, VarRole::QuatZ];
        let orientation = [0, 1, 2, 3].map(|i| self.push_var(q[i], VarFlag::Free, qroles[i], Some(e)));
        let id = self.push_entity(EntityKind::Body { position, orientation });
        self.constraints.push(ConstraintRecord {
            constraint: ConstraintKindRecord::Internal { kind: InternalKind::QuaternionNorm, entity: id },
            reference: false,
            aux: Vec::new(),
            fix_targets: Vec::new(),
        });
        id
    }

    /// A grounded body: all its variables are [`VarFlag::Fixed`].
    pub fn ground(&mut self, pose: Frame) -> EntityId {
        let b = self.body(pose);
        self.set_entity_flag(b, VarFlag::Fixed);
        b
    }

    /// A mate connector fixed in `body` at `local` (body coordinates).
    pub fn connector(&mut self, body: EntityId, local: Frame) -> Result<EntityId, BuildError> {
        self.expect(body, "a body", |k| matches!(k, EntityKind::Body { .. }))?;
        let local = Frame { origin: local.origin, rotation: normalize4(local.rotation) };
        Ok(self.push_entity(EntityKind::Connector { body, local }))
    }

    /// Adds an enforced constraint.
    pub fn constrain(&mut self, c: Constraint) -> Result<ConstraintId, BuildError> {
        self.add_constraint(c, false)
    }

    /// Adds a reference (driven) dimension: its value is measured and reported, not enforced.
    pub fn reference(&mut self, c: Constraint) -> Result<ConstraintId, BuildError> {
        if !c.is_dimension() {
            return Err(BuildError::NotADimension);
        }
        self.add_constraint(c, true)
    }

    fn add_constraint(&mut self, c: Constraint, reference: bool) -> Result<ConstraintId, BuildError> {
        self.validate(&c)?;
        let id = ConstraintId(self.constraints.len() as u32);
        let mut aux = Vec::new();
        let mut fix_targets = Vec::new();
        if let Constraint::PointOnCurve { point, curve } = &c {
            if let EntityKind::BSpline { .. } = self.entities[curve.index()].kind {
                let t0 = crate::residual::closest_spline_parameter(self, *curve, self.point(*point));
                aux.push(self.push_var(t0, VarFlag::Free, VarRole::CurveParameter, None));
            }
        }
        if let Constraint::Fix(e) = &c {
            fix_targets = self.entity_vars(*e).iter().map(|v| self.value(*v)).collect();
        }
        self.constraints.push(ConstraintRecord {
            constraint: ConstraintKindRecord::User(c),
            reference,
            aux,
            fix_targets,
        });
        Ok(id)
    }

    // ---------------------------------------------------------------- validation

    fn get(&self, e: EntityId) -> Result<&EntityKind, BuildError> {
        self.entities.get(e.index()).map(|x| &x.kind).ok_or(BuildError::UnknownEntity(e))
    }

    fn expect(
        &self,
        e: EntityId,
        expected: &'static str,
        ok: impl Fn(&EntityKind) -> bool,
    ) -> Result<(), BuildError> {
        if ok(self.get(e)?) {
            Ok(())
        } else {
            Err(BuildError::WrongKind { entity: e, expected })
        }
    }

    fn expect_point(&self, e: EntityId) -> Result<(), BuildError> {
        self.expect(e, "a point", |k| matches!(k, EntityKind::Point2 { .. }))
    }
    fn expect_line(&self, e: EntityId) -> Result<(), BuildError> {
        self.expect(e, "a line", |k| matches!(k, EntityKind::Line { .. }))
    }
    fn expect_round(&self, e: EntityId) -> Result<(), BuildError> {
        self.expect(e, "a circle or arc", |k| matches!(k, EntityKind::Circle { .. } | EntityKind::Arc { .. }))
    }
    fn expect_connector(&self, e: EntityId) -> Result<EntityId, BuildError> {
        match self.get(e)? {
            EntityKind::Connector { body, .. } => Ok(*body),
            _ => Err(BuildError::WrongKind { entity: e, expected: "a mate connector" }),
        }
    }
    fn expect_open(&self, e: EntityId) -> Result<(), BuildError> {
        self.expect(e, "a line, arc or spline", |k| {
            matches!(k, EntityKind::Line { .. } | EntityKind::Arc { .. } | EntityKind::BSpline { .. })
        })
    }
    fn distinct(a: EntityId, b: EntityId) -> Result<(), BuildError> {
        if a == b {
            Err(BuildError::SameEntity(a))
        } else {
            Ok(())
        }
    }

    fn validate(&self, c: &Constraint) -> Result<(), BuildError> {
        use Constraint as C;
        match c {
            C::Coincident(a, b) | C::HorizontalPoints(a, b) | C::VerticalPoints(a, b) => {
                self.expect_point(*a)?;
                self.expect_point(*b)?;
                Self::distinct(*a, *b)
            }
            C::Horizontal(l) | C::Vertical(l) => self.expect_line(*l),
            C::Parallel(a, b) | C::Perpendicular(a, b) | C::EqualLength(a, b) | C::Collinear(a, b) => {
                self.expect_line(*a)?;
                self.expect_line(*b)?;
                Self::distinct(*a, *b)
            }
            C::Tangent(a, b) => {
                Self::distinct(*a, *b)?;
                let la = matches!(self.get(*a)?, EntityKind::Line { .. });
                let lb = matches!(self.get(*b)?, EntityKind::Line { .. });
                match (la, lb) {
                    (true, true) => Err(BuildError::WrongKind { entity: *b, expected: "a circle or arc" }),
                    (true, false) => self.expect_round(*b),
                    (false, true) => self.expect_round(*a),
                    (false, false) => {
                        self.expect_round(*a)?;
                        self.expect_round(*b)
                    }
                }
            }
            C::TangentAtEnd { a, b, .. } => {
                Self::distinct(*a, *b)?;
                self.expect_open(*a)?;
                self.expect_open(*b)
            }
            C::CurvatureContinuity { spline, arc, .. } => {
                match self.get(*spline)? {
                    EntityKind::BSpline { degree, control_points, .. } => {
                        if *degree < 2 || control_points.len() < 3 {
                            return Err(BuildError::BadSpline("G2 needs degree >= 2 and 3 control points"));
                        }
                    }
                    _ => return Err(BuildError::WrongKind { entity: *spline, expected: "a B-spline" }),
                }
                self.expect(*arc, "an arc", |k| matches!(k, EntityKind::Arc { .. }))
            }
            C::EqualRadius(a, b) | C::Concentric(a, b) | C::Coradial(a, b) => {
                self.expect_round(*a)?;
                self.expect_round(*b)?;
                Self::distinct(*a, *b)
            }
            C::Midpoint { point, line } => {
                self.expect_point(*point)?;
                self.expect_line(*line)
            }
            C::Symmetric { a, b, axis } => {
                self.expect_point(*a)?;
                self.expect_point(*b)?;
                Self::distinct(*a, *b)?;
                self.expect_line(*axis)
            }
            C::Fix(e) => self.get(*e).map(|_| ()).and_then(|_| {
                self.expect(*e, "a fixable entity", |k| !matches!(k, EntityKind::Connector { .. }))
            }),
            C::PointOnCurve { point, curve } => {
                self.expect_point(*point)?;
                self.expect(*curve, "a curve", |k| {
                    matches!(
                        k,
                        EntityKind::Line { .. }
                            | EntityKind::Circle { .. }
                            | EntityKind::Arc { .. }
                            | EntityKind::Ellipse { .. }
                            | EntityKind::BSpline { .. }
                    )
                })
            }
            C::Distance { a, b, value } => {
                finite(*value)?;
                if *value < 0.0 {
                    return Err(BuildError::BadValue("distance must be non-negative"));
                }
                Self::distinct(*a, *b)?;
                let pa = matches!(self.get(*a)?, EntityKind::Point2 { .. });
                let pb = matches!(self.get(*b)?, EntityKind::Point2 { .. });
                match (pa, pb) {
                    (true, true) => Ok(()),
                    (true, false) => self.expect_line(*b),
                    (false, true) => self.expect_line(*a),
                    (false, false) => Err(BuildError::WrongKind { entity: *a, expected: "a point" }),
                }
            }
            C::Angle { a, b, value } => {
                finite(*value)?;
                self.expect_line(*a)?;
                self.expect_line(*b)?;
                Self::distinct(*a, *b)
            }
            C::Radius { curve, value } | C::Diameter { curve, value } => {
                finite(*value)?;
                if *value <= 0.0 {
                    return Err(BuildError::BadValue("radius must be positive"));
                }
                self.expect_round(*curve)
            }
            C::Length { line, value } => {
                finite(*value)?;
                if *value < 0.0 {
                    return Err(BuildError::BadValue("length must be non-negative"));
                }
                self.expect_line(*line)
            }
            C::VarValue { var, value } => {
                finite(*value)?;
                if var.index() >= self.vars.len() {
                    return Err(BuildError::UnknownVar(*var));
                }
                Ok(())
            }
            C::Mate(m) => self.validate_mate(m),
        }
    }

    fn validate_mate(&self, m: &Mate) -> Result<(), BuildError> {
        let (a, b) = mate_connectors(m);
        let ba = self.expect_connector(a)?;
        let bb = self.expect_connector(b)?;
        if ba == bb {
            return Err(BuildError::SameBody(ba));
        }
        match m {
            Mate::Tangent { kind: TangentKind::PlaneSphere { radius } | TangentKind::PlaneCylinder { radius }, .. } => {
                finite(*radius)?;
                if *radius < 0.0 {
                    return Err(BuildError::BadValue("radius must be non-negative"));
                }
            }
            Mate::Distance { target, .. } | Mate::Angle { target, .. } => match target {
                Target::Exact(v) => finite(*v)?,
                Target::Range { min, max } => {
                    finite(*min)?;
                    finite(*max)?;
                    if min > max {
                        return Err(BuildError::BadValue("range min exceeds max"));
                    }
                }
            },
            _ => {}
        }
        Ok(())
    }
}

pub(crate) fn mate_connectors(m: &Mate) -> (EntityId, EntityId) {
    match *m {
        Mate::Fasten { a, b }
        | Mate::Revolute { a, b }
        | Mate::Slider { a, b }
        | Mate::Cylindrical { a, b }
        | Mate::Planar { a, b }
        | Mate::Ball { a, b }
        | Mate::Parallel { a, b }
        | Mate::Tangent { a, b, .. }
        | Mate::Distance { a, b, .. }
        | Mate::Angle { a, b, .. } => (a, b),
    }
}

fn finite(v: f64) -> Result<(), BuildError> {
    if v.is_finite() {
        Ok(())
    } else {
        Err(BuildError::BadValue("not finite"))
    }
}

pub(crate) fn normalize4(q: [f64; 4]) -> [f64; 4] {
    let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    if n > 0.0 {
        [q[0] / n, q[1] / n, q[2] / n, q[3] / n]
    } else {
        [1.0, 0.0, 0.0, 0.0]
    }
}
