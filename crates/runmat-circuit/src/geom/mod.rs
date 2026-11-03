//! Board geometry in integer nanometres.
//!
//! Coordinates are `i64` nanometres (KiCad's internal unit), so placement, Gerber output and
//! polygon booleans are exact and platform-independent. Valid coordinates satisfy
//! `|v| < 2^38` nm (about 275 m), which keeps every exact predicate inside `i128`.
//!
//! - [`poly`]: polygon booleans (union, intersection, difference, xor) with non-zero or
//!   positive fill, and ring utilities.
//! - [`offset`]: polygon offsetting with round, miter and square joins.
//! - [`shape`]: copper and graphic primitives (circle, rect, rounded rect, oval, trapezoid,
//!   segment, arc, polygon) with their outlines and exact-enough distances.

pub mod offset;
pub mod poly;
pub mod shape;

use std::fmt;
use std::ops::{Add, Neg, Sub};

pub type Coord = i64;

pub const NM_PER_MM: i64 = 1_000_000;
pub const NM_PER_MIL: i64 = 25_400;
/// Largest absolute coordinate the exact predicates accept.
pub const COORD_LIMIT: Coord = 1 << 38;

/// Millimetres to nanometres, rounded.
pub fn mm(v: f64) -> Coord {
    (v * 1e6).round() as Coord
}
/// Nanometres to millimetres.
pub fn to_mm(v: Coord) -> f64 {
    v as f64 * 1e-6
}
pub fn mil(v: f64) -> Coord {
    (v * NM_PER_MIL as f64).round() as Coord
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Point {
    pub x: Coord,
    pub y: Coord,
}

impl Point {
    pub const ORIGIN: Point = Point { x: 0, y: 0 };
    pub const fn new(x: Coord, y: Coord) -> Self {
        Point { x, y }
    }
    pub fn mm(x: f64, y: f64) -> Self {
        Point::new(mm(x), mm(y))
    }
    pub fn cross(self, o: Point) -> i128 {
        self.x as i128 * o.y as i128 - self.y as i128 * o.x as i128
    }
    pub fn dot(self, o: Point) -> i128 {
        self.x as i128 * o.x as i128 + self.y as i128 * o.y as i128
    }
    pub fn len(self) -> f64 {
        (self.x as f64).hypot(self.y as f64)
    }
    pub fn dist(self, o: Point) -> f64 {
        (self - o).len()
    }
    pub fn to_f(self) -> (f64, f64) {
        (self.x as f64, self.y as f64)
    }
    pub fn from_f(x: f64, y: f64) -> Self {
        Point::new(x.round() as Coord, y.round() as Coord)
    }
    pub fn rotate(self, deg: f64) -> Point {
        Transform::rotation(deg).apply(self)
    }
}

impl Add for Point {
    type Output = Point;
    fn add(self, o: Point) -> Point {
        Point::new(self.x + o.x, self.y + o.y)
    }
}
impl Sub for Point {
    type Output = Point;
    fn sub(self, o: Point) -> Point {
        Point::new(self.x - o.x, self.y - o.y)
    }
}
impl Neg for Point {
    type Output = Point;
    fn neg(self) -> Point {
        Point::new(-self.x, -self.y)
    }
}

impl fmt::Display for Point {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {}) mm", to_mm(self.x), to_mm(self.y))
    }
}

/// Orientation of `c` relative to the directed line `a → b`: positive left, negative right.
pub fn orient(a: Point, b: Point, c: Point) -> i128 {
    (b - a).cross(c - a)
}

/// Axis-aligned bounding box (inclusive).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Rect {
    pub min: Point,
    pub max: Point,
}

impl Rect {
    pub const EMPTY: Rect = Rect {
        min: Point {
            x: Coord::MAX,
            y: Coord::MAX,
        },
        max: Point {
            x: Coord::MIN,
            y: Coord::MIN,
        },
    };
    pub fn new(a: Point, b: Point) -> Self {
        Rect {
            min: Point::new(a.x.min(b.x), a.y.min(b.y)),
            max: Point::new(a.x.max(b.x), a.y.max(b.y)),
        }
    }
    pub fn from_points(ps: impl IntoIterator<Item = Point>) -> Rect {
        let mut r = Rect::EMPTY;
        for p in ps {
            r = r.include(p);
        }
        r
    }
    pub fn is_empty(&self) -> bool {
        self.min.x > self.max.x || self.min.y > self.max.y
    }
    pub fn include(self, p: Point) -> Rect {
        Rect {
            min: Point::new(self.min.x.min(p.x), self.min.y.min(p.y)),
            max: Point::new(self.max.x.max(p.x), self.max.y.max(p.y)),
        }
    }
    pub fn union(self, o: Rect) -> Rect {
        if self.is_empty() {
            return o;
        }
        if o.is_empty() {
            return self;
        }
        self.include(o.min).include(o.max)
    }
    pub fn inflate(self, d: Coord) -> Rect {
        if self.is_empty() {
            return self;
        }
        Rect {
            min: Point::new(self.min.x - d, self.min.y - d),
            max: Point::new(self.max.x + d, self.max.y + d),
        }
    }
    pub fn intersects(&self, o: &Rect) -> bool {
        !(self.is_empty()
            || o.is_empty()
            || self.max.x < o.min.x
            || o.max.x < self.min.x
            || self.max.y < o.min.y
            || o.max.y < self.min.y)
    }
    pub fn contains(&self, p: Point) -> bool {
        p.x >= self.min.x && p.x <= self.max.x && p.y >= self.min.y && p.y <= self.max.y
    }
    pub fn width(&self) -> Coord {
        self.max.x - self.min.x
    }
    pub fn height(&self) -> Coord {
        self.max.y - self.min.y
    }
    pub fn center(&self) -> Point {
        Point::new(
            self.min.x + (self.max.x - self.min.x) / 2,
            self.min.y + (self.max.y - self.min.y) / 2,
        )
    }
}

/// Placement transform: optional mirror about the y axis (bottom side), then rotation
/// counter-clockwise in degrees, then translation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Transform {
    pub offset: Point,
    pub rotation_deg: f64,
    pub mirror: bool,
}

impl Default for Transform {
    fn default() -> Self {
        Transform::IDENTITY
    }
}

impl Transform {
    pub const IDENTITY: Transform = Transform {
        offset: Point::ORIGIN,
        rotation_deg: 0.0,
        mirror: false,
    };
    pub fn new(offset: Point, rotation_deg: f64, mirror: bool) -> Self {
        Transform {
            offset,
            rotation_deg,
            mirror,
        }
    }
    pub fn rotation(deg: f64) -> Self {
        Transform {
            offset: Point::ORIGIN,
            rotation_deg: deg,
            mirror: false,
        }
    }
    pub fn translation(p: Point) -> Self {
        Transform {
            offset: p,
            rotation_deg: 0.0,
            mirror: false,
        }
    }
    /// Exact for multiples of 90°; otherwise rounded to the nearest nanometre.
    pub fn apply(&self, p: Point) -> Point {
        let p = if self.mirror {
            Point::new(-p.x, p.y)
        } else {
            p
        };
        let r = rotate_exact(p, self.rotation_deg);
        r + self.offset
    }
    pub fn apply_angle(&self, deg: f64) -> f64 {
        let a = if self.mirror { 180.0 - deg } else { deg };
        normalize_deg(a + self.rotation_deg)
    }
    /// `self` after `inner`: `self.then_of(inner).apply(p) == self.apply(inner.apply(p))`
    /// for rotation/mirror/translation transforms.
    pub fn compose(&self, inner: &Transform) -> Transform {
        let offset = self.apply(inner.offset);
        let mirror = self.mirror != inner.mirror;
        let rot = if self.mirror {
            self.rotation_deg - inner.rotation_deg
        } else {
            self.rotation_deg + inner.rotation_deg
        };
        Transform {
            offset,
            rotation_deg: normalize_deg(rot),
            mirror,
        }
    }
}

pub fn normalize_deg(a: f64) -> f64 {
    let r = a.rem_euclid(360.0);
    if (r - 360.0).abs() < 1e-12 {
        0.0
    } else {
        r
    }
}

fn rotate_exact(p: Point, deg: f64) -> Point {
    let d = normalize_deg(deg);
    let q = d / 90.0;
    if (q - q.round()).abs() < 1e-12 {
        return match q.round() as i64 % 4 {
            0 => p,
            1 => Point::new(-p.y, p.x),
            2 => Point::new(-p.x, -p.y),
            _ => Point::new(p.y, -p.x),
        };
    }
    let (s, c) = d.to_radians().sin_cos();
    let (x, y) = p.to_f();
    Point::from_f(x * c - y * s, x * s + y * c)
}

/// Segments needed so a polygon approximating a circle of radius `r` deviates by at most
/// `tol` (at least 8, a multiple of 4 so axis extremes are vertices).
pub fn circle_segments(r: Coord, tol: Coord) -> usize {
    if r <= tol || r <= 0 {
        return 8;
    }
    let half = (1.0 - tol as f64 / r as f64).acos();
    let n = (std::f64::consts::PI / half).ceil() as usize;
    n.clamp(8, 720).div_ceil(4) * 4
}

/// Default approximation tolerance for arcs: 1 µm.
pub const ARC_TOL: Coord = 1_000;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transforms() {
        let t = Transform::new(Point::mm(10.0, 0.0), 90.0, false);
        assert_eq!(t.apply(Point::mm(1.0, 0.0)), Point::mm(10.0, 1.0));
        let m = Transform::new(Point::ORIGIN, 0.0, true);
        assert_eq!(m.apply(Point::mm(1.0, 2.0)), Point::mm(-1.0, 2.0));
        let a = Transform::new(Point::mm(1.0, 2.0), 30.0, true);
        let b = Transform::new(Point::mm(-3.0, 5.0), 45.0, false);
        let c = a.compose(&b);
        let p = Point::mm(0.7, -0.2);
        let d = c.apply(p) - a.apply(b.apply(p));
        assert!(d.x.abs() <= 2 && d.y.abs() <= 2, "{d:?}");
        assert_eq!(circle_segments(mm(0.5), ARC_TOL) % 4, 0);
    }
}
