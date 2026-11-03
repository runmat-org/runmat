//! Sketches: entities, constraints and the `solved { }` cache, behind a solver seam.

use crate::ids::{ConstraintId, SketchEntityId};
use crate::model::datum::PlaneRef;
use crate::value::ValueExpr;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Sketch {
    pub plane: PlaneRef,
    pub entities: Vec<SketchEntity>,
    #[serde(default)]
    pub constraints: Vec<SketchConstraint>,
    /// The `solved { }` block: the last solver output, so a file reloads without re-solving.
    #[serde(default)]
    pub solved: SolvedCache,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SketchEntity {
    pub id: SketchEntityId,
    pub geometry: SketchGeometry,
    #[serde(default)]
    pub construction: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "entity")]
pub enum SketchGeometry {
    Point {
        at: PointExpr,
    },
    Line {
        start: PointExpr,
        end: PointExpr,
    },
    /// Counter-clockwise from `start` to `end` about `center`.
    Arc {
        center: PointExpr,
        start: PointExpr,
        end: PointExpr,
    },
    Circle {
        center: PointExpr,
        radius: ValueExpr,
    },
}

/// Which defining point of an entity.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PointRole {
    /// A point entity's own position.
    At,
    Start,
    End,
    Center,
}

/// A sketch point as written: coordinates, or another entity's point (`l1.end`), which makes the
/// two the same point.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "point")]
pub enum PointExpr {
    Coords {
        x: ValueExpr,
        y: ValueExpr,
    },
    Of {
        entity: SketchEntityId,
        role: PointRole,
    },
}

impl PointExpr {
    pub fn mm(x: f64, y: f64) -> Self {
        PointExpr::Coords {
            x: ValueExpr::mm(x),
            y: ValueExpr::mm(y),
        }
    }
    pub fn of(entity: &str, role: PointRole) -> Self {
        PointExpr::Of {
            entity: SketchEntityId::new(entity),
            role,
        }
    }
}

/// A point or curve that a constraint names.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "target")]
pub enum SketchTarget {
    Entity {
        entity: SketchEntityId,
    },
    Point {
        entity: SketchEntityId,
        role: PointRole,
    },
}

impl SketchTarget {
    pub fn entity(id: &str) -> Self {
        SketchTarget::Entity {
            entity: SketchEntityId::new(id),
        }
    }
    pub fn point(id: &str, role: PointRole) -> Self {
        SketchTarget::Point {
            entity: SketchEntityId::new(id),
            role,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SketchConstraint {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<ConstraintId>,
    pub kind: SketchConstraintKind,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "constraint")]
pub enum SketchConstraintKind {
    Coincident {
        a: SketchTarget,
        b: SketchTarget,
    },
    Horizontal {
        line: SketchTarget,
    },
    Vertical {
        line: SketchTarget,
    },
    Parallel {
        a: SketchTarget,
        b: SketchTarget,
    },
    Perpendicular {
        a: SketchTarget,
        b: SketchTarget,
    },
    Tangent {
        a: SketchTarget,
        b: SketchTarget,
    },
    Equal {
        a: SketchTarget,
        b: SketchTarget,
    },
    Fix {
        target: SketchTarget,
    },
    Distance {
        a: SketchTarget,
        b: SketchTarget,
        value: ValueExpr,
    },
    Length {
        line: SketchTarget,
        value: ValueExpr,
    },
    Radius {
        curve: SketchTarget,
        value: ValueExpr,
    },
    Angle {
        a: SketchTarget,
        b: SketchTarget,
        value: ValueExpr,
    },
}

/// The `solved { }` cache: SI coordinates per entity point, keyed by name. Entries the solver
/// adds for hidden variables use [`SolvedKey::Hidden`].
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SolvedCache {
    #[serde(default, with = "crate::serde_rows")]
    pub points: BTreeMap<SolvedKey, [f64; 2]>,
    #[serde(default, with = "crate::serde_rows")]
    pub scalars: BTreeMap<SolvedKey, f64>,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "key")]
pub enum SolvedKey {
    Point {
        entity: SketchEntityId,
        role: PointRole,
    },
    Radius {
        entity: SketchEntityId,
    },
    /// A solver-owned variable with no authored name (e.g. a curve parameter of point-on-spline),
    /// keyed by the constraint that owns it.
    Hidden {
        constraint: ConstraintId,
        index: u32,
    },
}

/// Solved sketch geometry in sketch coordinates (SI).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SolvedSketch {
    #[serde(with = "crate::serde_rows")]
    pub points: BTreeMap<(SketchEntityId, PointRole), [f64; 2]>,
    pub radii: BTreeMap<SketchEntityId, f64>,
    pub status: SketchSolveStatus,
    /// Remaining DOF per entity, where the solver reports it.
    #[serde(default)]
    pub entity_dof: BTreeMap<SketchEntityId, u32>,
    #[serde(default)]
    pub dof: Option<u32>,
    /// Constraints reported redundant (consistent but implied by others), by index into
    /// `Sketch::constraints`.
    #[serde(default)]
    pub redundant: Vec<usize>,
    /// Constraints that contradict others, by index into `Sketch::constraints`.
    #[serde(default)]
    pub conflicting: Vec<usize>,
    /// The cache to write back into `solved { }`.
    #[serde(default)]
    pub cache: SolvedCache,
}

impl SolvedSketch {
    pub fn point(&self, entity: &SketchEntityId, role: PointRole) -> Option<[f64; 2]> {
        self.points.get(&(entity.clone(), role)).copied()
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SketchSolveStatus {
    WellDefined,
    UnderDefined,
    OverDefined,
    /// Explicit coordinates were taken as written; no constraint was solved.
    #[default]
    Explicit,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SketchSolveFailure {
    pub message: String,
    /// Constraint indices involved (conflict set), when known.
    #[serde(default)]
    pub constraints: Vec<usize>,
}
