//! Fastener sets: rules, not occurrences.
//!
//! `fasteners row_a { along = spar.publish.upper_flange, pitch = 4.5 * d, edge_distance = 2 * d,
//! fastener = hi_lok { diameter = d, grip = auto }, stack = [skin, spar], drill = through,
//! except { omit at station 2210 mm; oversize(+0.4 mm) at station 3380 mm } }`.
//!
//! A rule expands deterministically to instances with **stable keys**: `(chain, row, station)`,
//! where the station is measured in a declared basis (arc length along the published path, or
//! the projection onto an axis such as fuselage station) and quantised by a declared step. Keys
//! never come from an array index, so an exception written at a station survives a pitch change
//! when that station still carries a fastener, and is diagnosed when it does not.
//!
//! Irregular sets come from a [`ColumnarTable`] referenced by content digest (`inst("sha256:…")`)
//! through the [`TableSource`] seam, integrity-checked on load. Tables have two lossless
//! encodings: a compact binary one and a line-oriented text one.
//!
//! Expansion output is columnar ([`FastenerExpansion`]): about 60 bytes per instance, so a
//! 100k-fastener set stays a few megabytes and never becomes 100k occurrence nodes.

mod tables;
mod seams;
mod expansion;
pub use expansion::*;
pub use seams::*;
pub use tables::*;

use crate::assembly::model::{DefinitionRef, RelPath};
use crate::assembly::options::Condition;
use crate::assembly::source::{ConfiguredSummary, PathChain, PublishedGeometry};
use crate::ids::{ConfigurationName, PublishName};
use crate::regen::kernel::Transform;
use crate::units::{Dimension, UnitAuthority};
use crate::value::{Evaluator, ValueEnv, ValueError, ValueExpr};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::sync::Arc;

// ------------------------------------------------------------------------------ the model

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "spacing")]
pub enum Spacing {
    /// Fixed pitch from the first station; the remainder falls at the far end.
    Pitch { pitch: ValueExpr },
    /// The fewest equal spaces no longer than `pitch`.
    MaxPitch { pitch: ValueExpr },
    /// Exactly `count` fasteners, equally spaced.
    Count { count: ValueExpr },
}

/// How stations are measured.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "basis")]
pub enum StationBasis {
    /// Arc length from the chain's start.
    ArcLength,
    /// Projection onto an axis (fuselage station, wing station), in assembly coordinates.
    Axis {
        origin: [ValueExpr; 3],
        direction: [f64; 3],
    },
}

/// A parallel row: its key and its lateral offset (along `normal × tangent`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RowSpec {
    pub key: String,
    pub offset: ValueExpr,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AlongRule {
    /// The occurrence whose definition publishes the path, relative to this assembly.
    pub occurrence: RelPath,
    pub path: PublishName,
    /// The chains of the published edge set to follow, by key (empty: all of them).
    #[serde(default)]
    pub chains: Vec<String>,
    pub spacing: Spacing,
    pub edge_distance: ValueExpr,
    /// Empty: one row, key `"0"`, on the path.
    #[serde(default)]
    pub rows: Vec<RowSpec>,
    #[serde(default = "arc")]
    pub stations: StationBasis,
    /// The station quantisation step (default 0.01 mm).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub key_step: Option<ValueExpr>,
}

fn arc() -> StationBasis {
    StationBasis::ArcLength
}

/// Which columns of the table hold what.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TableColumns {
    pub key: String,
    pub x: String,
    pub y: String,
    pub z: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub axis: Option<[String; 3]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub diameter: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TableRule {
    /// `sha256:<hex>` of the table's binary encoding.
    pub table: String,
    pub columns: TableColumns,
    /// Table coordinates are in this occurrence's definition frame (empty: the assembly's).
    #[serde(default)]
    pub frame: RelPath,
}

// Authored data, built once per edit; variant sizes do not matter here.
#[allow(clippy::large_enum_variant)]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "layout")]
pub enum FastenerLayout {
    Along(AlongRule),
    Table(TableRule),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "grip")]
pub enum GripSpec {
    /// From the stack thickness at each instance (through the [`StackGauge`]).
    Auto,
    Fixed {
        length: ValueExpr,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FastenerSpec {
    /// The fastener family (a `.part` whose configurations are its dash numbers).
    pub family: DefinitionRef,
    pub diameter: ValueExpr,
    pub grip: GripSpec,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DrillSpec {
    /// Drill every stack member.
    pub through: bool,
    /// Hole diameter; `None`: the fastener diameter (oversize exceptions included).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub diameter: Option<ValueExpr>,
}

impl Default for DrillSpec {
    fn default() -> Self {
        Self {
            through: true,
            diameter: None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "at")]
pub enum ExceptionSelector {
    /// The instance at this station (exact key unless a tolerance is given).
    Station {
        station: ValueExpr,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        chain: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        row: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        tolerance: Option<ValueExpr>,
    },
    /// The instance with this key (table sets, or a written station key).
    Key { key: String },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "action")]
pub enum ExceptionAction {
    Omit,
    Oversize {
        delta: ValueExpr,
    },
    Replace {
        fastener: FastenerSpec,
    },
    /// Move along the path; the key (identity) stays.
    Shift {
        along: ValueExpr,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FastenerException {
    pub at: ExceptionSelector,
    pub action: ExceptionAction,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FastenerSet {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub when: Option<Condition>,
    pub layout: FastenerLayout,
    pub fastener: FastenerSpec,
    /// The parts the fasteners clamp, outermost first, relative to this assembly.
    #[serde(default)]
    pub stack: Vec<RelPath>,
    #[serde(default)]
    pub drill: DrillSpec,
    #[serde(default)]
    pub exceptions: Vec<FastenerException>,
}

// ------------------------------------------------------------------------------ keys

/// A fastener instance's stable key.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "by")]
pub enum FastenerKey {
    /// `chain/row@station` with the station in key steps.
    Station {
        chain: String,
        row: String,
        station: i64,
    },
    /// A table row key.
    Row { key: String },
}

impl fmt::Display for FastenerKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FastenerKey::Station {
                chain,
                row,
                station,
            } => write!(f, "{}/{}@{}", chain, row, station),
            FastenerKey::Row { key } => f.write_str(key),
        }
    }
}
