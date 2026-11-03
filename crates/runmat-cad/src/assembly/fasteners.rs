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

// ------------------------------------------------------------------------------ tables

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "type")]
pub enum ColumnData {
    F64 { unit: String, values: Vec<f64> },
    I64 { values: Vec<i64> },
    Text { values: Vec<String> },
}

impl ColumnData {
    pub fn len(&self) -> usize {
        match self {
            ColumnData::F64 { values, .. } => values.len(),
            ColumnData::I64 { values } => values.len(),
            ColumnData::Text { values } => values.len(),
        }
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Column {
    pub name: String,
    pub data: ColumnData,
}

/// A canonical columnar table (irregular instance sets, imported arrays).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ColumnarTable {
    pub columns: Vec<Column>,
}

const TABLE_MAGIC: &[u8; 8] = b"RMTBL\x00\x00\x01";

impl ColumnarTable {
    pub fn rows(&self) -> usize {
        self.columns.first().map(|c| c.data.len()).unwrap_or(0)
    }

    pub fn column(&self, name: &str) -> Option<&ColumnData> {
        self.columns
            .iter()
            .find(|c| c.name == name)
            .map(|c| &c.data)
    }

    pub fn validate(&self) -> Result<(), String> {
        let n = self.rows();
        let mut names = BTreeSet::new();
        for c in &self.columns {
            if !names.insert(c.name.as_str()) {
                return Err(format!("column `{}` appears twice", c.name));
            }
            if c.data.len() != n {
                return Err(format!(
                    "column `{}` has {} rows, expected {}",
                    c.name,
                    c.data.len(),
                    n
                ));
            }
        }
        Ok(())
    }

    /// The binary encoding: magic, column count, then per column its name, type tag, unit and
    /// values, all little-endian with length prefixes.
    pub fn encode_binary(&self) -> Vec<u8> {
        fn put_str(out: &mut Vec<u8>, s: &str) {
            out.extend_from_slice(&(s.len() as u32).to_le_bytes());
            out.extend_from_slice(s.as_bytes());
        }
        let mut out = Vec::with_capacity(16 + self.rows() * self.columns.len() * 8);
        out.extend_from_slice(TABLE_MAGIC);
        out.extend_from_slice(&(self.columns.len() as u32).to_le_bytes());
        out.extend_from_slice(&(self.rows() as u64).to_le_bytes());
        for c in &self.columns {
            put_str(&mut out, &c.name);
            match &c.data {
                ColumnData::F64 { unit, values } => {
                    out.push(0);
                    put_str(&mut out, unit);
                    for v in values {
                        out.extend_from_slice(&v.to_bits().to_le_bytes());
                    }
                }
                ColumnData::I64 { values } => {
                    out.push(1);
                    for v in values {
                        out.extend_from_slice(&v.to_le_bytes());
                    }
                }
                ColumnData::Text { values } => {
                    out.push(2);
                    for v in values {
                        put_str(&mut out, v);
                    }
                }
            }
        }
        out
    }

    pub fn decode_binary(bytes: &[u8]) -> Result<ColumnarTable, String> {
        struct R<'a> {
            b: &'a [u8],
            at: usize,
        }
        impl R<'_> {
            fn take(&mut self, n: usize) -> Result<&[u8], String> {
                if self.at + n > self.b.len() {
                    return Err("truncated table".into());
                }
                let s = &self.b[self.at..self.at + n];
                self.at += n;
                Ok(s)
            }
            fn u32(&mut self) -> Result<u32, String> {
                Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
            }
            fn u64(&mut self) -> Result<u64, String> {
                Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
            }
            fn string(&mut self) -> Result<String, String> {
                let n = self.u32()? as usize;
                String::from_utf8(self.take(n)?.to_vec()).map_err(|e| e.to_string())
            }
        }
        let mut r = R { b: bytes, at: 0 };
        if r.take(8)? != TABLE_MAGIC {
            return Err("not a RunMat columnar table".into());
        }
        let ncols = r.u32()? as usize;
        let nrows = r.u64()? as usize;
        let mut columns = Vec::with_capacity(ncols);
        for _ in 0..ncols {
            let name = r.string()?;
            let tag = r.take(1)?[0];
            let data = match tag {
                0 => {
                    let unit = r.string()?;
                    let mut values = Vec::with_capacity(nrows);
                    for _ in 0..nrows {
                        values.push(f64::from_bits(r.u64()?));
                    }
                    ColumnData::F64 { unit, values }
                }
                1 => {
                    let mut values = Vec::with_capacity(nrows);
                    for _ in 0..nrows {
                        values.push(r.u64()? as i64);
                    }
                    ColumnData::I64 { values }
                }
                2 => {
                    let mut values = Vec::with_capacity(nrows);
                    for _ in 0..nrows {
                        values.push(r.string()?);
                    }
                    ColumnData::Text { values }
                }
                t => return Err(format!("unknown column type {}", t)),
            };
            columns.push(Column { name, data });
        }
        if r.at != bytes.len() {
            return Err("trailing bytes after the table".into());
        }
        let t = ColumnarTable { columns };
        t.validate()?;
        Ok(t)
    }

    /// The text encoding: a header line of `name:f64[unit]`, `name:i64` or `name:text`
    /// fields, then one line per row; fields are tab-separated. Numbers are written in their
    /// shortest round-trip form, so the conversion is lossless. Text escapes `\\`, `\t`, `\n`.
    pub fn encode_text(&self) -> String {
        fn esc(s: &str) -> String {
            s.replace('\\', "\\\\")
                .replace('\t', "\\t")
                .replace('\n', "\\n")
        }
        let mut out = String::new();
        let header: Vec<String> = self
            .columns
            .iter()
            .map(|c| match &c.data {
                ColumnData::F64 { unit, .. } => format!("{}:f64[{}]", esc(&c.name), esc(unit)),
                ColumnData::I64 { .. } => format!("{}:i64", esc(&c.name)),
                ColumnData::Text { .. } => format!("{}:text", esc(&c.name)),
            })
            .collect();
        out.push_str(&header.join("\t"));
        out.push('\n');
        for r in 0..self.rows() {
            let fields: Vec<String> = self
                .columns
                .iter()
                .map(|c| match &c.data {
                    ColumnData::F64 { values, .. } => format!("{:?}", values[r]),
                    ColumnData::I64 { values } => values[r].to_string(),
                    ColumnData::Text { values } => esc(&values[r]),
                })
                .collect();
            out.push_str(&fields.join("\t"));
            out.push('\n');
        }
        out
    }

    pub fn decode_text(text: &str) -> Result<ColumnarTable, String> {
        fn unesc(s: &str) -> String {
            let mut out = String::new();
            let mut it = s.chars();
            while let Some(c) = it.next() {
                if c == '\\' {
                    match it.next() {
                        Some('t') => out.push('\t'),
                        Some('n') => out.push('\n'),
                        Some(o) => out.push(o),
                        None => out.push('\\'),
                    }
                } else {
                    out.push(c);
                }
            }
            out
        }
        let mut lines = text.lines();
        let header = lines.next().ok_or("empty table text")?;
        let mut columns = Vec::new();
        for field in header.split('\t') {
            let (name, ty) = field
                .rsplit_once(':')
                .ok_or_else(|| format!("header field {:?} has no type", field))?;
            let data = if let Some(unit) = ty.strip_prefix("f64[").and_then(|u| u.strip_suffix(']'))
            {
                ColumnData::F64 {
                    unit: unesc(unit),
                    values: Vec::new(),
                }
            } else if ty == "i64" {
                ColumnData::I64 { values: Vec::new() }
            } else if ty == "text" {
                ColumnData::Text { values: Vec::new() }
            } else {
                return Err(format!("unknown column type {:?}", ty));
            };
            columns.push(Column {
                name: unesc(name),
                data,
            });
        }
        for (ln, line) in lines.enumerate() {
            let fields: Vec<&str> = line.split('\t').collect();
            if fields.len() != columns.len() {
                return Err(format!("row {} has {} fields", ln + 1, fields.len()));
            }
            for (c, f) in columns.iter_mut().zip(fields) {
                match &mut c.data {
                    ColumnData::F64 { values, .. } => values.push(
                        f.parse::<f64>()
                            .map_err(|e| format!("row {}: {}", ln + 1, e))?,
                    ),
                    ColumnData::I64 { values } => values.push(
                        f.parse::<i64>()
                            .map_err(|e| format!("row {}: {}", ln + 1, e))?,
                    ),
                    ColumnData::Text { values } => values.push(unesc(f)),
                }
            }
        }
        let t = ColumnarTable { columns };
        t.validate()?;
        Ok(t)
    }

    /// `sha256:<hex>` of the binary encoding.
    pub fn digest(&self) -> String {
        format!(
            "sha256:{}",
            hex::encode(Sha256::digest(self.encode_binary()))
        )
    }
}

/// Fetches tables by digest.
pub trait TableSource {
    /// The table's binary encoding.
    fn fetch(&self, digest: &str) -> Result<Vec<u8>, String>;
}

#[derive(Clone, Debug, Default)]
pub struct MemoryTables {
    pub blobs: BTreeMap<String, Vec<u8>>,
}

impl MemoryTables {
    /// Store a table; returns its digest.
    pub fn put(&mut self, t: &ColumnarTable) -> String {
        let d = t.digest();
        self.blobs.insert(d.clone(), t.encode_binary());
        d
    }
}

impl TableSource for MemoryTables {
    fn fetch(&self, digest: &str) -> Result<Vec<u8>, String> {
        self.blobs
            .get(digest)
            .cloned()
            .ok_or_else(|| format!("no table `{}`", digest))
    }
}

/// Load and integrity-check a table.
pub fn load_table(
    source: &dyn TableSource,
    digest: &str,
) -> Result<Arc<ColumnarTable>, FastenerProblem> {
    let bytes = source
        .fetch(digest)
        .map_err(|message| FastenerProblem::TableIntegrity { message })?;
    let got = format!("sha256:{}", hex::encode(Sha256::digest(&bytes)));
    if got != digest {
        return Err(FastenerProblem::TableIntegrity {
            message: format!("table content hashes to {}, not {}", got, digest),
        });
    }
    ColumnarTable::decode_binary(&bytes)
        .map(Arc::new)
        .map_err(|message| FastenerProblem::TableIntegrity { message })
}

// ------------------------------------------------------------------------------ seams

/// A selected fastener: the family configuration (dash number) for a diameter and grip.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct FastenerPart {
    pub family: DefinitionRef,
    pub configuration: ConfigurationName,
    pub part_number: String,
}

pub trait FastenerCatalog {
    fn select(
        &self,
        family: &DefinitionRef,
        diameter: f64,
        grip: f64,
    ) -> Result<FastenerPart, String>;
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CatalogRow {
    pub family: DefinitionRef,
    pub diameter: f64,
    pub grip_min: f64,
    pub grip_max: f64,
    pub configuration: ConfigurationName,
    pub part_number: String,
}

/// A dash-number table (a family's configurations from table).
#[derive(Clone, Debug, Default)]
pub struct TableCatalog {
    pub rows: Vec<CatalogRow>,
}

impl FastenerCatalog for TableCatalog {
    fn select(
        &self,
        family: &DefinitionRef,
        diameter: f64,
        grip: f64,
    ) -> Result<FastenerPart, String> {
        let tol = 1e-9 * diameter.abs().max(1e-3);
        let mut best: Option<&CatalogRow> = None;
        for r in &self.rows {
            if &r.family != family || (r.diameter - diameter).abs() > tol {
                continue;
            }
            if grip < r.grip_min - 1e-12 || grip > r.grip_max + 1e-12 {
                continue;
            }
            best = match best {
                Some(b) if (b.grip_max, &b.part_number) <= (r.grip_max, &r.part_number) => Some(b),
                _ => Some(r),
            };
        }
        best.map(|r| FastenerPart {
            family: r.family.clone(),
            configuration: r.configuration.clone(),
            part_number: r.part_number.clone(),
        })
        .ok_or_else(|| {
            format!(
                "`{}` has no size for diameter {} m and grip {} m",
                family, diameter, grip
            )
        })
    }
}

/// A stack member at expansion time: its path, its definition and where it sits.
#[derive(Clone, Debug)]
pub struct StackMember {
    pub path: RelPath,
    pub definition: DefinitionRef,
    /// Member definition → assembly coordinates.
    pub transform: Transform,
    pub summary: Option<Arc<ConfiguredSummary>>,
}

/// Measures the clamped stack at a point along an axis (the kernel ray-casts; R0 uses
/// published nominal thicknesses).
pub trait StackGauge {
    fn grip(&self, members: &[StackMember], point: [f64; 3], axis: [f64; 3])
        -> Result<f64, String>;
}

/// The sum of each member's published value `thickness` (or another name).
#[derive(Clone, Debug)]
pub struct NominalThicknessGauge {
    pub publish: PublishName,
}

impl Default for NominalThicknessGauge {
    fn default() -> Self {
        Self {
            publish: PublishName::new("thickness"),
        }
    }
}

impl StackGauge for NominalThicknessGauge {
    fn grip(
        &self,
        members: &[StackMember],
        _point: [f64; 3],
        _axis: [f64; 3],
    ) -> Result<f64, String> {
        let mut total = 0.0;
        for m in members {
            let s = m
                .summary
                .as_ref()
                .ok_or_else(|| format!("`{}` has no configured summary", m.path))?;
            match s.publishes.get(&self.publish) {
                Some(PublishedGeometry::Value { value })
                    if value.dimension == Dimension::LENGTH =>
                {
                    total += value.si
                }
                _ => {
                    return Err(format!(
                        "`{}` publishes no length `{}`",
                        m.definition, self.publish
                    ))
                }
            }
        }
        Ok(total)
    }
}

// ------------------------------------------------------------------------------ expansion

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "problem")]
pub enum FastenerProblem {
    PathMissing {
        message: String,
    },
    PathTooShort {
        chain: String,
        length: f64,
        needed: f64,
    },
    ExceptionUnmatched {
        index: usize,
        what: String,
    },
    ExceptionAmbiguous {
        index: usize,
        matches: Vec<String>,
    },
    DuplicateKey {
        key: String,
    },
    NoFit {
        key: String,
        message: String,
    },
    TableIntegrity {
        message: String,
    },
    TableSchema {
        message: String,
    },
    GaugeFailed {
        key: String,
        message: String,
    },
    Value {
        what: String,
        error: ValueError,
    },
}

impl fmt::Display for FastenerProblem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FastenerProblem::PathMissing { message } => f.write_str(message),
            FastenerProblem::PathTooShort {
                chain,
                length,
                needed,
            } => write!(
                f,
                "chain `{}` is {:.6} m long; the edge distances need {:.6} m",
                chain, length, needed
            ),
            FastenerProblem::ExceptionUnmatched { index, what } => {
                write!(f, "exception {} ({}) matches no fastener", index, what)
            }
            FastenerProblem::ExceptionAmbiguous { index, matches } => write!(
                f,
                "exception {} matches {} fasteners: {}",
                index,
                matches.len(),
                matches.join(", ")
            ),
            FastenerProblem::DuplicateKey { key } => {
                write!(f, "two fasteners share the key `{}`", key)
            }
            FastenerProblem::NoFit { key, message } => write!(f, "`{}`: {}", key, message),
            FastenerProblem::TableIntegrity { message } => write!(f, "instance table: {}", message),
            FastenerProblem::TableSchema { message } => write!(f, "instance table: {}", message),
            FastenerProblem::GaugeFailed { key, message } => {
                write!(f, "grip at `{}`: {}", key, message)
            }
            FastenerProblem::Value { what, error } => write!(f, "{}: {}", what, error),
        }
    }
}

/// Holes one stack member receives: indices into the expansion plus diameters.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DrillList {
    pub member: RelPath,
    pub instances: Vec<u32>,
    pub diameters: Vec<f64>,
}

/// The expanded set, columnar, in key order. Positions and axes are in the owning assembly's
/// coordinates.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct FastenerExpansion {
    pub set: String,
    pub keys: Vec<FastenerKey>,
    pub positions: Vec<[f64; 3]>,
    pub axes: Vec<[f64; 3]>,
    pub diameters: Vec<f64>,
    pub grips: Vec<f64>,
    /// Per instance: index into `parts`, or `u32::MAX` when no size fits.
    pub selection: Vec<u32>,
    pub parts: Vec<FastenerPart>,
    pub omitted: Vec<FastenerKey>,
    pub drills: Vec<DrillList>,
}

impl FastenerExpansion {
    pub fn len(&self) -> usize {
        self.keys.len()
    }
    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    pub fn index_of(&self, key: &FastenerKey) -> Option<usize> {
        self.keys.binary_search(key).ok()
    }

    /// Counts per selected part, for the BOM.
    pub fn counts(&self) -> BTreeMap<&FastenerPart, u64> {
        let mut out = BTreeMap::new();
        for &s in &self.selection {
            if let Some(p) = self.parts.get(s as usize) {
                *out.entry(p).or_insert(0) += 1;
            }
        }
        out
    }

    /// A content digest (keys, positions, sizes): equal rules give equal digests everywhere.
    pub fn digest(&self) -> String {
        let mut h = Sha256::new();
        h.update(self.set.as_bytes());
        for i in 0..self.keys.len() {
            h.update(self.keys[i].to_string().as_bytes());
            for v in self.positions[i].iter().chain(&self.axes[i]) {
                h.update(v.to_bits().to_le_bytes());
            }
            h.update(self.diameters[i].to_bits().to_le_bytes());
            h.update(self.grips[i].to_bits().to_le_bytes());
            h.update(self.selection[i].to_le_bytes());
        }
        for p in &self.parts {
            h.update(p.part_number.as_bytes());
        }
        format!("sha256:{}", hex::encode(h.finalize()))
    }
}

/// What expansion needs from its caller.
pub struct ExpansionInputs<'a> {
    pub ev: &'a Evaluator<'a>,
    pub env: &'a ValueEnv,
    pub units: &'a dyn UnitAuthority,
    /// The published path, already in assembly coordinates (along rules).
    pub path: Option<&'a [PathChain]>,
    /// The table, already loaded (table rules), and its frame → assembly transform.
    pub table: Option<(&'a ColumnarTable, Transform)>,
    pub stack: &'a [StackMember],
    pub catalog: Option<&'a dyn FastenerCatalog>,
    pub gauge: Option<&'a dyn StackGauge>,
}

struct Raw {
    key: FastenerKey,
    station: f64,
    chain: String,
    row: String,
    /// Arc length along the chain (along rules).
    s: f64,
    position: [f64; 3],
    axis: [f64; 3],
    diameter: f64,
    spec: Option<FastenerSpec>,
}

fn len3(a: [f64; 3]) -> f64 {
    (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt()
}

fn norm3(a: [f64; 3]) -> [f64; 3] {
    let n = len3(a);
    if n > 0.0 {
        [a[0] / n, a[1] / n, a[2] / n]
    } else {
        a
    }
}

fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Point, tangent and normal at arc length `s` along a chain.
fn sample(chain: &PathChain, cum: &[f64], s: f64) -> ([f64; 3], [f64; 3], [f64; 3]) {
    let n = chain.points.len();
    let mut i = match cum.binary_search_by(|v| v.partial_cmp(&s).unwrap()) {
        Ok(i) => i,
        Err(i) => i.saturating_sub(1),
    };
    if i >= n - 1 {
        i = n - 2;
    }
    let seg = cum[i + 1] - cum[i];
    let t = if seg > 0.0 { (s - cum[i]) / seg } else { 0.0 };
    let a = chain.points[i];
    let b = chain.points[i + 1];
    let p = [
        a[0] + t * (b[0] - a[0]),
        a[1] + t * (b[1] - a[1]),
        a[2] + t * (b[2] - a[2]),
    ];
    let tangent = norm3([b[0] - a[0], b[1] - a[1], b[2] - a[2]]);
    let na = chain.normals.get(i).copied().unwrap_or([0.0, 0.0, 1.0]);
    let nb = chain.normals.get(i + 1).copied().unwrap_or(na);
    let normal = norm3([
        na[0] + t * (nb[0] - na[0]),
        na[1] + t * (nb[1] - na[1]),
        na[2] + t * (nb[2] - na[2]),
    ]);
    (p, tangent, normal)
}

/// Expand a fastener set. Problems are returned alongside whatever instances could be made.
pub fn expand(
    set: &FastenerSet,
    x: &ExpansionInputs<'_>,
) -> (FastenerExpansion, Vec<FastenerProblem>) {
    let mut problems = Vec::new();
    let mut out = FastenerExpansion {
        set: set.name.clone(),
        ..Default::default()
    };
    let len = |what: &str, e: &ValueExpr, problems: &mut Vec<FastenerProblem>| -> Option<f64> {
        match x.ev.eval_as(e, x.env, Dimension::LENGTH) {
            Ok(v) => Some(v),
            Err(error) => {
                problems.push(FastenerProblem::Value {
                    what: what.into(),
                    error,
                });
                None
            }
        }
    };
    let Some(base_d) = len("fastener diameter", &set.fastener.diameter, &mut problems) else {
        return (out, problems);
    };
    let mut raws: Vec<Raw> = Vec::new();
    let step;
    match &set.layout {
        FastenerLayout::Along(rule) => {
            step = match &rule.key_step {
                Some(e) => len("key step", e, &mut problems).unwrap_or(1e-5),
                None => 1e-5,
            };
            let Some(chains) = x.path else {
                problems.push(FastenerProblem::PathMissing {
                    message: format!("`{}` publishes no path `{}`", rule.occurrence, rule.path),
                });
                return (out, problems);
            };
            let Some(edge) = len("edge distance", &rule.edge_distance, &mut problems) else {
                return (out, problems);
            };
            let rows: Vec<(String, f64)> = if rule.rows.is_empty() {
                vec![("0".into(), 0.0)]
            } else {
                rule.rows
                    .iter()
                    .filter_map(|r| {
                        len(&format!("row {}", r.key), &r.offset, &mut problems)
                            .map(|o| (r.key.clone(), o))
                    })
                    .collect()
            };
            let basis = match &rule.stations {
                StationBasis::ArcLength => None,
                StationBasis::Axis { origin, direction } => {
                    let o = [
                        len("station origin", &origin[0], &mut problems).unwrap_or(0.0),
                        len("station origin", &origin[1], &mut problems).unwrap_or(0.0),
                        len("station origin", &origin[2], &mut problems).unwrap_or(0.0),
                    ];
                    Some((o, norm3(*direction)))
                }
            };
            for want in &rule.chains {
                if !chains.iter().any(|c| &c.key == want) {
                    problems.push(FastenerProblem::PathMissing {
                        message: format!("`{}` has no chain `{}`", rule.path, want),
                    });
                }
            }
            for chain in chains
                .iter()
                .filter(|c| rule.chains.is_empty() || rule.chains.contains(&c.key))
            {
                if chain.points.len() < 2 {
                    problems.push(FastenerProblem::PathMissing {
                        message: format!("chain `{}` has fewer than two points", chain.key),
                    });
                    continue;
                }
                let mut cum = Vec::with_capacity(chain.points.len());
                let mut acc = 0.0;
                cum.push(0.0);
                for w in chain.points.windows(2) {
                    acc += len3([w[1][0] - w[0][0], w[1][1] - w[0][1], w[1][2] - w[0][2]]);
                    cum.push(acc);
                }
                let total = acc;
                if total < 2.0 * edge {
                    problems.push(FastenerProblem::PathTooShort {
                        chain: chain.key.clone(),
                        length: total,
                        needed: 2.0 * edge,
                    });
                    continue;
                }
                let usable = total - 2.0 * edge;
                let stations: Vec<f64> = match &rule.spacing {
                    Spacing::Pitch { pitch } => {
                        let Some(p) = len("pitch", pitch, &mut problems) else {
                            continue;
                        };
                        if p <= 0.0 {
                            problems.push(FastenerProblem::Value {
                                what: "pitch".into(),
                                error: ValueError::OutOfRange {
                                    name: "pitch".into(),
                                    value: p.to_string(),
                                },
                            });
                            continue;
                        }
                        let n = ((usable / p) + 1e-9).floor() as usize + 1;
                        (0..n).map(|i| edge + i as f64 * p).collect()
                    }
                    Spacing::MaxPitch { pitch } => {
                        let Some(p) = len("pitch", pitch, &mut problems) else {
                            continue;
                        };
                        if p <= 0.0 {
                            continue;
                        }
                        let spaces = ((usable / p) - 1e-9).ceil().max(0.0) as usize;
                        if spaces == 0 {
                            vec![edge + usable / 2.0]
                        } else {
                            let d = usable / spaces as f64;
                            (0..=spaces).map(|i| edge + i as f64 * d).collect()
                        }
                    }
                    Spacing::Count { count } => {
                        let n = match x.ev.eval(count, x.env) {
                            Ok(q) if q.dimension.is_dimensionless() && q.si >= 1.0 => {
                                q.si.round() as usize
                            }
                            Ok(q) => {
                                problems.push(FastenerProblem::Value {
                                    what: "count".into(),
                                    error: ValueError::OutOfRange {
                                        name: "count".into(),
                                        value: q.si.to_string(),
                                    },
                                });
                                continue;
                            }
                            Err(error) => {
                                problems.push(FastenerProblem::Value {
                                    what: "count".into(),
                                    error,
                                });
                                continue;
                            }
                        };
                        if n == 1 {
                            vec![edge + usable / 2.0]
                        } else {
                            let d = usable / (n - 1) as f64;
                            (0..n).map(|i| edge + i as f64 * d).collect()
                        }
                    }
                };
                for s in stations {
                    let (p, tangent, normal) = sample(chain, &cum, s);
                    let station = match basis {
                        None => s,
                        Some((o, d)) => {
                            (p[0] - o[0]) * d[0] + (p[1] - o[1]) * d[1] + (p[2] - o[2]) * d[2]
                        }
                    };
                    let lateral = norm3(cross3(normal, tangent));
                    for (rk, off) in &rows {
                        let pos = [
                            p[0] + off * lateral[0],
                            p[1] + off * lateral[1],
                            p[2] + off * lateral[2],
                        ];
                        raws.push(Raw {
                            key: FastenerKey::Station {
                                chain: chain.key.clone(),
                                row: rk.clone(),
                                station: (station / step).round() as i64,
                            },
                            station,
                            chain: chain.key.clone(),
                            row: rk.clone(),
                            s,
                            position: pos,
                            axis: normal,
                            diameter: base_d,
                            spec: None,
                        });
                    }
                }
            }
        }
        FastenerLayout::Table(rule) => {
            step = 1e-5;
            let Some((table, frame)) = &x.table else {
                problems.push(FastenerProblem::TableIntegrity {
                    message: format!("table `{}` is not loaded", rule.table),
                });
                return (out, problems);
            };
            let cols = &rule.columns;
            let keys: Vec<String> = match table.column(&cols.key) {
                Some(ColumnData::Text { values }) => values.clone(),
                Some(ColumnData::I64 { values }) => values.iter().map(|v| v.to_string()).collect(),
                _ => {
                    problems.push(FastenerProblem::TableSchema {
                        message: format!("no text or integer key column `{}`", cols.key),
                    });
                    return (out, problems);
                }
            };
            let length_col =
                |name: &str, problems: &mut Vec<FastenerProblem>| -> Option<Vec<f64>> {
                    match table.column(name) {
                        Some(ColumnData::F64 { unit, values }) => match x.units.unit(unit) {
                            Some(u) if u.dimension == Dimension::LENGTH => {
                                Some(values.iter().map(|v| v * u.scale).collect())
                            }
                            _ => {
                                problems.push(FastenerProblem::TableSchema {
                                    message: format!(
                                        "column `{}` has unit `{}`, not a length",
                                        name, unit
                                    ),
                                });
                                None
                            }
                        },
                        _ => {
                            problems.push(FastenerProblem::TableSchema {
                                message: format!("no length column `{}`", name),
                            });
                            None
                        }
                    }
                };
            let (Some(xs), Some(ys), Some(zs)) = (
                length_col(&cols.x, &mut problems),
                length_col(&cols.y, &mut problems),
                length_col(&cols.z, &mut problems),
            ) else {
                return (out, problems);
            };
            let axis_cols: Option<[Vec<f64>; 3]> = cols.axis.as_ref().and_then(|a| {
                let get = |n: &str| match table.column(n) {
                    Some(ColumnData::F64 { values, .. }) => Some(values.clone()),
                    _ => None,
                };
                Some([get(&a[0])?, get(&a[1])?, get(&a[2])?])
            });
            if cols.axis.is_some() && axis_cols.is_none() {
                problems.push(FastenerProblem::TableSchema {
                    message: "axis columns are missing or not numeric".into(),
                });
            }
            let dia_col = match &cols.diameter {
                Some(n) => length_col(n, &mut problems),
                None => None,
            };
            for i in 0..table.rows() {
                let local = [xs[i], ys[i], zs[i]];
                let ax = axis_cols
                    .as_ref()
                    .map(|a| norm3([a[0][i], a[1][i], a[2][i]]))
                    .unwrap_or([0.0, 0.0, 1.0]);
                raws.push(Raw {
                    key: FastenerKey::Row {
                        key: keys[i].clone(),
                    },
                    station: 0.0,
                    chain: String::new(),
                    row: String::new(),
                    s: 0.0,
                    position: frame.apply(local),
                    axis: norm3(frame.apply_vector(ax)),
                    diameter: dia_col.as_ref().map(|d| d[i]).unwrap_or(base_d),
                    spec: None,
                });
            }
        }
    }

    // Key order, duplicates diagnosed (the first in rule order is kept).
    raws.sort_by(|a, b| a.key.cmp(&b.key));
    let mut dedup: Vec<Raw> = Vec::with_capacity(raws.len());
    for r in raws {
        if dedup.last().map(|l| l.key == r.key).unwrap_or(false) {
            problems.push(FastenerProblem::DuplicateKey {
                key: r.key.to_string(),
            });
            continue;
        }
        dedup.push(r);
    }
    let mut raws = dedup;

    // Exceptions.
    let mut omit = vec![false; raws.len()];
    for (ei, ex) in set.exceptions.iter().enumerate() {
        let hits: Vec<usize> = match &ex.at {
            ExceptionSelector::Key { key } => raws
                .iter()
                .enumerate()
                .filter(|(_, r)| r.key.to_string() == *key)
                .map(|(i, _)| i)
                .collect(),
            ExceptionSelector::Station {
                station,
                chain,
                row,
                tolerance,
            } => {
                let Some(st) = len("exception station", station, &mut problems) else {
                    continue;
                };
                let tol = match tolerance {
                    Some(t) => len("exception tolerance", t, &mut problems),
                    None => None,
                };
                let q = (st / step).round() as i64;
                raws.iter()
                    .enumerate()
                    .filter(|(_, r)| {
                        chain.as_ref().map(|c| *c == r.chain).unwrap_or(true)
                            && row.as_ref().map(|w| *w == r.row).unwrap_or(true)
                            && match tol {
                                Some(t) => (r.station - st).abs() <= t,
                                None => matches!(&r.key, FastenerKey::Station { station, .. } if *station == q),
                            }
                    })
                    .map(|(i, _)| i)
                    .collect()
            }
        };
        let row_filtered = matches!(&ex.at, ExceptionSelector::Station { row: Some(_), .. });
        let multi_row_ok = !row_filtered && {
            // One station across several rows is one exception per row, as written.
            let stations: BTreeSet<(String, i64)> = hits
                .iter()
                .filter_map(|&i| match &raws[i].key {
                    FastenerKey::Station { chain, station, .. } => Some((chain.clone(), *station)),
                    _ => None,
                })
                .collect();
            stations.len() == 1
        };
        if hits.is_empty() {
            problems.push(FastenerProblem::ExceptionUnmatched {
                index: ei,
                what: match &ex.at {
                    ExceptionSelector::Key { key } => format!("key {}", key),
                    ExceptionSelector::Station { .. } => "station".into(),
                },
            });
            continue;
        }
        if hits.len() > 1 && !multi_row_ok {
            problems.push(FastenerProblem::ExceptionAmbiguous {
                index: ei,
                matches: hits.iter().map(|&i| raws[i].key.to_string()).collect(),
            });
            continue;
        }
        for i in hits {
            match &ex.action {
                ExceptionAction::Omit => omit[i] = true,
                ExceptionAction::Oversize { delta } => {
                    if let Some(d) = len("oversize", delta, &mut problems) {
                        raws[i].diameter += d;
                    }
                }
                ExceptionAction::Replace { fastener } => {
                    if let Some(d) = len("replacement diameter", &fastener.diameter, &mut problems)
                    {
                        raws[i].diameter = d;
                    }
                    raws[i].spec = Some(fastener.clone());
                }
                ExceptionAction::Shift { along } => {
                    if let (Some(d), FastenerLayout::Along(_), Some(chains)) =
                        (len("shift", along, &mut problems), &set.layout, x.path)
                    {
                        if let Some(chain) = chains.iter().find(|c| c.key == raws[i].chain) {
                            let mut cum = vec![0.0];
                            let mut acc = 0.0;
                            for w in chain.points.windows(2) {
                                acc +=
                                    len3([w[1][0] - w[0][0], w[1][1] - w[0][1], w[1][2] - w[0][2]]);
                                cum.push(acc);
                            }
                            let s = (raws[i].s + d).clamp(0.0, acc);
                            let (p, _, normal) = sample(chain, &cum, s);
                            let delta = [
                                raws[i].position[0] - sample(chain, &cum, raws[i].s).0[0],
                                raws[i].position[1] - sample(chain, &cum, raws[i].s).0[1],
                                raws[i].position[2] - sample(chain, &cum, raws[i].s).0[2],
                            ];
                            raws[i].s = s;
                            raws[i].position = [p[0] + delta[0], p[1] + delta[1], p[2] + delta[2]];
                            raws[i].axis = normal;
                        }
                    }
                }
            }
        }
    }

    // Grip, selection, drills.
    let grip_fixed = match &set.fastener.grip {
        GripSpec::Fixed { length } => len("grip", length, &mut problems),
        GripSpec::Auto => None,
    };
    let mut part_index: BTreeMap<FastenerPart, u32> = BTreeMap::new();
    let mut drills: Vec<DrillList> = if set.drill.through {
        set.stack
            .iter()
            .map(|m| DrillList {
                member: m.clone(),
                instances: Vec::new(),
                diameters: Vec::new(),
            })
            .collect()
    } else {
        Vec::new()
    };
    let drill_d = set
        .drill
        .diameter
        .as_ref()
        .and_then(|e| len("drill diameter", e, &mut problems));
    let mut selected: Vec<(usize, u32, f64)> = Vec::new();
    for (i, r) in raws.iter_mut().enumerate() {
        if omit[i] {
            out.omitted.push(r.key.clone());
            continue;
        }
        let grip = match grip_fixed {
            Some(g) => g,
            None => match (&set.fastener.grip, x.gauge) {
                (GripSpec::Fixed { .. }, _) => f64::NAN,
                (GripSpec::Auto, Some(g)) => match g.grip(x.stack, r.position, r.axis) {
                    Ok(v) => v,
                    Err(message) => {
                        problems.push(FastenerProblem::GaugeFailed {
                            key: r.key.to_string(),
                            message,
                        });
                        f64::NAN
                    }
                },
                (GripSpec::Auto, None) => {
                    problems.push(FastenerProblem::GaugeFailed {
                        key: r.key.to_string(),
                        message: "no stack gauge is configured".into(),
                    });
                    f64::NAN
                }
            },
        };
        let family = r
            .spec
            .as_ref()
            .map(|s| s.family.clone())
            .unwrap_or_else(|| set.fastener.family.clone());
        let sel = match x.catalog {
            Some(cat) if grip.is_finite() => match cat.select(&family, r.diameter, grip) {
                Ok(p) => {
                    let n = part_index.len() as u32;
                    *part_index.entry(p).or_insert(n)
                }
                Err(message) => {
                    problems.push(FastenerProblem::NoFit {
                        key: r.key.to_string(),
                        message,
                    });
                    u32::MAX
                }
            },
            Some(_) => u32::MAX,
            None => {
                problems.push(FastenerProblem::NoFit {
                    key: r.key.to_string(),
                    message: "no fastener catalog is configured".into(),
                });
                u32::MAX
            }
        };
        selected.push((i, sel, grip));
    }
    // Parts in a canonical order (sorted), so selection indices are deterministic.
    let mut parts: Vec<(FastenerPart, u32)> = part_index.into_iter().collect();
    parts.sort();
    let remap: BTreeMap<u32, u32> = parts
        .iter()
        .enumerate()
        .map(|(new, (_, old))| (*old, new as u32))
        .collect();
    out.parts = parts.into_iter().map(|(p, _)| p).collect();
    for (k, (i, sel, grip)) in selected.into_iter().enumerate() {
        let r = &raws[i];
        out.keys.push(r.key.clone());
        out.positions.push(r.position);
        out.axes.push(r.axis);
        out.diameters.push(r.diameter);
        out.grips.push(grip);
        out.selection
            .push(remap.get(&sel).copied().unwrap_or(u32::MAX));
        for d in &mut drills {
            d.instances.push(k as u32);
            d.diameters.push(drill_d.unwrap_or(r.diameter));
        }
    }
    out.drills = drills;
    (out, problems)
}
