//! `options { }`: option and effectivity axes, `when` conditions, rules, named configurations,
//! and the exact region algebra the validator uses.
//!
//! An axis is an enumeration, a boolean, or an ordered integer axis: unit (serial / MSN)
//! effectivity, lot effectivity, plain integers and calendar dates (as day numbers). A condition
//! is a boolean formula over axis atoms. Compiled, a condition is a **region**: a finite union of
//! **boxes**, each box holding one set per axis (a bitset for enumerations and booleans, a sorted
//! list of disjoint closed intervals for ordered axes). Intersection, complement and emptiness are
//! exact, so the validator *proves* that alternatives never overlap inside the legal space instead
//! of sampling configurations, and every failure carries a witness configuration.
//!
//! Regions are kept disjoint (complements are built prefix-wise), and every operation is bounded
//! by [`RegionLimits::max_boxes`]; exceeding it is reported (`cad.asm.condition_too_complex`),
//! never silently approximated.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt;

/// An axis name (`msn`, `engine`, `cabin`).
pub type AxisName = String;

/// A calendar date. Serialised as `YYYY-MM-DD`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(into = "String", try_from = "String")]
pub struct CivilDate {
    pub year: i32,
    pub month: u8,
    pub day: u8,
}

impl CivilDate {
    pub fn new(year: i32, month: u8, day: u8) -> Self {
        Self { year, month, day }
    }

    /// Days since 1970-01-01 (proleptic Gregorian; Howard Hinnant's algorithm).
    pub fn day_number(self) -> i64 {
        let y = self.year as i64 - if self.month <= 2 { 1 } else { 0 };
        let era = if y >= 0 { y } else { y - 399 } / 400;
        let yoe = y - era * 400;
        let m = self.month as i64;
        let doy = (153 * (m + if m > 2 { -3 } else { 9 }) + 2) / 5 + self.day as i64 - 1;
        let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
        era * 146097 + doe - 719468
    }

    pub fn from_day_number(z: i64) -> Self {
        let z = z + 719468;
        let era = if z >= 0 { z } else { z - 146096 } / 146097;
        let doe = z - era * 146097;
        let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
        let y = yoe + era * 400;
        let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
        let mp = (5 * doy + 2) / 153;
        let d = doy - (153 * mp + 2) / 5 + 1;
        let m = if mp < 10 { mp + 3 } else { mp - 9 };
        CivilDate {
            year: (y + if m <= 2 { 1 } else { 0 }) as i32,
            month: m as u8,
            day: d as u8,
        }
    }
}

impl fmt::Display for CivilDate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:04}-{:02}-{:02}", self.year, self.month, self.day)
    }
}

impl From<CivilDate> for String {
    fn from(d: CivilDate) -> String {
        d.to_string()
    }
}

impl TryFrom<String> for CivilDate {
    type Error = String;
    fn try_from(s: String) -> Result<Self, Self::Error> {
        let parts: Vec<&str> = s.split('-').collect();
        if parts.len() != 3 {
            return Err(format!("date {:?} is not YYYY-MM-DD", s));
        }
        let year = parts[0].parse::<i32>().map_err(|e| e.to_string())?;
        let month = parts[1].parse::<u8>().map_err(|e| e.to_string())?;
        let day = parts[2].parse::<u8>().map_err(|e| e.to_string())?;
        if !(1..=12).contains(&month) || !(1..=31).contains(&day) {
            return Err(format!("date {:?} is out of range", s));
        }
        let d = CivilDate { year, month, day };
        if CivilDate::from_day_number(d.day_number()) != d {
            return Err(format!("date {:?} does not exist", s));
        }
        Ok(d)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "domain")]
pub enum AxisDomain {
    /// A finite set of named values, in declared order.
    Enum {
        values: Vec<String>,
    },
    Bool,
    /// Unit (serial number, MSN, line number) effectivity: an inclusive range.
    Unit {
        min: i64,
        max: i64,
    },
    /// Lot or block-point effectivity: an inclusive range of lot numbers.
    Lot {
        min: i64,
        max: i64,
    },
    /// A plain integer option (seat count, stretch number).
    Integer {
        min: i64,
        max: i64,
    },
    /// Date effectivity: an inclusive range of days.
    Date {
        min: CivilDate,
        max: CivilDate,
    },
}

impl AxisDomain {
    fn is_ordered(&self) -> bool {
        !matches!(self, AxisDomain::Enum { .. } | AxisDomain::Bool)
    }

    fn bounds(&self) -> Option<(i64, i64)> {
        match self {
            AxisDomain::Unit { min, max }
            | AxisDomain::Lot { min, max }
            | AxisDomain::Integer { min, max } => Some((*min, *max)),
            AxisDomain::Date { min, max } => Some((min.day_number(), max.day_number())),
            _ => None,
        }
    }

    fn cardinality(&self) -> usize {
        match self {
            AxisDomain::Enum { values } => values.len(),
            AxisDomain::Bool => 2,
            _ => 0,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AxisDecl {
    pub name: AxisName,
    pub domain: AxisDomain,
    /// The value an unassigned axis takes in a configuration. `None`: it must be assigned.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default: Option<AxisValue>,
}

/// A value on an axis.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", untagged)]
pub enum AxisValue {
    Bool(bool),
    Int(i64),
    Date(CivilDate),
    Enum(String),
}

impl fmt::Display for AxisValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AxisValue::Bool(b) => write!(f, "{}", b),
            AxisValue::Int(i) => write!(f, "{}", i),
            AxisValue::Date(d) => write!(f, "{}", d),
            AxisValue::Enum(e) => f.write_str(e),
        }
    }
}

/// A `when` condition.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "when")]
pub enum Condition {
    True,
    False,
    /// `axis == value`.
    Is {
        axis: AxisName,
        value: AxisValue,
    },
    /// `axis in [values]`.
    In {
        axis: AxisName,
        values: Vec<AxisValue>,
    },
    /// `axis in min..max` (inclusive; an open end is unbounded). Ordered axes only.
    Range {
        axis: AxisName,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        min: Option<AxisValue>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        max: Option<AxisValue>,
    },
    Not {
        of: Box<Condition>,
    },
    All {
        of: Vec<Condition>,
    },
    Any {
        of: Vec<Condition>,
    },
    /// A named change group's effectivity (`when eco.ECO_1234`).
    Group {
        name: String,
    },
}

impl Condition {
    pub fn is(axis: &str, value: AxisValue) -> Self {
        Condition::Is {
            axis: axis.into(),
            value,
        }
    }
    pub fn is_enum(axis: &str, value: &str) -> Self {
        Self::is(axis, AxisValue::Enum(value.into()))
    }
    pub fn range(axis: &str, min: Option<i64>, max: Option<i64>) -> Self {
        Condition::Range {
            axis: axis.into(),
            min: min.map(AxisValue::Int),
            max: max.map(AxisValue::Int),
        }
    }
    pub fn negate(c: Condition) -> Self {
        Condition::Not { of: Box::new(c) }
    }
    pub fn all(of: Vec<Condition>) -> Self {
        Condition::All { of }
    }
    pub fn any(of: Vec<Condition>) -> Self {
        Condition::Any { of }
    }
    /// `a requires b`.
    pub fn requires(a: Condition, b: Condition) -> Self {
        Condition::any(vec![Condition::negate(a), b])
    }
    /// `a excludes b`.
    pub fn excludes(a: Condition, b: Condition) -> Self {
        Condition::negate(Condition::all(vec![a, b]))
    }

    /// Every axis the condition reads (groups expanded by the caller).
    pub fn axes(&self, out: &mut Vec<AxisName>) {
        match self {
            Condition::True | Condition::False | Condition::Group { .. } => {}
            Condition::Is { axis, .. }
            | Condition::In { axis, .. }
            | Condition::Range { axis, .. } => out.push(axis.clone()),
            Condition::Not { of } => of.axes(out),
            Condition::All { of } | Condition::Any { of } => of.iter().for_each(|c| c.axes(out)),
        }
    }
}

impl fmt::Display for Condition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Condition::True => f.write_str("true"),
            Condition::False => f.write_str("false"),
            Condition::Is { axis, value } => write!(f, "{} == {}", axis, value),
            Condition::In { axis, values } => {
                write!(f, "{} in [", axis)?;
                for (i, v) in values.iter().enumerate() {
                    if i > 0 {
                        f.write_str(", ")?;
                    }
                    write!(f, "{}", v)?;
                }
                f.write_str("]")
            }
            Condition::Range { axis, min, max } => {
                write!(f, "{} in ", axis)?;
                if let Some(m) = min {
                    write!(f, "{}", m)?;
                }
                f.write_str("..")?;
                if let Some(m) = max {
                    write!(f, "{}", m)?;
                }
                Ok(())
            }
            Condition::Not { of } => write!(f, "not ({})", of),
            Condition::All { of } | Condition::Any { of } => {
                let sep = if matches!(self, Condition::All { .. }) {
                    " and "
                } else {
                    " or "
                };
                f.write_str("(")?;
                for (i, c) in of.iter().enumerate() {
                    if i > 0 {
                        f.write_str(sep)?;
                    }
                    write!(f, "{}", c)?;
                }
                f.write_str(")")
            }
            Condition::Group { name } => write!(f, "eco.{}", name),
        }
    }
}

/// A rule that every legal configuration satisfies (`requires`, `excludes`, or any condition).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct OptionRule {
    pub name: String,
    pub condition: Condition,
}

/// A named change group (an ECO's effectivity), referenced as `when eco.<name>`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChangeGroup {
    pub name: String,
    pub effectivity: Condition,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

/// A named configuration: one value per axis (unassigned axes take their default).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct VariantConfiguration {
    pub name: String,
    pub values: BTreeMap<AxisName, AxisValue>,
}

/// The `options { }` block of the top `.assembly`.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct OptionsBlock {
    #[serde(default)]
    pub axes: Vec<AxisDecl>,
    #[serde(default)]
    pub rules: Vec<OptionRule>,
    #[serde(default)]
    pub groups: Vec<ChangeGroup>,
    #[serde(default)]
    pub configurations: Vec<VariantConfiguration>,
}

// ------------------------------------------------------------------------------ problems

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "problem")]
pub enum OptionProblem {
    UnknownAxis {
        axis: String,
    },
    UnknownGroup {
        group: String,
    },
    BadValue {
        axis: String,
        value: String,
        reason: String,
    },
    /// A group refers to itself through other groups.
    GroupCycle {
        group: String,
    },
    TooComplex {
        boxes: usize,
    },
}

impl fmt::Display for OptionProblem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OptionProblem::UnknownAxis { axis } => write!(f, "no option axis `{}`", axis),
            OptionProblem::UnknownGroup { group } => write!(f, "no change group `{}`", group),
            OptionProblem::BadValue {
                axis,
                value,
                reason,
            } => write!(f, "`{}` is not a value of `{}`: {}", value, axis, reason),
            OptionProblem::GroupCycle { group } => {
                write!(f, "change group `{}` refers to itself", group)
            }
            OptionProblem::TooComplex { boxes } => write!(
                f,
                "the condition expands past {} boxes and cannot be verified exactly",
                boxes
            ),
        }
    }
}

// ------------------------------------------------------------------------------ the space

/// One axis' set inside a box.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AxisSet {
    /// Bit `i` set: value `i` of the enumeration (or `false`/`true`) is in the set.
    Bits(Vec<u64>),
    /// Disjoint, sorted, inclusive, non-adjacent intervals.
    Intervals(Vec<(i64, i64)>),
}

impl AxisSet {
    fn is_empty(&self) -> bool {
        match self {
            AxisSet::Bits(b) => b.iter().all(|w| *w == 0),
            AxisSet::Intervals(iv) => iv.is_empty(),
        }
    }

    fn intersect(&self, other: &AxisSet) -> AxisSet {
        match (self, other) {
            (AxisSet::Bits(a), AxisSet::Bits(b)) => {
                AxisSet::Bits(a.iter().zip(b).map(|(x, y)| x & y).collect())
            }
            (AxisSet::Intervals(a), AxisSet::Intervals(b)) => {
                let mut out = Vec::new();
                let (mut i, mut j) = (0, 0);
                while i < a.len() && j < b.len() {
                    let lo = a[i].0.max(b[j].0);
                    let hi = a[i].1.min(b[j].1);
                    if lo <= hi {
                        out.push((lo, hi));
                    }
                    if a[i].1 < b[j].1 {
                        i += 1;
                    } else {
                        j += 1;
                    }
                }
                AxisSet::Intervals(out)
            }
            _ => unreachable!("axis sets of one axis share a representation"),
        }
    }

    /// The complement within `full`.
    fn complement(&self, full: &AxisSet) -> AxisSet {
        match (self, full) {
            (AxisSet::Bits(a), AxisSet::Bits(f)) => {
                AxisSet::Bits(a.iter().zip(f).map(|(x, m)| !x & m).collect())
            }
            (AxisSet::Intervals(a), AxisSet::Intervals(f)) => {
                let mut out = Vec::new();
                for &(flo, fhi) in f {
                    let mut cur = flo;
                    for &(lo, hi) in a {
                        if hi < cur || lo > fhi {
                            continue;
                        }
                        if lo > cur {
                            out.push((cur, lo - 1));
                        }
                        cur = cur.max(hi.saturating_add(1));
                        if cur > fhi {
                            break;
                        }
                    }
                    if cur <= fhi {
                        out.push((cur, fhi));
                    }
                }
                AxisSet::Intervals(out)
            }
            _ => unreachable!("axis sets of one axis share a representation"),
        }
    }

    fn first(&self) -> Option<i64> {
        match self {
            AxisSet::Bits(b) => {
                for (w, word) in b.iter().enumerate() {
                    if *word != 0 {
                        return Some((w * 64 + word.trailing_zeros() as usize) as i64);
                    }
                }
                None
            }
            AxisSet::Intervals(iv) => iv.first().map(|i| i.0),
        }
    }

    fn contains(&self, v: i64) -> bool {
        match self {
            AxisSet::Bits(b) => {
                v >= 0
                    && (v as usize) / 64 < b.len()
                    && (b[(v as usize) / 64] >> ((v as usize) % 64)) & 1 == 1
            }
            AxisSet::Intervals(iv) => iv.iter().any(|&(lo, hi)| lo <= v && v <= hi),
        }
    }
}

/// A product of per-axis sets.
pub type RegionBox = Vec<AxisSet>;

/// A finite disjoint union of boxes.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct Region {
    pub boxes: Vec<RegionBox>,
}

impl Region {
    pub fn is_empty(&self) -> bool {
        self.boxes.is_empty()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct RegionLimits {
    pub max_boxes: usize,
}

impl Default for RegionLimits {
    fn default() -> Self {
        Self { max_boxes: 50_000 }
    }
}

/// A configuration point: one value per axis, in axis order (enum/bool as indices, ordered axes
/// as integers or day numbers).
pub type Point = Vec<i64>;

/// The compiled option space of one top assembly.
#[derive(Clone, Debug)]
pub struct OptionSpace {
    pub axes: Vec<AxisDecl>,
    index: BTreeMap<AxisName, usize>,
    groups: BTreeMap<String, Condition>,
    full: RegionBox,
    /// The legal region: domains intersected with every rule.
    legal: Region,
    pub limits: RegionLimits,
    /// Problems compiling the rules.
    pub rule_problems: Vec<(String, OptionProblem)>,
}

impl OptionSpace {
    pub fn new(block: &OptionsBlock) -> Self {
        Self::with_limits(block, RegionLimits::default())
    }

    pub fn with_limits(block: &OptionsBlock, limits: RegionLimits) -> Self {
        let index = block
            .axes
            .iter()
            .enumerate()
            .map(|(i, a)| (a.name.clone(), i))
            .collect();
        let full: RegionBox = block
            .axes
            .iter()
            .map(|a| match a.domain.bounds() {
                Some((lo, hi)) if lo <= hi => AxisSet::Intervals(vec![(lo, hi)]),
                Some(_) => AxisSet::Intervals(Vec::new()),
                None => {
                    let n = a.domain.cardinality();
                    let mut bits = vec![0u64; n.div_ceil(64).max(1)];
                    for i in 0..n {
                        bits[i / 64] |= 1 << (i % 64);
                    }
                    AxisSet::Bits(bits)
                }
            })
            .collect();
        let groups = block
            .groups
            .iter()
            .map(|g| (g.name.clone(), g.effectivity.clone()))
            .collect();
        let mut space = OptionSpace {
            axes: block.axes.clone(),
            index,
            groups,
            legal: Region {
                boxes: vec![full.clone()],
            },
            full,
            limits,
            rule_problems: Vec::new(),
        };
        if space.full.iter().any(|s| s.is_empty()) {
            space.legal = Region::default();
        }
        let mut legal = space.legal.clone();
        for r in &block.rules {
            match space.compile(&r.condition) {
                Ok(reg) => match space.intersect(&legal, &reg) {
                    Ok(l) => legal = l,
                    Err(p) => space.rule_problems.push((r.name.clone(), p)),
                },
                Err(p) => space.rule_problems.push((r.name.clone(), p)),
            }
        }
        space.legal = legal;
        space
    }

    pub fn axis_index(&self, name: &str) -> Option<usize> {
        self.index.get(name).copied()
    }

    pub fn legal(&self) -> &Region {
        &self.legal
    }

    pub fn full_region(&self) -> Region {
        Region {
            boxes: vec![self.full.clone()],
        }
    }

    fn encode(&self, axis: usize, v: &AxisValue) -> Result<i64, OptionProblem> {
        let decl = &self.axes[axis];
        let bad = |reason: &str| OptionProblem::BadValue {
            axis: decl.name.clone(),
            value: v.to_string(),
            reason: reason.into(),
        };
        match (&decl.domain, v) {
            (AxisDomain::Enum { values }, AxisValue::Enum(e)) => values
                .iter()
                .position(|x| x == e)
                .map(|i| i as i64)
                .ok_or_else(|| bad("not one of the axis values")),
            (AxisDomain::Bool, AxisValue::Bool(b)) => Ok(*b as i64),
            (AxisDomain::Date { .. }, AxisValue::Date(d)) => Ok(d.day_number()),
            (
                AxisDomain::Unit { .. } | AxisDomain::Lot { .. } | AxisDomain::Integer { .. },
                AxisValue::Int(i),
            ) => Ok(*i),
            _ => Err(bad("wrong value type for the axis")),
        }
    }

    /// Decode a point coordinate back to an axis value.
    pub fn decode(&self, axis: usize, v: i64) -> AxisValue {
        match &self.axes[axis].domain {
            AxisDomain::Enum { values } => AxisValue::Enum(values[v as usize].clone()),
            AxisDomain::Bool => AxisValue::Bool(v != 0),
            AxisDomain::Date { .. } => AxisValue::Date(CivilDate::from_day_number(v)),
            _ => AxisValue::Int(v),
        }
    }

    fn atom(
        &self,
        axis: &str,
        set: impl FnOnce(&AxisSet, usize) -> Result<AxisSet, OptionProblem>,
    ) -> Result<Region, OptionProblem> {
        let i = self
            .axis_index(axis)
            .ok_or_else(|| OptionProblem::UnknownAxis { axis: axis.into() })?;
        let mut b = self.full.clone();
        b[i] = set(&self.full[i], i)?;
        Ok(if b[i].is_empty() {
            Region::default()
        } else {
            Region { boxes: vec![b] }
        })
    }

    fn values_set(&self, i: usize, values: &[AxisValue]) -> Result<AxisSet, OptionProblem> {
        let full = &self.full[i];
        let mut codes = values
            .iter()
            .map(|v| self.encode(i, v))
            .collect::<Result<Vec<_>, _>>()?;
        codes.sort_unstable();
        codes.dedup();
        Ok(match full {
            AxisSet::Bits(f) => {
                let mut bits = vec![0u64; f.len()];
                for c in codes {
                    bits[c as usize / 64] |= 1 << (c as usize % 64);
                }
                AxisSet::Bits(bits)
            }
            AxisSet::Intervals(_) => {
                let mut iv: Vec<(i64, i64)> = Vec::new();
                for c in codes {
                    match iv.last_mut() {
                        Some(last) if last.1 + 1 == c => last.1 = c,
                        _ => iv.push((c, c)),
                    }
                }
                AxisSet::Intervals(iv).intersect(full)
            }
        })
    }

    /// Compile a condition to a region of the full space (rules not applied).
    pub fn compile(&self, c: &Condition) -> Result<Region, OptionProblem> {
        self.compile_in(c, &mut Vec::new())
    }

    fn compile_in(&self, c: &Condition, stack: &mut Vec<String>) -> Result<Region, OptionProblem> {
        match c {
            Condition::True => Ok(self.full_region()),
            Condition::False => Ok(Region::default()),
            Condition::Is { axis, value } => {
                self.atom(axis, |_, i| self.values_set(i, std::slice::from_ref(value)))
            }
            Condition::In { axis, values } => self.atom(axis, |_, i| self.values_set(i, values)),
            Condition::Range { axis, min, max } => self.atom(axis, |full, i| {
                if !self.axes[i].domain.is_ordered() {
                    return Err(OptionProblem::BadValue {
                        axis: axis.clone(),
                        value: "a range".into(),
                        reason: "ranges apply to ordered axes only".into(),
                    });
                }
                let lo = match min {
                    Some(v) => self.encode(i, v)?,
                    None => i64::MIN,
                };
                let hi = match max {
                    Some(v) => self.encode(i, v)?,
                    None => i64::MAX,
                };
                Ok(if lo <= hi {
                    AxisSet::Intervals(vec![(lo, hi)]).intersect(full)
                } else {
                    AxisSet::Intervals(Vec::new())
                })
            }),
            Condition::Not { of } => {
                let r = self.compile_in(of, stack)?;
                self.complement(&r)
            }
            Condition::All { of } => {
                let mut acc = self.full_region();
                for x in of {
                    let r = self.compile_in(x, stack)?;
                    acc = self.intersect(&acc, &r)?;
                    if acc.is_empty() {
                        break;
                    }
                }
                Ok(acc)
            }
            Condition::Any { of } => {
                // a ∪ b = a ∪ (b ∖ a), which keeps the union disjoint.
                let mut acc = Region::default();
                for x in of {
                    let r = self.compile_in(x, stack)?;
                    let rest = self.subtract(&r, &acc)?;
                    acc.boxes.extend(rest.boxes);
                    self.check(&acc)?;
                }
                Ok(acc)
            }
            Condition::Group { name } => {
                if stack.contains(name) {
                    return Err(OptionProblem::GroupCycle {
                        group: name.clone(),
                    });
                }
                let g = self
                    .groups
                    .get(name)
                    .ok_or_else(|| OptionProblem::UnknownGroup {
                        group: name.clone(),
                    })?;
                stack.push(name.clone());
                let r = self.compile_in(g, stack);
                stack.pop();
                r
            }
        }
    }

    fn check(&self, r: &Region) -> Result<(), OptionProblem> {
        if r.boxes.len() > self.limits.max_boxes {
            Err(OptionProblem::TooComplex {
                boxes: self.limits.max_boxes,
            })
        } else {
            Ok(())
        }
    }

    fn box_intersect(a: &RegionBox, b: &RegionBox) -> Option<RegionBox> {
        let mut out = Vec::with_capacity(a.len());
        for (x, y) in a.iter().zip(b) {
            let s = x.intersect(y);
            if s.is_empty() {
                return None;
            }
            out.push(s);
        }
        Some(out)
    }

    pub fn intersect(&self, a: &Region, b: &Region) -> Result<Region, OptionProblem> {
        let mut out = Vec::new();
        for x in &a.boxes {
            for y in &b.boxes {
                if let Some(z) = Self::box_intersect(x, y) {
                    out.push(z);
                    if out.len() > self.limits.max_boxes {
                        return Err(OptionProblem::TooComplex {
                            boxes: self.limits.max_boxes,
                        });
                    }
                }
            }
        }
        Ok(Region { boxes: out })
    }

    /// The complement of one box, as disjoint boxes: for axis `i`, axes before `i` keep the
    /// box's set and axis `i` takes the complement.
    fn box_complement(&self, b: &RegionBox) -> Vec<RegionBox> {
        let mut out = Vec::new();
        for i in 0..b.len() {
            let c = b[i].complement(&self.full[i]);
            if c.is_empty() {
                continue;
            }
            let mut nb = Vec::with_capacity(b.len());
            for (j, full) in self.full.iter().enumerate() {
                nb.push(match j.cmp(&i) {
                    std::cmp::Ordering::Less => b[j].clone(),
                    std::cmp::Ordering::Equal => c.clone(),
                    std::cmp::Ordering::Greater => full.clone(),
                });
            }
            out.push(nb);
        }
        out
    }

    pub fn complement(&self, r: &Region) -> Result<Region, OptionProblem> {
        let mut acc = self.full_region();
        for b in &r.boxes {
            let c = Region {
                boxes: self.box_complement(b),
            };
            acc = self.intersect(&acc, &c)?;
            if acc.is_empty() {
                break;
            }
        }
        Ok(acc)
    }

    /// `a ∖ b`.
    pub fn subtract(&self, a: &Region, b: &Region) -> Result<Region, OptionProblem> {
        if b.is_empty() || a.is_empty() {
            return Ok(a.clone());
        }
        let mut acc = a.clone();
        for bx in &b.boxes {
            let c = Region {
                boxes: self.box_complement(bx),
            };
            acc = self.intersect(&acc, &c)?;
            if acc.is_empty() {
                break;
            }
        }
        Ok(acc)
    }

    /// A point of the region (the first value of every axis set of its first box).
    pub fn witness(&self, r: &Region) -> Option<Point> {
        let b = r.boxes.first()?;
        b.iter().map(|s| s.first()).collect()
    }

    /// The witness as `axis = value` text, for messages and repairs.
    pub fn describe_point(&self, p: &Point) -> BTreeMap<String, String> {
        p.iter()
            .enumerate()
            .map(|(i, v)| (self.axes[i].name.clone(), self.decode(i, *v).to_string()))
            .collect()
    }

    pub fn region_contains(&self, r: &Region, p: &Point) -> bool {
        r.boxes
            .iter()
            .any(|b| b.iter().zip(p).all(|(s, v)| s.contains(*v)))
    }

    /// Evaluate a condition at a point (no region expansion: linear in the formula).
    pub fn eval(&self, c: &Condition, p: &Point) -> Result<bool, OptionProblem> {
        self.eval_in(c, p, &mut Vec::new())
    }

    fn eval_in(
        &self,
        c: &Condition,
        p: &Point,
        stack: &mut Vec<String>,
    ) -> Result<bool, OptionProblem> {
        Ok(match c {
            Condition::True => true,
            Condition::False => false,
            Condition::Is { axis, value } => {
                let i = self
                    .axis_index(axis)
                    .ok_or_else(|| OptionProblem::UnknownAxis { axis: axis.clone() })?;
                p[i] == self.encode(i, value)?
            }
            Condition::In { axis, values } => {
                let i = self
                    .axis_index(axis)
                    .ok_or_else(|| OptionProblem::UnknownAxis { axis: axis.clone() })?;
                let mut hit = false;
                for v in values {
                    if p[i] == self.encode(i, v)? {
                        hit = true;
                    }
                }
                hit
            }
            Condition::Range { axis, min, max } => {
                let i = self
                    .axis_index(axis)
                    .ok_or_else(|| OptionProblem::UnknownAxis { axis: axis.clone() })?;
                let lo = match min {
                    Some(v) => self.encode(i, v)?,
                    None => i64::MIN,
                };
                let hi = match max {
                    Some(v) => self.encode(i, v)?,
                    None => i64::MAX,
                };
                lo <= p[i] && p[i] <= hi
            }
            Condition::Not { of } => !self.eval_in(of, p, stack)?,
            Condition::All { of } => {
                for x in of {
                    if !self.eval_in(x, p, stack)? {
                        return Ok(false);
                    }
                }
                true
            }
            Condition::Any { of } => {
                for x in of {
                    if self.eval_in(x, p, stack)? {
                        return Ok(true);
                    }
                }
                false
            }
            Condition::Group { name } => {
                if stack.contains(name) {
                    return Err(OptionProblem::GroupCycle {
                        group: name.clone(),
                    });
                }
                let g = self
                    .groups
                    .get(name)
                    .ok_or_else(|| OptionProblem::UnknownGroup {
                        group: name.clone(),
                    })?;
                stack.push(name.clone());
                let r = self.eval_in(g, p, stack);
                stack.pop();
                r?
            }
        })
    }

    /// A configuration point from axis values (defaults fill the rest). Errors name each axis
    /// that is missing or wrong.
    pub fn point(
        &self,
        values: &BTreeMap<AxisName, AxisValue>,
    ) -> Result<Point, Vec<OptionProblem>> {
        let mut problems = Vec::new();
        for k in values.keys() {
            if self.axis_index(k).is_none() {
                problems.push(OptionProblem::UnknownAxis { axis: k.clone() });
            }
        }
        let mut p = Vec::with_capacity(self.axes.len());
        for (i, a) in self.axes.iter().enumerate() {
            let v = values.get(&a.name).or(a.default.as_ref());
            match v {
                None => problems.push(OptionProblem::BadValue {
                    axis: a.name.clone(),
                    value: "(unassigned)".into(),
                    reason: "the axis has no default".into(),
                }),
                Some(v) => match self.encode(i, v) {
                    Ok(c) if self.full[i].contains(c) => p.push(c),
                    Ok(_) => problems.push(OptionProblem::BadValue {
                        axis: a.name.clone(),
                        value: v.to_string(),
                        reason: "outside the axis domain".into(),
                    }),
                    Err(e) => problems.push(e),
                },
            }
        }
        if problems.is_empty() {
            Ok(p)
        } else {
            Err(problems)
        }
    }

    pub fn is_legal(&self, p: &Point) -> bool {
        self.region_contains(&self.legal, p)
    }

    /// Classify a condition over a region of configurations: always, never, or it varies (with
    /// a witness of each side).
    pub fn classify(&self, c: &Condition, within: &Region) -> Result<Coverage, OptionProblem> {
        let r = self.compile(c)?;
        let inside = self.intersect(within, &r)?;
        let outside = self.subtract(within, &r)?;
        Ok(match (inside.is_empty(), outside.is_empty()) {
            (true, _) => Coverage::Never,
            (false, true) => Coverage::Always,
            (false, false) => Coverage::Varies {
                active: self.witness(&inside).unwrap_or_default(),
                inactive: self.witness(&outside).unwrap_or_default(),
            },
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Coverage {
    Always,
    Never,
    Varies { active: Point, inactive: Point },
}

#[cfg(test)]
mod tests {
    use super::*;

    fn block() -> OptionsBlock {
        OptionsBlock {
            axes: vec![
                AxisDecl {
                    name: "msn".into(),
                    domain: AxisDomain::Unit { min: 1, max: 1000 },
                    default: None,
                },
                AxisDecl {
                    name: "engine".into(),
                    domain: AxisDomain::Enum {
                        values: vec!["a".into(), "b".into()],
                    },
                    default: Some(AxisValue::Enum("a".into())),
                },
                AxisDecl {
                    name: "freighter".into(),
                    domain: AxisDomain::Bool,
                    default: Some(AxisValue::Bool(false)),
                },
            ],
            rules: vec![OptionRule {
                name: "freighter_needs_b".into(),
                condition: Condition::requires(
                    Condition::is("freighter", AxisValue::Bool(true)),
                    Condition::is_enum("engine", "b"),
                ),
            }],
            groups: vec![],
            configurations: vec![],
        }
    }

    #[test]
    fn dates_round_trip() {
        for d in [
            CivilDate::new(1970, 1, 1),
            CivilDate::new(2000, 2, 29),
            CivilDate::new(2027, 12, 31),
            CivilDate::new(1600, 3, 1),
        ] {
            assert_eq!(CivilDate::from_day_number(d.day_number()), d);
        }
        assert_eq!(CivilDate::new(1970, 1, 2).day_number(), 1);
        assert!(CivilDate::try_from("2027-02-30".to_string()).is_err());
    }

    #[test]
    fn complement_and_overlap_are_exact() {
        let s = OptionSpace::new(&block());
        let a = s
            .compile(&Condition::range("msn", Some(1), Some(40)))
            .unwrap();
        let b = s.compile(&Condition::range("msn", Some(41), None)).unwrap();
        assert!(s.intersect(&a, &b).unwrap().is_empty());
        let both = s
            .compile(&Condition::any(vec![
                Condition::range("msn", Some(1), Some(40)),
                Condition::range("msn", Some(41), None),
            ]))
            .unwrap();
        assert!(s.complement(&both).unwrap().is_empty());
        let c = s
            .compile(&Condition::range("msn", Some(40), Some(41)))
            .unwrap();
        let w = s.witness(&s.intersect(&a, &c).unwrap()).unwrap();
        assert_eq!(w[0], 40);
    }

    #[test]
    fn rules_shape_the_legal_space() {
        let s = OptionSpace::new(&block());
        let mut v = BTreeMap::new();
        v.insert("msn".into(), AxisValue::Int(7));
        v.insert("freighter".into(), AxisValue::Bool(true));
        let p = s.point(&v).unwrap();
        assert!(!s.is_legal(&p), "a freighter with engine a breaks the rule");
        v.insert("engine".into(), AxisValue::Enum("b".into()));
        assert!(s.is_legal(&s.point(&v).unwrap()));
        let cov = s
            .classify(&Condition::is_enum("engine", "b"), s.legal())
            .unwrap();
        assert!(matches!(cov, Coverage::Varies { .. }));
    }
}
