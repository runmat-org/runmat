//! Hole tables: one row per hole of the drawn model, tagged, located from a datum origin in the
//! view's plane, with a size callout.
//!
//! Hole geometry comes through [`HoleGeometry`], which the kernel side implements: every hole
//! instance by feature and semantic key (position, axis, diameter, depth, thread, counterbore,
//! countersink, in model coordinates), the frame a view projects onto, and the point a datum
//! origin reference names. This module stays kernel-free.
//!
//! Tags: by size (`A1, A2 … B1`, one letter per distinct callout, in size order) or
//! sequential (`H1, H2 …`). Tags are kept across edits: `keep` pins a hole (`feature:key`) to
//! its tag, so a new hole does not retag the drawing. Holes whose axis is not along the view
//! direction cannot be located in the view's plane; they are left out and reported.

use crate::diagnostics::{Cause, DiagCode, DiagnosticReport, Severity, Site, Subject};
use crate::drawing::{ModelRef, View};
use crate::ids::FeatureId;
use crate::model::datum::Frame;
use crate::refs::reference::Reference;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ThreadCallout {
    /// `M8x1.25`, `1/4-20 UNC`.
    pub designation: String,
    /// Tolerance class (`6H`, `2B`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub class: Option<String>,
    /// Thread depth (SI metres); `None`: the full hole.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub depth: Option<f64>,
}

/// One hole of the model, in model coordinates (SI).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HoleInstance {
    pub feature: FeatureId,
    /// The hole's semantic key within its feature.
    pub key: String,
    /// The hole's start point on its face.
    pub position: [f64; 3],
    /// Into the material.
    pub axis: [f64; 3],
    pub diameter: f64,
    /// `None`: through all.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub depth: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thread: Option<ThreadCallout>,
    /// (diameter, depth).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub counterbore: Option<(f64, f64)>,
    /// (diameter, included angle in radians).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub countersink: Option<(f64, f64)>,
}

/// The kernel-side seam for hole tables.
pub trait HoleGeometry {
    /// Every hole instance of the model, by feature and key.
    fn holes(&self, model: &ModelRef) -> Result<Vec<HoleInstance>, String>;
    /// The view's projection frame in model coordinates: X right, Y up, Z towards the viewer.
    fn view_frame(&self, view: &View) -> Result<Frame, String>;
    /// The model point a reference names (a datum origin: a vertex, a hole centre, a frame).
    fn point(&self, model: &ModelRef, reference: &Reference) -> Result<[f64; 3], String>;
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HoleTagging {
    #[default]
    BySize,
    Sequential,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HoleSort {
    /// Feature order, then key.
    #[default]
    Feature,
    /// By X, then Y.
    X,
    /// By Y, then X.
    Y,
    /// By distance from the origin.
    Distance,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LengthUnit {
    #[default]
    Millimeter,
    Inch,
}

impl LengthUnit {
    fn per_metre(self) -> f64 {
        match self {
            LengthUnit::Millimeter => 1000.0,
            LengthUnit::Inch => 1.0 / 0.0254,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HoleTableSpec {
    /// Only these hole features (empty: all).
    #[serde(default)]
    pub features: Vec<FeatureId>,
    #[serde(default)]
    pub tagging: HoleTagging,
    #[serde(default)]
    pub sort: HoleSort,
    #[serde(default)]
    pub units: LengthUnit,
    #[serde(default = "two")]
    pub decimals: u32,
    /// `feature:key` → tag, kept across edits.
    #[serde(default)]
    pub keep: BTreeMap<String, String>,
}

fn two() -> u32 {
    2
}

impl Default for HoleTableSpec {
    fn default() -> Self {
        Self {
            features: Vec::new(),
            tagging: HoleTagging::default(),
            sort: HoleSort::default(),
            units: LengthUnit::default(),
            decimals: 2,
            keep: BTreeMap::new(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HoleRow {
    pub tag: String,
    pub feature: FeatureId,
    pub key: String,
    /// In the table's units, from the datum origin along the view's X and Y.
    pub x: f64,
    pub y: f64,
    pub x_text: String,
    pub y_text: String,
    /// The size callout (`Ø6.60 THRU`, `M8x1.25-6H ↧ 12.00`, `⌴ Ø11.00 ↧ 6.50`).
    pub size: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HoleSize {
    pub letter: String,
    pub size: String,
    pub quantity: u32,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct HoleTableData {
    pub table: String,
    pub rows: Vec<HoleRow>,
    /// One per distinct callout, in size order (by-size tagging only).
    pub sizes: Vec<HoleSize>,
    pub diagnostics: DiagnosticReport,
}

fn site(table: &str, param: &str) -> Site {
    Site::with_param(
        Subject::Value {
            name: format!("table.{}", table),
        },
        param,
    )
}

fn letters(mut i: usize) -> String {
    let mut s = Vec::new();
    loop {
        s.push((b'A' + (i % 26) as u8) as char);
        if i < 26 {
            break;
        }
        i = i / 26 - 1;
    }
    s.iter().rev().collect()
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn norm(a: [f64; 3]) -> f64 {
    dot(a, a).sqrt()
}

/// The size callout of a hole, in the table's units.
pub fn hole_callout(h: &HoleInstance, units: LengthUnit, decimals: u32) -> String {
    let f = |v: f64| format!("{:.*}", decimals as usize, v * units.per_metre());
    let depth = |d: Option<f64>| match d {
        None => "THRU".to_owned(),
        Some(d) => format!("↧ {}", f(d)),
    };
    let mut parts = vec![format!("Ø{} {}", f(h.diameter), depth(h.depth))];
    if let Some(t) = &h.thread {
        let mut s = t.designation.clone();
        if let Some(c) = &t.class {
            s.push('-');
            s.push_str(c);
        }
        s.push(' ');
        s.push_str(&depth(t.depth.or(h.depth)));
        parts.push(s);
    }
    if let Some((d, dp)) = h.counterbore {
        parts.push(format!("⌴ Ø{} ↧ {}", f(d), f(dp)));
    }
    if let Some((d, a)) = h.countersink {
        parts.push(format!("⌵ Ø{} × {:.0}°", f(d), a.to_degrees()));
    }
    parts.join(", ")
}

/// Hole-table rows for one table on one view.
pub fn hole_table(
    table: &str,
    spec: &HoleTableSpec,
    view: &View,
    origin: &Reference,
    geometry: &dyn HoleGeometry,
) -> HoleTableData {
    let mut out = HoleTableData {
        table: table.to_owned(),
        ..Default::default()
    };
    let fail = |out: &mut HoleTableData, param: &str, m: String| {
        out.diagnostics.push(
            DiagCode::DrawingHoleTable,
            Severity::Error,
            site(table, param),
            m,
            Cause::Root,
            Vec::new(),
        );
    };
    let holes = match geometry.holes(&view.model) {
        Ok(h) => h,
        Err(m) => {
            fail(&mut out, "view", format!("hole table `{}`: {}", table, m));
            return out;
        }
    };
    let frame = match geometry.view_frame(view) {
        Ok(f) => f,
        Err(m) => {
            fail(&mut out, "view", format!("hole table `{}`: {}", table, m));
            return out;
        }
    };
    let o = match geometry.point(&view.model, origin) {
        Ok(p) => p,
        Err(m) => {
            fail(
                &mut out,
                "origin",
                format!("hole table `{}` origin `{}` {}", table, origin, m),
            );
            return out;
        }
    };
    let known: BTreeSet<&FeatureId> = holes.iter().map(|h| &h.feature).collect();
    for f in &spec.features {
        if !known.contains(f) {
            fail(
                &mut out,
                "features",
                format!(
                    "hole table `{}` lists `{}`, which makes no holes in `{}`",
                    table, f, view.model.file
                ),
            );
        }
    }
    // Feature order as the model reports it.
    let mut feature_order: Vec<&FeatureId> = Vec::new();
    for h in &holes {
        if !feature_order.contains(&&h.feature) {
            feature_order.push(&h.feature);
        }
    }
    let k = spec.units.per_metre();
    let round = |v: f64| {
        let p = 10f64.powi(spec.decimals as i32);
        (v * p).round() / p
    };
    struct Placed<'h> {
        hole: &'h HoleInstance,
        x: f64,
        y: f64,
        size: String,
    }
    let mut placed: Vec<Placed> = Vec::new();
    let mut skewed = Vec::new();
    for h in &holes {
        if !spec.features.is_empty() && !spec.features.contains(&h.feature) {
            continue;
        }
        let a = norm(h.axis);
        if a == 0.0 || (dot(h.axis, frame.z).abs() / a - 1.0).abs() > 1e-6 {
            skewed.push(format!("{}:{}", h.feature, h.key));
            continue;
        }
        let d = [
            h.position[0] - o[0],
            h.position[1] - o[1],
            h.position[2] - o[2],
        ];
        placed.push(Placed {
            hole: h,
            x: round(dot(d, frame.x) * k),
            y: round(dot(d, frame.y) * k),
            size: hole_callout(h, spec.units, spec.decimals),
        });
    }
    if !skewed.is_empty() {
        out.diagnostics.push(
            DiagCode::DrawingHoleTable,
            Severity::Warning,
            site(table, "view"),
            format!(
                "hole table `{}` leaves out {} hole(s) not along the view direction: {}",
                table,
                skewed.len(),
                skewed.join(", ")
            ),
            Cause::Root,
            Vec::new(),
        );
    }
    let feat_idx = |f: &FeatureId| feature_order.iter().position(|x| *x == f).unwrap_or(0);
    placed.sort_by(|a, b| {
        let by_xy = |p: &Placed, q: &Placed, first_x: bool| {
            let (p1, p2, q1, q2) = if first_x {
                (p.x, p.y, q.x, q.y)
            } else {
                (p.y, p.x, q.y, q.x)
            };
            p1.partial_cmp(&q1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(p2.partial_cmp(&q2).unwrap_or(std::cmp::Ordering::Equal))
        };
        let o = match spec.sort {
            HoleSort::Feature => feat_idx(&a.hole.feature)
                .cmp(&feat_idx(&b.hole.feature))
                .then_with(|| a.hole.key.cmp(&b.hole.key)),
            HoleSort::X => by_xy(a, b, true),
            HoleSort::Y => by_xy(a, b, false),
            HoleSort::Distance => (a.x * a.x + a.y * a.y)
                .partial_cmp(&(b.x * b.x + b.y * b.y))
                .unwrap_or(std::cmp::Ordering::Equal),
        };
        o.then_with(|| (&a.hole.feature, &a.hole.key).cmp(&(&b.hole.feature, &b.hole.key)))
    });

    // Size letters: distinct callouts by diameter, then text.
    let mut sizes: Vec<(f64, String)> = Vec::new();
    for p in &placed {
        if !sizes.iter().any(|(_, s)| *s == p.size) {
            sizes.push((p.hole.diameter, p.size.clone()));
        }
    }
    sizes.sort_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.1.cmp(&b.1))
    });
    let letter_of = |size: &str| {
        letters(
            sizes
                .iter()
                .position(|(_, s)| s == size)
                .expect("every size is listed"),
        )
    };
    // Kept tags first; then the next free number per letter (or overall), in row order.
    let mut used: BTreeSet<String> = BTreeSet::new();
    let mut tags: Vec<Option<String>> = vec![None; placed.len()];
    for (i, p) in placed.iter().enumerate() {
        let id = format!("{}:{}", p.hole.feature, p.hole.key);
        if let Some(t) = spec.keep.get(&id) {
            if !used.insert(t.clone()) {
                fail(
                    &mut out,
                    "keep",
                    format!("hole table `{}` keeps tag `{}` for two holes", table, t),
                );
                continue;
            }
            tags[i] = Some(t.clone());
        }
    }
    let mut next: BTreeMap<String, u32> = BTreeMap::new();
    for (i, p) in placed.iter().enumerate() {
        if tags[i].is_some() {
            continue;
        }
        let prefix = match spec.tagging {
            HoleTagging::BySize => letter_of(&p.size),
            HoleTagging::Sequential => "H".into(),
        };
        let n = next.entry(prefix.clone()).or_insert(0);
        let tag = loop {
            *n += 1;
            let t = format!("{}{}", prefix, n);
            if !used.contains(&t) {
                break t;
            }
        };
        used.insert(tag.clone());
        tags[i] = Some(tag);
    }
    let text = |v: f64| format!("{:.*}", spec.decimals as usize, v);
    for (p, tag) in placed.iter().zip(tags) {
        out.rows.push(HoleRow {
            tag: tag.expect("every hole is tagged"),
            feature: p.hole.feature.clone(),
            key: p.hole.key.clone(),
            x: p.x,
            y: p.y,
            x_text: text(p.x),
            y_text: text(p.y),
            size: p.size.clone(),
        });
    }
    if spec.tagging == HoleTagging::BySize {
        out.sizes = sizes
            .iter()
            .enumerate()
            .map(|(i, (_, s))| HoleSize {
                letter: letters(i),
                size: s.clone(),
                quantity: placed.iter().filter(|p| &p.size == s).count() as u32,
            })
            .collect();
    }
    out
}
