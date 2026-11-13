//! Graded, interface-conforming triangle mesh of a cross-section.
//!
//! The mesh is a mapped tensor grid. Rows lie on every layer interface, the signal band faces
//! and the mask surfaces; columns pass through every trace corner and follow the sloped
//! sidewalls across the signal band, so trace boundaries and dielectric interfaces are mesh
//! lines. Spacing grows geometrically from `edge_size` at trace edges and corners (where the
//! field is singular) to `max_size`, by the growth ratio. Each quad splits along its shorter
//! diagonal into two linear triangles.
use super::{
    geometry::{Geometry, KeyColumn},
    CrossSectionError, CrossSectionMeshControls,
};
use serde::{Deserialize, Serialize};

/// Default element size at trace edges: the smallest feature divided by this.
pub(super) const EDGE_DIVISIONS: f64 = 1000.0;
/// Default largest element, in dielectric heights; growth from the edges sets the size near
/// the traces, this caps it in the far field.
pub(super) const MAX_SIZE_HEIGHTS: f64 = 2.0;
/// Resident bytes per mesh node before factorization (coordinates, connectivity, operator
/// rows); the factor itself is quoted by the symbolic analysis.
pub(super) const BYTES_PER_NODE: u64 = 256;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CrossSectionMeshSummary {
    pub nodes: usize,
    pub elements: usize,
    pub edge_size_m: f64,
    pub max_size_m: f64,
    pub growth: f64,
}

#[derive(Debug, Clone)]
pub(super) struct CrossSectionMesh {
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    pub triangles: Vec<[usize; 3]>,
    pub columns: usize,
    pub rows: usize,
    summary: CrossSectionMeshSummary,
}

/// Points from `a` to `b` (both included) whose spacing follows
/// h(x) = min(h_max, h_a + (g − 1)(x − a), h_b + (g − 1)(b − x)): equal steps of ∫ dx / h.
pub(super) fn graded(a: f64, b: f64, h_a: f64, h_b: f64, growth: f64, h_max: f64) -> Vec<f64> {
    if b <= a {
        return vec![a];
    }
    const SAMPLES: usize = 2048;
    let rate = growth - 1.0;
    let size = |x: f64| h_max.min(h_a + rate * (x - a)).min(h_b + rate * (b - x));
    let step = (b - a) / SAMPLES as f64;
    let mut cumulative = Vec::with_capacity(SAMPLES + 1);
    cumulative.push(0.0);
    let mut total = 0.0;
    for i in 0..SAMPLES {
        let x0 = a + step * i as f64;
        let x1 = x0 + step;
        total += 0.5 * step * (1.0 / size(x0) + 1.0 / size(x1));
        cumulative.push(total);
    }
    let count = (total - 1e-9).ceil().max(1.0) as usize;
    let mut points = Vec::with_capacity(count + 1);
    points.push(a);
    let mut sample = 0;
    for k in 1..count {
        let target = total * k as f64 / count as f64;
        while cumulative[sample + 1] < target {
            sample += 1;
        }
        let f = (target - cumulative[sample]) / (cumulative[sample + 1] - cumulative[sample]);
        points.push(a + step * (sample as f64 + f));
    }
    points.push(b);
    points
}

/// Graded lines through sorted key positions with a target size at each key.
fn lines(keys: &[(f64, f64)], growth: f64, h_max: f64) -> Vec<f64> {
    let mut out = vec![keys[0].0];
    for pair in keys.windows(2) {
        let (a, h_a) = pair[0];
        let (b, h_b) = pair[1];
        out.extend(graded(a, b, h_a, h_b, growth, h_max).into_iter().skip(1));
    }
    out
}

impl CrossSectionMesh {
    pub fn build(
        geometry: &Geometry,
        controls: &CrossSectionMeshControls,
    ) -> Result<Self, CrossSectionError> {
        let growth = controls.growth;
        if !(growth.is_finite() && growth > 1.0 && growth <= 2.0) {
            return Err(CrossSectionError::Invalid(
                "mesh growth must be in (1, 2]".into(),
            ));
        }
        let edge = match controls.edge_size_m {
            Some(size) if size.is_finite() && size > 0.0 => size,
            Some(_) => {
                return Err(CrossSectionError::Invalid(
                    "mesh edge size must be positive".into(),
                ))
            }
            None => geometry.feature / EDGE_DIVISIONS,
        };
        let h_max = match controls.max_size_m {
            Some(size) if size.is_finite() && size >= edge => size,
            Some(_) => {
                return Err(CrossSectionError::Invalid(
                    "mesh max size must be finite and at least the edge size".into(),
                ))
            }
            None => (geometry.height * MAX_SIZE_HEIGHTS).max(edge),
        };
        let tolerance = 1e-9 * (geometry.y_high - geometry.y_low);
        // Rows.
        let mut y_keys = geometry.y_keys();
        y_keys.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut merged: Vec<(f64, bool)> = Vec::new();
        for (y, at_edge) in y_keys {
            match merged.last_mut() {
                Some(last) if (y - last.0).abs() <= tolerance => last.1 |= at_edge,
                _ => merged.push((y, at_edge)),
            }
        }
        let sized: Vec<(f64, f64)> = merged
            .iter()
            .enumerate()
            .map(|(i, &(y, at_edge))| {
                if at_edge {
                    return (y, edge);
                }
                let below = (i > 0).then(|| y - merged[i - 1].0);
                let above = merged.get(i + 1).map(|next| next.0 - y);
                let local = below.into_iter().chain(above).fold(h_max, f64::min);
                (y, (0.5 * local).max(edge))
            })
            .collect();
        let y_lines = lines(&sized, growth, h_max);
        // Columns: parametric positions between consecutive key columns.
        let keys = geometry.x_keys();
        for pair in keys.windows(2) {
            if !(pair[1].low > pair[0].low && pair[1].high > pair[0].high) {
                return Err(CrossSectionError::Invalid(
                    "trace edges cross or touch; conductors must be separated".into(),
                ));
            }
        }
        let mut parametric: Vec<(usize, f64)> = Vec::new();
        for (k, pair) in keys.windows(2).enumerate() {
            let size = |key: &KeyColumn| if key.edge { edge } else { h_max };
            // Grade on the wider of the two faces so both are resolved.
            let width = (pair[1].low - pair[0].low).max(pair[1].high - pair[0].high);
            let points = graded(0.0, width, size(&pair[0]), size(&pair[1]), growth, h_max);
            for point in &points[..points.len() - 1] {
                parametric.push((k, point / width));
            }
        }
        parametric.push((keys.len() - 2, 1.0));
        let (rows, columns) = (y_lines.len(), parametric.len());
        let nodes = rows
            .checked_mul(columns)
            .ok_or_else(|| CrossSectionError::Invalid("mesh size overflows".into()))?;
        if let Some(budget) = crate::solve::shared_sparse::default_memory_budget() {
            let required = (nodes as u64).saturating_mul(BYTES_PER_NODE);
            if required > budget {
                return Err(CrossSectionError::Budget {
                    required_bytes: required,
                    budget_bytes: budget,
                });
            }
        }
        let mut x = Vec::with_capacity(nodes);
        let mut y = Vec::with_capacity(nodes);
        for &row_y in &y_lines {
            let key_x: Vec<f64> = keys.iter().map(|key| key.x_at(row_y)).collect();
            for &(k, s) in &parametric {
                x.push(key_x[k] + s * (key_x[k + 1] - key_x[k]));
                y.push(row_y);
            }
        }
        let mut triangles = Vec::with_capacity(2 * (rows - 1) * (columns - 1));
        let id = |i: usize, j: usize| i * columns + j;
        let length = |p: usize, q: usize| (x[p] - x[q]).hypot(y[p] - y[q]);
        for i in 0..rows - 1 {
            for j in 0..columns - 1 {
                let (a, b, c, d) = (id(i, j), id(i, j + 1), id(i + 1, j + 1), id(i + 1, j));
                if length(a, c) <= length(b, d) {
                    triangles.push([a, b, c]);
                    triangles.push([a, c, d]);
                } else {
                    triangles.push([a, b, d]);
                    triangles.push([b, c, d]);
                }
            }
        }
        let summary = CrossSectionMeshSummary {
            nodes,
            elements: triangles.len(),
            edge_size_m: edge,
            max_size_m: h_max,
            growth,
        };
        Ok(Self {
            x,
            y,
            triangles,
            columns,
            rows,
            summary,
        })
    }

    pub fn summary(&self) -> CrossSectionMeshSummary {
        self.summary
    }
}
