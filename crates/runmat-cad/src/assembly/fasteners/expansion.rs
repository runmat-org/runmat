//! Expanding a fastener set: stations along its rows, the fastener, grip and drill at each,
//! exceptions, and the problems expansion reports.

use super::*;

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

pub(super) struct Raw {
    pub(super) key: FastenerKey,
    pub(super) station: f64,
    pub(super) chain: String,
    pub(super) row: String,
    /// Arc length along the chain (along rules).
    pub(super) s: f64,
    pub(super) position: [f64; 3],
    pub(super) axis: [f64; 3],
    pub(super) diameter: f64,
    pub(super) spec: Option<FastenerSpec>,
}

pub(super) fn len3(a: [f64; 3]) -> f64 {
    (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt()
}

pub(super) fn norm3(a: [f64; 3]) -> [f64; 3] {
    let n = len3(a);
    if n > 0.0 {
        [a[0] / n, a[1] / n, a[2] / n]
    } else {
        a
    }
}

pub(super) fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Point, tangent and normal at arc length `s` along a chain.
pub(super) fn sample(chain: &PathChain, cum: &[f64], s: f64) -> ([f64; 3], [f64; 3], [f64; 3]) {
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
