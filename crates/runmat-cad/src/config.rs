//! Configurations at scale: columnar design tables, typed checks, change impact and
//! affected-only verification.
//!
//! - **Columnar tables.** `configurations from table inst("sha256:…")`: one configuration per
//!   row of a [`ColumnarTable`] (the same canonical encoding fastener tables use), with a key
//!   column and column → input bindings. Each row value takes its column's unit through the
//!   unit authority and is checked against the input's declared kind. [`materialize`] turns
//!   the table into ordinary table configurations once, at load, so regeneration sees one form.
//! - **Impact.** [`affected_configurations`] diffs two versions of a document and, with the
//!   static dependency graph, returns which configurations can change: a changed default
//!   reaches the configurations that do not override it, a changed row reaches its own
//!   configuration, a changed feature reaches every configuration that does not suppress it.
//!   [`affected_by_published`] does the same when only values published by other files moved.
//! - **Verify.** [`verify`] regenerates only those configurations (Creo's instance verify,
//!   limited to affected instances) and reports each one's errors and final key.

use crate::assembly::fasteners::{load_table, ColumnData, ColumnarTable, TableSource};
use crate::ids::{ConfigurationName, FeatureId, InputName};
use crate::model::items::{ConfigurationTable, TableRow};
use crate::model::PartDocument;
use crate::regen::deps::DependencyGraph;
use crate::regen::engine::{regenerate, RegenCache, RegenContext, RegenRequest};
use crate::regen::hash::CacheKey;
use crate::units::UnitAuthority;
use crate::value::ValueExpr;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

/// `configurations from table`, columnar.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ColumnarConfigurations {
    /// Configurations are named `<name>.<row key>`.
    pub name: String,
    /// `sha256:<hex>` of the table's binary encoding.
    pub table: String,
    pub key: String,
    /// Column → input.
    pub inputs: BTreeMap<String, InputName>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "problem")]
pub enum TableProblem {
    Load {
        message: String,
    },
    NoKeyColumn {
        column: String,
    },
    NoColumn {
        column: String,
    },
    NotNumeric {
        column: String,
    },
    UnknownUnit {
        column: String,
        unit: String,
    },
    UnknownInput {
        input: String,
    },
    KindMismatch {
        column: String,
        input: String,
        expected: String,
        found: String,
    },
    DuplicateKey {
        key: String,
    },
}

/// Expand a columnar table into a configuration table, typed against the document's inputs.
pub fn expand_columnar(
    doc: &PartDocument,
    spec: &ColumnarConfigurations,
    table: &ColumnarTable,
    units: &dyn UnitAuthority,
) -> Result<ConfigurationTable, Vec<TableProblem>> {
    let mut problems = Vec::new();
    let keys: Vec<String> = match table.column(&spec.key) {
        Some(ColumnData::Text { values }) => values.clone(),
        Some(ColumnData::I64 { values }) => values.iter().map(|v| v.to_string()).collect(),
        _ => {
            return Err(vec![TableProblem::NoKeyColumn {
                column: spec.key.clone(),
            }])
        }
    };
    let mut seen = BTreeSet::new();
    for k in &keys {
        if !seen.insert(k) {
            problems.push(TableProblem::DuplicateKey { key: k.clone() });
        }
    }
    let mut columns = Vec::new();
    let mut data: Vec<Vec<ValueExpr>> = Vec::new();
    for (col, input) in &spec.inputs {
        let Some(decl) = doc.inputs.iter().find(|i| &i.name == input) else {
            problems.push(TableProblem::UnknownInput {
                input: input.0.clone(),
            });
            continue;
        };
        let expected = units.kind(&decl.kind);
        let values: Vec<ValueExpr> = match table.column(col) {
            None => {
                problems.push(TableProblem::NoColumn {
                    column: col.clone(),
                });
                continue;
            }
            Some(ColumnData::F64 { unit, values }) => {
                let Some(u) = units.unit(unit) else {
                    problems.push(TableProblem::UnknownUnit {
                        column: col.clone(),
                        unit: unit.clone(),
                    });
                    continue;
                };
                if let Some(k) = &expected {
                    if k.dimension != u.dimension {
                        problems.push(TableProblem::KindMismatch {
                            column: col.clone(),
                            input: input.0.clone(),
                            expected: format!("{} ({})", k.id, k.si_symbol),
                            found: u.dimension.to_string(),
                        });
                        continue;
                    }
                }
                values.iter().map(|v| ValueExpr::qty(*v, unit)).collect()
            }
            Some(ColumnData::I64 { values }) => values
                .iter()
                .map(|v| ValueExpr::number(*v as f64))
                .collect(),
            Some(ColumnData::Text { .. }) => {
                problems.push(TableProblem::NotNumeric {
                    column: col.clone(),
                });
                continue;
            }
        };
        columns.push(input.clone());
        data.push(values);
    }
    if !problems.is_empty() {
        return Err(problems);
    }
    let rows = keys
        .iter()
        .enumerate()
        .map(|(i, k)| TableRow {
            key: k.clone(),
            values: data.iter().map(|c| c[i].clone()).collect(),
        })
        .collect();
    Ok(ConfigurationTable {
        name: spec.name.clone(),
        columns,
        rows,
    })
}

/// Load and expand every columnar table into the document's configuration tables.
pub fn materialize(
    doc: &PartDocument,
    specs: &[ColumnarConfigurations],
    tables: &dyn TableSource,
    units: &dyn UnitAuthority,
) -> (PartDocument, Vec<(String, TableProblem)>) {
    let mut out = doc.clone();
    let mut problems = Vec::new();
    for spec in specs {
        match load_table(tables, &spec.table) {
            Err(p) => problems.push((
                spec.name.clone(),
                TableProblem::Load {
                    message: p.to_string(),
                },
            )),
            Ok(t) => match expand_columnar(doc, spec, &t, units) {
                Ok(ct) => out.configuration_tables.push(ct),
                Err(ps) => problems.extend(ps.into_iter().map(|p| (spec.name.clone(), p))),
            },
        }
    }
    (out, problems)
}

/// Why a configuration is affected.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "because")]
pub enum Because {
    Default {
        input: InputName,
    },
    Override {
        input: InputName,
    },
    Feature {
        feature: FeatureId,
    },
    Suppression,
    New,
    /// A value published by another file that the configuration reads changed.
    Published {
        value: crate::value::ValueXref,
    },
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Impact {
    /// `None` is the default configuration.
    pub affected: BTreeMap<Option<ConfigurationName>, BTreeSet<Because>>,
    pub unaffected: BTreeSet<ConfigurationName>,
    pub removed: BTreeSet<ConfigurationName>,
}

/// Which configurations a change from `old` to `new` can change.
pub fn affected_configurations(old: &PartDocument, new: &PartDocument) -> Impact {
    let deps = DependencyGraph::analyse(new);
    let reaches = |inputs: &[InputName]| !deps.affected_by_inputs(inputs).features.is_empty();
    // Changed defaults.
    let changed_defaults: Vec<InputName> = new
        .inputs
        .iter()
        .filter(|i| old.inputs.iter().find(|o| o.name == i.name) != Some(*i))
        .map(|i| i.name.clone())
        .collect();
    // Changed features (content or order), and everything downstream.
    let old_index: BTreeMap<&FeatureId, (usize, &crate::model::Feature)> = old
        .features
        .iter()
        .enumerate()
        .map(|(i, f)| (&f.id, (i, f)))
        .collect();
    let roots: BTreeSet<FeatureId> = new
        .features
        .iter()
        .enumerate()
        .filter(|(i, f)| {
            old_index
                .get(&f.id)
                .map(|(oi, of)| oi != i || *of != *f)
                .unwrap_or(true)
        })
        .map(|(_, f)| f.id.clone())
        .collect();
    let changed_features = deps.downstream(&roots);
    let old_cfg: BTreeMap<ConfigurationName, crate::model::items::Configuration> = old
        .all_configurations()
        .into_iter()
        .map(|c| (c.name.clone(), c))
        .collect();
    let mut impact = Impact::default();
    let mut judge = |name: Option<ConfigurationName>,
                     overrides: &BTreeMap<InputName, ValueExpr>,
                     suppress: &BTreeSet<FeatureId>,
                     prev: Option<&crate::model::items::Configuration>| {
        let mut why = BTreeSet::new();
        for d in &changed_defaults {
            if !overrides.contains_key(d) && reaches(std::slice::from_ref(d)) {
                why.insert(Because::Default { input: d.clone() });
            }
        }
        match prev {
            None if name.is_some() => {
                why.insert(Because::New);
            }
            Some(p) => {
                let keys: BTreeSet<&InputName> =
                    p.overrides.keys().chain(overrides.keys()).collect();
                for k in keys {
                    if p.overrides.get(k) != overrides.get(k) && reaches(std::slice::from_ref(k)) {
                        why.insert(Because::Override { input: k.clone() });
                    }
                }
                if &p.suppress != suppress {
                    why.insert(Because::Suppression);
                }
            }
            None => {}
        }
        for f in &changed_features {
            if !suppress.contains(f) {
                why.insert(Because::Feature { feature: f.clone() });
            }
        }
        match (why.is_empty(), name) {
            (true, Some(n)) => {
                impact.unaffected.insert(n);
            }
            (true, None) => {}
            (false, n) => {
                impact.affected.insert(n, why);
            }
        }
    };
    judge(None, &BTreeMap::new(), &BTreeSet::new(), None);
    let new_cfgs = new.all_configurations();
    for c in &new_cfgs {
        judge(
            Some(c.name.clone()),
            &c.overrides,
            &c.suppress,
            old_cfg.get(&c.name),
        );
    }
    let new_names: BTreeSet<&ConfigurationName> = new_cfgs.iter().map(|c| &c.name).collect();
    impact.removed = old_cfg
        .keys()
        .filter(|n| !new_names.contains(n))
        .cloned()
        .collect();
    impact
}

/// Which configurations a change of values published by other files can change: a
/// configuration is affected when an unsuppressed feature reads a changed value (in its own
/// parameters or lets, or through an input default the configuration does not override), or
/// when one of its overrides reads one. Used when an upstream file's published value moves and
/// this file did not change.
pub fn affected_by_published(doc: &PartDocument, changed: &[crate::value::ValueXref]) -> Impact {
    let deps = DependencyGraph::analyse(doc);
    let changed_set: BTreeSet<&crate::value::ValueXref> = changed.iter().collect();
    let mut impact = Impact::default();
    let mut judge = |name: Option<ConfigurationName>,
                     overrides: &BTreeMap<InputName, ValueExpr>,
                     suppress: &BTreeSet<FeatureId>| {
        let overridden: BTreeSet<InputName> = overrides.keys().cloned().collect();
        let mut why = BTreeSet::new();
        let mut roots = deps.published_readers(changed, &overridden);
        // Overrides that read a changed value act like changed inputs.
        let mut hit_overrides = Vec::new();
        for (input, expr) in overrides {
            let mut xs = Vec::new();
            expr.collect_published(&mut xs);
            for x in xs.into_iter().filter(|x| changed_set.contains(x)) {
                why.insert(Because::Published { value: x.clone() });
                hit_overrides.push(input.clone());
            }
        }
        if !hit_overrides.is_empty() {
            roots.extend(deps.affected_by_inputs(&hit_overrides).features);
        }
        let reached: BTreeSet<FeatureId> = deps
            .downstream(&roots)
            .into_iter()
            .filter(|f| !suppress.contains(f))
            .collect();
        for f in &roots {
            if suppress.contains(f) {
                continue;
            }
            let reads = deps
                .feature_published
                .get(f)
                .into_iter()
                .flatten()
                .chain(
                    deps.feature_inputs
                        .get(f)
                        .into_iter()
                        .flatten()
                        .filter(|i| !overridden.contains(*i))
                        .flat_map(|i| deps.input_published.get(i).into_iter().flatten()),
                )
                .filter(|x| changed_set.contains(x));
            for x in reads {
                why.insert(Because::Published { value: x.clone() });
            }
        }
        for f in reached {
            why.insert(Because::Feature { feature: f });
        }
        match (why.is_empty(), name) {
            (true, Some(n)) => {
                impact.unaffected.insert(n);
            }
            (true, None) => {}
            (false, n) => {
                impact.affected.insert(n, why);
            }
        }
    };
    judge(None, &BTreeMap::new(), &BTreeSet::new());
    for c in doc.all_configurations() {
        judge(Some(c.name.clone()), &c.overrides, &c.suppress);
    }
    impact
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VerifyRow {
    pub configuration: Option<ConfigurationName>,
    pub errors: usize,
    pub first_error: Option<String>,
    pub final_key: CacheKey,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct VerifyReport {
    pub rows: Vec<VerifyRow>,
    pub skipped: BTreeSet<ConfigurationName>,
}

impl VerifyReport {
    pub fn ok(&self) -> bool {
        self.rows.iter().all(|r| r.errors == 0)
    }
}

/// Regenerate exactly the configurations `impact` marks affected (all of them when `impact`
/// is `None`), sharing one kernel cache.
pub fn verify(doc: &PartDocument, ctx: &RegenContext<'_>, impact: Option<&Impact>) -> VerifyReport {
    let mut cache = RegenCache::default();
    let mut report = VerifyReport::default();
    let mut targets: Vec<Option<ConfigurationName>> = vec![None];
    targets.extend(doc.all_configurations().into_iter().map(|c| Some(c.name)));
    for t in targets {
        if let Some(imp) = impact {
            if !imp.affected.contains_key(&t) {
                if let Some(n) = t {
                    report.skipped.insert(n);
                }
                continue;
            }
        }
        let r = regenerate(
            doc,
            ctx,
            &RegenRequest {
                configuration: t.clone(),
            },
            &mut cache,
        );
        let errors: Vec<_> = r.diagnostics.errors().collect();
        report.rows.push(VerifyRow {
            configuration: t,
            errors: errors.len(),
            first_error: errors.first().map(|d| format!("{} {}", d.code, d.message)),
            final_key: r.final_key.clone(),
        });
    }
    report
}
