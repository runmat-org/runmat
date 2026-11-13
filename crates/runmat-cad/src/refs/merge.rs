//! Re-resolution after merge: every reference whose target changed is reported, never
//! silently accepted.
//!
//! A three-way text merge can keep a reference's text intact while the entity it names
//! changes: one side renames or reorders, the other adds a feature that now makes the named
//! face; one side re-picks, the other edits the maker. After the merged document regenerates,
//! [`recheck_part_merge`] compares, site by site (`feature.round#edges[1]`, `region.mount`),
//! the entities each side resolved for the *same reference text* with what the merge resolves:
//!
//! - both sides agree, the merge differs → **redirected**;
//! - the sides differ: the side that kept the base's target defers to the side that changed it
//!   (that change is intended); when both changed it, any merged target is reported;
//! - a side resolved and the merge does not → **broken**.
//!
//! Entities compare by their structured names, never by per-run positions, so the comparison
//! is stable across kernels and runs. [`recheck_assembly_merge`] does the same for occurrence
//! paths: a path that now names a different occurrence id (a rename on one side plus a new
//! occurrence under the old name on the other) is redirected.

use crate::assembly::resolve::ResolvedAssembly;
use crate::diagnostics::{Cause, DiagCode, DiagnosticReport, Repair, Severity, Site, Subject};
use crate::ids::{ItemUuid, RegionName};
use crate::refs::name::EntityName;
use crate::refs::reference::Reference;
use crate::refs::resolve::ResolveOutcome;
use crate::regen::engine::{ItemOutcome, RegenResult};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// What one reference site resolved to.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SiteResolution {
    pub site: Site,
    pub reference: Reference,
    /// `None`: waiting or failed.
    pub entities: Option<Vec<Vec<EntityName>>>,
}

fn sorted(mut v: Vec<Vec<EntityName>>) -> Vec<Vec<EntityName>> {
    for e in &mut v {
        e.sort();
    }
    v.sort();
    v
}

/// Every reference site of a regeneration, by site key.
pub fn sites(r: &RegenResult) -> BTreeMap<String, SiteResolution> {
    let mut out = BTreeMap::new();
    for f in &r.features {
        for rec in &f.references {
            let site = Site::feature(&f.id, Some(&rec.param));
            let entities = match &rec.outcome {
                ResolveOutcome::Resolved(_) => Some(sorted(rec.entities.clone())),
                _ => None,
            };
            out.insert(
                site.key(),
                SiteResolution {
                    site,
                    reference: rec.reference.clone(),
                    entities,
                },
            );
        }
    }
    let mut item = |subject: Subject, o: &ItemOutcome| {
        for (k, rec) in o.references.iter().enumerate() {
            let site = Site {
                subject: subject.clone(),
                param: Some(format!("select[{}]", k)),
            };
            let entities = match &rec.outcome {
                ResolveOutcome::Resolved(_) => Some(sorted(rec.entities.clone())),
                _ => None,
            };
            out.insert(
                site.key(),
                SiteResolution {
                    site,
                    reference: rec.reference.clone(),
                    entities,
                },
            );
        }
    };
    for (name, o) in &r.regions {
        item(
            Subject::Region {
                region: RegionName::new(name.as_str()),
            },
            o,
        );
    }
    for (name, o) in &r.parts {
        item(Subject::Part { part: name.clone() }, o);
    }
    out
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "finding")]
pub enum MergeFinding {
    Redirected {
        site: Site,
        reference: Reference,
        expected: Vec<Vec<EntityName>>,
        merged: Vec<Vec<EntityName>>,
        /// Both sides changed the target, so no side's target is authoritative.
        both_changed: bool,
    },
    Broken {
        site: Site,
        reference: Reference,
        expected: Vec<Vec<EntityName>>,
    },
}

#[derive(Clone, Debug, Default)]
pub struct MergeReport {
    pub findings: Vec<MergeFinding>,
    pub diagnostics: DiagnosticReport,
    pub checked: usize,
}

fn show(v: &[Vec<EntityName>]) -> String {
    let parts: Vec<String> = v
        .iter()
        .map(|e| {
            e.first()
                .map(|n| n.to_string())
                .unwrap_or_else(|| "?".into())
        })
        .collect();
    format!("[{}]", parts.join(", "))
}

/// Recheck a merged part document against both sides (and the base, when known).
pub fn recheck_part_merge(
    base: Option<&RegenResult>,
    ours: &RegenResult,
    theirs: &RegenResult,
    merged: &RegenResult,
) -> MergeReport {
    let b = base.map(sites);
    let (o, t, m) = (sites(ours), sites(theirs), sites(merged));
    let mut rep = MergeReport::default();
    for (key, ms) in &m {
        // A side's expectation counts only where it wrote the same reference text.
        let same = |x: &BTreeMap<String, SiteResolution>| {
            x.get(key)
                .filter(|s| s.reference == ms.reference)
                .and_then(|s| s.entities.clone())
        };
        let (eo, et) = (same(&o), same(&t));
        let eb = b.as_ref().and_then(same);
        rep.checked += 1;
        let (expected, both_changed): (Option<Vec<Vec<EntityName>>>, bool) = match (eo, et) {
            (None, None) => continue,
            (Some(x), None) | (None, Some(x)) => (Some(x), false),
            (Some(x), Some(y)) if x == y => (Some(x), false),
            (Some(x), Some(y)) => match &eb {
                Some(base) if &x == base => (Some(y), false),
                Some(base) if &y == base => (Some(x), false),
                // Both changed (or no base): only a target equal to one side is acceptable.
                _ => {
                    if ms.entities.as_ref() == Some(&x) || ms.entities.as_ref() == Some(&y) {
                        continue;
                    }
                    (Some(x), true)
                }
            },
        };
        let Some(expected) = expected else { continue };
        match &ms.entities {
            Some(got) if *got == expected && !both_changed => {}
            Some(got) => {
                rep.diagnostics.push(
                    DiagCode::MergeRedirected,
                    Severity::Error,
                    ms.site.clone(),
                    format!(
                        "after the merge `{}` resolves to {} instead of {}{}",
                        ms.reference,
                        show(got),
                        show(&expected),
                        if both_changed {
                            " (both sides changed its target)"
                        } else {
                            ""
                        }
                    ),
                    Cause::Root,
                    vec![Repair::RePick {
                        site: ms.site.clone(),
                        candidates: Vec::new(),
                    }],
                );
                rep.findings.push(MergeFinding::Redirected {
                    site: ms.site.clone(),
                    reference: ms.reference.clone(),
                    expected,
                    merged: got.clone(),
                    both_changed,
                });
            }
            None => {
                rep.diagnostics.push(
                    DiagCode::MergeBroken,
                    Severity::Error,
                    ms.site.clone(),
                    format!(
                        "after the merge `{}` no longer resolves (it resolved to {} before)",
                        ms.reference,
                        show(&expected)
                    ),
                    Cause::Root,
                    vec![Repair::RePick {
                        site: ms.site.clone(),
                        candidates: Vec::new(),
                    }],
                );
                rep.findings.push(MergeFinding::Broken {
                    site: ms.site.clone(),
                    reference: ms.reference.clone(),
                    expected,
                });
            }
        }
    }
    rep
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PathRedirect {
    pub path: String,
    pub expected: ItemUuid,
    pub merged: ItemUuid,
}

/// Recheck a merged product: occurrence paths that name a different occurrence than both
/// sides agreed on (or than the side that changed it).
pub fn recheck_assembly_merge(
    base: Option<&ResolvedAssembly>,
    ours: &ResolvedAssembly,
    theirs: &ResolvedAssembly,
    merged: &ResolvedAssembly,
) -> (Vec<PathRedirect>, DiagnosticReport) {
    let index = |r: &ResolvedAssembly| -> BTreeMap<String, ItemUuid> {
        (0..r.nodes.len() as u32)
            .map(|i| (r.path(i), r.node(i).uuid))
            .collect()
    };
    let (o, t, m) = (index(ours), index(theirs), index(merged));
    let b = base.map(index);
    let mut out = Vec::new();
    let mut diags = DiagnosticReport::default();
    for (path, got) in &m {
        let (eo, et) = (o.get(path), t.get(path));
        let expected = match (eo, et) {
            (None, None) => continue,
            (Some(x), None) | (None, Some(x)) => *x,
            (Some(x), Some(y)) if x == y => *x,
            (Some(x), Some(y)) => match b.as_ref().and_then(|b| b.get(path)) {
                Some(base) if base == x => *y,
                Some(base) if base == y => *x,
                _ => {
                    if got == x || got == y {
                        continue;
                    }
                    *x
                }
            },
        };
        if *got != expected {
            diags.push(
                DiagCode::MergeRedirected,
                Severity::Error,
                Site::of(Subject::Occurrence { path: path.clone() }),
                format!(
                    "after the merge `{}` names a different occurrence ({} instead of {})",
                    path, got, expected
                ),
                Cause::Root,
                Vec::new(),
            );
            out.push(PathRedirect {
                path: path.clone(),
                expected,
                merged: *got,
            });
        }
    }
    (out, diags)
}
