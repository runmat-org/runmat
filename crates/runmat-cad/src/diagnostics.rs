//! Regeneration diagnostics: root causes separate from their cascades, stable codes, repair
//! suggestions as typed edit operations, and source-span placeholders the syntax layer fills.

use crate::ids::{FeatureId, FilePath, PartName, PublishName, RegionName, RevisionId};
use crate::model::items::PublishEntry;
use crate::refs::{EntityName, RefProblem, Reference, WaitCause, WaitReason, XrefProblem};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum DiagCode {
    #[serde(rename = "cad.ref.maker_missing")]
    RefMakerMissing,
    #[serde(rename = "cad.ref.forward_reference")]
    RefForward,
    #[serde(rename = "cad.ref.blocked")]
    RefBlocked,
    #[serde(rename = "cad.ref.not_generated")]
    RefNotGenerated,
    #[serde(rename = "cad.ref.consumed")]
    RefConsumed,
    #[serde(rename = "cad.ref.instance_removed")]
    RefInstanceRemoved,
    #[serde(rename = "cad.ref.not_adjacent")]
    RefNotAdjacent,
    #[serde(rename = "cad.ref.edge_replaced")]
    RefEdgeReplaced,
    #[serde(rename = "cad.ref.no_piece")]
    RefNoPiece,
    #[serde(rename = "cad.ref.no_vertex")]
    RefNoVertex,
    #[serde(rename = "cad.ref.kind_mismatch")]
    RefKindMismatch,
    #[serde(rename = "cad.ref.ambiguous")]
    RefAmbiguous,
    #[serde(rename = "cad.ref.empty")]
    RefEmpty,
    #[serde(rename = "cad.ref.waiting")]
    RefWaiting,
    #[serde(rename = "cad.ref.dormant")]
    RefDormant,
    #[serde(rename = "cad.ref.via_alias")]
    RefViaAlias,
    #[serde(rename = "cad.ref.fallback_used")]
    RefFallbackUsed,
    #[serde(rename = "cad.ref.fallback_failed")]
    RefFallbackFailed,
    #[serde(rename = "cad.ref.fallback_disabled")]
    RefFallbackDisabled,
    #[serde(rename = "cad.ref.unknown_publish")]
    RefUnknownPublish,
    #[serde(rename = "cad.xref.no_catalog")]
    XrefNoCatalog,
    #[serde(rename = "cad.xref.unknown_file")]
    XrefUnknownFile,
    #[serde(rename = "cad.xref.not_published")]
    XrefNotPublished,
    #[serde(rename = "cad.xref.revision_missing")]
    XrefRevisionMissing,
    #[serde(rename = "cad.xref.live_across_partition")]
    XrefLiveAcrossPartition,
    #[serde(rename = "cad.xref.publisher_broken")]
    XrefPublisherBroken,
    #[serde(rename = "cad.xref.pin_behind")]
    XrefPinBehind,
    #[serde(rename = "cad.value.failed")]
    ValueFailed,
    #[serde(rename = "cad.sketch.solve_failed")]
    SketchSolveFailed,
    #[serde(rename = "cad.sketch.redundant")]
    SketchRedundant,
    #[serde(rename = "cad.sketch.open_profile")]
    SketchOpenProfile,
    #[serde(rename = "cad.kernel.not_supported")]
    KernelNotSupported,
    #[serde(rename = "cad.kernel.failed")]
    KernelFailed,
    #[serde(rename = "cad.kernel.history_incomplete")]
    KernelHistoryIncomplete,
    #[serde(rename = "cad.regen.cascade")]
    RegenCascade,
    #[serde(rename = "cad.regen.frozen_stale")]
    RegenFrozenStale,
    #[serde(rename = "cad.regen.frozen_missing")]
    RegenFrozenMissing,
    #[serde(rename = "cad.regen.invalid")]
    RegenInvalid,
    #[serde(rename = "cad.region.expectation")]
    RegionExpectation,
}

impl fmt::Display for DiagCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let v = serde_json::to_value(self).expect("code serializes");
        f.write_str(v.as_str().unwrap_or("cad.unknown"))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Severity {
    Error,
    Warning,
    Info,
}

/// The item a diagnostic is about.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "subject")]
pub enum Subject {
    Feature { feature: FeatureId },
    Part { part: PartName },
    Region { region: RegionName },
    Publish { publish: PublishName },
    Material { index: usize },
    Value { name: String },
    Document,
}

/// Where in an item: a parameter path such as `edges[1]` or `constraints[3]`.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Site {
    pub subject: Subject,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub param: Option<String>,
}

impl Site {
    pub fn feature(f: &FeatureId, param: Option<&str>) -> Self {
        Site {
            subject: Subject::Feature { feature: f.clone() },
            param: param.map(str::to_owned),
        }
    }

    /// The key the syntax layer's span map uses: `feature.round#edges[1]`.
    pub fn key(&self) -> String {
        let base = match &self.subject {
            Subject::Feature { feature } => format!("feature.{}", feature),
            Subject::Part { part } => format!("part.{}", part),
            Subject::Region { region } => format!("region.{}", region),
            Subject::Publish { publish } => format!("publish.{}", publish),
            Subject::Material { index } => format!("material[{}]", index),
            Subject::Value { name } => name.clone(),
            Subject::Document => "document".into(),
        };
        match &self.param {
            Some(p) => format!("{}#{}", base, p),
            None => base,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct SourceSpan {
    pub file: FilePath,
    pub start: u32,
    pub end: u32,
}

/// Site key → span, supplied by the syntax layer (M1).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpanMap {
    pub spans: BTreeMap<String, SourceSpan>,
}

/// A repair: a typed model edit. Each maps to a registry command with a preview.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "repair")]
pub enum Repair {
    /// Pick again; `candidates` are same-feature entities to offer first.
    RePick {
        site: Site,
        candidates: Vec<Reference>,
    },
    /// Replace the reference with one of these, each selecting a single piece.
    Disambiguate {
        site: Site,
        candidates: Vec<Reference>,
    },
    /// Use the relational form: the edges between these two faces.
    UseRelational {
        site: Site,
        between: Box<[Reference; 2]>,
    },
    /// Write the alias target into the text.
    RewriteReference {
        site: Site,
        reference: Reference,
    },
    Unsuppress {
        feature: FeatureId,
    },
    /// Move `feature` so it runs after `after`.
    Reorder {
        feature: FeatureId,
        after: FeatureId,
    },
    /// Move `feature` so it runs before `before`.
    MoveBefore {
        feature: FeatureId,
        before: FeatureId,
    },
    RemoveReference {
        site: Site,
    },
    PinToRevision {
        site: Site,
        file: FilePath,
        revision: RevisionId,
    },
    UpdatePin {
        site: Site,
        file: FilePath,
        from: RevisionId,
        to: RevisionId,
    },
    AddPublish {
        file: FilePath,
        entry: PublishEntry,
    },
    /// Release the other file so it can be pinned.
    ReleaseFile {
        file: FilePath,
    },
    /// Point the region or reference at the blend face that replaced the edge.
    UseBlendFace {
        site: Site,
        face: EntityName,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "cause")]
pub enum Cause {
    Root,
    /// Failed because of diagnostic `root`.
    Cascade {
        root: u32,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Diagnostic {
    pub id: u32,
    pub code: DiagCode,
    pub severity: Severity,
    pub site: Site,
    pub message: String,
    pub cause: Cause,
    #[serde(default)]
    pub repairs: Vec<Repair>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub span: Option<SourceSpan>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct DiagnosticReport {
    pub diagnostics: Vec<Diagnostic>,
}

impl DiagnosticReport {
    pub fn push(
        &mut self,
        code: DiagCode,
        severity: Severity,
        site: Site,
        message: String,
        cause: Cause,
        repairs: Vec<Repair>,
    ) -> u32 {
        let id = self.diagnostics.len() as u32;
        self.diagnostics.push(Diagnostic {
            id,
            code,
            severity,
            site,
            message,
            cause,
            repairs,
            span: None,
        });
        id
    }

    pub fn roots(&self) -> impl Iterator<Item = &Diagnostic> {
        self.diagnostics
            .iter()
            .filter(|d| d.cause == Cause::Root && d.severity == Severity::Error)
    }

    pub fn cascades_of(&self, root: u32) -> impl Iterator<Item = &Diagnostic> {
        self.diagnostics
            .iter()
            .filter(move |d| d.cause == Cause::Cascade { root })
    }

    pub fn with_code(&self, code: DiagCode) -> Vec<&Diagnostic> {
        self.diagnostics.iter().filter(|d| d.code == code).collect()
    }

    pub fn errors(&self) -> impl Iterator<Item = &Diagnostic> {
        self.diagnostics
            .iter()
            .filter(|d| d.severity == Severity::Error)
    }

    /// Fill spans from the syntax layer's map.
    pub fn attach_spans(&mut self, spans: &SpanMap) {
        for d in &mut self.diagnostics {
            d.span = spans
                .spans
                .get(&d.site.key())
                .or_else(|| {
                    let mut s = d.site.clone();
                    s.param = None;
                    spans.spans.get(&s.key())
                })
                .cloned();
        }
    }
}

/// Code, severity, message and repairs for a reference problem at `site`.
pub fn describe_ref_problem(
    site: &Site,
    reference: &Reference,
    problem: &RefProblem,
) -> (DiagCode, String, Vec<Repair>) {
    let consumer = match &site.subject {
        Subject::Feature { feature } => Some(feature.clone()),
        _ => None,
    };
    match problem {
        RefProblem::MakerMissing { maker } => (
            DiagCode::RefMakerMissing,
            format!(
                "`{}` refers to feature `{}`, which is not in the file",
                reference, maker
            ),
            vec![
                Repair::RePick {
                    site: site.clone(),
                    candidates: Vec::new(),
                },
                Repair::RemoveReference { site: site.clone() },
            ],
        ),
        RefProblem::ForwardReference { maker } => (
            DiagCode::RefForward,
            format!(
                "`{}` refers to `{}`, which comes later in the history",
                reference, maker
            ),
            consumer
                .map(|c| {
                    vec![Repair::Reorder {
                        feature: c,
                        after: maker.clone(),
                    }]
                })
                .unwrap_or_default(),
        ),
        RefProblem::Blocked { maker, root } => (
            DiagCode::RefBlocked,
            if maker == root {
                format!("`{}` waits on `{}`, which failed", reference, maker)
            } else {
                format!(
                    "`{}` waits on `{}`, which could not run because `{}` failed",
                    reference, maker, root
                )
            },
            Vec::new(),
        ),
        RefProblem::NotGenerated {
            maker,
            name,
            candidates,
        } => (
            DiagCode::RefNotGenerated,
            format!("`{}` ran but no longer makes `{}`", maker, name),
            vec![Repair::RePick {
                site: site.clone(),
                candidates: candidates.clone(),
            }],
        ),
        RefProblem::Consumed { name, by } => {
            let mut repairs = vec![Repair::RePick {
                site: site.clone(),
                candidates: Vec::new(),
            }];
            if let Some(c) = consumer {
                repairs.push(Repair::MoveBefore {
                    feature: c,
                    before: by.clone(),
                });
            }
            (
                DiagCode::RefConsumed,
                format!("`{}` was removed by `{}`", name, by),
                repairs,
            )
        }
        RefProblem::InstanceRemoved {
            pattern,
            instance,
            available,
        } => (
            DiagCode::RefInstanceRemoved,
            format!(
                "`{}` no longer has instance {} (instances now: {})",
                pattern,
                instance,
                available
                    .iter()
                    .map(|k| k.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            vec![
                Repair::RePick {
                    site: site.clone(),
                    candidates: Vec::new(),
                },
                Repair::RemoveReference { site: site.clone() },
            ],
        ),
        RefProblem::NotAdjacent { a, b } => (
            DiagCode::RefNotAdjacent,
            format!("no edge runs between `{}` and `{}`", a, b),
            vec![Repair::RePick {
                site: site.clone(),
                candidates: Vec::new(),
            }],
        ),
        RefProblem::EdgeReplaced { by, blend } => (
            DiagCode::RefEdgeReplaced,
            format!(
                "the edge `{}` was replaced by `{}` from `{}`",
                reference, blend, by
            ),
            vec![Repair::UseBlendFace {
                site: site.clone(),
                face: blend.clone(),
            }],
        ),
        RefProblem::NoPieceMatches { base, pieces } => (
            DiagCode::RefNoPiece,
            format!(
                "none of the {} pieces of `{}` matches the qualifier",
                pieces, base
            ),
            vec![Repair::RePick {
                site: site.clone(),
                candidates: Vec::new(),
            }],
        ),
        RefProblem::NoVertex => (
            DiagCode::RefNoVertex,
            format!("no vertex touches everything in `{}`", reference),
            vec![Repair::RePick {
                site: site.clone(),
                candidates: Vec::new(),
            }],
        ),
        RefProblem::KindMismatch { expected, found } => (
            DiagCode::RefKindMismatch,
            format!(
                "`{}` selects {}s; a {} is needed here",
                reference, found, expected
            ),
            vec![Repair::RePick {
                site: site.clone(),
                candidates: Vec::new(),
            }],
        ),
        RefProblem::Ambiguous { count, candidates } => (
            DiagCode::RefAmbiguous,
            format!(
                "`{}` now selects {} entities; one is needed here",
                reference, count
            ),
            vec![Repair::Disambiguate {
                site: site.clone(),
                candidates: candidates.clone(),
            }],
        ),
        RefProblem::Xref { problem } => {
            let (code, repairs) = match problem {
                XrefProblem::NoCatalog => (DiagCode::XrefNoCatalog, Vec::new()),
                XrefProblem::UnknownFile { .. } => (
                    DiagCode::XrefUnknownFile,
                    vec![Repair::RemoveReference { site: site.clone() }],
                ),
                XrefProblem::NotPublished { .. } => (
                    DiagCode::XrefNotPublished,
                    vec![Repair::RePick {
                        site: site.clone(),
                        candidates: Vec::new(),
                    }],
                ),
                XrefProblem::RevisionMissing { file, .. } => (
                    DiagCode::XrefRevisionMissing,
                    vec![Repair::ReleaseFile { file: file.clone() }],
                ),
                XrefProblem::LiveAcrossPartition {
                    file,
                    latest_release,
                } => (
                    DiagCode::XrefLiveAcrossPartition,
                    match latest_release {
                        Some(r) => vec![Repair::PinToRevision {
                            site: site.clone(),
                            file: file.clone(),
                            revision: r.clone(),
                        }],
                        None => vec![Repair::ReleaseFile { file: file.clone() }],
                    },
                ),
                XrefProblem::PublisherBroken { .. } => (DiagCode::XrefPublisherBroken, Vec::new()),
            };
            (code, problem.to_string(), repairs)
        }
        RefProblem::UnknownPublish { name } => (
            DiagCode::RefUnknownPublish,
            format!("this file publishes no `{}`", name),
            vec![Repair::RemoveReference { site: site.clone() }],
        ),
        RefProblem::FallbackFailed { inner, matches } => {
            let (_, msg, repairs) = describe_ref_problem(site, reference, inner);
            (
                DiagCode::RefFallbackFailed,
                format!(
                    "{}; the geometric fallback found {} matches, not one",
                    msg, matches
                ),
                repairs,
            )
        }
        RefProblem::FallbackDisabled { inner } => {
            let (_, msg, repairs) = describe_ref_problem(site, reference, inner);
            (
                DiagCode::RefFallbackDisabled,
                format!("{}; the geometric fallback is turned off", msg),
                repairs,
            )
        }
        RefProblem::Empty => (
            DiagCode::RefEmpty,
            format!("`{}` selects nothing", reference),
            vec![Repair::RePick {
                site: site.clone(),
                candidates: Vec::new(),
            }],
        ),
    }
}

pub fn describe_wait(reference: &Reference, w: &WaitReason) -> (String, Vec<Repair>) {
    let why = match &w.cause {
        WaitCause::Suppressed => format!("`{}` is suppressed", w.maker),
        WaitCause::SuppressedInConfiguration => {
            format!("`{}` is suppressed in this configuration", w.maker)
        }
        WaitCause::RolledBack => format!("`{}` is past the rollback bar", w.maker),
        WaitCause::MakerWaiting { root } => format!("`{}` is waiting on `{}`", w.maker, root),
    };
    let root = match &w.cause {
        WaitCause::MakerWaiting { root } => root.clone(),
        _ => w.maker.clone(),
    };
    (
        format!(
            "`{}` is waiting: {} (the reference is kept, not lost)",
            reference, why
        ),
        vec![Repair::Unsuppress { feature: root }],
    )
}
