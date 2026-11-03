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
    #[serde(rename = "cad.xref.via_alias")]
    XrefViaAlias,
    #[serde(rename = "cad.merge.redirected")]
    MergeRedirected,
    #[serde(rename = "cad.merge.broken")]
    MergeBroken,
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
    #[serde(rename = "cad.asm.unknown_definition")]
    AsmUnknownDefinition,
    #[serde(rename = "cad.asm.cycle")]
    AsmCycle,
    #[serde(rename = "cad.asm.duplicate_name")]
    AsmDuplicateName,
    #[serde(rename = "cad.asm.ambiguous_variant")]
    AsmAmbiguousVariant,
    #[serde(rename = "cad.asm.incomplete_variant")]
    AsmIncompleteVariant,
    #[serde(rename = "cad.asm.absent_in_configuration")]
    AsmAbsentInConfiguration,
    #[serde(rename = "cad.asm.unknown_axis")]
    AsmUnknownAxis,
    #[serde(rename = "cad.asm.bad_axis_value")]
    AsmBadAxisValue,
    #[serde(rename = "cad.asm.illegal_configuration")]
    AsmIllegalConfiguration,
    #[serde(rename = "cad.asm.unknown_configuration")]
    AsmUnknownConfiguration,
    #[serde(rename = "cad.asm.condition_too_complex")]
    AsmConditionTooComplex,
    #[serde(rename = "cad.asm.override_target_missing")]
    AsmOverrideTargetMissing,
    #[serde(rename = "cad.asm.override_not_allowed")]
    AsmOverrideNotAllowed,
    #[serde(rename = "cad.asm.override_shadowed")]
    AsmOverrideShadowed,
    #[serde(rename = "cad.asm.input_type")]
    AsmInputType,
    #[serde(rename = "cad.asm.unknown_occurrence")]
    AsmUnknownOccurrence,
    #[serde(rename = "cad.asm.unknown_connector")]
    AsmUnknownConnector,
    #[serde(rename = "cad.asm.anchor_unpublished")]
    AsmAnchorUnpublished,
    #[serde(rename = "cad.asm.mate_redundant")]
    AsmMateRedundant,
    #[serde(rename = "cad.asm.mate_conflict")]
    AsmMateConflict,
    #[serde(rename = "cad.asm.mate_not_converged")]
    AsmMateNotConverged,
    #[serde(rename = "cad.asm.under_defined")]
    AsmUnderDefined,
    #[serde(rename = "cad.asm.limit_active")]
    AsmLimitActive,
    #[serde(rename = "cad.asm.mate_drives_instance")]
    AsmMateDrivesInstance,
    #[serde(rename = "cad.asm.representation_unavailable")]
    AsmRepresentationUnavailable,
    #[serde(rename = "cad.asm.unlocked")]
    AsmUnlocked,
    #[serde(rename = "cad.asm.via_alias")]
    AsmViaAlias,
    #[serde(rename = "cad.asm.live_across_partition")]
    AsmLiveAcrossPartition,
    #[serde(rename = "cad.asm.too_many_children")]
    AsmTooManyChildren,
    #[serde(rename = "cad.asm.interference")]
    AsmInterference,
    #[serde(rename = "cad.asm.clearance")]
    AsmClearance,
    #[serde(rename = "cad.asm.check_failed")]
    AsmCheckFailed,
    #[serde(rename = "cad.asm.bad_pattern")]
    AsmBadPattern,
    #[serde(rename = "cad.fastener.path_missing")]
    FastenerPathMissing,
    #[serde(rename = "cad.fastener.path_too_short")]
    FastenerPathTooShort,
    #[serde(rename = "cad.fastener.exception_unmatched")]
    FastenerExceptionUnmatched,
    #[serde(rename = "cad.fastener.exception_ambiguous")]
    FastenerExceptionAmbiguous,
    #[serde(rename = "cad.fastener.duplicate_key")]
    FastenerDuplicateKey,
    #[serde(rename = "cad.fastener.no_fit")]
    FastenerNoFit,
    #[serde(rename = "cad.fastener.table_integrity")]
    FastenerTableIntegrity,
    #[serde(rename = "cad.fastener.table_schema")]
    FastenerTableSchema,
    #[serde(rename = "cad.fastener.gauge_failed")]
    FastenerGaugeFailed,
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
    Feature {
        feature: FeatureId,
    },
    Part {
        part: PartName,
    },
    Region {
        region: RegionName,
    },
    Publish {
        publish: PublishName,
    },
    Material {
        index: usize,
    },
    Value {
        name: String,
    },
    Document,
    /// An occurrence, by its display path (`wing.left/box/rib@ribs[3]`).
    Occurrence {
        path: String,
    },
    Mate {
        name: String,
    },
    Connector {
        name: String,
    },
    Pattern {
        name: String,
    },
    FastenerSet {
        name: String,
    },
    Axis {
        name: String,
    },
    Configuration {
        name: String,
    },
    Override {
        index: usize,
    },
}

/// Where in an item: a parameter path such as `edges[1]` or `constraints[3]`.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Site {
    pub subject: Subject,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub param: Option<String>,
}

impl Site {
    pub fn of(subject: Subject) -> Self {
        Site {
            subject,
            param: None,
        }
    }

    pub fn with_param(subject: Subject, param: &str) -> Self {
        Site {
            subject,
            param: Some(param.to_owned()),
        }
    }

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
            Subject::Occurrence { path } => format!("occurrence.{}", path),
            Subject::Mate { name } => format!("mate.{}", name),
            Subject::Connector { name } => format!("connector.{}", name),
            Subject::Pattern { name } => format!("pattern.{}", name),
            Subject::FastenerSet { name } => format!("fasteners.{}", name),
            Subject::Axis { name } => format!("option.{}", name),
            Subject::Configuration { name } => format!("configuration.{}", name),
            Subject::Override { index } => format!("override[{}]", index),
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
    /// Suppress a mate that over-constrains or conflicts.
    SuppressMate {
        assembly: FilePath,
        mate: String,
    },
    /// Delete an override whose target no longer exists.
    RemoveOverride {
        assembly: FilePath,
        index: usize,
    },
    /// Point an override at one of these occurrence paths instead.
    RetargetOverride {
        assembly: FilePath,
        index: usize,
        candidates: Vec<String>,
    },
    /// Narrow an occurrence's `when` so it excludes this configuration (written as axis values).
    ExcludeConfiguration {
        assembly: FilePath,
        occurrence: String,
        witness: BTreeMap<String, String>,
    },
    /// Rewrite an occurrence path written against an old name to its current name.
    RewritePath {
        site: Site,
        from: String,
        to: String,
    },
    /// Delete a fastener exception that no longer matches an instance.
    RemoveException {
        set: String,
        index: usize,
    },
    /// Add a `publish` entry to the definition so the anchor can be used.
    PublishAnchor {
        file: FilePath,
        name: PublishName,
    },
    /// Pin a live occurrence source to the latest released revision.
    PinOccurrence {
        assembly: FilePath,
        occurrence: String,
        revision: RevisionId,
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
