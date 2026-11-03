//! Body slots across regenerations: the MinHash lineage signature stored beside each item
//! that names one body (a part output, a material assignment), and the comparison that tells,
//! on the next regeneration, whether the slot still holds the same body.
//!
//! Names decide identity. The signature is the fixed-size second opinion a stored slot keeps
//! when the names it was written against are no longer at hand: a split that tied, an edit that
//! re-made the maker feature, a merge that brought another branch's history. After each
//! regeneration [`record_slots`] writes one [`BodySlot`] per item (its body names and its
//! [`BodySignature`]); the editor keeps the table with the file's derived state. The next
//! regeneration compares ([`compare_slots`], also run by [`crate::regen::regenerate`] when
//! [`crate::regen::RegenContext::slots`] is set):
//!
//! - **kept**: the item resolves and the body's signature agrees with the stored one on at
//!   least [`SLOT_KEEP`] of [`SIGNATURE_LEN`] positions;
//! - **drifted**: the item resolves, but to a body whose lineage has moved away from the stored
//!   one. When another live body agrees with the stored signature better, by
//!   [`SLOT_MARGIN_POSITIONS`], it is offered as a repair: the slot may have moved;
//! - **lost**: the item no longer resolves. Live bodies agreeing with the stored signature on at
//!   least [`SLOT_KEEP`] positions are offered as repairs, best first;
//! - **new**: nothing was stored for the item.
//!
//! Nothing is retargeted silently: drift and loss are warnings with typed repairs.

use crate::diagnostics::{Cause, DiagCode, DiagnosticReport, Repair, Severity, Site, Subject};
use crate::ids::PartName;
use crate::model::PartDocument;
use crate::refs::naming::{BodySignature, SIGNATURE_LEN};
use crate::refs::{EntityName, Reference};
use crate::regen::engine::{ItemOutcome, RegenResult};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

/// Agreement (of [`SIGNATURE_LEN`] positions) at or above which a body still fills its slot.
pub const SLOT_KEEP: u32 = (SIGNATURE_LEN as u32) / 2;
/// How many more positions another body must agree on to be offered for a drifted slot.
pub const SLOT_MARGIN_POSITIONS: u32 = (SIGNATURE_LEN as u32) / 8;

/// The item a slot belongs to.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "slot")]
pub enum SlotKey {
    Part { part: PartName },
    Material { index: usize },
}

impl SlotKey {
    fn site(&self) -> Site {
        match self {
            SlotKey::Part { part } => Site {
                subject: Subject::Part { part: part.clone() },
                param: Some("body".into()),
            },
            SlotKey::Material { index } => Site {
                subject: Subject::Material { index: *index },
                param: Some("body".into()),
            },
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BodySlot {
    pub key: SlotKey,
    /// The body's own names when the slot was written.
    pub names: Vec<EntityName>,
    pub signature: BodySignature,
}

/// Every stored slot of one document, sorted by key.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SlotTable {
    pub slots: Vec<BodySlot>,
}

impl SlotTable {
    pub fn get(&self, key: &SlotKey) -> Option<&BodySlot> {
        self.slots
            .binary_search_by(|s| s.key.cmp(key))
            .ok()
            .map(|i| &self.slots[i])
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "status")]
pub enum SlotStatus {
    Kept {
        agreement: u32,
    },
    Drifted {
        agreement: u32,
        /// A live body that agrees better: `(body index, agreement)`.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        better: Option<(usize, u32)>,
    },
    Lost {
        /// Live bodies agreeing on at least [`SLOT_KEEP`] positions, best first.
        candidates: Vec<(usize, u32)>,
    },
    New,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SlotRow {
    pub key: SlotKey,
    /// The live body the item resolves to.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub body: Option<usize>,
    pub status: SlotStatus,
}

/// The live body an item resolved to: the body whose own names include every name resolved.
fn resolved_body(result: &RegenResult, o: &ItemOutcome) -> Option<usize> {
    if !o.ok || o.entities.len() != 1 {
        return None;
    }
    let want: BTreeSet<&EntityName> = o.entities[0].iter().collect();
    result.state.bodies.iter().position(|b| {
        !want.is_empty() && want.iter().all(|n| b.names.body.contains(*n))
    })
}

fn items<'a>(
    doc: &PartDocument,
    result: &'a RegenResult,
) -> Vec<(SlotKey, Option<&'a ItemOutcome>)> {
    let mut out: Vec<(SlotKey, Option<&ItemOutcome>)> = doc
        .parts
        .iter()
        .map(|p| {
            (
                SlotKey::Part {
                    part: p.name.clone(),
                },
                result.parts.get(&p.name),
            )
        })
        .collect();
    for (i, _) in doc.materials.iter().enumerate() {
        out.push((SlotKey::Material { index: i }, result.materials.get(i)));
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

/// One slot per item that resolved to a single body.
pub fn record_slots(doc: &PartDocument, result: &RegenResult) -> SlotTable {
    let mut slots = Vec::new();
    for (key, o) in items(doc, result) {
        let Some(b) = o.and_then(|o| resolved_body(result, o)) else {
            continue;
        };
        let body = &result.state.bodies[b];
        slots.push(BodySlot {
            key,
            names: body.names.body.iter().cloned().collect(),
            signature: body.names.signature.clone(),
        });
    }
    SlotTable { slots }
}

/// Compare stored slots with a regeneration. Returns one row per item and the warnings.
pub fn compare_slots(
    stored: &SlotTable,
    doc: &PartDocument,
    result: &RegenResult,
) -> (Vec<SlotRow>, DiagnosticReport) {
    let mut rows = Vec::new();
    let mut diags = DiagnosticReport::default();
    let body_ref = |b: usize| -> Option<Reference> {
        result.state.bodies[b]
            .names
            .body
            .iter()
            .next()
            .cloned()
            .map(Reference::body)
    };
    for (key, o) in items(doc, result) {
        let body = o.and_then(|o| resolved_body(result, o));
        let Some(slot) = stored.get(&key) else {
            rows.push(SlotRow {
                key,
                body,
                status: SlotStatus::New,
            });
            continue;
        };
        let mut ranked: Vec<(usize, u32)> = result
            .state
            .bodies
            .iter()
            .enumerate()
            .map(|(i, b)| (i, b.names.signature.agreement(&slot.signature)))
            .collect();
        ranked.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        let status = match body {
            Some(b) => {
                let agreement = result.state.bodies[b]
                    .names
                    .signature
                    .agreement(&slot.signature);
                if agreement >= SLOT_KEEP {
                    SlotStatus::Kept { agreement }
                } else {
                    let better = ranked
                        .iter()
                        .find(|(i, a)| *i != b && *a >= agreement + SLOT_MARGIN_POSITIONS)
                        .copied();
                    let mut repairs = Vec::new();
                    if let Some(r) = better.and_then(|(i, _)| body_ref(i)) {
                        repairs.push(Repair::RewriteReference {
                            site: key.site(),
                            reference: r,
                        });
                    }
                    diags.push(
                        DiagCode::RefBodySlotDrift,
                        Severity::Warning,
                        key.site(),
                        format!(
                            "the body this item names has drifted from the one it was written \
                             against: its lineage agrees on {} of {} signature positions{}",
                            agreement,
                            SIGNATURE_LEN,
                            match better {
                                Some((_, a)) => format!(", another body agrees on {}", a),
                                None => String::new(),
                            }
                        ),
                        Cause::Root,
                        repairs,
                    );
                    SlotStatus::Drifted { agreement, better }
                }
            }
            None => {
                let candidates: Vec<(usize, u32)> = ranked
                    .iter()
                    .filter(|(_, a)| *a >= SLOT_KEEP)
                    .copied()
                    .collect();
                if !candidates.is_empty() {
                    let repairs = candidates
                        .iter()
                        .filter_map(|(i, _)| body_ref(*i))
                        .map(|r| Repair::RewriteReference {
                            site: key.site(),
                            reference: r,
                        })
                        .collect();
                    diags.push(
                        DiagCode::RefBodySlotDrift,
                        Severity::Warning,
                        key.site(),
                        format!(
                            "the item no longer resolves; {} live bod{} carry its lineage (best \
                             agreement {} of {})",
                            candidates.len(),
                            if candidates.len() == 1 { "y" } else { "ies" },
                            candidates[0].1,
                            SIGNATURE_LEN
                        ),
                        Cause::Root,
                        repairs,
                    );
                }
                SlotStatus::Lost { candidates }
            }
        };
        rows.push(SlotRow { key, body, status });
    }
    (rows, diags)
}
