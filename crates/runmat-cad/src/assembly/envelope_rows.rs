//! The top assembly's envelopes as Outline rows (`envelope:<name>`): the box, its size, whether it
//! keeps components out, its owner, and the values Envelope starts from on the row.

use crate::assembly::model::AssemblyDocument;
use crate::assembly::resolve::ResolvedAssembly;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EnvelopeRow {
    pub name: String,
    /// Interference checks list components inside it (other than its owner).
    pub keep_out: bool,
    /// The one component allowed inside, by its path.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub owner: Option<String>,
    /// The box in the assembly's coordinates (m).
    pub min: [f64; 3],
    pub max: [f64; 3],
    /// What is wrong with it: its box doesn't evaluate, or its owner is gone.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub problem: Option<String>,
}

/// One row per envelope the top assembly declares, in document order.
pub fn envelope_rows(tree: &ResolvedAssembly, top: Option<&AssemblyDocument>) -> Vec<EnvelopeRow> {
    let Some(doc) = top else {
        return Vec::new();
    };
    doc.envelopes
        .iter()
        .map(|e| {
            let resolved = tree
                .envelopes
                .iter()
                .find(|r| r.declared_by == 0 && r.name == e.name);
            let problem = match (resolved, &e.owner) {
                (None, _) => Some("its box doesn't evaluate".to_owned()),
                (Some(r), Some(o)) if r.occupant.is_none() => {
                    Some(format!("its owner `{o}` isn't in the assembly"))
                }
                _ => None,
            };
            let (min, max) = resolved.map_or(([0.0; 3], [0.0; 3]), |r| (r.world.min, r.world.max));
            EnvelopeRow {
                name: e.name.clone(),
                keep_out: e.keep_out,
                owner: e.owner.clone(),
                min,
                max,
                problem,
            }
        })
        .collect()
}
