//! Typed source-output identity for a placement graph edge.

use super::{CandidateOutputResidency, PlacementGraphCandidate, ResourceByteBound};
use serde::{Deserialize, Serialize};

const MAX_GRAPH_OUTPUT_ROLES: usize = 256;

/// Zero-based public output selected as the source of a graph edge.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SourceOutputOrdinal(u32);

impl SourceOutputOrdinal {
    /// Construct an ordinal from an already selected public-output position.
    pub const fn new(ordinal: u32) -> Self {
        Self(ordinal)
    }

    /// Return the zero-based position for bounds checking against output roles.
    pub const fn index(self) -> u32 {
        self.0
    }
}

pub(super) fn output_arity(
    candidate: &PlacementGraphCandidate,
) -> Result<Option<usize>, &'static str> {
    let Some(outputs) = candidate.output_facts.as_ref() else {
        if matches!(
            candidate.descriptor.output_residency,
            CandidateOutputResidency::ProviderAndHost { .. }
        ) {
            // Only the request's single-node, edge-free whole-work envelope
            // may admit this unresolved aggregate; graph validation owns it.
            return Ok(None);
        }
        return Ok(Some(1));
    };
    if outputs.is_empty() || outputs.len() > MAX_GRAPH_OUTPUT_ROLES {
        return Err("placement output role count is invalid");
    }
    match candidate.descriptor.output_residency {
        CandidateOutputResidency::ProviderAndHost {
            device_id,
            host_bytes,
        } => {
            if !outputs
                .iter()
                .any(|fact| matches!(fact.residency, CandidateOutputResidency::Host))
                || !outputs.iter().any(|role| {
                    matches!(role.residency, CandidateOutputResidency::Provider { device_id: owner } if owner == device_id)
                })
                || outputs.iter().any(|role| {
                    !matches!(role.residency, CandidateOutputResidency::Host)
                        && !matches!(role.residency, CandidateOutputResidency::Provider { device_id: owner } if owner == device_id)
                })
            {
                return Err("mixed placement output roles do not match their provider owner");
            }
            if sum_bounds(
                outputs
                    .iter()
                    .filter(|fact| matches!(fact.residency, CandidateOutputResidency::Host))
                    .map(|fact| fact.retained_bytes),
            ) != Some(host_bytes)
                || sum_bounds(
                    outputs
                        .iter()
                        .filter(|fact| {
                            matches!(fact.residency, CandidateOutputResidency::Provider { .. })
                        })
                        .map(|fact| fact.retained_bytes),
                ) != Some(candidate.resources.retained_bytes)
            {
                return Err("mixed placement output byte bounds do not match aggregate resources");
            }
        }
        CandidateOutputResidency::Unknown
        | CandidateOutputResidency::ConditionalHostOrProvider { .. } => {
            return Err("placement output roles require exact residency");
        }
        aggregate => {
            if outputs.iter().any(|fact| fact.residency != aggregate) {
                return Err("placement output roles do not match candidate residency");
            }
            if sum_bounds(outputs.iter().map(|fact| fact.retained_bytes))
                != Some(candidate.resources.retained_bytes)
            {
                return Err("placement output byte bounds do not match aggregate resources");
            }
        }
    }
    Ok(Some(outputs.len()))
}

impl PlacementGraphCandidate {
    /// Resolve a previously validated graph edge to this candidate's public
    /// output role. Returns None for an unvalidated or out-of-range ordinal.
    pub fn source_output_residency(
        &self,
        ordinal: SourceOutputOrdinal,
    ) -> Option<CandidateOutputResidency> {
        match self.output_facts.as_ref() {
            Some(outputs) => outputs
                .get(ordinal.index() as usize)
                .map(|fact| fact.residency),
            None if ordinal.index() == 0 => Some(self.descriptor.output_residency),
            None => None,
        }
    }
}

fn sum_bounds(bounds: impl Iterator<Item = ResourceByteBound>) -> Option<ResourceByteBound> {
    let mut known = 0u64;
    let mut unknown = None;
    for bound in bounds {
        match bound {
            ResourceByteBound::Known(bytes) => known = known.checked_add(bytes)?,
            ResourceByteBound::Unavailable(reason) => {
                if unknown.replace(reason).is_some() {
                    return None;
                }
            }
        }
    }
    match unknown {
        Some(reason) if known == 0 => Some(ResourceByteBound::Unavailable(reason)),
        Some(_) => None,
        None => Some(ResourceByteBound::Known(known)),
    }
}
