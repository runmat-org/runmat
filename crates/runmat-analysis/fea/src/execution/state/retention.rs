//! Retained state inventory; shared original payloads are distinct from encoded wire copies.
use super::*;
use serde::ser::{SerializeSeq, SerializeStruct};
use std::sync::Arc;

impl CanonicalSemanticState {
    /// Share only identical original histories. Each state's declaration and identity remain owned.
    /// Imported state validation precedes this operation; equality is checked in full, not by hash.
    pub fn share_original_nonlinear_payload(&mut self, original: &Self) {
        if self.source_result != original.source_result
            || self.producer != original.producer
            || self.topology_digest != original.topology_digest
        {
            return;
        }
        if let (
            CanonicalSemanticStatePayload::NonlinearSolid { state },
            CanonicalSemanticStatePayload::NonlinearSolid { state: previous },
        ) = (&mut self.payload, &original.payload)
        {
            if state.as_ref() == previous.as_ref() {
                *state = Arc::clone(previous);
            }
        }
    }
}

fn shared_payload(a: &CanonicalSemanticState, b: &CanonicalSemanticState) -> bool {
    matches!((&a.payload, &b.payload), (
        CanonicalSemanticStatePayload::NonlinearSolid { state: a },
        CanonicalSemanticStatePayload::NonlinearSolid { state: b },
    ) if Arc::ptr_eq(a, b))
}

struct Inventory<I>(I);
struct Entry<'a> {
    state: &'a CanonicalSemanticState,
    include_payload: bool,
}
impl Serialize for Entry<'_> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let s = self.state;
        let mut entry = serializer.serialize_struct("RetainedSemanticState", 12)?;
        entry.serialize_field("schema_version", &s.schema_version)?;
        entry.serialize_field("identity", &s.identity)?;
        entry.serialize_field("content_digest", &s.content_digest)?;
        entry.serialize_field("producer", &s.producer)?;
        entry.serialize_field("state_id", &s.state_id)?;
        entry.serialize_field("kind", &s.kind)?;
        entry.serialize_field("completeness", &s.completeness)?;
        entry.serialize_field("persistence", &s.persistence)?;
        entry.serialize_field("topology_digest", &s.topology_digest)?;
        entry.serialize_field("source_result", &s.source_result)?;
        entry.serialize_field("payload", &self.include_payload.then_some(&s.payload))?;
        // Charge both atomic counters even for aliases; null payload still charges its handle.
        entry.serialize_field("shared_control_bytes", &(2 * std::mem::size_of::<usize>()))?;
        entry.end()
    }
}
impl<'a, I: Iterator<Item = &'a CanonicalSemanticState> + Clone> Serialize for Inventory<I> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut sequence = serializer.serialize_seq(None)?;
        for (ordinal, state) in self.0.clone().enumerate() {
            let include_payload = !self
                .0
                .clone()
                .take(ordinal)
                .any(|prior| shared_payload(prior, state));
            sequence.serialize_element(&Entry {
                state,
                include_payload,
            })?;
        }
        sequence.end()
    }
}

/// Bound retained typed state storage at its typed resident size (states are held as typed
/// Rust values, never as a JSON value tree). Independently decoded equal payloads remain charged
/// separately until actual pointer sharing is established. This is neither wire-output admission
/// nor decoder-workspace admission, and is not measured RSS. `limits` bounds the traversal size.
pub fn semantic_state_retained_storage_bytes<'a>(
    states: impl Iterator<Item = &'a CanonicalSemanticState> + Clone,
    limits: CanonicalLimits,
) -> AnalysisExecutionResult<u64> {
    let bytes = runmat_canonical_codec::typed_resident_bytes_estimate(&Inventory(states))?;
    if bytes > limits.maximum_encoded_bytes as u64 {
        return Err(crate::execution::AnalysisExecutionError::Limit(
            "retained semantic state typed storage exceeds the admitted limit",
        ));
    }
    Ok(bytes)
}
