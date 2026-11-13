//! Canonical serde projection without retaining a complete JSON value tree.
//! Sequences retain encoded bytes; maps retain only sorted keys and encoded values.
use crate::{
    checkpoint, encoding_error, CanonicalCodecControl, CanonicalCodecError, CanonicalLimits,
};
use serde::Serialize;
mod compound;
mod probe;
mod serializer;
#[cfg(test)]
mod tests;
use serializer::Serializer;

struct State<'a> {
    domain: &'a str,
    limits: CanonicalLimits,
    control: &'a dyn CanonicalCodecControl,
    wire_bytes: usize,
    stream: Option<Stream<'a>>,
}
/// Streamed output: final bytes of the root buffer leave for the sink once they can no longer
/// be rewritten. Only an open declared-length map writing into the root buffer can rewrite its
/// own bytes (when it falls back to sorted holding), so its header pins everything after it.
struct Stream<'a> {
    root: *const Vec<u8>,
    /// Root-buffer bytes already handed to the sink; offsets recorded by open maps are absolute.
    drained: usize,
    /// Absolute header offsets of open streaming maps in the root buffer, innermost last.
    pinned: Vec<usize>,
    sink: &'a mut dyn FnMut(&[u8]),
}
/// Root-buffer bytes retained before they are handed to the sink. Tests drain after a few
/// bytes so every encoder path runs with drained, pinned and rewritten root offsets.
#[cfg(not(test))]
const STREAM_CHUNK_BYTES: usize = 1 << 20;
#[cfg(test)]
const STREAM_CHUNK_BYTES: usize = 7;
impl State<'_> {
    fn check(&self) -> Result<(), CanonicalCodecError> {
        checkpoint(self.control)
    }
    /// Absolute position of the end of `output` (root offsets include drained bytes).
    fn position(&self, output: &Vec<u8>) -> usize {
        match &self.stream {
            Some(stream) if std::ptr::eq(output, stream.root) => stream.drained + output.len(),
            _ => output.len(),
        }
    }
    /// Index in `output` of an absolute position recorded by `position`.
    fn index(&self, output: &Vec<u8>, position: usize) -> usize {
        match &self.stream {
            Some(stream) if std::ptr::eq(output, stream.root) => position - stream.drained,
            _ => position,
        }
    }
    /// Pin a streaming map header at `position` when it is written into the root buffer.
    fn pin(&mut self, output: &Vec<u8>, position: usize) -> bool {
        match &mut self.stream {
            Some(stream) if std::ptr::eq(output, stream.root) => {
                stream.pinned.push(position);
                true
            }
            _ => false,
        }
    }
    fn unpin(&mut self, position: usize) {
        if let Some(stream) = &mut self.stream {
            debug_assert_eq!(stream.pinned.last(), Some(&position));
            stream.pinned.pop();
        }
    }
    /// Hand final root-buffer bytes to the sink once a chunk has accumulated.
    fn flush(&mut self, output: &mut Vec<u8>, force: bool) {
        let Some(stream) = &mut self.stream else {
            return;
        };
        if !std::ptr::eq(output, stream.root) || (!force && output.len() < STREAM_CHUNK_BYTES) {
            return;
        }
        let end = stream
            .pinned
            .first()
            .map_or(output.len(), |position| position - stream.drained);
        if end == 0 {
            return;
        }
        (stream.sink)(&output[..end]);
        output.drain(..end);
        stream.drained += end;
    }
    fn charge(&mut self, bytes: usize) -> Result<(), CanonicalCodecError> {
        let next = self.wire_bytes.checked_add(bytes).ok_or_else(|| {
            CanonicalCodecError::resource_limit(
                "canonical encoding",
                format!("contract `{}` encoded byte count overflow", self.domain),
            )
        })?;
        if next > self.limits.maximum_encoded_bytes {
            return Err(CanonicalCodecError::resource_limit(
                "canonical encoding",
                format!(
                    "contract `{}` requires at least {next} bytes, exceeding its {}-byte limit",
                    self.domain, self.limits.maximum_encoded_bytes,
                ),
            ));
        }
        self.wire_bytes = next;
        Ok(())
    }
}
fn invalid(reason: &'static str) -> CanonicalCodecError {
    CanonicalCodecError::invalid("canonical encoding", reason)
}

pub(crate) fn encode<T: Serialize>(
    prefix: &[u8],
    domain: &str,
    value: &T,
    limits: CanonicalLimits,
    control: &dyn CanonicalCodecControl,
) -> Result<Vec<u8>, CanonicalCodecError> {
    // Reserve the profiled output once so it never regrows; the encoding workspace profile
    // assumes this. Shape limits are the encoder's own; a profile it cannot produce leaves
    // the output to grow as before.
    let capacity = crate::profile_canonical_encoding_with_control(
        value,
        CanonicalLimits::new(
            limits.maximum_encoded_bytes,
            usize::MAX,
            usize::MAX,
            usize::MAX,
        ),
        control,
    )
    .ok()
    .and_then(|profile| usize::try_from(profile.encoded_bytes_upper_bound).ok())
    .filter(|capacity| *capacity <= limits.maximum_encoded_bytes);
    encode_with_capacity(prefix, domain, value, limits, control, capacity)
}

pub(crate) fn encode_with_capacity<T: Serialize>(
    prefix: &[u8],
    domain: &str,
    value: &T,
    limits: CanonicalLimits,
    control: &dyn CanonicalCodecControl,
    capacity: Option<usize>,
) -> Result<Vec<u8>, CanonicalCodecError> {
    checkpoint(control)?;
    if capacity.is_some_and(|capacity| capacity > limits.maximum_encoded_bytes) {
        return Err(CanonicalCodecError::resource_limit(
            "canonical encoding",
            "requested output capacity exceeds the codec bound",
        ));
    }
    let mut output = Vec::new();
    if let Some(capacity) = capacity {
        output.try_reserve_exact(capacity).map_err(|error| {
            CanonicalCodecError::resource_limit("canonical encoding", error.to_string())
        })?;
    }
    output.extend_from_slice(prefix);
    minicbor::Encoder::new(&mut output)
        .array(2)
        .and_then(|encoder| encoder.str(domain))
        .map_err(encoding_error)?;
    let mut state = State {
        domain,
        limits,
        control,
        wire_bytes: 0,
        stream: None,
    };
    state.charge(output.len())?;
    value.serialize(Serializer::child(&mut output, &mut state, value))?;
    checkpoint(control)?;
    debug_assert_eq!(state.wire_bytes, output.len());
    Ok(output)
}

/// Same bytes as `encode`, handed to `sink` in order as they become final. Only struct fields
/// that arrive before a smaller key and maps that leave strictly increasing key order are held,
/// exactly as `encode` holds them; nothing else of the encoding is retained.
pub(crate) fn stream<T: Serialize>(
    prefix: &[u8],
    domain: &str,
    value: &T,
    limits: CanonicalLimits,
    control: &dyn CanonicalCodecControl,
    sink: &mut dyn FnMut(&[u8]),
) -> Result<u64, CanonicalCodecError> {
    checkpoint(control)?;
    let mut output = Vec::new();
    output.extend_from_slice(prefix);
    minicbor::Encoder::new(&mut output)
        .array(2)
        .and_then(|encoder| encoder.str(domain))
        .map_err(encoding_error)?;
    let mut state = State {
        domain,
        limits,
        control,
        wire_bytes: 0,
        stream: Some(Stream {
            root: &output,
            drained: 0,
            pinned: Vec::new(),
            sink,
        }),
    };
    state.charge(output.len())?;
    value.serialize(Serializer::child(&mut output, &mut state, value))?;
    checkpoint(control)?;
    state.flush(&mut output, true);
    debug_assert!(output.is_empty());
    u64::try_from(state.wire_bytes).map_err(|_| invalid("encoded byte count overflow"))
}
