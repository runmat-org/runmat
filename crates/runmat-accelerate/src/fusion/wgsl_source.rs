//! Allocation-free byte geometry for the exact shared WGSL source writer.

use std::fmt;

const WORD_HELPER_MARKER: &[u8] = b"fn runmat_f64_from_words(";

/// The scanner starts at the generated body, excluding any semantic preamble,
/// just like the established word-helper injection decision.
pub(super) struct CountedWgslSource {
    bytes: usize,
    marker_prefix: usize,
    has_word_helper: bool,
}

/// A source fragment whose bytes have already been checked by its own writer.
/// Formatting it into the shared shader writer measures the surrounding shell
/// without constructing a synthetic body or binding string.
pub(super) struct MeasuredFragment(pub(super) usize);

/// Count and emit the same ordered shader sections without copying their text.
#[derive(Clone, Copy)]
pub(super) struct WgslShaderFragments<Bindings, Body, FinalWrites> {
    pub(super) bindings: Bindings,
    pub(super) body: Body,
    pub(super) final_writes: FinalWrites,
}

/// A body admitted by the counted producer writer cannot grow past its
/// reserved capacity, even if a declaration changes between count and emit.
pub(super) struct BoundedWgslBody {
    source: String,
    limit: Option<usize>,
}

impl BoundedWgslBody {
    pub(super) fn new(limit: Option<usize>) -> Result<Self, std::collections::TryReserveError> {
        let mut source = String::new();
        if let Some(bytes) = limit {
            source.try_reserve_exact(bytes)?;
        }
        Ok(Self { source, limit })
    }

    pub(super) fn from_source(source: String, limit: Option<usize>) -> Result<Self, fmt::Error> {
        if limit.is_some_and(|bytes| source.len() > bytes) {
            return Err(fmt::Error);
        }
        Ok(Self { source, limit })
    }

    pub(super) fn into_source(self) -> String {
        self.source
    }
}

impl fmt::Write for BoundedWgslBody {
    fn write_str(&mut self, text: &str) -> fmt::Result {
        if self.limit.is_some_and(|limit| {
            self.source
                .len()
                .checked_add(text.len())
                .is_none_or(|next| next > limit)
        }) {
            return Err(fmt::Error);
        }
        fmt::Write::write_str(&mut self.source, text)
    }
}

impl fmt::Display for MeasuredFragment {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        const FILL: &str = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx";
        let mut remaining = self.0;
        while remaining >= FILL.len() {
            formatter.write_str(FILL)?;
            remaining -= FILL.len();
        }
        formatter.write_str(&FILL[..remaining])
    }
}

impl CountedWgslSource {
    pub(super) fn with_preamble_bytes(bytes: usize) -> Self {
        Self {
            bytes,
            marker_prefix: 0,
            has_word_helper: false,
        }
    }

    pub(super) fn bytes(&self) -> usize {
        self.bytes
    }

    pub(super) fn has_word_helper(&self) -> bool {
        self.has_word_helper
    }
}

impl fmt::Write for CountedWgslSource {
    fn write_str(&mut self, text: &str) -> fmt::Result {
        self.bytes = self.bytes.checked_add(text.len()).ok_or(fmt::Error)?;
        for byte in text.bytes() {
            if byte == WORD_HELPER_MARKER[self.marker_prefix] {
                self.marker_prefix += 1;
                if self.marker_prefix == WORD_HELPER_MARKER.len() {
                    self.has_word_helper = true;
                    self.marker_prefix = 0;
                }
            } else {
                // The marker has only a one-byte proper prefix that appears
                // inside itself (`f`), so mismatch recovery needs no table.
                self.marker_prefix = usize::from(byte == WORD_HELPER_MARKER[0]);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fmt::Write;

    #[test]
    fn marker_across_writer_fragments_matches_generated_only_source() {
        let mut source = CountedWgslSource::with_preamble_bytes(9);
        source.write_str("fn runmat_f64_").unwrap();
        source.write_str("from_words(").unwrap();
        assert!(source.has_word_helper());
        assert_eq!(source.bytes(), 9 + WORD_HELPER_MARKER.len());
        let preamble_only = CountedWgslSource::with_preamble_bytes(WORD_HELPER_MARKER.len());
        assert!(!preamble_only.has_word_helper());
    }
}
