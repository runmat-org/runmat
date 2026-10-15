//! Checked numerical inputs/results; no renderer precision in this contract.
use runmat_accelerate_api::{ProviderExecutionError, ProviderExecutionResult};

pub struct ExactHistogramEdges {
    words: Vec<[u32; 2]>,
    bins: u32,
}
impl ExactHistogramEdges {
    pub fn new(edges: &[f64]) -> ProviderExecutionResult<Self> {
        if edges.len() < 2 || !edges.windows(2).all(|pair| pair[0] < pair[1]) {
            return Err(ProviderExecutionError::from_source(
                HistogramError::InvalidEdges,
            ));
        }
        let bins = u32::try_from(edges.len() - 1)
            .map_err(|_| ProviderExecutionError::from_source(HistogramError::BinRange))?;
        let words = edges
            .iter()
            .map(|value| {
                let bits = value.to_bits();
                [bits as u32, (bits >> 32) as u32]
            })
            .collect();
        Ok(Self { words, bins })
    }
    pub fn bin_count(&self) -> usize {
        self.bins as usize
    }
    pub(crate) fn bin_count_u32(&self) -> u32 {
        self.bins
    }
    pub(crate) fn storage_bytes(&self) -> Vec<u8> {
        self.words
            .iter()
            .flat_map(|words| words.iter().flat_map(|word| word.to_le_bytes()))
            .collect()
    }
}

pub struct ExactHistogramCounts {
    counts: Vec<u32>,
}
impl ExactHistogramCounts {
    pub(super) fn admit(counts: Vec<u32>, samples: u32) -> ProviderExecutionResult<Self> {
        let total = counts.iter().map(|count| u64::from(*count)).sum::<u64>();
        if total != u64::from(samples) {
            return Err(ProviderExecutionError::from_source(
                HistogramError::CountCoverage,
            ));
        }
        Ok(Self { counts })
    }
    /// Every u32 count widens exactly; there is no f32 numerical intermediate.
    pub fn into_double_counts(self) -> Vec<f64> {
        self.counts.into_iter().map(f64::from).collect()
    }
}

#[derive(Debug, thiserror::Error)]
pub(super) enum HistogramError {
    #[error("hist: sample count exceeds u32")]
    SampleRange,
    #[error("hist: count byte size overflow")]
    CountByteOverflow,
    #[error("hist: numerical buffers exceed actual device limits")]
    DeviceBufferLimit,
    #[error("hist: source byte size overflow")]
    SourceByteOverflow,
    #[error("hist: invalid or unbindable numerical sample extent")]
    SourceExtent,
    #[error("hist: binary64 source word index exceeds u32")]
    SourceWordRange,
    #[error("hist: dispatch stride overflow")]
    DispatchOverflow,
    #[error("hist: no legal histogram workgroups")]
    NoWorkgroups,
    #[error("hist: numerical edges must be strictly increasing")]
    InvalidEdges,
    #[error("hist: bin count exceeds u32")]
    BinRange,
    #[error("hist: numerical count evidence does not cover every sample exactly once")]
    CountCoverage,
}
