//! MATLAB sample-leading arrays to canonical sample-major, row-major tensor values.
//! Shape authentication belongs to the caller; this conversion checks its own cardinality.
use crate::signal_materialization::NumericValues;
use crate::{AnalysisBridgeError, AnalysisBridgeResult};

/// Required in frozen signal admission; absent/unknown old layouts are rejected.
#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CanonicalGeneratorValueLayout {
    SampleMajorRowMajor,
}

pub(crate) fn canonical_sample_layout(
    values: NumericValues,
    samples: usize,
    extents: &[usize],
) -> AnalysisBridgeResult<NumericValues> {
    if samples == 0 || extents.contains(&0) {
        return Err(invalid("generator layout has an empty axis"));
    }
    let width = extents.iter().try_fold(1_usize, |size, extent| {
        size.checked_mul(*extent)
            .ok_or_else(|| invalid("generator tensor cardinality overflows"))
    })?;
    let total = samples
        .checked_mul(width)
        .ok_or_else(|| invalid("generator sample cardinality overflows"))?;
    // Only the final value buffer is allocated, not a second index/order inventory.
    // MATLAB's first tail axis varies fastest. Canonical tensors' last axis does.
    let source_index = |canonical: usize| {
        let sample = canonical / width;
        let mut tail = canonical % width;
        let mut column = 0;
        let mut stride = width;
        for extent in extents.iter().rev() {
            stride /= extent;
            column += (tail % extent) * stride;
            tail /= extent;
        }
        sample + samples * column
    };
    match values {
        NumericValues::Real(values) if values.len() == total => Ok(NumericValues::Real(
            (0..total)
                .map(|index| values[source_index(index)])
                .collect(),
        )),
        NumericValues::Complex(values) if values.len() == total => Ok(NumericValues::Complex(
            (0..total)
                .map(|index| values[source_index(index)])
                .collect(),
        )),
        _ => Err(invalid("generator value count differs from its layout")),
    }
}

fn invalid(message: &str) -> AnalysisBridgeError {
    AnalysisBridgeError::Invalid(message.into())
}

#[cfg(test)]
mod tests;
