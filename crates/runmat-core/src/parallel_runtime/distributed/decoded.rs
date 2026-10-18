//! Decoded partitions retain their values through asynchronous assembly.

use runmat_execution_runner::OwnedPartition;
use runmat_gc::{Trace, Tracer};
use runmat_runtime::execution::{value_codec, RootedResult, RootedValuesBuilder};
use runmat_runtime::parallel::distribution::DistributedPartitionValue;
use runmat_runtime::RuntimeError;

pub(super) struct DecodedPartitions(pub(super) Vec<DistributedPartitionValue>);

impl Trace for DecodedPartitions {
    fn trace(&self, tracer: &mut dyn Tracer) {
        for part in &self.0 {
            part.value.trace(tracer);
        }
    }
}

pub(super) fn decode_parts(
    parts: Vec<OwnedPartition>,
) -> Result<RootedResult<DecodedPartitions>, RuntimeError> {
    let (layouts, payloads): (Vec<_>, Vec<_>) = parts
        .into_iter()
        .map(|part| (part.layout, part.value))
        .unzip();
    // Preserve the store's existing per-partition limits. A combined codec
    // budget would reject valid large distributed values whose shards each fit.
    let mut values = RootedValuesBuilder::with_capacity(payloads.len());
    for payload in payloads {
        values.push_owned(value_codec::decode_inline_value_owned(&payload).map_err(super::error)?);
    }
    let values = values
        .finish()
        .map_err(|error| error.into_runtime_error())?;
    values.with_unbounded_values(|values| {
        let partitions = layouts
            .into_iter()
            .zip(values)
            .map(|(layout, value)| DistributedPartitionValue { layout, value })
            .collect();
        RootedResult::new(DecodedPartitions(partitions)).map_err(|error| error.into_runtime_error())
    })?
}
