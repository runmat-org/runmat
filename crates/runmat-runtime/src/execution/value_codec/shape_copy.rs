//! Convert wire dimensions directly into their final shared shape owner.
use super::ValueCodecError;
use crate::context::{
    host_allocation::{HostAllocationError, HostAllocationPlan, HostAllocationReceipt},
    RuntimeContext,
};
use runmat_value::{SharedShape, SharedShapeBuilder};

enum Output {
    Exact(SharedShapeBuilder),
    Ordinary(Vec<usize>),
}
pub(super) struct ShapeCopy<'a> {
    output: Output,
    source: &'a [u64],
    offset: usize,
    receipt: Option<HostAllocationReceipt>,
    runtime: Option<&'a RuntimeContext>,
}
impl<'a> ShapeCopy<'a> {
    pub(super) fn new(
        source: &'a [u64],
        runtime: Option<&'a RuntimeContext>,
        path: &str,
    ) -> Result<Self, ValueCodecError> {
        let allocation = |e| ValueCodecError::allocation(path, e);
        let raw = SharedShape::requested_bytes_for_rank(source.len())
            .map_err(|e| allocation(e.into()))?;
        let retained = HostAllocationPlan::retained_exact_output_bytes(
            u64::try_from(raw).map_err(|_| allocation(HostAllocationError::Overflow))?,
            1,
        )
        .map_err(allocation)?;
        let receipt = HostAllocationReceipt::admit(
            runtime,
            HostAllocationPlan::new(0, retained).map_err(allocation)?,
        )
        .map_err(allocation)?;
        let output = if let Some(receipt) = &receipt {
            let runtime = runtime.expect("managed shape receipt has context");
            let permit = receipt.output_permit(runtime).map_err(allocation)?;
            let output = SharedShape::begin_exact(source.len(), &permit);
            receipt.checkpoint(runtime).map_err(allocation)?;
            Output::Exact(output.map_err(|e| allocation(e.into()))?)
        } else {
            Output::Ordinary(Vec::with_capacity(source.len()))
        };
        Ok(Self {
            output,
            source,
            offset: 0,
            receipt,
            runtime,
        })
    }
    fn checkpoint(&self, path: &str) -> Result<(), ValueCodecError> {
        if let Some(receipt) = &self.receipt {
            receipt
                .checkpoint(self.runtime.expect("managed shape context"))
                .map_err(|e| ValueCodecError::allocation(path, e))?;
        }
        if self.runtime.is_some_and(RuntimeContext::is_cancelled) {
            return Err(ValueCodecError::allocation(
                path,
                HostAllocationError::Cancelled,
            ));
        }
        Ok(())
    }
    pub(super) fn advance(&mut self, work: usize, path: &str) -> Result<usize, ValueCodecError> {
        self.checkpoint(path)?;
        let end = self.source.len().min(self.offset.saturating_add(work));
        let count = end - self.offset;
        for &dimension in &self.source[self.offset..end] {
            let dimension = usize::try_from(dimension).map_err(|_| {
                ValueCodecError::invalid(path, "shape dimension exceeds platform limits")
            })?;
            match &mut self.output {
                Output::Exact(output) => output
                    .try_push(dimension)
                    .expect("one reserved slot per dimension"),
                Output::Ordinary(output) => output.push(dimension),
            }
        }
        self.offset = end;
        self.checkpoint(path)?;
        Ok(count.max(1))
    }
    pub(super) fn is_complete(&self) -> bool {
        self.offset == self.source.len()
    }
    pub(super) fn finish(self, path: &str) -> Result<SharedShape, ValueCodecError> {
        self.checkpoint(path)?;
        if !self.is_complete() {
            return Err(ValueCodecError::invalid(
                path,
                "shape conversion is incomplete",
            ));
        }
        let shape = match self.output {
            Output::Exact(output) => output.try_finish().expect("completed shape"),
            Output::Ordinary(output) => SharedShape::new_unmanaged(output),
        };
        if let Some(receipt) = &self.receipt {
            receipt
                .checkpoint(self.runtime.expect("managed shape context"))
                .map_err(|e| ValueCodecError::allocation(path, e))?;
        }
        Ok(shape)
    }
}
