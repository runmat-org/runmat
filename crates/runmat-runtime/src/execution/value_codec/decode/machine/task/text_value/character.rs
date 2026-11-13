//! Incremental portable character decoding into final exact native backing.

use super::super::super::*;
use crate::context::host_allocation::{
    HostAllocationError, HostAllocationPlan, HostAllocationPolicy, HostAllocationReceipt,
};
use crate::context::RuntimeContext;
use runmat_value::{
    CharArrayAdmission, ExactArrayStorage, ExactArrayStorageBuilder, SharedShape,
    SharedShapeBuilder, Utf16CodeUnit,
};

pub(in crate::execution::value_codec::decode::machine) struct ManagedCharacter<'a> {
    // Physical builders precede the stage receipt in destruction order.
    shape: Option<SharedShapeBuilder>,
    final_shape: Option<SharedShape>,
    output: Option<ExactArrayStorageBuilder<Utf16CodeUnit>>,
    admission: Option<CharArrayAdmission>,
    source_shape: &'a [u64],
    source: &'a [u16],
    offset: usize,
    receipt: HostAllocationReceipt,
    runtime: &'a RuntimeContext,
}

impl<'a> ManagedCharacter<'a> {
    pub(super) fn new(
        source_shape: &'a [u64],
        source: &'a [u16],
        runtime: Option<&'a RuntimeContext>,
        path: &str,
    ) -> Result<Option<Self>, ValueCodecError> {
        if HostAllocationReceipt::policy(runtime)
            .map_err(|error| ValueCodecError::allocation(path, error))?
            == HostAllocationPolicy::Unbounded
        {
            return Ok(None);
        }
        let payload =
            ExactArrayStorage::<Utf16CodeUnit>::requested_bytes(source.len()).map_err(|error| {
                ValueCodecError::allocation(path, HostAllocationError::Numeric(error))
            })?;
        let shape = SharedShape::requested_bytes_for_rank(source_shape.len()).map_err(|error| {
            ValueCodecError::allocation(path, HostAllocationError::Numeric(error))
        })?;
        let bytes = payload
            .checked_add(shape)
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or_else(|| ValueCodecError::allocation(path, HostAllocationError::Overflow))?;
        let retained = HostAllocationPlan::retained_exact_output_bytes(
            bytes,
            usize::from(!source.is_empty()) + 1,
        )
        .map_err(|error| ValueCodecError::allocation(path, error))?;
        let plan = HostAllocationPlan::new(0, retained)
            .map_err(|error| ValueCodecError::allocation(path, error))?;
        let receipt = HostAllocationReceipt::admit(runtime, plan)
            .map_err(|error| ValueCodecError::allocation(path, error))?
            .ok_or_else(|| {
                ValueCodecError::allocation(path, HostAllocationError::StaleAuthority)
            })?;
        Ok(Some(Self {
            shape: None,
            final_shape: None,
            output: None,
            admission: None,
            source_shape,
            source,
            offset: 0,
            receipt,
            runtime: runtime.expect("managed receipt has a selected runtime"),
        }))
    }

    pub(super) fn advance(
        &mut self,
        maximum_work: usize,
        path: &str,
    ) -> Result<(usize, Option<Value>), ValueCodecError> {
        self.checkpoint(path)?;
        if let Some(admission) = self.admission.as_mut() {
            let completed = admission
                .advance(std::num::NonZeroUsize::new(maximum_work).expect("decode work is nonzero"))
                .map_err(|error| ValueCodecError::invalid(path, error))?;
            let complete = admission.is_complete();
            self.checkpoint(path)?;
            if !complete {
                return Ok((completed.max(1), None));
            }
            let value = self
                .admission
                .take()
                .expect("completed character admission")
                .into_array()
                .unwrap_or_else(|_| unreachable!("completed character geometry"));
            return Ok((completed.max(1), Some(Value::CharArray(value))));
        }
        if self.final_shape.is_none() {
            if self.shape.is_none() {
                let permit = self
                    .receipt
                    .output_permit(self.runtime)
                    .map_err(|error| ValueCodecError::allocation(path, error))?;
                self.shape = Some(
                    SharedShape::begin_exact(self.source_shape.len(), &permit)
                        .map_err(|error| self.allocation_error(path, error))?,
                );
                self.checkpoint(path)?;
            }
            let shape = self.shape.as_mut().expect("active final shape builder");
            let start = shape.len();
            let end = self
                .source_shape
                .len()
                .min(start.saturating_add(maximum_work));
            for &extent in &self.source_shape[start..end] {
                let extent = usize::try_from(extent)
                    .map_err(|_| ValueCodecError::invalid(path, "value exceeds platform limits"))?;
                shape.try_push(extent).map_err(|_| {
                    ValueCodecError::invalid(path, "character shape exceeds admitted rank")
                })?;
            }
            if end == self.source_shape.len() {
                self.final_shape = Some(
                    self.shape
                        .take()
                        .expect("complete final shape builder")
                        .try_finish()
                        .unwrap_or_else(|_| unreachable!("complete shape prefix")),
                );
            }
            self.checkpoint(path)?;
            return Ok(((end - start).max(1), None));
        }
        if self.output.is_none() {
            let permit = self
                .receipt
                .output_permit(self.runtime)
                .map_err(|error| ValueCodecError::allocation(path, error))?;
            self.output = Some(
                ExactArrayStorage::try_begin(self.source.len(), &permit)
                    .map_err(|error| self.allocation_error(path, error))?,
            );
            self.checkpoint(path)?;
        }
        let end = self
            .source
            .len()
            .min(self.offset.saturating_add(maximum_work));
        let output = self
            .output
            .as_mut()
            .expect("active final character builder");
        for &unit in &self.source[self.offset..end] {
            output
                .try_push(Utf16CodeUnit::from_raw(unit))
                .map_err(|_| {
                    ValueCodecError::invalid(path, "character payload exceeds admitted size")
                })?;
        }
        let completed = (end - self.offset).max(1);
        self.offset = end;
        self.checkpoint(path)?;
        if end != self.source.len() {
            return Ok((completed, None));
        }
        let output = self
            .output
            .take()
            .expect("complete final character builder")
            .try_finish()
            .unwrap_or_else(|_| unreachable!("complete character prefix"));
        self.admission = Some(
            CharArrayAdmission::new(
                output,
                self.final_shape.take().expect("completed final shape"),
            )
            .map_err(|error| ValueCodecError::invalid(path, error))?,
        );
        self.checkpoint(path)?;
        Ok((completed, None))
    }

    fn checkpoint(&self, path: &str) -> Result<(), ValueCodecError> {
        self.receipt
            .checkpoint(self.runtime)
            .map_err(|error| ValueCodecError::allocation(path, error))
    }

    fn allocation_error(
        &self,
        path: &str,
        source: runmat_value::ExactHostNumericAllocationError,
    ) -> ValueCodecError {
        ValueCodecError::allocation(
            path,
            self.receipt
                .checkpoint(self.runtime)
                .err()
                .unwrap_or(HostAllocationError::Numeric(source)),
        )
    }
}

#[cfg(test)]
#[path = "character/tests.rs"]
mod tests;
