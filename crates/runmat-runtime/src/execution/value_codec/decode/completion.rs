//! Final decoded argument collection and its construction-root owner. Child
//! payloads and private decoder control blocks have independent admission.

use runmat_value::{OutputListStorage, OutputListStorageBuilder};

use crate::context::host_allocation::{
    HostAllocationError, HostAllocationPlan, HostAllocationReceipt,
};
use crate::context::RuntimeContext;
use crate::execution::{RootedValueChildren, RootedValueMember, RootedValues, RootedValuesBuilder};

use super::{CooperativeValueCodecError, ValueCodecError};

pub(super) struct DecodedCollection<'a> {
    values: DecodedValues,
    children: RootedValueChildren<'a>,
    receipt: Option<HostAllocationReceipt>,
    runtime: Option<&'a RuntimeContext>,
}

enum DecodedValues {
    Unbounded(RootedValuesBuilder),
    Managed(OutputListStorageBuilder),
}

impl<'a> DecodedCollection<'a> {
    pub(super) fn new(
        count: usize,
        runtime: Option<&'a RuntimeContext>,
    ) -> Result<Self, CooperativeValueCodecError> {
        let Some(runtime) = runtime else {
            return Ok(Self::unbounded(count));
        };
        let raw = OutputListStorage::requested_bytes(count)
            .map_err(|source| allocation(HostAllocationError::Numeric(source)))?;
        let raw = u64::try_from(raw).map_err(|_| allocation(HostAllocationError::Overflow))?;
        let retained =
            HostAllocationPlan::retained_exact_output_bytes(raw, usize::from(count != 0))
                .map_err(allocation)?;
        let plan = HostAllocationPlan::new(0, retained).map_err(allocation)?;
        let Some(receipt) =
            HostAllocationReceipt::admit(Some(runtime), plan).map_err(allocation)?
        else {
            return Ok(Self::unbounded(count));
        };
        let permit = receipt.output_permit(runtime).map_err(allocation)?;
        let values = OutputListStorage::try_begin(count, &permit).map_err(|source| {
            allocation(
                receipt
                    .checkpoint(runtime)
                    .err()
                    .unwrap_or(HostAllocationError::Numeric(source)),
            )
        })?;
        receipt.checkpoint(runtime).map_err(allocation)?;
        Ok(Self {
            values: DecodedValues::Managed(values),
            children: RootedValueChildren::in_context(runtime).map_err(allocation)?,
            receipt: Some(receipt),
            runtime: Some(runtime),
        })
    }

    fn unbounded(count: usize) -> Self {
        Self {
            values: DecodedValues::Unbounded(RootedValuesBuilder::with_capacity(count)),
            children: RootedValueChildren::new(),
            receipt: None,
            runtime: None,
        }
    }

    pub(super) fn push(
        &mut self,
        member: RootedValueMember,
    ) -> Result<(), CooperativeValueCodecError> {
        match &mut self.values {
            DecodedValues::Unbounded(values) => values.push_rooted(member),
            DecodedValues::Managed(values) => {
                // Keep each construction root through final root publication.
                // The admitted Value moves into its final backing exactly once.
                let value = member.into_child(&mut self.children).map_err(allocation)?;
                values
                    .try_push(value)
                    .map_err(|_| allocation(HostAllocationError::Overflow))?;
            }
        }
        Ok(())
    }

    pub(super) fn checkpoint(&self) -> Result<(), CooperativeValueCodecError> {
        if let (Some(receipt), Some(runtime)) = (&self.receipt, self.runtime) {
            receipt.checkpoint(runtime).map_err(allocation)?;
        }
        Ok(())
    }

    pub(super) fn finish(self) -> Result<RootedValues, CooperativeValueCodecError> {
        // Keep the intact carrier's field drop order on checkpoint failure:
        // admitted Values retire before their construction roots.
        self.checkpoint()?;
        let Self {
            values,
            children,
            receipt: _receipt,
            runtime,
        } = self;
        match values {
            DecodedValues::Unbounded(values) => Ok(values.finish()?),
            DecodedValues::Managed(values) => {
                let values = values
                    .try_finish()
                    .map_err(|_| allocation(HostAllocationError::Overflow))?;
                let result = RootedValues::from_output_storage_in_context(
                    values,
                    runtime.expect("managed collection retains its runtime"),
                )
                .map_err(publication)?;
                drop(children);
                Ok(result)
            }
        }
    }
}

fn allocation(source: HostAllocationError) -> CooperativeValueCodecError {
    ValueCodecError::allocation("$", source).into()
}

fn publication(error: crate::RuntimeError) -> CooperativeValueCodecError {
    let mut cursor: &(dyn std::error::Error + 'static) = &error;
    loop {
        if let Some(source) = cursor.downcast_ref::<crate::execution::ExecutionServiceError>() {
            return CooperativeValueCodecError::Execution(source.clone());
        }
        if let Some(source) = cursor.downcast_ref::<HostAllocationError>() {
            return allocation(*source);
        }
        let Some(source) = cursor.source() else { break };
        cursor = source;
    }
    CooperativeValueCodecError::Execution(
        crate::execution::ExecutionServiceError::LocalizedRuntimeFailure(error.into()),
    )
}
