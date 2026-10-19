//! Final decoded argument collection only. Child payloads, task stacks and
//! construction-root metadata have separate, still-open admission owners.

use runmat_value::{OutputListStorage, OutputListStorageBuilder};

use crate::context::host_allocation::{
    HostAllocationError, HostAllocationPlan, HostAllocationReceipt,
};
use crate::context::RuntimeContext;
use crate::execution::{RootedValueChildren, RootedValueMember, RootedValues, RootedValuesBuilder};

use super::{CooperativeValueCodecError, ValueCodecError};

pub(super) enum DecodedCollection<'a> {
    Unbounded(RootedValuesBuilder),
    Managed {
        values: OutputListStorageBuilder,
        children: RootedValueChildren,
        receipt: HostAllocationReceipt,
        runtime: &'a RuntimeContext,
    },
}

impl<'a> DecodedCollection<'a> {
    pub(super) fn new(
        count: usize,
        runtime: Option<&'a RuntimeContext>,
    ) -> Result<Self, CooperativeValueCodecError> {
        let Some(runtime) = runtime else {
            return Ok(Self::Unbounded(RootedValuesBuilder::with_capacity(count)));
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
            return Ok(Self::Unbounded(RootedValuesBuilder::with_capacity(count)));
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
        Ok(Self::Managed {
            values,
            children: RootedValueChildren::new(),
            receipt,
            runtime,
        })
    }

    pub(super) fn push(
        &mut self,
        member: RootedValueMember,
    ) -> Result<(), CooperativeValueCodecError> {
        match self {
            Self::Unbounded(values) => values.push_rooted(member),
            Self::Managed {
                values, children, ..
            } => {
                // Keep each construction root through final root publication.
                // The admitted Value moves into its final backing exactly once.
                let value = member.into_child(children);
                values
                    .try_push(value)
                    .map_err(|_| allocation(HostAllocationError::Overflow))?;
            }
        }
        Ok(())
    }

    pub(super) fn checkpoint(&self) -> Result<(), CooperativeValueCodecError> {
        if let Self::Managed {
            receipt, runtime, ..
        } = self
        {
            receipt.checkpoint(runtime).map_err(allocation)?;
        }
        Ok(())
    }

    pub(super) fn finish(self) -> Result<RootedValues, CooperativeValueCodecError> {
        // Keep the intact carrier's field drop order on checkpoint failure:
        // admitted Values retire before their construction roots.
        self.checkpoint()?;
        match self {
            Self::Unbounded(values) => Ok(values.finish()?),
            Self::Managed {
                values,
                children,
                receipt: _receipt,
                runtime,
            } => {
                let values = values
                    .try_finish()
                    .map_err(|_| allocation(HostAllocationError::Overflow))?;
                let result = RootedValues::from_output_storage_in_context(values, runtime)
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
