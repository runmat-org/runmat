//! Original Value-domain execution metadata snapshots. Wire contracts remain
//! unchanged; retaining a Value never clones nested facts or result shapes.

use crate::immutable_metadata::ImmutableMetadata;
use crate::{ExactHostAllocationPermit, ExactHostNumericAllocationError};

mod quote;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExecutionMetadataQuoteError {
    Allocation(ExactHostNumericAllocationError),
    /// std tree nodes have no public exact physical allocation geometry.
    OpaqueFactTree,
    CallableIdentity,
    FactDepth,
}
impl From<ExactHostNumericAllocationError> for ExecutionMetadataQuoteError {
    fn from(error: ExactHostNumericAllocationError) -> Self {
        Self::Allocation(error)
    }
}
impl std::fmt::Display for ExecutionMetadataQuoteError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Allocation(error) => error.fmt(f),
            Self::OpaqueFactTree => f.write_str(
                "execution metadata contains tree allocations without an exact physical owner",
            ),
            Self::CallableIdentity => {
                f.write_str("callable fact identity requires its original admitted metadata owner")
            }
            Self::FactDepth => {
                f.write_str("execution metadata fact depth exceeds bounded quote traversal")
            }
        }
    }
}
impl std::error::Error for ExecutionMetadataQuoteError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Allocation(error) => Some(error),
            _ => None,
        }
    }
}

macro_rules! snapshot {
    ($name:ident, $wire:ty, $quote:path, $normalize:path) => {
        #[derive(Debug, Clone)]
        pub struct $name(ImmutableMetadata<$wire>);
        impl $name {
            pub fn new(mut value: $wire) -> Self {
                $normalize(&mut value);
                Self(ImmutableMetadata::new(value))
            }
            pub fn shares_allocation_with(&self, other: &Self) -> bool {
                self.0.ptr_eq(&other.0)
            }
            pub fn has_retained_metadata_claim(&self) -> bool {
                self.0.has_claim()
            }
            pub fn metadata_unclaimed_bytes(
                &self,
            ) -> Result<Option<usize>, ExecutionMetadataQuoteError> {
                if self.0.has_claim() {
                    return Ok(None);
                }
                let bytes = $quote(self.as_ref())?;
                Ok(self.0.unclaimed_bytes(|_| Ok(bytes))?)
            }
            /// Covers only original metadata, not service authority, results
            /// or GC roots. Refusal leaves the backing and all aliases intact.
            pub fn try_promote_metadata(
                &self,
                permit: &dyn ExactHostAllocationPermit,
            ) -> Result<(), ExecutionMetadataQuoteError> {
                if self.0.has_claim() {
                    return Ok(());
                }
                let bytes = $quote(self.as_ref())?;
                Ok(self.0.promote(permit, |_| Ok(bytes))?)
            }
            /// Explicit ordinary wire extraction, never a finite retain path.
            pub fn into_wire_unbounded(self) -> $wire {
                self.0.into_inner()
            }
        }
        impl From<$wire> for $name {
            fn from(value: $wire) -> Self {
                Self::new(value)
            }
        }
        impl From<Box<$wire>> for $name {
            fn from(value: Box<$wire>) -> Self {
                Self::new(*value)
            }
        }
        impl AsRef<$wire> for $name {
            fn as_ref(&self) -> &$wire {
                self.0.get()
            }
        }
        impl std::ops::Deref for $name {
            type Target = $wire;
            fn deref(&self) -> &Self::Target {
                self.as_ref()
            }
        }
        impl PartialEq for $name {
            fn eq(&self, other: &Self) -> bool {
                self.as_ref() == other.as_ref()
            }
        }
        impl Eq for $name {}
    };
}

snapshot!(
    SharedFutureHandle,
    runmat_execution::FutureHandle,
    quote::future,
    quote::unchanged
);
snapshot!(
    SharedTaskHandle,
    runmat_execution::TaskHandle,
    quote::task,
    quote::unchanged
);
snapshot!(
    SharedJobHandle,
    runmat_execution::JobHandle,
    quote::job,
    quote::unchanged
);
snapshot!(
    SharedDistributedHandle,
    runmat_execution::DistributedValueHandle,
    quote::distributed,
    quote::normalize_distributed
);
snapshot!(
    SharedCompositeHandle,
    runmat_execution::CompositeHandle,
    quote::composite,
    quote::normalize_composite
);
