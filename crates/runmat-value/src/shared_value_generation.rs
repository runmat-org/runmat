//! One immutable physical Value generation, independent of GC root policy.
//!
//! A GC activation traces the contained Value as an ordinary child. An
//! external workspace or read carrier pairs this owner with its own root
//! lease; storing a root lease in this payload would make self-referential
//! closure activations uncollectible.

use runmat_execution::resource::ExactShared;
use std::alloc::Layout;

use crate::{
    ExactHostAllocationClaim, ExactHostAllocationPermit, ExactHostNumericAllocationError, Value,
};

#[derive(Debug)]
struct SharedValuePayload {
    value: Value,
    // The Value and shared header retire before this physical claim refunds.
    _claim: Option<ExactHostAllocationClaim>,
}

#[derive(Clone, Debug)]
pub struct SharedValueGeneration(ExactShared<SharedValuePayload>);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CallableGenerationQuote {
    pub requested_bytes: usize,
    pub retained_claims: usize,
}

/// Caller-owned preparation state keeps failed backing alive without a
/// second allocation for an error carrier. On success both owners move into
/// the shared generation; on failure this input must outlive its root lease.
#[derive(Debug)]
pub struct CallableGenerationInput {
    // Field order preserves the physical backing until before claim refund.
    value: Option<Value>,
    claim: Option<ExactHostAllocationClaim>,
}

impl CallableGenerationInput {
    pub fn new(value: Value) -> Self {
        Self {
            value: Some(value),
            claim: None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CallableGenerationPrepareError {
    AlreadyConsumedOrReserved,
    NotCallable,
    Allocation(ExactHostNumericAllocationError),
}

impl std::fmt::Display for CallableGenerationPrepareError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AlreadyConsumedOrReserved => {
                formatter.write_str("callable input was already consumed or reserved")
            }
            Self::NotCallable => formatter.write_str("input is not a callable value"),
            Self::Allocation(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for CallableGenerationPrepareError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Allocation(error) => Some(error),
            Self::AlreadyConsumedOrReserved | Self::NotCallable => None,
        }
    }
}

impl SharedValueGeneration {
    pub const fn requested_bytes() -> usize {
        ExactShared::<SharedValuePayload>::requested_bytes()
    }

    pub fn prepare(
        value: Value,
        permit: &dyn ExactHostAllocationPermit,
    ) -> Result<Self, ExactHostNumericAllocationError> {
        let claim = permit.reserve_exact_claim(Self::requested_bytes())?;
        ExactShared::try_new(SharedValuePayload {
            value,
            _claim: Some(claim),
        })
        .map(Self)
        .map_err(|_| ExactHostNumericAllocationError::AllocationFailed)
    }

    /// Quote the physical backing moved into one immutable callable
    /// generation. This includes the shared generation header and the
    /// callable's own strings and capture-list backing. Captured Values,
    /// lexical environments, and bound-authority graphs retain their own
    /// independent allocation and root policies.
    pub fn callable_quote(
        value: &Value,
    ) -> Result<Option<CallableGenerationQuote>, ExactHostNumericAllocationError> {
        let mut requested = Self::requested_bytes();
        let mut add_capacity = |capacity: usize, element: usize| {
            let bytes = capacity
                .checked_mul(element)
                .ok_or(ExactHostNumericAllocationError::LayoutOverflow)?;
            requested = requested
                .checked_add(bytes)
                .ok_or(ExactHostNumericAllocationError::LayoutOverflow)?;
            Ok::<_, ExactHostNumericAllocationError>(())
        };
        match value {
            Value::FunctionHandle(handle)
            | Value::ExternalFunctionHandle(handle)
            | Value::MethodFunctionHandle(handle) => {
                if !handle.name_has_retained_claim() {
                    add_capacity(handle.name_capacity(), 1)?;
                }
            }
            Value::BoundFunctionHandle(handle) => {
                add_capacity(handle.name.capacity(), 1)?;
                if let Some(source_file) = &handle.source_file {
                    add_capacity(source_file.capacity(), 1)?;
                }
            }
            Value::Closure(closure) => {
                add_capacity(closure.function_name.capacity(), 1)?;
                if let Some(source_file) = &closure.source_file {
                    add_capacity(source_file.len(), 1)?;
                }
                let captures = Layout::array::<Value>(closure.captures.capacity())
                    .map_err(|_| ExactHostNumericAllocationError::LayoutOverflow)?
                    .size();
                requested = requested
                    .checked_add(captures)
                    .ok_or(ExactHostNumericAllocationError::LayoutOverflow)?;
            }
            _ => return Ok(None),
        }
        Ok(Some(CallableGenerationQuote {
            requested_bytes: requested,
            retained_claims: 1,
        }))
    }

    /// Move a callable and its quoted shallow backing into one exact shared
    /// generation. A failed input retains its Value and any issued claim;
    /// no error allocation or nested child allocation is certified here.
    pub fn prepare_callable(
        input: &mut CallableGenerationInput,
        permit: &dyn ExactHostAllocationPermit,
    ) -> Result<Self, CallableGenerationPrepareError> {
        if input.claim.is_some() {
            return Err(CallableGenerationPrepareError::AlreadyConsumedOrReserved);
        }
        let value = input
            .value
            .as_ref()
            .ok_or(CallableGenerationPrepareError::AlreadyConsumedOrReserved)?;
        let quote = Self::callable_quote(value)
            .map_err(CallableGenerationPrepareError::Allocation)?
            .ok_or(CallableGenerationPrepareError::NotCallable)?;
        let claim = permit
            .reserve_exact_claim(quote.requested_bytes)
            .map_err(CallableGenerationPrepareError::Allocation)?;
        let value = input.value.take().expect("quoted callable input");
        ExactShared::try_new(SharedValuePayload {
            value,
            _claim: Some(claim),
        })
        .map(Self)
        .map_err(|payload| {
            input.value = Some(payload.value);
            input.claim = payload._claim;
            CallableGenerationPrepareError::Allocation(
                ExactHostNumericAllocationError::AllocationFailed,
            )
        })
    }

    pub fn unbounded(value: Value) -> Self {
        Self(ExactShared::new(SharedValuePayload {
            value,
            _claim: None,
        }))
    }

    pub fn value(&self) -> &Value {
        &self.0.get().value
    }

    pub fn has_retained_claim(&self) -> bool {
        self.0.get()._claim.is_some()
    }

    pub fn ptr_eq(&self, other: &Self) -> bool {
        self.0.ptr_eq(&other.0)
    }
}
