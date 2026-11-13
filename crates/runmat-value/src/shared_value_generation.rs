//! One immutable physical Value generation, independent of GC root policy.
//!
//! A GC activation traces the contained Value as an ordinary child. An
//! external workspace or read carrier pairs this owner with its own root
//! lease; storing a root lease in this payload would make self-referential
//! closure activations uncollectible.

use runmat_execution::resource::ExactShared;

use crate::{
    ExactArrayClaimPromotionError, ExactHostAllocationClaim, ExactHostAllocationPermit,
    ExactHostNumericAllocationError, Value,
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
    CapturePromotion(ExactArrayClaimPromotionError),
}

impl std::fmt::Display for CallableGenerationPrepareError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AlreadyConsumedOrReserved => {
                formatter.write_str("callable input was already consumed or reserved")
            }
            Self::NotCallable => formatter.write_str("input is not a callable value"),
            Self::Allocation(error) => error.fmt(formatter),
            Self::CapturePromotion(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for CallableGenerationPrepareError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Allocation(error) => Some(error),
            Self::CapturePromotion(error) => Some(error),
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

    /// Quote the generation header/name metadata and any unclaimed physical
    /// capture owner separately. The capture claim attaches to its shared
    /// backing, surviving generation retirement while aliases remain.
    /// Captured Values, lexical environments, and bound-authority graphs
    /// retain their own independent allocation and root policies.
    pub fn callable_quote(
        value: &Value,
    ) -> Result<Option<CallableGenerationQuote>, ExactHostNumericAllocationError> {
        let mut requested = Self::requested_bytes();
        let mut retained_claims = 1usize;
        match value {
            Value::ScopedFunctionHandle(_) => {
                return Err(ExactHostNumericAllocationError::UnsupportedClass)
            }
            Value::FunctionHandle(handle)
            | Value::ExternalFunctionHandle(handle)
            | Value::MethodFunctionHandle(handle) => {
                if let Some(bytes) = handle.name_unclaimed_bytes()? {
                    requested = requested
                        .checked_add(bytes)
                        .ok_or(ExactHostNumericAllocationError::LayoutOverflow)?;
                    retained_claims += 1;
                }
            }
            Value::BoundFunctionHandle(handle) => {
                let (bytes, claims) = handle.metadata_quote()?;
                requested = requested
                    .checked_add(bytes)
                    .ok_or(ExactHostNumericAllocationError::LayoutOverflow)?;
                retained_claims += claims;
            }
            Value::Closure(closure) => {
                let (bytes, claims) = closure.metadata_quote()?;
                requested = requested
                    .checked_add(bytes)
                    .ok_or(ExactHostNumericAllocationError::LayoutOverflow)?;
                retained_claims += claims;
                let capture_bytes =
                    closure
                        .captures
                        .unclaimed_shared_bytes()
                        .map_err(|error| match error {
                            ExactArrayClaimPromotionError::Allocation(error) => error,
                            ExactArrayClaimPromotionError::UnsharedBacking => {
                                unreachable!("constructed closure captures have shared backing")
                            }
                        })?;
                if let Some(capture_bytes) = capture_bytes {
                    requested = requested
                        .checked_add(capture_bytes)
                        .ok_or(ExactHostNumericAllocationError::LayoutOverflow)?;
                    retained_claims += 1;
                }
            }
            _ => return Ok(None),
        }
        Ok(Some(CallableGenerationQuote {
            requested_bytes: requested,
            retained_claims,
        }))
    }

    /// Promote a Closure's original capture owner, then move callable
    /// metadata into one exact shared generation. A failed input retains its
    /// Value and any capture claim; no captured child allocation is certified.
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
        let _quote = Self::callable_quote(value)
            .map_err(CallableGenerationPrepareError::Allocation)?
            .ok_or(CallableGenerationPrepareError::NotCallable)?;
        if let Value::FunctionHandle(handle)
        | Value::ExternalFunctionHandle(handle)
        | Value::MethodFunctionHandle(handle) = value
        {
            handle
                .try_promote_name_claim(permit)
                .map_err(CallableGenerationPrepareError::Allocation)?;
        }
        if let Value::Closure(closure) = value {
            closure
                .try_promote_metadata(permit)
                .map_err(CallableGenerationPrepareError::Allocation)?;
            closure
                .captures
                .try_promote_shared_claim(permit)
                .map_err(CallableGenerationPrepareError::CapturePromotion)?;
        }
        if let Value::BoundFunctionHandle(handle) = value {
            handle
                .try_promote_metadata(permit)
                .map_err(CallableGenerationPrepareError::Allocation)?;
        }
        let generation_bytes = Self::callable_quote(value)
            .map_err(CallableGenerationPrepareError::Allocation)?
            .expect("quoted callable")
            .requested_bytes;
        let claim = permit
            .reserve_exact_claim(generation_bytes)
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
