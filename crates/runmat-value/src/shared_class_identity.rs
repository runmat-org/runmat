//! One immutable class spelling owner shared by every Value-domain alias.

use runmat_execution::resource::ExactShared;
use runmat_types::{ClassIdentity, StaticClassIdentity};
use std::sync::OnceLock;

use crate::{ExactHostAllocationClaim, ExactHostAllocationPermit, ExactHostNumericAllocationError};

#[derive(Debug)]
struct SharedClassIdentityPayload {
    identity: ClassIdentity,
    // Drops after the spelling. ExactShared retires its physical header before
    // it drops the payload, so this one claim covers both original allocations.
    claim: OnceLock<ExactHostAllocationClaim>,
}

#[derive(Debug, Clone)]
pub struct SharedClassIdentity(ExactShared<SharedClassIdentityPayload>);

#[derive(Debug)]
pub struct SharedClassIdentityAdmissionError {
    identity: SharedClassIdentity,
    source: ExactHostNumericAllocationError,
}

impl SharedClassIdentityAdmissionError {
    pub fn source(&self) -> ExactHostNumericAllocationError {
        self.source
    }

    pub fn try_into_unclaimed_identity(self) -> Result<SharedClassIdentity, Self> {
        if self.identity.has_retained_claim() {
            Err(self)
        } else {
            Ok(self.identity)
        }
    }
}

impl SharedClassIdentity {
    pub fn shared_storage_identity(&self) -> crate::SharedStorageIdentity {
        crate::SharedStorageIdentity::of(self.0.get())
    }
    pub fn new(identity: ClassIdentity) -> Self {
        Self(ExactShared::new(SharedClassIdentityPayload {
            identity,
            claim: OnceLock::new(),
        }))
    }

    pub fn as_identity(&self) -> &ClassIdentity {
        &self.0.get().identity
    }

    pub fn display_name(&self) -> &str {
        self.as_identity().display_name()
    }

    pub fn has_retained_claim(&self) -> bool {
        self.0.get().claim.get().is_some()
    }

    pub fn shares_allocation_with(&self, other: &Self) -> bool {
        self.0.ptr_eq(&other.0)
    }

    pub fn requested_bytes_for_identity(
        identity: &ClassIdentity,
    ) -> Result<usize, ExactHostNumericAllocationError> {
        Self::requested_bytes_for_capacity(identity.allocation_capacity())
    }

    pub fn requested_bytes_for_capacity(
        capacity: usize,
    ) -> Result<usize, ExactHostNumericAllocationError> {
        capacity
            .checked_add(ExactShared::<SharedClassIdentityPayload>::requested_bytes())
            .ok_or(ExactHostNumericAllocationError::LayoutOverflow)
    }

    pub fn requested_bytes(&self) -> Result<usize, ExactHostNumericAllocationError> {
        Self::requested_bytes_for_identity(self.as_identity())
    }

    /// Attach one physical claim to the original spelling and shared header.
    /// Every alias observes the same claim; refusal returns this owner intact.
    pub fn try_admit_exact(
        self,
        permit: &dyn ExactHostAllocationPermit,
    ) -> Result<Self, SharedClassIdentityAdmissionError> {
        if self.has_retained_claim() {
            return Ok(self);
        }
        let bytes = match self.requested_bytes() {
            Ok(bytes) => bytes,
            Err(source) => {
                return Err(SharedClassIdentityAdmissionError {
                    identity: self,
                    source,
                })
            }
        };
        let claim = match permit.reserve_exact_claim(bytes) {
            Ok(claim) => claim,
            Err(_source) if self.has_retained_claim() => return Ok(self),
            Err(source) => {
                return Err(SharedClassIdentityAdmissionError {
                    identity: self,
                    source,
                })
            }
        };
        let _ = self.0.get().claim.set(claim);
        Ok(self)
    }

    pub(crate) fn try_new_claimed(
        identity: ClassIdentity,
        claim: ExactHostAllocationClaim,
    ) -> Result<Self, ExactHostNumericAllocationError> {
        let claims = OnceLock::new();
        let _ = claims.set(claim);
        ExactShared::try_new(SharedClassIdentityPayload {
            identity,
            claim: claims,
        })
        .map(Self)
        .map_err(|_| ExactHostNumericAllocationError::AllocationFailed)
    }
}

impl std::ops::Deref for SharedClassIdentity {
    type Target = ClassIdentity;
    fn deref(&self) -> &Self::Target {
        self.as_identity()
    }
}

impl std::fmt::Display for SharedClassIdentity {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.as_identity().fmt(formatter)
    }
}

impl PartialEq for SharedClassIdentity {
    fn eq(&self, other: &Self) -> bool {
        self.as_identity() == other.as_identity()
    }
}
impl Eq for SharedClassIdentity {}
impl PartialEq<ClassIdentity> for SharedClassIdentity {
    fn eq(&self, other: &ClassIdentity) -> bool {
        self.as_identity() == other
    }
}
impl PartialEq<SharedClassIdentity> for ClassIdentity {
    fn eq(&self, other: &SharedClassIdentity) -> bool {
        self == other.as_identity()
    }
}
impl PartialEq<StaticClassIdentity> for SharedClassIdentity {
    fn eq(&self, other: &StaticClassIdentity) -> bool {
        self.as_identity() == other
    }
}
impl PartialEq<SharedClassIdentity> for StaticClassIdentity {
    fn eq(&self, other: &SharedClassIdentity) -> bool {
        other == self
    }
}
impl AsRef<str> for SharedClassIdentity {
    fn as_ref(&self) -> &str {
        self.display_name()
    }
}
impl std::borrow::Borrow<str> for SharedClassIdentity {
    fn borrow(&self) -> &str {
        self.display_name()
    }
}

impl From<ClassIdentity> for SharedClassIdentity {
    fn from(identity: ClassIdentity) -> Self {
        Self::new(identity)
    }
}
impl From<String> for SharedClassIdentity {
    fn from(identity: String) -> Self {
        Self::new(identity.into())
    }
}
impl From<&str> for SharedClassIdentity {
    fn from(identity: &str) -> Self {
        Self::new(identity.into())
    }
}
impl From<StaticClassIdentity> for SharedClassIdentity {
    fn from(identity: StaticClassIdentity) -> Self {
        Self::new(identity.into())
    }
}
