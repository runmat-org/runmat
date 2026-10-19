//! Retained class-access identity used by poll-local dispatch scopes.

use crate::context::host_allocation::{
    HostAllocationError, HostAllocationPlan, HostAllocationReceipt, HostOutputPermit,
};
use crate::context::RuntimeContext;
use runmat_execution::resource::ExactShared;
use runmat_types::ClassIdentity;
use runmat_value::{
    exact_capacity_vec, ExactHostAllocationClaim, ExactHostAllocationPermit,
    ExactHostNumericAllocationError,
};
use std::fmt;
use std::ops::Deref;

pub struct ClassAccessIdentity(IdentityOwner);

enum IdentityOwner {
    Unbounded(ClassIdentity),
    Managed(ExactShared<ManagedIdentity>),
}

struct ManagedIdentity {
    identity: ClassIdentity,
    _text_claim: ExactHostAllocationClaim,
    _header_claim: ExactHostAllocationClaim,
}

impl ClassAccessIdentity {
    pub(crate) fn retained_quote_for(
        identity: &ClassIdentity,
    ) -> Result<(u64, usize), HostAllocationError> {
        let bytes = identity
            .display_name()
            .len()
            .checked_add(ExactShared::<ManagedIdentity>::requested_bytes())
            .ok_or(HostAllocationError::Overflow)?;
        Ok((
            u64::try_from(bytes).map_err(|_| HostAllocationError::Overflow)?,
            2,
        ))
    }

    pub(crate) fn retained_quote(&self) -> Result<(u64, usize), HostAllocationError> {
        match &self.0 {
            IdentityOwner::Managed(_) => Ok((0, 0)),
            IdentityOwner::Unbounded(identity) => Self::retained_quote_for(identity),
        }
    }

    pub(crate) fn admit(self, context: &RuntimeContext) -> Result<Self, HostAllocationError> {
        let policy = HostAllocationReceipt::policy(Some(context))?;
        if matches!(&self.0, IdentityOwner::Managed(_)) {
            return Ok(self);
        }
        if policy == crate::context::host_allocation::HostAllocationPolicy::Unbounded {
            return Ok(self);
        }
        let (bytes, claims) = self.retained_quote()?;
        let retained = HostAllocationPlan::retained_exact_output_bytes(bytes, claims)?;
        let receipt =
            HostAllocationReceipt::admit(Some(context), HostAllocationPlan::new(0, retained)?)?
                .ok_or(HostAllocationError::MissingManagedOwner)?;
        let result = receipt
            .output_permit(context)
            .and_then(|permit| self.admit_with_permit(permit));
        receipt.checkpoint(context)?;
        result
    }

    pub(crate) fn admit_with_permit(
        self,
        permit: HostOutputPermit,
    ) -> Result<Self, HostAllocationError> {
        let identity = match self.0 {
            IdentityOwner::Unbounded(identity) => identity,
            IdentityOwner::Managed(owner) => {
                return Ok(Self(IdentityOwner::Managed(owner)));
            }
        };
        Self::managed_from_identity(&identity, permit)
    }

    pub(crate) fn admitted_clone_with_permit(
        &self,
        permit: HostOutputPermit,
    ) -> Result<Self, HostAllocationError> {
        match &self.0 {
            IdentityOwner::Managed(owner) => Ok(Self(IdentityOwner::Managed(owner.clone()))),
            IdentityOwner::Unbounded(identity) => Self::managed_from_identity(identity, permit),
        }
    }

    pub(crate) fn admitted_from_identity(
        identity: &ClassIdentity,
        permit: HostOutputPermit,
    ) -> Result<Self, HostAllocationError> {
        Self::managed_from_identity(identity, permit)
    }

    fn managed_from_identity(
        identity: &ClassIdentity,
        permit: HostOutputPermit,
    ) -> Result<Self, HostAllocationError> {
        let bytes = identity.display_name().as_bytes();
        let text_claim = permit
            .reserve_exact_claim(bytes.len())
            .map_err(HostAllocationError::Numeric)?;
        let header_claim = permit
            .reserve_exact_claim(ExactShared::<ManagedIdentity>::requested_bytes())
            .map_err(HostAllocationError::Numeric)?;
        let mut owned = exact_capacity_vec(bytes.len()).map_err(HostAllocationError::Numeric)?;
        owned.extend_from_slice(bytes);
        let name = String::from_utf8(owned).map_err(|_| {
            HostAllocationError::Numeric(ExactHostNumericAllocationError::AllocationFailed)
        })?;
        let identity = ClassIdentity::new(name).map_err(|_| {
            HostAllocationError::Numeric(ExactHostNumericAllocationError::AllocationFailed)
        })?;
        let owner = ExactShared::try_new(ManagedIdentity {
            identity,
            _text_claim: text_claim,
            _header_claim: header_claim,
        })
        .map_err(|_| {
            HostAllocationError::Numeric(ExactHostNumericAllocationError::AllocationFailed)
        })?;
        Ok(Self(IdentityOwner::Managed(owner)))
    }
}

impl Clone for ClassAccessIdentity {
    fn clone(&self) -> Self {
        Self(match &self.0 {
            IdentityOwner::Unbounded(identity) => IdentityOwner::Unbounded(identity.clone()),
            IdentityOwner::Managed(owner) => IdentityOwner::Managed(owner.clone()),
        })
    }
}

impl From<ClassIdentity> for ClassAccessIdentity {
    fn from(identity: ClassIdentity) -> Self {
        Self(IdentityOwner::Unbounded(identity))
    }
}

impl Deref for ClassAccessIdentity {
    type Target = ClassIdentity;

    fn deref(&self) -> &Self::Target {
        match &self.0 {
            IdentityOwner::Unbounded(identity) => identity,
            IdentityOwner::Managed(owner) => &owner.get().identity,
        }
    }
}

impl fmt::Debug for ClassAccessIdentity {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.deref().fmt(formatter)
    }
}

impl fmt::Display for ClassAccessIdentity {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.deref().fmt(formatter)
    }
}

impl AsRef<str> for ClassAccessIdentity {
    fn as_ref(&self) -> &str {
        self.display_name()
    }
}

impl PartialEq for ClassAccessIdentity {
    fn eq(&self, other: &Self) -> bool {
        self.deref() == other.deref()
    }
}

impl PartialEq<ClassIdentity> for ClassAccessIdentity {
    fn eq(&self, other: &ClassIdentity) -> bool {
        self.deref() == other
    }
}

impl Eq for ClassAccessIdentity {}
