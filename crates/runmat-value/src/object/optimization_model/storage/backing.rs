use super::*;
use crate::ExactHostNumericAllocationError;

pub(in crate::object) trait Backing {
    fn identity(&self) -> Option<SharedStorageIdentity>;
    fn quote(&self) -> Result<Option<usize>, ExactArrayClaimPromotionError>;
    fn promote(
        &self,
        permit: &dyn ExactHostAllocationPermit,
    ) -> Result<(), ExactArrayClaimPromotionError>;
}
pub(in crate::object) struct MetadataRef<'a>(Option<&'a dyn Backing>);
impl<'a> MetadataRef<'a> {
    pub(in crate::object) fn new<T: Backing>(value: &'a T) -> Self {
        Self(Some(value))
    }
    pub(super) fn empty() -> Self {
        Self(None)
    }
    pub(super) fn quote(&self) -> Result<OptimizationMetadataQuote, ExactArrayClaimPromotionError> {
        match self.0 {
            Some(value) => Ok(OptimizationMetadataQuote {
                owner: value.identity(),
                bytes: value.quote()?,
            }),
            None => Ok(OptimizationMetadataQuote {
                owner: None,
                bytes: None,
            }),
        }
    }
    pub(super) fn promote(
        &self,
        permit: &dyn ExactHostAllocationPermit,
    ) -> Result<(), ExactArrayClaimPromotionError> {
        self.0.map_or(Ok(()), |value| value.promote(permit))
    }
}
impl<T> Backing for ExactArrayStorage<T> {
    fn identity(&self) -> Option<SharedStorageIdentity> {
        self.shared_storage_identity()
    }
    fn quote(&self) -> Result<Option<usize>, ExactArrayClaimPromotionError> {
        self.unclaimed_shared_bytes()
    }
    fn promote(
        &self,
        permit: &dyn ExactHostAllocationPermit,
    ) -> Result<(), ExactArrayClaimPromotionError> {
        self.try_promote_shared_claim(permit)
    }
}
impl Backing for SharedShape {
    fn identity(&self) -> Option<SharedStorageIdentity> {
        Some(self.shared_storage_identity())
    }
    fn quote(&self) -> Result<Option<usize>, ExactArrayClaimPromotionError> {
        self.unclaimed_shared_bytes()
            .map_err(ExactArrayClaimPromotionError::Allocation)
    }
    fn promote(
        &self,
        permit: &dyn ExactHostAllocationPermit,
    ) -> Result<(), ExactArrayClaimPromotionError> {
        self.try_promote_shared_claim(permit)
            .map_err(ExactArrayClaimPromotionError::Allocation)
    }
}
impl Backing for Utf8Text {
    fn identity(&self) -> Option<SharedStorageIdentity> {
        self.shared_storage_identity()
    }
    fn quote(&self) -> Result<Option<usize>, ExactArrayClaimPromotionError> {
        self.unclaimed_bytes()
            .map_err(ExactArrayClaimPromotionError::Allocation)
    }
    fn promote(
        &self,
        permit: &dyn ExactHostAllocationPermit,
    ) -> Result<(), ExactArrayClaimPromotionError> {
        self.clone()
            .try_admit_exact(permit)
            .map(|_| ())
            .map_err(|error| ExactArrayClaimPromotionError::Allocation(error.source()))
    }
}

pub(super) fn invalid() -> ExactArrayClaimPromotionError {
    ExactArrayClaimPromotionError::Allocation(ExactHostNumericAllocationError::UnsupportedClass)
}

impl Backing for crate::SharedClassIdentity {
    fn identity(&self) -> Option<SharedStorageIdentity> {
        Some(self.shared_storage_identity())
    }
    fn quote(&self) -> Result<Option<usize>, ExactArrayClaimPromotionError> {
        if self.has_retained_claim() {
            Ok(None)
        } else {
            self.requested_bytes()
                .map(Some)
                .map_err(ExactArrayClaimPromotionError::Allocation)
        }
    }
    fn promote(
        &self,
        permit: &dyn ExactHostAllocationPermit,
    ) -> Result<(), ExactArrayClaimPromotionError> {
        self.clone()
            .try_admit_exact(permit)
            .map(|_| ())
            .map_err(|source| ExactArrayClaimPromotionError::Allocation(source.source()))
    }
}
