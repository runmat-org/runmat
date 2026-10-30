//! RunMat-owned shader/control backing, charged before allocation.
//! This does not account Naga, WGPU, allocator-private or driver-private storage.
use runmat_execution::resource::{
    ProviderHostMetadataAdmission, ProviderHostMetadataError, RetainedHostAllocationClaim,
};
#[cfg(any(feature = "wgpu", test))]
use std::ops::Range;
use std::{alloc::Layout, fmt, sync::Arc};

pub(crate) type Claim = Option<Arc<dyn RetainedHostAllocationClaim>>;

#[derive(Debug)]
pub(crate) enum Error {
    Overflow,
    Metadata(ProviderHostMetadataError),
    Allocation(runmat_value::ExactHostNumericAllocationError),
    Formatting(fmt::Error),
}
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Overflow => f.write_str("shader control buffer layout overflow"),
            Self::Metadata(error) => fmt::Display::fmt(error, f),
            Self::Allocation(error) => fmt::Display::fmt(error, f),
            Self::Formatting(error) => fmt::Display::fmt(error, f),
        }
    }
}
impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Overflow => None,
            Self::Metadata(error) => Some(error),
            Self::Allocation(error) => Some(error),
            Self::Formatting(error) => Some(error),
        }
    }
}

/// The Runtime issuer adds its own canonical claim header exactly once.
/// Zero bytes mean no physical backing allocation, not a free nonempty block.
pub(crate) fn claim_bytes(
    metadata: Option<&dyn ProviderHostMetadataAdmission>,
    bytes: usize,
) -> Result<Claim, Error> {
    if bytes == 0 {
        return Ok(None);
    }
    let bytes = u64::try_from(bytes).map_err(|_| Error::Overflow)?;
    metadata
        .map(|owner| owner.claim_record(bytes).map_err(Error::Metadata))
        .transpose()
}

#[derive(Debug)]
pub(crate) struct ClaimedVec<T> {
    // Declaration order matters: release actual backing before its funding.
    values: Vec<T>,
    claim: Claim,
}
impl<T> ClaimedVec<T> {
    pub(crate) fn with_capacity(
        capacity: usize,
        metadata: Option<&dyn ProviderHostMetadataAdmission>,
    ) -> Result<Self, Error> {
        let layout = Layout::array::<T>(capacity).map_err(|_| Error::Overflow)?;
        let claim = claim_bytes(metadata, layout.size())?;
        let values = runmat_value::exact_capacity_vec(capacity).map_err(Error::Allocation)?;
        Ok(Self { values, claim })
    }
    pub(crate) fn len(&self) -> usize {
        self.values.len()
    }
    #[cfg(any(feature = "wgpu", test))]
    pub(crate) fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
    pub(crate) fn capacity(&self) -> usize {
        self.values.capacity()
    }
    pub(crate) fn as_slice(&self) -> &[T] {
        &self.values
    }
    #[cfg(any(feature = "wgpu", test))]
    pub(crate) fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.values
    }
    fn reserve(
        &mut self,
        additional: usize,
        metadata: Option<&dyn ProviderHostMetadataAdmission>,
    ) -> Result<(), Error> {
        let required = self.len().checked_add(additional).ok_or(Error::Overflow)?;
        if required <= self.capacity() {
            return Ok(());
        }
        // This is the actual requested growth policy, not an estimated quote
        // for another allocator: exact_capacity_vec requests precisely this
        // checked Layout. Old and new blocks/claims overlap during relocation.
        let capacity = required
            .max(self.capacity().checked_mul(2).ok_or(Error::Overflow)?)
            .max(4);
        let mut next = Self::with_capacity(capacity, metadata)?;
        next.values.append(&mut self.values);
        drop(std::mem::replace(self, next));
        Ok(())
    }
    #[cfg(any(feature = "wgpu", test))]
    pub(crate) fn push(
        &mut self,
        value: T,
        metadata: Option<&dyn ProviderHostMetadataAdmission>,
    ) -> Result<(), Error> {
        self.reserve(1, metadata)?;
        self.values.push(value);
        Ok(())
    }
    pub(crate) fn extend_from_slice(
        &mut self,
        values: &[T],
        metadata: Option<&dyn ProviderHostMetadataAdmission>,
    ) -> Result<(), Error>
    where
        T: Copy,
    {
        self.reserve(values.len(), metadata)?;
        self.values.extend_from_slice(values);
        Ok(())
    }
    #[cfg(any(feature = "wgpu", test))]
    pub(crate) fn truncate(&mut self, len: usize) {
        self.values.truncate(len);
    }
    #[cfg(any(feature = "wgpu", test))]
    pub(crate) fn dedup(&mut self)
    where
        T: PartialEq,
    {
        self.values.dedup();
    }
    /// Coupled ownership transfer. The successor must retain this SAME claim
    /// until the returned backing is deallocated (collection field first).
    /// This is never an uncharged detach or a refund at logical completion.
    pub(crate) fn into_parts(self) -> (Vec<T>, Claim) {
        (self.values, self.claim)
    }
}

#[derive(Debug)]
pub(crate) struct ClaimedString {
    bytes: ClaimedVec<u8>,
}
impl ClaimedString {
    pub(crate) fn with_capacity(
        bytes: usize,
        metadata: Option<&dyn ProviderHostMetadataAdmission>,
    ) -> Result<Self, Error> {
        Ok(Self {
            bytes: ClaimedVec::with_capacity(bytes, metadata)?,
        })
    }
    #[cfg(any(feature = "wgpu", test))]
    pub(crate) fn copy_from(
        source: &str,
        metadata: Option<&dyn ProviderHostMetadataAdmission>,
    ) -> Result<Self, Error> {
        let mut result = Self::with_capacity(source.len(), metadata)?;
        result.push_str(source, metadata)?;
        Ok(result)
    }
    pub(crate) fn as_str(&self) -> &str {
        std::str::from_utf8(self.bytes.as_slice()).expect("shader writer preserves UTF-8")
    }
    pub(crate) fn len(&self) -> usize {
        self.bytes.len()
    }
    pub(crate) fn push_str(
        &mut self,
        source: &str,
        metadata: Option<&dyn ProviderHostMetadataAdmission>,
    ) -> Result<(), Error> {
        self.bytes.extend_from_slice(source.as_bytes(), metadata)
    }
    pub(crate) fn append_fmt(
        &mut self,
        arguments: fmt::Arguments<'_>,
        metadata: Option<&dyn ProviderHostMetadataAdmission>,
    ) -> Result<(), Error> {
        let bytes = formatted_len(arguments)?;
        self.bytes.reserve(bytes, metadata)?;
        struct Writer<'a>(&'a mut Vec<u8>);
        impl fmt::Write for Writer<'_> {
            fn write_str(&mut self, source: &str) -> fmt::Result {
                if self.0.capacity() - self.0.len() < source.len() {
                    return Err(fmt::Error);
                }
                self.0.extend_from_slice(source.as_bytes());
                Ok(())
            }
        }
        let initial = self.len();
        fmt::write(&mut Writer(&mut self.bytes.values), arguments).map_err(Error::Formatting)?;
        if self.len().checked_sub(initial) != Some(bytes) {
            return Err(Error::Overflow);
        }
        Ok(())
    }
    #[cfg(any(feature = "wgpu", test))]
    pub(crate) fn remove_range(&mut self, range: Range<usize>) -> Result<(), Error> {
        if self.as_str().get(range.clone()).is_none() {
            return Err(Error::Overflow);
        }
        let next_len = self.len() - range.len();
        self.bytes.values.copy_within(range.end.., range.start);
        self.bytes.truncate(next_len);
        Ok(())
    }
    /// Coupled transfer; the successor owns both String backing and its claim.
    pub(crate) fn into_parts(self) -> (String, Claim) {
        let (bytes, claim) = self.bytes.into_parts();
        (
            String::from_utf8(bytes).expect("shader writer preserves UTF-8"),
            claim,
        )
    }
}
impl fmt::Display for ClaimedString {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

pub(crate) fn formatted_len(arguments: fmt::Arguments<'_>) -> Result<usize, Error> {
    struct Count {
        bytes: usize,
        overflow: bool,
    }
    impl fmt::Write for Count {
        fn write_str(&mut self, source: &str) -> fmt::Result {
            self.bytes = self.bytes.checked_add(source.len()).ok_or_else(|| {
                self.overflow = true;
                fmt::Error
            })?;
            Ok(())
        }
    }
    let mut count = Count {
        bytes: 0,
        overflow: false,
    };
    match fmt::write(&mut count, arguments) {
        Ok(()) => Ok(count.bytes),
        Err(_) if count.overflow => Err(Error::Overflow),
        Err(error) => Err(Error::Formatting(error)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_execution::resource::RetainedHostViewGrant;
    use std::sync::atomic::{AtomicU64, Ordering};

    #[derive(Debug)]
    struct Admission {
        live: Arc<AtomicU64>,
        limit: AtomicU64,
    }
    #[derive(Debug)]
    struct Funding {
        live: Arc<AtomicU64>,
        bytes: u64,
    }
    impl Drop for Funding {
        fn drop(&mut self) {
            self.live.fetch_sub(self.bytes, Ordering::SeqCst);
        }
    }
    impl ProviderHostMetadataAdmission for Admission {
        fn claim_view(
            &self,
            _: u64,
        ) -> Result<Box<dyn RetainedHostViewGrant>, ProviderHostMetadataError> {
            Err(ProviderHostMetadataError::OwnerFailure)
        }
        fn claim_record(
            &self,
            bytes: u64,
        ) -> Result<Arc<dyn RetainedHostAllocationClaim>, ProviderHostMetadataError> {
            self.live
                .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |live| {
                    live.checked_add(bytes)
                        .filter(|&next| next <= self.limit.load(Ordering::SeqCst))
                })
                .map_err(|_| ProviderHostMetadataError::ExceedsCapacity)?;
            Ok(Arc::new(Funding {
                live: self.live.clone(),
                bytes,
            }))
        }
    }

    #[test]
    fn shader_buffers_refuse_before_growth_and_retain_original_funding() {
        let admission = Admission {
            live: Arc::new(AtomicU64::new(0)),
            limit: AtomicU64::new(0),
        };
        assert!(matches!(
            ClaimedString::copy_from("abcd", Some(&admission)),
            Err(Error::Metadata(ProviderHostMetadataError::ExceedsCapacity))
        ));
        assert_eq!(admission.live.load(Ordering::SeqCst), 0);
        admission.limit.store(4, Ordering::SeqCst);
        let mut source = ClaimedString::copy_from("abcd", Some(&admission)).unwrap();
        assert_eq!(admission.live.load(Ordering::SeqCst), 4);
        // Growth requests eight bytes while the old four-byte backing lives.
        admission.limit.store(11, Ordering::SeqCst);
        assert!(matches!(
            source.push_str("e", Some(&admission)),
            Err(Error::Metadata(ProviderHostMetadataError::ExceedsCapacity))
        ));
        assert_eq!(source.as_str(), "abcd");
        assert_eq!(admission.live.load(Ordering::SeqCst), 4);
        admission.limit.store(12, Ordering::SeqCst);
        source.push_str("e", Some(&admission)).unwrap();
        assert_eq!(source.as_str(), "abcde");
        assert_eq!(admission.live.load(Ordering::SeqCst), 8);
        // Removing text does not refund a still-allocated backing block.
        source.remove_range(1..4).unwrap();
        assert_eq!(source.as_str(), "ae");
        assert_eq!(admission.live.load(Ordering::SeqCst), 8);
        drop(source);
        assert_eq!(admission.live.load(Ordering::SeqCst), 0);
    }
}
