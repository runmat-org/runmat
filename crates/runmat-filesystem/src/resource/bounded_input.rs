use super::{
    FsFilePayloadContract, FsResourceContract, FsResourceLease, FsResourceLeaseHandle,
    FsResourceRevision,
};
use crate::{FileHandle, FsMetadata};
use std::io::{self, Read, Seek, SeekFrom};
use std::sync::Arc;

/// One retained opened source, bounded by the extent observed on that same
/// handle. Growth or path replacement cannot expand the bytes exposed to a
/// caller; truncation or in-place mutation may still change contents and can
/// cause a short read. This is a source-byte bound, not a memory-peak claim.
pub struct FsBoundedInput {
    handle: Box<dyn FileHandle>,
    opened_metadata: FsMetadata,
    position: u64,
    lease: FsResourceLeaseHandle,
    contract: FsResourceContract,
}

impl FsBoundedInput {
    /// Providers may opt in only when `handle` reads directly into the caller
    /// buffer and does not retain a payload-sized hidden staging allocation.
    pub fn new(
        handle: Box<dyn FileHandle>,
        opened_metadata: FsMetadata,
        lease: Arc<dyn FsResourceLease>,
    ) -> io::Result<Self> {
        Self::new_with_lease(handle, opened_metadata, FsResourceLeaseHandle::Owned(lease))
    }

    pub fn new_with_lease(
        mut handle: Box<dyn FileHandle>,
        opened_metadata: FsMetadata,
        lease: FsResourceLeaseHandle,
    ) -> io::Result<Self> {
        let contract = lease.lease().contract();
        if !opened_metadata.is_file()
            || contract.opened_input_handle != Some(FsFilePayloadContract::CallerOwnedBuffer)
            || !lease.lease().is_valid()
        {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "opened source has no valid direct-read resource authority",
            ));
        }
        let opened_at = handle.seek(SeekFrom::Start(0))?;
        if opened_at != 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "opened source did not seek to its reported start",
            ));
        }
        check_lease(lease.lease(), contract)?;
        Ok(Self {
            handle,
            opened_metadata,
            position: 0,
            lease,
            contract,
        })
    }

    pub fn opened_extent(&self) -> u64 {
        self.opened_metadata.len()
    }

    pub fn resource_revision(&self) -> FsResourceRevision {
        self.contract.revision
    }

    pub fn is_valid(&self) -> bool {
        self.lease.lease().is_valid() && self.lease.lease().contract() == self.contract
    }

    pub async fn metadata_async(&self) -> io::Result<FsMetadata> {
        self.check_valid()?;
        Ok(self.opened_metadata.clone())
    }

    fn check_valid(&self) -> io::Result<()> {
        check_lease(self.lease.lease(), self.contract)
    }
}

fn check_lease(lease: &dyn FsResourceLease, contract: FsResourceContract) -> io::Result<()> {
    if lease.is_valid() && lease.contract() == contract {
        Ok(())
    } else {
        Err(io::Error::new(
            io::ErrorKind::PermissionDenied,
            "opened source resource authority was revoked",
        ))
    }
}

impl Read for FsBoundedInput {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        self.check_valid()?;
        let remaining = self.opened_extent().saturating_sub(self.position);
        let allowed = usize::try_from(remaining)
            .unwrap_or(usize::MAX)
            .min(buffer.len());
        if allowed == 0 {
            return Ok(0);
        }
        let count = self.handle.read(&mut buffer[..allowed])?;
        if count > allowed {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "opened source reported more bytes than its bounded read",
            ));
        }
        self.check_valid()?;
        self.position = self
            .position
            .checked_add(count as u64)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "source offset overflow"))?;
        Ok(count)
    }
}

impl Seek for FsBoundedInput {
    fn seek(&mut self, position: SeekFrom) -> io::Result<u64> {
        self.check_valid()?;
        let base = match position {
            SeekFrom::Start(offset) => i128::from(offset),
            SeekFrom::Current(delta) => i128::from(self.position) + i128::from(delta),
            SeekFrom::End(delta) => i128::from(self.opened_extent()) + i128::from(delta),
        };
        if base < 0 || base > i128::from(self.opened_extent()) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "seek exceeds opened source extent",
            ));
        }
        let offset = base as u64;
        let actual = self.handle.seek(SeekFrom::Start(offset))?;
        if actual != offset {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "opened source reported a different seek position",
            ));
        }
        self.check_valid()?;
        self.position = offset;
        Ok(offset)
    }
}
