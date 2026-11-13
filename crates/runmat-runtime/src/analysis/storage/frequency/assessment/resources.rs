//! Shared profiling and bounded original object reads; no numerical reduction or codec substitute.
use super::*;
use crate::analysis::storage::{AnalysisPublicationError, AssessmentStoreError as Error};
use runmat_analysis_fea::execution::AnalysisChunkDescriptor;

pub(super) fn add(left: u64, right: u64) -> Result<u64, Error> {
    left.checked_add(right).ok_or(Error::ResourceLimit)
}
pub(super) struct Reader<'a, R> {
    catalog: &'a PublishedObjectCatalog,
    remaining_encoded: u64,
    maximum: u64,
    read: R,
}
impl<'a, R: FnMut(Digest, u64) -> Result<Option<Vec<u8>>, Error>> Reader<'a, R> {
    pub(super) fn new(
        catalog: &'a PublishedObjectCatalog,
        remaining_encoded: u64,
        maximum: u64,
        read: R,
    ) -> Self {
        Self {
            catalog,
            remaining_encoded,
            maximum,
            read,
        }
    }
    pub(super) fn resident(&self, value: &impl serde::Serialize) -> Result<u64, Error> {
        crate::analysis::storage::publication::resident(value, self.maximum).map_err(Into::into)
    }
    pub(super) fn admit(&self, bytes: u64) -> Result<(), Error> {
        if bytes > self.maximum {
            return Err(Error::ResourceLimit);
        }
        Ok(())
    }
    fn verified(&mut self, digest: Digest, retained: u64) -> Result<Vec<u8>, Error> {
        let reference = self
            .catalog
            .objects
            .iter()
            .find(|reference| reference.logical_digest == digest)
            .ok_or_else(|| Error::Invalid("observation object is not in its catalog".into()))?;
        let limit = self
            .maximum
            .checked_sub(retained)
            .ok_or(Error::ResourceLimit)?
            .min(self.remaining_encoded);
        if reference.encoded_length > limit {
            return Err(Error::ResourceLimit);
        }
        let bytes = (self.read)(digest, limit)?
            .ok_or_else(|| Error::Invalid("observation object is missing".into()))?;
        if bytes.len() as u64 > limit {
            return Err(Error::ResourceLimit);
        }
        if bytes.len() as u64 != reference.encoded_length || Digest::sha256(&bytes) != digest {
            return Err(Error::Invalid(
                "observation object differs from its catalog".into(),
            ));
        }
        self.remaining_encoded -= reference.encoded_length;
        Ok(bytes)
    }
    pub(super) fn admit_decode(&self, bytes: &[u8], retained: u64) -> Result<(), Error> {
        let decoded =
            runmat_canonical_codec::canonical_payload_typed_resident_bytes_estimate(bytes)
                .map_err(|error| Error::from(AnalysisPublicationError::from(error)))?;
        self.admit(add(retained, decoded)?)
    }
    pub(super) fn manifest(
        &mut self,
        digest: Digest,
        retained: u64,
    ) -> Result<AnalysisObjectManifest, Error> {
        let bytes = self.verified(digest, retained)?;
        self.admit_decode(&bytes, retained)?;
        AnalysisObjectManifest::from_canonical_bytes(&bytes)
            .map_err(|error| Error::Invalid(error.to_string()))
    }
    pub(super) fn payload(
        &mut self,
        chunks: &[AnalysisChunkDescriptor],
        encoded_length: u64,
        digest: Digest,
        retained: u64,
    ) -> Result<Vec<u8>, Error> {
        let capacity = usize::try_from(encoded_length).map_err(|_| Error::ResourceLimit)?;
        // Exact reservation avoids geometric buffer growth; even a forged declaration is
        // checked against the phase before the allocator or source callback is entered.
        self.admit(add(retained, encoded_length)?)?;
        let mut payload = Vec::new();
        payload
            .try_reserve_exact(capacity)
            .map_err(|_| Error::ResourceLimit)?;
        for chunk in chunks {
            let bytes = self.verified(chunk.digest, add(retained, encoded_length)?)?;
            crate::analysis::storage::validate_chunk(chunk, &bytes).map_err(Error::Invalid)?;
            if payload
                .len()
                .checked_add(bytes.len())
                .filter(|length| *length <= capacity)
                .is_none()
            {
                return Err(Error::Invalid(
                    "scalar payload chunks exceed declared partition".into(),
                ));
            }
            payload.extend_from_slice(&bytes);
        }
        if payload.len() != capacity || Digest::sha256(&payload) != digest {
            return Err(Error::Invalid(
                "scalar payload differs from its partition".into(),
            ));
        }
        Ok(payload)
    }
}
