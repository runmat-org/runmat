use sha2::{Digest as _, Sha256};

use crate::transfer::OpaqueObject;
use crate::{TransportError, TransportResult};

pub fn verify_ciphertext(object: &OpaqueObject, bytes: &[u8]) -> TransportResult<()> {
    if u64::try_from(bytes.len()).ok() != Some(object.ciphertext_size_bytes)
        || format!("sha256:{:x}", Sha256::digest(bytes)) != object.ciphertext_digest
    {
        return Err(TransportError::Integrity);
    }
    Ok(())
}
