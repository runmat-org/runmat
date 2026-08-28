use rand::RngCore as _;
use runmat_execution::{Digest, ProgramRevision};
use runmat_execution_artifact::encryption::{
    decode_encrypted_run_object, encode_encrypted_run_object, EncryptedRunObject,
    EncryptionContext, EncryptionPurpose, RunKeyMaterial, RunObjectEncryption,
};
use sha2::{Digest as _, Sha256};

use crate::{NativeExecutionError, NativeExecutionResult};

pub(super) fn open_object(
    run_key: &RunKeyMaterial,
    bytes: &[u8],
    run_id: &str,
    purpose: EncryptionPurpose,
) -> NativeExecutionResult<Vec<u8>> {
    let object = decode_encrypted_run_object(bytes, 64 * 1024 * 1024).map_err(protocol)?;
    if object.context.run_identity != run_id || object.context.purpose != purpose {
        return Err(protocol("encrypted object scope or purpose is invalid"));
    }
    RunObjectEncryption.open(run_key, &object).map_err(protocol)
}

pub(super) fn seal_object(
    run_key: &RunKeyMaterial,
    run_id: &str,
    purpose: EncryptionPurpose,
    plaintext: &[u8],
) -> NativeExecutionResult<Vec<u8>> {
    let mut salt = [0_u8; 32];
    rand::rngs::OsRng.fill_bytes(&mut salt);
    let object: EncryptedRunObject = RunObjectEncryption
        .seal_with_entropy(
            run_key,
            salt,
            EncryptionContext {
                schema_version: 1,
                run_identity: run_id.to_string(),
                purpose,
                object_digest: Digest::sha256(plaintext),
                task_identity: None,
                attempt_identity: None,
                chunk_index: 0,
                total_length: plaintext.len() as u64,
                key_epoch: 1,
            },
            plaintext,
        )
        .map_err(protocol)?;
    encode_encrypted_run_object(&object).map_err(protocol)
}

pub(super) fn ciphertext_digest(bytes: &[u8]) -> String {
    format!("sha256:{:x}", Sha256::digest(bytes))
}

pub(super) fn project_revision_identity(revision: &ProgramRevision) -> String {
    format!("sha256:{}", hex(revision.source_digest().bytes()))
}

fn hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(HEX[(byte >> 4) as usize] as char);
        output.push(HEX[(byte & 0x0f) as usize] as char);
    }
    output
}

fn protocol(error: impl std::fmt::Display) -> NativeExecutionError {
    NativeExecutionError::Protocol(error.to_string())
}
