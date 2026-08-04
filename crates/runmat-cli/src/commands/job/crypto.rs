use anyhow::{bail, Result};
use runmat_execution_artifact::encryption::{
    decode_encrypted_run_object, EncryptionPurpose, RunKeyMaterial, RunObjectEncryption,
};

const MAX_REMOTE_ARTIFACT_BYTES: usize = 64 * 1024 * 1024;

pub(crate) fn open_run_object(
    key: &RunKeyMaterial,
    run_id: &str,
    purpose: EncryptionPurpose,
    ciphertext: &[u8],
) -> Result<Vec<u8>> {
    let object = decode_encrypted_run_object(ciphertext, MAX_REMOTE_ARTIFACT_BYTES)?;
    if object.context.run_identity != run_id || object.context.purpose != purpose {
        bail!("encrypted remote artifact has the wrong run scope or purpose");
    }
    RunObjectEncryption
        .open(key, &object)
        .map_err(anyhow::Error::from)
}
