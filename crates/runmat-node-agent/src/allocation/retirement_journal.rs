//! Bounded opaque receipt collection by the process-tree owner after retirement.
//! Task semantics remain encrypted and are checked only by the replacement driver.

use std::io::Read as _;

use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
use runmat_execution::security::{
    MAX_RETIREMENT_ATTEMPTS, MAX_RETIREMENT_ENCRYPTED_RECEIPT_BYTES,
    RETIREMENT_JOURNAL_STAGING_FILE,
};
use runmat_execution_artifact::encryption::{decode_encrypted_run_object, EncryptionPurpose};

use super::WorkerRetirementScope;
use crate::{AgentError, AgentResult};

pub fn retired_encrypted_receipts(scope: &WorkerRetirementScope) -> AgentResult<Vec<String>> {
    if scope.retired_monotonic.is_none() || scope.clock_domain_nonce.iter().all(|value| *value == 0)
    {
        return Err(reject("retired worker lacks node clock observation"));
    }
    let mut receipts = Vec::new();
    for file in std::fs::read_dir(&scope.journal_directory)? {
        let file = file?;
        if !file.file_type()?.is_file() {
            return Err(reject("retired worker journal contains a non-regular file"));
        }
        if file.file_name() == RETIREMENT_JOURNAL_STAGING_FILE {
            if file.metadata()?.len() > MAX_RETIREMENT_ENCRYPTED_RECEIPT_BYTES as u64 {
                return Err(reject("retired worker staging slot exceeds its budget"));
            }
            // Killing a writer before atomic replacement leaves this partial
            // slot. It cannot supersede a prior receipt or establish admission.
            continue;
        }
        if receipts.len() == MAX_RETIREMENT_ATTEMPTS {
            return Err(reject("retired worker journal exceeds attempt capacity"));
        }
        if file.path().extension().and_then(|value| value.to_str()) != Some("receipt")
            || file
                .path()
                .file_stem()
                .and_then(|value| value.to_str())
                .and_then(|value| value.parse::<runmat_execution::identity::AttemptId>().ok())
                .is_none()
        {
            return Err(reject("retired worker journal contains a foreign file"));
        }
        let mut reader = std::fs::File::open(file.path())?;
        let mut bytes = Vec::with_capacity(MAX_RETIREMENT_ENCRYPTED_RECEIPT_BYTES);
        reader
            .by_ref()
            .take((MAX_RETIREMENT_ENCRYPTED_RECEIPT_BYTES + 1) as u64)
            .read_to_end(&mut bytes)?;
        if bytes.is_empty() || bytes.len() > MAX_RETIREMENT_ENCRYPTED_RECEIPT_BYTES {
            return Err(reject("retired worker receipt exceeds its budget"));
        }
        let object = decode_encrypted_run_object(&bytes, MAX_RETIREMENT_ENCRYPTED_RECEIPT_BYTES)
            .map_err(|error| reject(error.to_string()))?;
        if object.context.run_identity != scope.run_id
            || object.context.purpose != EncryptionPurpose::RetirementReceipt
            || object.context.task_identity.is_some()
            || object.context.attempt_identity.is_some()
        {
            return Err(reject("retired worker receipt has foreign clear scope"));
        }
        receipts.push((file.file_name(), URL_SAFE_NO_PAD.encode(bytes)));
    }
    receipts.sort_by(|left, right| left.0.cmp(&right.0));
    Ok(receipts.into_iter().map(|(_, encoded)| encoded).collect())
}

fn reject(message: impl Into<String>) -> AgentError {
    AgentError::AllocationRejected(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_execution::{identity::AttemptId, Digest};
    use runmat_execution_artifact::encryption::{
        encode_encrypted_run_object, EncryptionContext, RunKeyMaterial, RunObjectEncryption,
    };
    use runmat_process_host::clock::HostMonotonicReading;

    #[test]
    fn interrupted_atomic_publication_preserves_only_the_committed_receipt() {
        let directory = tempfile::tempdir().unwrap();
        let scope = WorkerRetirementScope {
            run_id: "run-1".into(),
            allocation_lease_id: "lease-1".into(),
            allocation_fencing_token: 1,
            driver_fencing_token: 1,
            process_id: 101,
            process_launch_nonce: [7; 16],
            clock_domain_nonce: [11; 16],
            journal_directory: directory.path().to_path_buf(),
            retired_monotonic: Some(HostMonotonicReading {
                ticks: 200,
                ticks_per_second: 1000,
            }),
            retired_at_unix_millis: 1000,
        };
        let plaintext = b"synthetic private journal";
        let object = RunObjectEncryption
            .seal_with_entropy(
                &RunKeyMaterial::from_entropy([9; 32]).unwrap(),
                [13; 32],
                EncryptionContext {
                    schema_version: 1,
                    run_identity: scope.run_id.clone(),
                    purpose: EncryptionPurpose::RetirementReceipt,
                    object_digest: Digest::sha256(plaintext),
                    task_identity: None,
                    attempt_identity: None,
                    chunk_index: 0,
                    total_length: plaintext.len() as u64,
                    key_epoch: 1,
                },
                plaintext,
            )
            .unwrap();
        let bytes = encode_encrypted_run_object(&object).unwrap();
        let receipt = directory
            .path()
            .join(format!("{}.receipt", AttemptId::derive(&[b"committed"])));
        std::fs::write(&receipt, &bytes).unwrap();
        let staging = directory.path().join(RETIREMENT_JOURNAL_STAGING_FILE);
        std::fs::write(&staging, b"interrupted ciphertext").unwrap();
        assert_eq!(
            retired_encrypted_receipts(&scope).unwrap(),
            vec![URL_SAFE_NO_PAD.encode(&bytes)]
        );
        std::fs::remove_file(&receipt).unwrap();
        assert!(
            retired_encrypted_receipts(&scope).unwrap().is_empty(),
            "staging does not establish admission"
        );
        std::fs::write(
            &staging,
            vec![0; MAX_RETIREMENT_ENCRYPTED_RECEIPT_BYTES + 1],
        )
        .unwrap();
        assert!(retired_encrypted_receipts(&scope).is_err());
        std::fs::remove_file(&staging).unwrap();
        std::fs::write(directory.path().join("foreign.tmp"), b"unexpected").unwrap();
        assert!(retired_encrypted_receipts(&scope).is_err());
    }
}
