use runmat_execution::Digest;
use runmat_execution_artifact::encryption::{
    decode_run_key_envelope, encode_run_key_envelope, open_run_key, seal_run_key,
    EncryptionContext, EncryptionPurpose, NativeExecutionEncryption, PortableExecutionEncryption,
    PortableExecutionPrivateKey, RunKeyMaterial, RunObjectEncryption,
};

fn context(run: &str, plaintext: &[u8]) -> EncryptionContext {
    EncryptionContext {
        schema_version: 1,
        run_identity: run.to_string(),
        purpose: EncryptionPurpose::Bundle,
        object_digest: Digest::sha256(plaintext),
        task_identity: None,
        attempt_identity: None,
        chunk_index: 0,
        total_length: plaintext.len() as u64,
        key_epoch: 1,
    }
}

#[test]
fn portable_recipient_secret_roundtrips_through_exact_bytes() {
    let provider = PortableExecutionEncryption;
    let (recipient, private_key) = provider
        .recipient_from_entropy_with_derived_fingerprint([13; 32], 1, u64::MAX)
        .unwrap();
    let restored = PortableExecutionPrivateKey::from_bytes(&private_key.to_bytes()).unwrap();
    assert_eq!(restored.public_key_bytes(), recipient.public_key.as_slice());
    let run_key = RunKeyMaterial::from_entropy([17; 32]).unwrap();
    let envelope = provider
        .seal_run_key_with_entropy([19; 32], &recipient, &run_key, "run_portable", 1)
        .unwrap();
    let opened = provider
        .open_run_key(
            &restored,
            &envelope,
            &recipient.fingerprint,
            "run_portable",
            1,
        )
        .unwrap();
    assert_eq!(
        opened.expose_for_recipient_envelope(),
        run_key.expose_for_recipient_envelope()
    );
    assert!(PortableExecutionPrivateKey::from_bytes(&[1; 31]).is_err());
}

#[test]
fn portable_run_object_vector_roundtrips_and_is_deterministic() {
    let key = RunKeyMaterial::from_entropy([7; 32]).unwrap();
    let plaintext = b"content blind execution";
    let encrypted = RunObjectEncryption
        .seal_with_entropy(&key, [9; 32], context("run_vector", plaintext), plaintext)
        .unwrap();
    assert_eq!(
        hex::encode(&encrypted.ciphertext),
        "75f74e481ed6e979227eb2f441b7b7477234c28254b32bca428d57c7782fa3ac47bc88d7f36a7e"
    );
    assert_eq!(
        RunObjectEncryption.open(&key, &encrypted).unwrap(),
        plaintext
    );
}

#[test]
fn context_swap_nonce_reuse_and_corruption_fail_closed() {
    let key = RunKeyMaterial::from_entropy([3; 32]).unwrap();
    let plaintext = b"bundle";
    let first = RunObjectEncryption
        .seal_with_entropy(&key, [4; 32], context("run_a", plaintext), plaintext)
        .unwrap();
    let second = RunObjectEncryption
        .seal_with_entropy(&key, [5; 32], context("run_a", plaintext), plaintext)
        .unwrap();
    assert_ne!(first.ciphertext, second.ciphertext);

    let mut swapped = first.clone();
    swapped.context.run_identity = "run_b".to_string();
    assert!(RunObjectEncryption.open(&key, &swapped).is_err());

    let mut corrupt = first;
    corrupt.ciphertext[0] ^= 1;
    assert!(RunObjectEncryption.open(&key, &corrupt).is_err());
}

#[test]
fn run_key_envelope_is_bound_to_recipient_run_and_epoch() {
    let provider = NativeExecutionEncryption;
    let (recipient, private_key) = provider
        .generate_recipient("node-key", 1, u64::MAX)
        .unwrap();
    let run_key = RunKeyMaterial::from_entropy([11; 32]).unwrap();
    let envelope = seal_run_key(&provider, &recipient, &run_key, "run_a", 7).unwrap();
    let encoded = encode_run_key_envelope(&envelope).unwrap();
    assert_eq!(
        decode_run_key_envelope(&encoded, 64 * 1024).unwrap(),
        envelope
    );
    let mut trailing = encoded;
    trailing.push(0);
    assert!(decode_run_key_envelope(&trailing, 64 * 1024).is_err());
    let opened = open_run_key(&provider, &private_key, &envelope, "node-key", "run_a", 7).unwrap();
    assert_eq!(
        opened.expose_for_recipient_envelope(),
        run_key.expose_for_recipient_envelope()
    );
    assert!(open_run_key(&provider, &private_key, &envelope, "node-key", "run_b", 7).is_err());
    assert!(open_run_key(&provider, &private_key, &envelope, "other-key", "run_a", 7).is_err());
    assert!(open_run_key(&provider, &private_key, &envelope, "node-key", "run_a", 8).is_err());
}
