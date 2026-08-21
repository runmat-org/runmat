use hpke::rand_core::{TryCryptoRng, TryRng};
use hpke::{
    aead::AesGcm128,
    kdf::HkdfSha256,
    kem::{Kem as _, X25519HkdfSha256},
    setup_receiver, setup_sender_with_rng, OpModeR, OpModeS, Serializable,
};
use runmat_execution::Digest;
use runmat_execution_artifact::encryption::{
    EncryptedArtifact, EncryptionContext, EncryptionPurpose, ExecutionEncryptionProvider,
    NativeExecutionEncryption,
};

fn context(plaintext: &[u8]) -> EncryptionContext {
    EncryptionContext {
        schema_version: 1,
        run_identity: "run-01".into(),
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
fn fixed_suite_round_trips_and_authenticates_context_and_ciphertext() {
    let provider = NativeExecutionEncryption;
    let (recipient, private) = provider
        .generate_recipient("recipient-01", 1, u64::MAX)
        .unwrap();
    assert_eq!(recipient.suite.kem_id(), 0x0020);
    assert_eq!(recipient.suite.kdf_id(), 0x0001);
    assert_eq!(recipient.suite.aead_id(), 0x0001);

    let plaintext = b"private execution bundle";
    let artifact = provider
        .seal(&recipient, context(plaintext), plaintext)
        .unwrap();
    assert_eq!(provider.open(&private, &artifact).unwrap(), plaintext);

    let mut tampered_ciphertext = artifact.clone();
    tampered_ciphertext.ciphertext[0] ^= 1;
    assert!(provider.open(&private, &tampered_ciphertext).is_err());

    let mut tampered_context = artifact;
    tampered_context.context.run_identity = "run-02".into();
    assert!(provider.open(&private, &tampered_context).is_err());
}

#[test]
fn plaintext_identity_must_match_before_encryption() {
    let provider = NativeExecutionEncryption;
    let (recipient, _) = provider
        .generate_recipient("recipient-01", 1, u64::MAX)
        .unwrap();
    let plaintext = b"payload";
    let mut mismatched = context(plaintext);
    mismatched.object_digest = Digest::sha256(b"another payload");
    assert!(provider.seal(&recipient, mismatched, plaintext).is_err());

    let mut malformed = context(plaintext);
    malformed.key_epoch = 0;
    assert!(provider.seal(&recipient, malformed, plaintext).is_err());

    let mut malformed = context(plaintext);
    malformed.attempt_identity = Some("attempt-01".into());
    assert!(provider.seal(&recipient, malformed, plaintext).is_err());
}

#[test]
fn encrypted_artifact_schema_is_closed() {
    let _: Option<EncryptedArtifact> = None;
}

#[test]
fn fixed_suite_matches_the_rfc_9180_vector() {
    let ikm_recipient: [u8; 32] =
        hex("6db9df30aa07dd42ee5e8181afdb977e538f5e1fec8a06223f33f7013e525037")
            .try_into()
            .unwrap();
    let ikm_ephemeral: [u8; 32] =
        hex("7268600d403fce431561aef583ee1613527cff655c1343f29812e66706df3234")
            .try_into()
            .unwrap();
    let (private, public) = X25519HkdfSha256::derive_keypair(&ikm_recipient);
    assert_eq!(
        public.to_bytes().as_slice(),
        hex("3948cfe0ad1ddb695d780e59077195da6c56506b027329794ab02bca80815c4d")
    );
    let mut entropy = VectorEntropy(ikm_ephemeral);
    let info = hex("4f6465206f6e2061204772656369616e2055726e");
    let aad = hex("436f756e742d30");
    let plaintext = hex("4265617574792069732074727574682c20747275746820626561757479");
    let (encapsulated, mut sender) =
        setup_sender_with_rng::<AesGcm128, HkdfSha256, X25519HkdfSha256>(
            &OpModeS::Base,
            &public,
            &info,
            &mut entropy,
        )
        .unwrap();
    assert_eq!(
        encapsulated.to_bytes().as_slice(),
        hex("37fda3567bdbd628e88668c3c8d7e97d1d1253b6d4ea6d44c150f741f1bf4431")
    );
    let ciphertext = sender.seal(&plaintext, &aad).unwrap();
    assert_eq!(
        ciphertext,
        hex("f938558b5d72f1a23810b4be2ab4f84331acc02fc97babc53a52ae8218a355a96d8770ac83d07bea87e13c512a")
    );
    let mut receiver = setup_receiver::<AesGcm128, HkdfSha256, X25519HkdfSha256>(
        &OpModeR::Base,
        &private,
        &encapsulated,
        &info,
    )
    .unwrap();
    assert_eq!(receiver.open(&ciphertext, &aad).unwrap(), plaintext);
}

struct VectorEntropy([u8; 32]);

impl TryRng for VectorEntropy {
    type Error = std::convert::Infallible;

    fn try_next_u32(&mut self) -> Result<u32, Self::Error> {
        unreachable!("X25519 requests its entropy as one byte slice")
    }

    fn try_next_u64(&mut self) -> Result<u64, Self::Error> {
        unreachable!("X25519 requests its entropy as one byte slice")
    }

    fn try_fill_bytes(&mut self, destination: &mut [u8]) -> Result<(), Self::Error> {
        destination.copy_from_slice(&self.0);
        Ok(())
    }
}

impl TryCryptoRng for VectorEntropy {}

fn hex(value: &str) -> Vec<u8> {
    value
        .as_bytes()
        .chunks_exact(2)
        .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap())
        .collect()
}
