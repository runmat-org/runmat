#![cfg(target_arch = "wasm32")]

use runmat_execution::Digest;
use runmat_execution_artifact::encryption::{EncryptionContext, EncryptionPurpose};
use runmat_wasm::BrowserExecutionRecipient;
use wasm_bindgen_test::wasm_bindgen_test;

wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test]
fn webcrypto_entropy_drives_the_shared_hpke_provider() {
    let plaintext = b"browser execution bundle".to_vec();
    let recipient = BrowserExecutionRecipient::new("browser-test".into(), 1, u64::MAX).unwrap();
    let context = EncryptionContext {
        schema_version: 1,
        run_identity: "run-browser".into(),
        purpose: EncryptionPurpose::Bundle,
        object_digest: Digest::sha256(&plaintext),
        task_identity: None,
        attempt_identity: None,
        chunk_index: 0,
        total_length: plaintext.len() as u64,
        key_epoch: 1,
    };
    let context = serde_wasm_bindgen::to_value(&context).unwrap();
    let encrypted = recipient.seal(context, plaintext.clone()).unwrap();
    assert_eq!(recipient.open(encrypted).unwrap(), plaintext);
}
