//! Build-time configuration identity. Runtime artifact owners combine this
//! with source dependencies, exact semantic inputs, schemas and acceptance.
use sha2::{Digest as _, Sha256};

/// Emit the actual configuration of the package whose build script calls this
/// helper, including dependency-local optimization and feature selections.
pub fn emit_configuration() {
    let compiler = std::process::Command::new(std::env::var_os("RUSTC").expect("Cargo compiler"))
        .arg("--version")
        .output()
        .expect("compiler identity");
    assert!(compiler.status.success(), "compiler identity failed");
    let fields = std::env::vars()
        .filter(|(key, _)| {
            key.starts_with("CARGO_FEATURE_")
                || key.starts_with("CARGO_CFG_")
                || matches!(
                    key.as_str(),
                    "TARGET"
                        | "HOST"
                        | "OPT_LEVEL"
                        | "DEBUG"
                        | "PROFILE"
                        | "CARGO_ENCODED_RUSTFLAGS"
                        | "CARGO_PKG_VERSION"
                        | "CARGO_PKG_NAME"
                )
        })
        .collect::<Vec<_>>();
    println!(
        "cargo:rustc-env=RUNMAT_BUILD_CONFIGURATION={}",
        configuration_digest(&fields, &compiler.stdout)
    );
}

fn configuration_digest(fields: &[(String, String)], compiler: &[u8]) -> String {
    let mut fields = fields.to_vec();
    fields.sort();
    let mut hash = Sha256::new();
    hash.update(b"runmat-package-build-configuration/v1\0");
    hash.update((compiler.len() as u64).to_be_bytes());
    hash.update(compiler);
    for (key, value) in fields {
        hash.update((key.len() as u64).to_be_bytes());
        hash.update(key.as_bytes());
        hash.update((value.len() as u64).to_be_bytes());
        hash.update(value.as_bytes());
    }
    hex::encode(hash.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn configuration_order_is_irrelevant_but_features_profile_and_compiler_invalidate() {
        let fields = vec![
            ("OPT_LEVEL".into(), "3".into()),
            ("CARGO_FEATURE_EXACT".into(), "1".into()),
        ];
        let original = configuration_digest(&fields, b"compiler-a");
        let reversed = fields.iter().cloned().rev().collect::<Vec<_>>();
        assert_eq!(original, configuration_digest(&reversed, b"compiler-a"));
        assert_ne!(original, configuration_digest(&fields, b"compiler-b"));
        for index in 0..fields.len() {
            let mut changed = fields.clone();
            changed[index].1 = "changed".into();
            assert_ne!(original, configuration_digest(&changed, b"compiler-a"));
        }
    }

    /// SHA-256 moved to sha2 0.11 (hardware SHA instructions where the CPU has them). Every digest
    /// RunMat persists or compares must stay byte-identical to the 0.10 software implementation.
    #[test]
    fn sha256_matches_published_vectors_the_previous_implementation_and_the_pinned_golden() {
        let vectors: [(&[u8], &str); 3] = [
            (
                b"",
                "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            ),
            (
                b"abc",
                "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            ),
            (
                b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
                "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1",
            ),
        ];
        for (message, expected) in vectors {
            assert_eq!(hex::encode(Sha256::digest(message)), expected);
        }
        let million = vec![b'a'; 1_000_000];
        assert_eq!(
            hex::encode(Sha256::digest(&million)),
            "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"
        );

        // Every padding boundary, one-shot and split across incremental updates, against 0.10.
        let bytes = (0..1025u32)
            .map(|i| (i.wrapping_mul(131) >> 3) as u8)
            .collect::<Vec<_>>();
        for len in 0..bytes.len() {
            let message = &bytes[..len];
            let previous = {
                use sha2_mac::Digest as _;
                sha2_mac::Sha256::digest(message).to_vec()
            };
            assert_eq!(
                Sha256::digest(message).as_slice(),
                previous.as_slice(),
                "len {len}"
            );
            let mut split = Sha256::new();
            let (head, tail) = message.split_at(len / 3);
            split.update(head);
            split.update(tail);
            assert_eq!(
                split.finalize().as_slice(),
                previous.as_slice(),
                "split len {len}"
            );
        }

        // Golden configuration digest, computed independently with Python hashlib.
        let fields = vec![
            ("OPT_LEVEL".into(), "3".into()),
            ("CARGO_FEATURE_EXACT".into(), "1".into()),
        ];
        assert_eq!(
            configuration_digest(&fields, b"compiler-a"),
            "e3a48a01e93acb51be7c816e4b7e9d6687e7f0a4d86040030d14691d3c335f7a"
        );
    }
}
