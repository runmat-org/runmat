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
    format!("{:x}", hash.finalize())
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
}
