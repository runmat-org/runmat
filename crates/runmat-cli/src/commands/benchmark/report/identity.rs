use serde::Serialize;
use sha2::{Digest, Sha256};
use std::io::Read;

#[derive(Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub(super) enum Identity {
    Sha256 { digest: String },
    Unavailable,
}

pub(super) fn source(source: &runmat_core::abi::SourceInput) -> Identity {
    source
        .captured_text()
        .map_or(Identity::Unavailable, content)
}

pub(super) fn content(text: &str) -> Identity {
    Identity::Sha256 {
        digest: hex::encode(Sha256::digest(text.as_bytes())),
    }
}

pub(super) fn executable() -> Identity {
    let digest = (|| -> std::io::Result<String> {
        let mut file = std::fs::File::open(std::env::current_exe()?)?;
        let mut hash = Sha256::new();
        let mut bytes = [0; 8192];
        loop {
            let count = file.read(&mut bytes)?;
            if count == 0 {
                break;
            }
            hash.update(&bytes[..count]);
        }
        Ok(hex::encode(hash.finalize()))
    })();
    digest.map_or(Identity::Unavailable, |digest| Identity::Sha256 { digest })
}
