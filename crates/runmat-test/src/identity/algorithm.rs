use sha2::{Digest, Sha256};

use crate::version::TEST_IDENTITY_ALGORITHM_VERSION;

pub(crate) fn digest(kind: &str, parts: &[&str]) -> String {
    let mut hasher = Sha256::new();
    write_part(&mut hasher, "runmat-test");
    write_part(&mut hasher, &TEST_IDENTITY_ALGORITHM_VERSION.to_string());
    write_part(&mut hasher, kind);
    for part in parts {
        write_part(&mut hasher, part);
    }
    format!(
        "v{}:sha256:{:x}",
        TEST_IDENTITY_ALGORITHM_VERSION,
        hasher.finalize()
    )
}

fn write_part(hasher: &mut Sha256, value: &str) {
    hasher.update((value.len() as u64).to_be_bytes());
    hasher.update(value.as_bytes());
}
