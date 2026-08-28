use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionHpkeSuite {
    X25519HkdfSha256Aes128GcmV1,
}

impl ExecutionHpkeSuite {
    pub const fn kem_id(self) -> u16 {
        0x0020
    }

    pub const fn kdf_id(self) -> u16 {
        0x0001
    }

    pub const fn aead_id(self) -> u16 {
        0x0001
    }
}
