#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpaqueFramePayload {
    pub encryption_suite: String,
    pub key_epoch: u64,
    pub ciphertext: Vec<u8>,
}

impl OpaqueFramePayload {
    pub fn contains_only_ciphertext_metadata(&self) -> bool {
        !self.encryption_suite.is_empty() && self.key_epoch > 0 && !self.ciphertext.is_empty()
    }
}
