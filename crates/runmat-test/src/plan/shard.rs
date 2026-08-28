use sha2::{Digest, Sha256};

use crate::identity::TestId;

pub fn shard_for(test_id: &TestId, shard_count: u32) -> Option<u32> {
    if shard_count == 0 {
        return None;
    }
    let digest = Sha256::digest(test_id.as_str().as_bytes());
    let value = u64::from_be_bytes(digest[..8].try_into().expect("eight digest bytes"));
    Some((value % u64::from(shard_count)) as u32)
}
