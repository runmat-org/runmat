//! Chained-hash cache keys.
//!
//! `key(feature) = sha256(domain ‖ previous key ‖ canonical feature content ‖ resolved
//! reference ids ‖ kernel id and version ‖ solver and unit fingerprints)`. Canonical content is
//! the evaluated, SI form (the kernel operation, or the datum or solved sketch), encoded as
//! serde JSON over sorted maps, so equal inputs give equal bytes on every platform. Resolved
//! reference ids are the entities' structured names, never per-run positions.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fmt;

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct CacheKey(pub String);

impl fmt::Display for CacheKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl CacheKey {
    /// The key before the first feature.
    pub fn root(document: &str, configuration: Option<&str>) -> CacheKey {
        let mut h = Sha256::new();
        part(&mut h, b"runmat-cad/regen/v1/root");
        part(&mut h, document.as_bytes());
        part(&mut h, configuration.unwrap_or("").as_bytes());
        CacheKey(hex::encode::<[u8; 32]>(h.finalize().into()))
    }

    /// The key of step `index` of the compound feature keyed `self`: the step's kernel result
    /// is cached under it, so a frozen compound feature finds every step again.
    pub fn step(&self, index: usize) -> CacheKey {
        let mut h = Sha256::new();
        part(&mut h, b"runmat-cad/regen/v1/step");
        part(&mut h, self.0.as_bytes());
        part(&mut h, &(index as u64).to_be_bytes());
        CacheKey(hex::encode::<[u8; 32]>(h.finalize().into()))
    }
}

fn part(h: &mut Sha256, bytes: &[u8]) {
    h.update((bytes.len() as u64).to_be_bytes());
    h.update(bytes);
}

/// Inputs to one feature's key.
pub struct KeyInputs<'a, C: Serialize, R: Serialize> {
    pub previous: &'a CacheKey,
    pub feature: &'a str,
    pub content: &'a C,
    pub references: &'a R,
    pub kernel: &'a str,
    pub environment: &'a str,
}

pub fn chain<C: Serialize, R: Serialize>(inputs: KeyInputs<'_, C, R>) -> CacheKey {
    let mut h = Sha256::new();
    part(&mut h, b"runmat-cad/regen/v1/feature");
    part(&mut h, inputs.previous.0.as_bytes());
    part(&mut h, inputs.feature.as_bytes());
    part(
        &mut h,
        &serde_json::to_vec(inputs.content).expect("canonical content serializes"),
    );
    part(
        &mut h,
        &serde_json::to_vec(inputs.references).expect("reference ids serialize"),
    );
    part(&mut h, inputs.kernel.as_bytes());
    part(&mut h, inputs.environment.as_bytes());
    CacheKey(hex::encode::<[u8; 32]>(h.finalize().into()))
}
