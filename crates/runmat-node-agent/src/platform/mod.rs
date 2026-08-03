mod attestation;

#[cfg(target_os = "linux")]
mod linux;
#[cfg(target_os = "macos")]
mod macos;
#[cfg(target_os = "windows")]
mod windows;

use std::collections::BTreeMap;

pub use attestation::{AttestationEvidence, AttestationProvider, NoAttestation};

pub fn capabilities() -> BTreeMap<String, String> {
    #[cfg(target_os = "linux")]
    let platform = linux::capabilities();
    #[cfg(target_os = "macos")]
    let platform = macos::capabilities();
    #[cfg(target_os = "windows")]
    let platform = windows::capabilities();
    #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
    let platform = BTreeMap::new();
    platform
}
