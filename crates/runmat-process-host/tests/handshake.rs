use std::collections::BTreeSet;

use runmat_process_host::ipc::{negotiate_handshake, HostHandshake};
use runmat_process_host::HostCapability;

#[test]
fn handshake_negotiates_the_stricter_frame_bound() {
    let mut local = HostHandshake::new("runmat.test", 1, 4096);
    local.capabilities = BTreeSet::from([HostCapability::StdioIpc]);
    let remote = HostHandshake::new("runmat.test", 1, 1024);
    let limits = negotiate_handshake(&local, &remote).unwrap();
    assert_eq!(limits.max_message_bytes, 1024);
}

#[test]
fn handshake_rejects_a_different_protocol() {
    let local = HostHandshake::new("runmat.test", 1, 4096);
    let remote = HostHandshake::new("runmat.execution", 1, 4096);
    assert!(negotiate_handshake(&local, &remote)
        .unwrap_err()
        .to_string()
        .contains("protocol mismatch"));
}
