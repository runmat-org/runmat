use runmat_process_host::ipc::LocalEndpoint;

#[test]
fn all_local_endpoint_variants_are_non_network() {
    assert!(!LocalEndpoint::Stdio.is_network());
    #[cfg(unix)]
    assert!(!LocalEndpoint::UnixSocket {
        path: "/tmp/runmat.sock".into()
    }
    .is_network());
    #[cfg(windows)]
    assert!(!LocalEndpoint::NamedPipe {
        name: r"\\.\pipe\runmat".into()
    }
    .is_network());
}
