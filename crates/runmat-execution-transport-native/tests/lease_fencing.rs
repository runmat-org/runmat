use runmat_execution_transport_native::identity::LeaseAuthority;
use runmat_execution_transport_native::TransportError;

#[test]
fn stale_or_expired_lease_authority_fails_closed() {
    let lease = LeaseAuthority {
        lease_id: "lease-1".to_string(),
        fencing_token: 7,
        expires_at_millis: 1_000,
    };
    lease.validate(7, 999).unwrap();
    assert_eq!(lease.validate(6, 999), Err(TransportError::StaleAuthority));
    assert_eq!(
        lease.validate(7, 1_000),
        Err(TransportError::StaleAuthority)
    );
}
