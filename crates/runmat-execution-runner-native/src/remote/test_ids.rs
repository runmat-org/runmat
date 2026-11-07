//! Server-issued identifiers for remote driver fixtures. A `DriverAuthority`
//! holds a typed organization and driver-lease identifier, so fixtures build
//! them in the Server's prefixed form.
use runmat_server_client::ids::{ExecutionDriverLeaseId, OrgId};

/// Organization identifier shared by driver authorities and endpoint evidence.
pub(crate) const TEST_ORG_ID: &str = "org_00000000000000000000000000000001";

pub(crate) fn test_org_id() -> OrgId {
    OrgId::parse(TEST_ORG_ID).unwrap()
}

/// Driver-lease identifier for the lease that holds `fence`.
pub(crate) fn test_driver_lease_id(fence: u64) -> ExecutionDriverLeaseId {
    ExecutionDriverLeaseId::parse(&format!("drv_{fence:032x}")).unwrap()
}
