use super::ProcessIdentity;
use runmat_package_cache::LeaseOwner;

pub fn process_lease_owner(identity: ProcessIdentity) -> LeaseOwner {
    LeaseOwner::new(format!("process-{}", identity.pid))
        .expect("numeric process identity is a valid lease owner")
}
