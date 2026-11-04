use crate::ids::{ExecutionDriverLeaseId, OrgId, SimulationLeaseId};

#[derive(Clone)]
pub(super) enum LeaseAuthority {
    Person,
    Driver {
        lease: ExecutionDriverLeaseId,
        fencing_token: u64,
    },
}

pub(super) enum LeaseAction {
    Usage,
    Settle,
}

impl LeaseAuthority {
    pub(super) fn start_url(&self, base: &str, org: &OrgId) -> String {
        match self {
            Self::Person => format!("{base}/v1/orgs/{org}/simulation/leases"),
            Self::Driver { lease, fencing_token } => format!(
                "{base}/v1/execution/drivers/{lease}/simulation/leases?fencingToken={fencing_token}"),
        }
    }

    pub(super) fn lease_url(
        &self,
        base: &str,
        simulation: &SimulationLeaseId,
        action: LeaseAction,
    ) -> String {
        let action = match action {
            LeaseAction::Usage => "usage",
            LeaseAction::Settle => "settle",
        };
        match self {
            Self::Person => format!("{base}/v1/simulation/leases/{simulation}/{action}"),
            Self::Driver { lease, fencing_token } => format!(
                "{base}/v1/execution/drivers/{lease}/simulation/leases/{simulation}/{action}?fencingToken={fencing_token}"),
        }
    }
}
