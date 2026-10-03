//! One owner for placement catalog/compiler/authority/policy revisions. Local
//! and complete-route decisions share the stable base, while policy-specific
//! inputs remain distinct and typed.

use runmat_execution::{
    Digest, PlacementExecutionAuthorityRevision, PlacementRevision, ProgramRevision,
};

use super::{ExplorationMode, PlacementPolicy, PlacementSessionConfig};

fn base(
    program: Option<ProgramRevision>,
    authority: PlacementExecutionAuthorityRevision,
    policy: Digest,
) -> PlacementRevision {
    PlacementRevision {
        program,
        catalog: Digest::sha256(b"runmat-placement-catalog-v1"),
        compiler: Digest::sha256(b"runmat-placement-compiler-v1"),
        authority,
        policy,
    }
}

pub(super) fn local(
    program: Option<ProgramRevision>,
    provider: Digest,
    policy: PlacementPolicy,
) -> PlacementRevision {
    // Preserve the existing local profile identity while moving its authority.
    base(
        program,
        PlacementExecutionAuthorityRevision::Provider(provider),
        Digest::sha256(format!(
            "local:{}:{}",
            policy.absolute_margin_ns, policy.relative_margin_basis_points
        )),
    )
}

pub(super) fn whole_work(
    program: Option<ProgramRevision>,
    provider: Digest,
    config: PlacementSessionConfig,
) -> PlacementRevision {
    base(
        program,
        PlacementExecutionAuthorityRevision::Provider(provider),
        whole_work_policy(config),
    )
}

fn whole_work_policy(config: PlacementSessionConfig) -> Digest {
    let mut bytes = b"runmat-whole-work-policy.v1\0".to_vec();
    let graph = PlacementPolicy::default();
    bytes.extend_from_slice(&graph.absolute_margin_ns.to_le_bytes());
    bytes.extend_from_slice(&graph.relative_margin_basis_points.to_le_bytes());
    bytes.extend_from_slice(&config.leave_absolute_margin_ns.to_le_bytes());
    bytes.extend_from_slice(&config.leave_relative_margin_basis_points.to_le_bytes());
    bytes.extend_from_slice(&config.exploration_max_overhead_basis_points.to_le_bytes());
    match config.exploration {
        ExplorationMode::Disabled => bytes.push(0),
        ExplorationMode::TransactionalEvery { interval } => {
            bytes.push(1);
            bytes.extend_from_slice(&interval.to_le_bytes());
        }
    }
    Digest::sha256(bytes)
}

pub(super) fn whole_work_host_service(
    program: Option<ProgramRevision>,
    host_service: Digest,
    config: PlacementSessionConfig,
) -> PlacementRevision {
    base(
        program,
        PlacementExecutionAuthorityRevision::HostService(host_service),
        whole_work_policy(config),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn common_revision_owner_distinguishes_policy_and_provider() {
        let provider = Digest::sha256(b"provider one");
        let other = Digest::sha256(b"provider two");
        let first = whole_work(None, provider, PlacementSessionConfig::default());
        let mut changed = PlacementSessionConfig::default();
        changed.leave_absolute_margin_ns += 1;
        assert_ne!(first.policy, whole_work(None, provider, changed).policy);
        assert_ne!(first.authority, whole_work(None, other, changed).authority);
        assert_eq!(
            first.catalog,
            local(None, provider, PlacementPolicy::default()).catalog
        );
        assert_eq!(
            first.compiler,
            local(None, provider, PlacementPolicy::default()).compiler
        );
        let host = whole_work_host_service(None, provider, PlacementSessionConfig::default());
        assert_ne!(first.authority, host.authority);
        assert_ne!(
            serde_json::to_string(&first).unwrap(),
            serde_json::to_string(&host).unwrap()
        );
    }
}
