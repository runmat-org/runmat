//! Test adapters use Core's actual host contracts, never reconstructed host
//! fingerprints or display-name dispatch metadata.

pub(crate) fn host_inventory() -> runmat_execution::host::ExecutionHostInventory {
    runmat_core::RunMatSession::with_options(false, false)
        .expect("fixture compiler session")
        .execution_host_inventory(runmat_execution::security::ExecutionTrustTier::CustomerTrusted)
        .expect("actual fixture execution host")
}

pub(crate) fn host_requirement() -> runmat_execution::host::ExecutionHostRequirement {
    runmat_execution::host::ExecutionHostRequirement::portable(
        runmat_core::program_environment(runmat_core::CompatMode::Matlab),
        runmat_types::CapabilitySet(std::collections::BTreeSet::from([
            runmat_types::CapabilityRequirement::HostRuntime,
        ])),
        runmat_types::InteropManifest::empty(),
        std::collections::BTreeSet::from([
            runmat_execution::security::ExecutionTrustTier::CustomerTrusted,
        ]),
    )
    .expect("fixture semantic host contract")
}
