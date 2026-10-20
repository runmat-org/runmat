//! Shared contract fixtures for this crate's integration tests.
#![allow(dead_code)]
use std::collections::BTreeSet;
use std::sync::OnceLock;

use runmat_execution::{
    host::{ExecutionHostInventory, ExecutionHostRequirement},
    resource::{
        AcceleratorAllocationDomainId, AcceleratorClass, AcceleratorDevice, AcceleratorDeviceId,
        AcceleratorFeature, AcceleratorProvider, AcceleratorProviderId,
        AcceleratorProviderVersion, AcceleratorRequest,
    },
    security::ExecutionTrustTier,
    Digest,
};

/// Requirement for portable work built against this host's MATLAB-mode environment.
pub(crate) fn host_requirement() -> ExecutionHostRequirement {
    ExecutionHostRequirement::portable(
        runmat_core::program_environment(runmat_core::CompatMode::Matlab),
        runmat_types::CapabilitySet(BTreeSet::from([
            runmat_types::CapabilityRequirement::HostRuntime,
        ])),
        runmat_types::InteropManifest::empty(),
        BTreeSet::from([ExecutionTrustTier::CustomerTrusted]),
    )
    .unwrap()
}

/// What a native worker on this host actually advertises.
pub(crate) fn host_inventory() -> ExecutionHostInventory {
    static INVENTORY: OnceLock<ExecutionHostInventory> = OnceLock::new();
    INVENTORY
        .get_or_init(|| {
            runmat_core::RunMatSession::with_options(false, false)
                .unwrap()
                .execution_host_inventory(ExecutionTrustTier::CustomerTrusted)
                .unwrap()
        })
        .clone()
}

pub(crate) fn accelerator_class(name: &str) -> AcceleratorClass {
    AcceleratorClass::new(name).unwrap()
}

pub(crate) fn accelerator_request(class: &str, minimum_allocation_bytes: u64) -> AcceleratorRequest {
    AcceleratorRequest {
        class: accelerator_class(class),
        count: 1,
        minimum_allocation_bytes,
        provider: None,
        required_features: BTreeSet::new(),
    }
}

pub(crate) fn accelerator_device(class: &str, max_allocation_bytes: u64) -> AcceleratorDevice {
    AcceleratorDevice {
        id: AcceleratorDeviceId::new("device:runner-test").unwrap(),
        allocation_domain: AcceleratorAllocationDomainId::new("allocation:runner-test").unwrap(),
        class: accelerator_class(class),
        provider: AcceleratorProvider {
            id: AcceleratorProviderId::new("runmat.test").unwrap(),
            version: AcceleratorProviderVersion::new("1.0.0").unwrap(),
            abi_fingerprint: Digest::sha256(b"runner-test-accelerator-abi"),
        },
        max_allocation_bytes,
        features: BTreeSet::from([AcceleratorFeature::Compute]),
        inventory_epoch: 1,
    }
}

/// A minimal program task request in the given scope and pool.
pub(crate) fn task_request(
    name: &str,
    scope_id: runmat_execution::ExecutionScopeId,
    pool_id: runmat_execution::PoolId,
    resources: runmat_execution::resource::ResourceRequest,
    retry: runmat_execution::task::RetryPolicy,
) -> runmat_execution::task::TaskRequest {
    runmat_execution::task::TaskRequest {
        schema_version: runmat_execution::task::TASK_REQUEST_SCHEMA_VERSION,
        id: runmat_execution::TaskId::derive(&[name.as_bytes()]),
        scope_id,
        pool_id,
        executable: runmat_execution::TaskExecutableIdentity::ProgramArtifact(
            runmat_execution::identity::ArtifactId::derive(&[name.as_bytes()]),
        ),
        callable: runmat_execution::task::Callable::for_program(
            "test",
            &runmat_execution::ProgramCallable::semantic(
                runmat_execution::ProgramFunctionId(0),
                Some(name.into()),
            ),
        ),
        host: host_requirement(),
        invocation_context: runmat_execution::ProgramInvocationContext::Direct,
        inputs: Vec::new(),
        environment: runmat_execution::EnvironmentSnapshot::default(),
        console: None,
        outputs: runmat_execution::OutputContract {
            requested_outputs: 0,
            owned_work: None,
        },
        resources,
        retry,
        deadline_unix_millis: None,
    }
}

/// One device per requested accelerator, sized to the request.
pub(crate) fn accelerator_devices(requests: &[AcceleratorRequest]) -> Vec<AcceleratorDevice> {
    requests
        .iter()
        .flat_map(|request| (0..request.count).map(move |ordinal| (request, ordinal)))
        .enumerate()
        .map(|(index, (request, _))| {
            let mut device =
                accelerator_device(request.class.as_str(), request.minimum_allocation_bytes);
            device.id = AcceleratorDeviceId::new(format!("device:runner-test-{index}")).unwrap();
            device.allocation_domain =
                AcceleratorAllocationDomainId::new(format!("allocation:runner-test-{index}"))
                    .unwrap();
            if let Some(provider) = &request.provider {
                device.provider = provider.clone();
            }
            device.features.extend(request.required_features.iter().cloned());
            device
        })
        .collect()
}
