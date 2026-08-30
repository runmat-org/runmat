use std::collections::{BTreeMap, BTreeSet};

use runmat_types::{
    CapabilityRequirement, ForeignAdapterId, ForeignArtifactIdentity, ForeignCapability,
    InteropManifest, WasmInteropPolicy,
};

use super::{foreign_error, ForeignErrorKind, ForeignPlatform};
use crate::RuntimeError;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ForeignAdapterDescriptor {
    pub adapter: ForeignAdapterId,
    pub version: u32,
    pub capabilities: BTreeSet<CapabilityRequirement>,
    pub foreign_capabilities: BTreeSet<ForeignCapability>,
    pub artifact_identities: BTreeSet<ForeignArtifactIdentity>,
    pub supports_wasm: bool,
    pub supports_host_bridge: bool,
    pub execution_stack: runmat_types::ExecutionStackRequirement,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InteropAdmissionPlan {
    pub adapters: Vec<ForeignAdapterId>,
    pub foreign_type_count: usize,
}

pub fn admit_interop_manifest(
    manifest: &InteropManifest,
    available: &BTreeMap<ForeignAdapterId, ForeignAdapterDescriptor>,
    platform: ForeignPlatform,
) -> Result<InteropAdmissionPlan, RuntimeError> {
    manifest.validate().map_err(|error| {
        foreign_error(
            ForeignErrorKind::InvalidManifest,
            format!("{}: {}", error.path, error.message),
        )
    })?;

    for requirement in &manifest.foreign_types {
        match (platform, requirement.wasm) {
            (ForeignPlatform::Native, _) | (_, WasmInteropPolicy::Portable) => {}
            (
                ForeignPlatform::Wasm {
                    host_bridge_available: true,
                },
                WasmInteropPolicy::HostBridge,
            ) => {}
            (ForeignPlatform::Wasm { .. }, _) => {
                return Err(foreign_error(
                    ForeignErrorKind::UnsupportedOnWasm,
                    format!(
                        "foreign type {}:{} is unavailable in this browser host",
                        requirement.type_identity.family, requirement.type_identity.name
                    ),
                ));
            }
        }
        let Some(adapter) = available.get(requirement.type_identity.family.as_str()) else {
            return Err(foreign_error(
                ForeignErrorKind::AdapterUnavailable,
                format!(
                    "no adapter is registered for foreign family {}",
                    requirement.type_identity.family
                ),
            ));
        };
        if !requirement
            .capabilities
            .iter()
            .all(|capability| adapter.foreign_capabilities.contains(capability))
        {
            return Err(foreign_error(
                ForeignErrorKind::CapabilityUnavailable,
                format!(
                    "adapter {} cannot satisfy all capabilities for {}",
                    adapter.adapter, requirement.type_identity.name
                ),
            ));
        }
        match platform {
            ForeignPlatform::Native => {}
            ForeignPlatform::Wasm {
                host_bridge_available,
            } => {
                let supported = match requirement.wasm {
                    WasmInteropPolicy::Portable => adapter.supports_wasm,
                    WasmInteropPolicy::HostBridge => {
                        host_bridge_available && adapter.supports_host_bridge
                    }
                    WasmInteropPolicy::Reject => false,
                };
                if !supported {
                    return Err(foreign_error(
                        ForeignErrorKind::UnsupportedOnWasm,
                        format!("adapter {} is unavailable on wasm", adapter.adapter),
                    ));
                }
            }
        }
    }

    for requirement in &manifest.adapters {
        let Some(adapter) = available.get(&requirement.adapter) else {
            return Err(foreign_error(
                ForeignErrorKind::AdapterUnavailable,
                format!("required adapter {} is not registered", requirement.adapter),
            ));
        };
        if adapter.version < requirement.minimum_version {
            return Err(foreign_error(
                ForeignErrorKind::CapabilityUnavailable,
                format!(
                    "adapter {} version {} is older than required version {}",
                    adapter.adapter, adapter.version, requirement.minimum_version
                ),
            ));
        }
        if !requirement.capabilities.0.is_subset(&adapter.capabilities)
            || !requirement
                .artifact_identities
                .iter()
                .all(|artifact| adapter.artifact_identities.contains(artifact))
        {
            return Err(foreign_error(
                ForeignErrorKind::CapabilityUnavailable,
                format!(
                    "adapter {} cannot satisfy its manifest contract",
                    adapter.adapter
                ),
            ));
        }
    }

    Ok(InteropAdmissionPlan {
        adapters: manifest
            .adapters
            .iter()
            .map(|requirement| requirement.adapter.clone())
            .collect(),
        foreign_type_count: manifest.foreign_types.len(),
    })
}

#[cfg(test)]
mod tests {
    use runmat_types::{
        CapabilitySet, ExecutionStackRequirement, ForeignAdapterContractReference,
        ForeignAdapterRequirement, ForeignAffinity, ForeignLifetime, ForeignOwnership,
        ForeignRequirement, ForeignTypeIdentity, PlannedForeignAdapter,
        INTEROP_MANIFEST_SCHEMA_VERSION,
    };

    use super::*;

    fn adapter_id(value: &str) -> ForeignAdapterId {
        ForeignAdapterId::new(value).unwrap()
    }

    fn artifact(value: &str) -> ForeignArtifactIdentity {
        ForeignArtifactIdentity::new(value).unwrap()
    }

    fn descriptor() -> ForeignAdapterDescriptor {
        ForeignAdapterDescriptor {
            adapter: adapter_id("java"),
            version: 2,
            capabilities: BTreeSet::from([CapabilityRequirement::ForeignRuntime]),
            foreign_capabilities: BTreeSet::from([
                ForeignCapability::Invoke,
                ForeignCapability::Callback,
            ]),
            artifact_identities: BTreeSet::from([artifact("jre:21")]),
            supports_wasm: false,
            supports_host_bridge: true,
            execution_stack: runmat_types::ExecutionStackRequirement::Any,
        }
    }

    fn manifest(wasm: WasmInteropPolicy) -> InteropManifest {
        InteropManifest {
            schema_version: INTEROP_MANIFEST_SCHEMA_VERSION,
            foreign_types: vec![ForeignRequirement {
                type_identity: ForeignTypeIdentity {
                    family: "java".into(),
                    name: "java.lang.Object".into(),
                    version: 1,
                },
                ownership: ForeignOwnership::Shared,
                affinity: ForeignAffinity::OriginProcess,
                lifetime: ForeignLifetime::Session,
                capabilities: vec![ForeignCapability::Invoke],
                wasm,
            }],
            adapters: vec![ForeignAdapterRequirement {
                adapter: adapter_id("java"),
                minimum_version: 2,
                capabilities: CapabilitySet(BTreeSet::from([
                    CapabilityRequirement::ForeignRuntime,
                ])),
                execution_stack: ExecutionStackRequirement::Process,
                artifact_identities: vec![artifact("jre:21")],
            }],
            adapter_contracts: Vec::new(),
        }
    }

    #[test]
    fn admits_exact_native_adapter_and_artifact_contracts() {
        let available = BTreeMap::from([(adapter_id("java"), descriptor())]);
        let plan = admit_interop_manifest(
            &manifest(WasmInteropPolicy::Reject),
            &available,
            ForeignPlatform::Native,
        )
        .unwrap();
        assert_eq!(plan.adapters, vec![adapter_id("java")]);
        assert_eq!(plan.foreign_type_count, 1);
    }

    #[test]
    fn wasm_rejects_native_only_requirements_before_execution() {
        let available = BTreeMap::from([(adapter_id("java"), descriptor())]);
        let error = admit_interop_manifest(
            &manifest(WasmInteropPolicy::Reject),
            &available,
            ForeignPlatform::Wasm {
                host_bridge_available: true,
            },
        )
        .unwrap_err();
        assert_eq!(error.identifier(), Some("RunMat:Foreign:UnsupportedOnWasm"));
    }

    #[test]
    fn host_bridge_requires_both_host_and_adapter_support() {
        let mut adapter = descriptor();
        adapter.supports_host_bridge = false;
        let available = BTreeMap::from([(adapter_id("java"), adapter)]);
        let error = admit_interop_manifest(
            &manifest(WasmInteropPolicy::HostBridge),
            &available,
            ForeignPlatform::Wasm {
                host_bridge_available: true,
            },
        )
        .unwrap_err();
        assert_eq!(error.identifier(), Some("RunMat:Foreign:UnsupportedOnWasm"));
    }

    #[test]
    fn planned_contracts_never_enter_runtime_admission() {
        let manifest = InteropManifest {
            adapter_contracts: vec![
                ForeignAdapterContractReference::planned(PlannedForeignAdapter::DotNet),
                ForeignAdapterContractReference::planned(PlannedForeignAdapter::WindowsCom),
            ],
            ..InteropManifest::empty()
        };
        let available = BTreeMap::from([
            (
                adapter_id(PlannedForeignAdapter::DotNet.adapter_id()),
                ForeignAdapterDescriptor {
                    adapter: adapter_id(PlannedForeignAdapter::DotNet.adapter_id()),
                    ..descriptor()
                },
            ),
            (
                adapter_id(PlannedForeignAdapter::WindowsCom.adapter_id()),
                ForeignAdapterDescriptor {
                    adapter: adapter_id(PlannedForeignAdapter::WindowsCom.adapter_id()),
                    ..descriptor()
                },
            ),
        ]);
        let plan = admit_interop_manifest(&manifest, &available, ForeignPlatform::Native).unwrap();
        assert!(plan.adapters.is_empty());
        assert_eq!(plan.foreign_type_count, 0);
    }
}
