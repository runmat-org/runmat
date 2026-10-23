use std::collections::{BTreeMap, BTreeSet};

use object::{Object, ObjectSection, ObjectSymbol};
use runmat_execution::host::{
    native_source_evaluation_service_abi, NativeAbiProbe, NativeCpuBaseline,
    NativeDeploymentServiceKind, NativeNumericalBehavior, NativeNumericalPolicy,
    NativeOperatingSystem, NativeOptimization, NATIVE_NUMERICAL_POLICY_SCHEMA_VERSION,
};
use runmat_execution::Digest;
use runmat_execution_artifact::target_pack::{
    ComponentId, NativeBuildInput, NativeBuildInputs, NativeLinkInput, NativeLinkRetention,
    TargetPackComponent, TargetPackComponentKind, TargetPackManifest, TargetPackService,
    TargetPackServiceBehavior, TargetPackServiceKind, TARGET_PACK_SCHEMA_VERSION,
};
use runmat_execution_artifact::{ObjectDescriptor, ObjectNamespace};
use runmat_types::{CapabilityRequirement, CapabilitySet, EffectSet};

use super::tree::ProducedHostedTree;
use crate::{
    archive::{
        member_matches_probe_target, read_native_abi_probe, read_native_accelerator_profile,
    },
    AotError, AotResult,
};

/// Declaration supplied by the target-build owner. It is an inventory and
/// linkage recipe, NOT a trusted compile grant or qualified toolchain record.
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HostedPackDeclaration {
    pub schema_version: u16,
    /// Issued by target-build qualification; architecture alone cannot prove
    /// which ISA extensions the selected runtime requires.
    pub cpu_baseline: NativeCpuBaseline,
    pub numerical_behavior: NativeNumericalBehavior,
    pub optimization: NativeOptimization,
    pub available_capabilities: CapabilitySet,
    pub platform_service_abi: Digest,
    pub services: Vec<HostedPackService>,
    pub native_inputs: Vec<HostedPackNativeInput>,
    pub linker_tree: Digest,
    pub sysroot_tree: Digest,
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HostedPackService {
    pub kind: TargetPackServiceKind,
    pub abi: Digest,
    pub behavior: TargetPackServiceBehavior,
}

#[derive(serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum HostedPackNativeInput {
    Library { name: String },
    Framework { name: String },
}

pub struct ProducedTargetPack {
    pub manifest: TargetPackManifest,
    /// Raw components only. Caller publication writes these bytes in an
    /// immutable root under their canonical component IDs.
    pub components: BTreeMap<ComponentId, Vec<u8>>,
}

pub struct ContentAddressedPackBuildInputs {
    pub linker: ProducedHostedTree,
    pub sysroot: ProducedHostedTree,
    /// Exact native implementations required by the runtime link. SDK import
    /// stubs are not implementations and do not belong here.
    pub native_archives: Vec<ContentAddressedNativeArchive>,
}

pub struct ContentAddressedNativeArchive {
    pub id: ComponentId,
    pub bytes: Vec<u8>,
}

pub fn build_hosted_target_pack(
    runtime_archive: Vec<u8>,
    numerical_backend: Option<Vec<u8>>,
    numerical_evidence: Option<Vec<u8>>,
    declaration: HostedPackDeclaration,
) -> AotResult<ProducedTargetPack> {
    build_target_pack(
        runtime_archive,
        numerical_backend,
        numerical_evidence,
        declaration,
        None,
    )
}

pub fn build_content_addressed_target_pack(
    runtime_archive: Vec<u8>,
    numerical_backend: Option<Vec<u8>>,
    numerical_evidence: Option<Vec<u8>>,
    declaration: HostedPackDeclaration,
    build_inputs: ContentAddressedPackBuildInputs,
) -> AotResult<ProducedTargetPack> {
    build_target_pack(
        runtime_archive,
        numerical_backend,
        numerical_evidence,
        declaration,
        Some(build_inputs),
    )
}

fn build_target_pack(
    runtime_archive: Vec<u8>,
    numerical_backend: Option<Vec<u8>>,
    numerical_evidence: Option<Vec<u8>>,
    declaration: HostedPackDeclaration,
    build_inputs: Option<ContentAddressedPackBuildInputs>,
) -> AotResult<ProducedTargetPack> {
    if declaration.schema_version != 1
        || declaration
            .platform_service_abi
            .bytes()
            .iter()
            .all(|byte| *byte == 0)
        || declaration
            .linker_tree
            .bytes()
            .iter()
            .all(|byte| *byte == 0)
        || declaration
            .sysroot_tree
            .bytes()
            .iter()
            .all(|byte| *byte == 0)
        || declaration
            .services
            .windows(2)
            .any(|pair| pair[0].kind >= pair[1].kind)
    {
        return Err(invalid(
            "hosted pack declaration is not bounded, sorted or complete",
        ));
    }
    let mut runtime_capabilities = BTreeSet::from([
        CapabilityRequirement::HostRuntime,
        CapabilityRequirement::NativeCode,
    ]);
    let accelerator_profile = read_native_accelerator_profile(&runtime_archive)?;
    if declaration
        .available_capabilities
        .0
        .contains(&CapabilityRequirement::Accelerator)
    {
        if !accelerator_profile.has_backend() {
            return Err(invalid(
                "accelerator inventory has no target-built backend implementation",
            ));
        }
        runtime_capabilities.insert(CapabilityRequirement::Accelerator);
    }
    if !runtime_capabilities.is_subset(&declaration.available_capabilities.0) {
        return Err(invalid(
            "hosted runtime omits its native-code/host-runtime inventory",
        ));
    }
    let probe = read_native_abi_probe(&runtime_archive)?;
    let target =
        runmat_native_codegen::NativeTarget::from_probe(probe.clone(), declaration.cpu_baseline)
            .map_err(invalid)?;
    let selected_target = target.native_parts().map_err(invalid)?.0.clone();
    let abi = probe.binding().map_err(invalid)?;
    let probe_bytes = probe.canonical_bytes().to_vec();
    let source_symbol_present = archive_defines_service_factory(
        &runtime_archive,
        &probe,
        runmat_native_codegen::aot::AOT_SOURCE_SERVICE_FACTORY_SYMBOL,
    )?;
    let source_declared = declaration
        .services
        .iter()
        .any(|service| service.kind == NativeDeploymentServiceKind::SourceEvaluation);
    if source_declared != source_symbol_present {
        return Err(invalid(
            "SourceEvaluation inventory differs from actual target-built runtime symbol",
        ));
    }
    if declaration.services.iter().any(|service| {
        service.kind == NativeDeploymentServiceKind::SourceEvaluation
            && service.abi != native_source_evaluation_service_abi()
    }) {
        return Err(invalid(
            "SourceEvaluation factory ABI differs from the canonical linked ABI",
        ));
    }
    if declaration.services.iter().any(|service| {
        service.kind == NativeDeploymentServiceKind::Effect(runmat_types::EffectKind::Clock)
    }) && (!archive_defines_service_factory(
        &runtime_archive,
        &probe,
        runmat_native_codegen::aot::AOT_CLOCK_SERVICE_FACTORY_SYMBOL,
    )? || declaration.services.iter().any(|service| {
        service.kind == NativeDeploymentServiceKind::Effect(runmat_types::EffectKind::Clock)
            && service.abi != runmat_execution::host::native_clock_service_abi()
    })) {
        return Err(invalid(
            "Clock inventory lacks its exact target-built linked factory/ABI",
        ));
    }
    let local_declared = declaration
        .services
        .iter()
        .any(|service| service.kind.is_local_effect_service());
    if local_declared
        && (!archive_defines_service_factory(
            &runtime_archive,
            &probe,
            runmat_native_codegen::aot::AOT_LOCAL_SERVICE_FACTORY_SYMBOL,
        )? || declaration.services.iter().any(|service| {
            service.kind.is_local_effect_service()
                && service.abi != runmat_execution::host::native_local_service_abi()
        }))
    {
        return Err(invalid(
            "local effect inventory lacks its exact target-built linked factory/ABI",
        ));
    }
    for (kind, symbol, abi) in [
        (
            NativeDeploymentServiceKind::Effect(runmat_types::EffectKind::UserInterface),
            runmat_native_codegen::aot::AOT_UI_SERVICE_FACTORY_SYMBOL,
            runmat_execution::host::native_ui_service_abi(),
        ),
        (
            NativeDeploymentServiceKind::Effect(runmat_types::EffectKind::HostCallback),
            runmat_native_codegen::aot::AOT_HOST_CALLBACK_FACTORY_SYMBOL,
            runmat_execution::host::native_host_callback_service_abi(),
        ),
        (
            NativeDeploymentServiceKind::HostedDynamicDispatch,
            runmat_native_codegen::aot::AOT_HOSTED_DYNAMIC_FACTORY_SYMBOL,
            runmat_execution::host::native_hosted_dynamic_dispatch_abi(),
        ),
    ] {
        if let Some(service) = declaration
            .services
            .iter()
            .find(|service| service.kind == kind)
        {
            if service.behavior
                != runmat_execution::host::NativeDeploymentServiceBehavior::HostedAmbient
                || service.abi != abi
                || !archive_defines_service_factory(&runtime_archive, &probe, symbol)?
            {
                return Err(invalid(
                    "linked external service lacks its qualified target-built factory/ABI",
                ));
            }
        }
    }
    let network_declared = declaration.services.iter().any(|service| {
        service.kind == NativeDeploymentServiceKind::Effect(runmat_types::EffectKind::Network)
    });
    if network_declared
        && (!archive_defines_service_factory(
            &runtime_archive,
            &probe,
            runmat_native_codegen::aot::AOT_NETWORK_SERVICE_FACTORY_SYMBOL,
        )? || declaration.services.iter().any(|service| {
            service.kind == NativeDeploymentServiceKind::Effect(runmat_types::EffectKind::Network)
                && (service.abi != runmat_execution::host::native_network_service_abi()
                    || service.behavior
                        != runmat_execution::host::NativeDeploymentServiceBehavior::HostedAmbient)
        }))
    {
        return Err(invalid(
            "Network inventory lacks its exact target-built linked factory/ABI",
        ));
    }
    let runtime_id = component_id("runtime.a")?;
    let notice_id = component_id(runmat_runtime::third_party_notices::FILE_NAME)?;
    let probe_id = component_id("abi.probe")?;
    let numerical_id = component_id("numerical.a")?;
    let evidence_id = component_id("numerical.evidence")?;
    let mut components = BTreeMap::new();
    components.insert(probe_id.clone(), probe_bytes);
    components.insert(runtime_id.clone(), runtime_archive);
    components.insert(
        notice_id.clone(),
        runmat_runtime::third_party_notices::bytes().to_vec(),
    );
    let backend_id = if let Some(bytes) = numerical_backend {
        components.insert(numerical_id.clone(), bytes);
        numerical_id
    } else {
        runtime_id.clone()
    };
    let evidence_component = if let Some(bytes) = numerical_evidence {
        components.insert(evidence_id.clone(), bytes);
        Some(evidence_id)
    } else {
        None
    };
    let mut component_kinds = BTreeMap::new();
    let mut component_dependencies = BTreeMap::<ComponentId, Vec<ComponentId>>::new();
    let selected_build_inputs = match build_inputs {
        None => NativeBuildInputs {
            linker: NativeBuildInput::HostedDeclared {
                fingerprint: declaration.linker_tree,
            },
            sysroot: NativeBuildInput::HostedDeclared {
                fingerprint: declaration.sysroot_tree,
            },
        },
        Some(inputs) => {
            if selected_target.target.operating_system != NativeOperatingSystem::Linux
                || !declaration.native_inputs.is_empty()
                || inputs.native_archives.is_empty()
            {
                return Err(invalid(
                    "content-addressed production currently requires the designated Linux inventory gate, explicit native implementation archives, and no ambient libraries/frameworks; final loader closure remains separately qualified",
                ));
            }
            let linker = append_build_tree(
                "linker.tree",
                TargetPackComponentKind::Toolchain,
                declaration.linker_tree,
                inputs.linker,
                &mut components,
                &mut component_kinds,
                &mut component_dependencies,
            )?;
            let sysroot = append_build_tree(
                "sysroot.tree",
                TargetPackComponentKind::Sysroot,
                declaration.sysroot_tree,
                inputs.sysroot,
                &mut components,
                &mut component_kinds,
                &mut component_dependencies,
            )?;
            let mut previous = None;
            for archive in inputs.native_archives {
                if previous.as_ref().is_some_and(|id| id >= &archive.id)
                    || components.contains_key(&archive.id)
                    || archive.bytes.is_empty()
                {
                    return Err(invalid(
                        "content-addressed native implementation archives must be nonempty, sorted and unique",
                    ));
                }
                previous = Some(archive.id.clone());
                component_kinds.insert(archive.id.clone(), TargetPackComponentKind::StaticArchive);
                component_dependencies
                    .entry(runtime_id.clone())
                    .or_default()
                    .push(archive.id.clone());
                components.insert(archive.id, archive.bytes);
            }
            NativeBuildInputs {
                linker: NativeBuildInput::ContentAddressed { component: linker },
                sysroot: NativeBuildInput::ContentAddressed { component: sysroot },
            }
        }
    };
    let backend = Digest::sha256(&components[&backend_id]);
    let numerical = NativeNumericalPolicy {
        schema_version: NATIVE_NUMERICAL_POLICY_SCHEMA_VERSION,
        backend,
        behavior: declaration.numerical_behavior,
    };
    numerical.validate().map_err(invalid)?;
    if numerical.required_evidence().is_some() != evidence_component.is_some()
        || numerical.required_evidence().is_some_and(|digest| {
            evidence_component
                .as_ref()
                .is_none_or(|id| Digest::sha256(&components[id]) != digest)
        })
    {
        return Err(invalid(
            "numerical qualification evidence bytes are missing or mismatched",
        ));
    }
    let mut descriptors = Vec::new();
    for (id, bytes) in &components {
        let kind = if let Some(kind) = component_kinds.get(id) {
            *kind
        } else if id == &probe_id {
            TargetPackComponentKind::NativeAbiProbe
        } else if id == &notice_id {
            TargetPackComponentKind::RuntimeData
        } else if Some(id) == evidence_component.as_ref() {
            TargetPackComponentKind::NumericalEvidence
        } else {
            TargetPackComponentKind::StaticArchive
        };
        let capabilities = if kind == TargetPackComponentKind::StaticArchive {
            CapabilitySet(runtime_capabilities.clone())
        } else {
            CapabilitySet::default()
        };
        descriptors.push(TargetPackComponent {
            id: id.clone(),
            kind,
            object: ObjectDescriptor::new(
                ObjectNamespace::ProgramArtifact,
                id.as_str(),
                if id == &notice_id {
                    "text/plain;charset=utf-8"
                } else {
                    "application/octet-stream"
                },
                bytes,
            )
            .map_err(invalid)?,
            dependencies: if let Some(dependencies) = component_dependencies.get(id) {
                let mut dependencies = dependencies.clone();
                if id == &runtime_id {
                    dependencies.push(notice_id.clone());
                    dependencies.sort();
                }
                dependencies
            } else if id == &runtime_id {
                vec![notice_id.clone()]
            } else {
                Vec::new()
            },
            capabilities,
            effects: EffectSet::default(),
        });
    }
    let mut link_inputs = vec![NativeLinkInput::Component {
        component: runtime_id.clone(),
        retention: NativeLinkRetention::Referenced,
    }];
    if backend_id != runtime_id {
        link_inputs.push(NativeLinkInput::Component {
            component: backend_id.clone(),
            retention: NativeLinkRetention::Referenced,
        });
    }
    for id in component_dependencies
        .get(&runtime_id)
        .into_iter()
        .flatten()
    {
        link_inputs.push(NativeLinkInput::Component {
            component: id.clone(),
            retention: NativeLinkRetention::Referenced,
        });
    }
    for input in &declaration.native_inputs {
        link_inputs.push(match input {
            HostedPackNativeInput::Library { name } => NativeLinkInput::HostedLibrary {
                required_by: runtime_id.clone(),
                name: name.clone(),
            },
            HostedPackNativeInput::Framework { name } => NativeLinkInput::HostedFramework {
                required_by: runtime_id.clone(),
                name: name.clone(),
            },
        });
    }
    let environment = runmat_core::program_environment(runmat_core::CompatMode::Matlab);
    let manifest = TargetPackManifest {
        schema_version: TARGET_PACK_SCHEMA_VERSION,
        target: selected_target,
        abi,
        abi_probe_component: probe_id,
        runtime: environment.runtime_fingerprint,
        catalog: environment.catalog_fingerprint,
        numerical,
        numerical_backend_component: backend_id,
        numerical_evidence_component: evidence_component,
        optimization: declaration.optimization,
        capabilities: declaration.available_capabilities,
        runtime_component: runtime_id.clone(),
        components: descriptors,
        link_inputs,
        build_inputs: selected_build_inputs,
        platform_service_abi: declaration.platform_service_abi,
        services: declaration
            .services
            .into_iter()
            .map(|service| TargetPackService {
                kind: service.kind,
                abi: service.abi,
                component: runtime_id.clone(),
                behavior: service.behavior,
            })
            .collect(),
    };
    manifest.validate().map_err(invalid)?;
    Ok(ProducedTargetPack {
        manifest,
        components,
    })
}

fn append_build_tree(
    name: &str,
    kind: TargetPackComponentKind,
    expected: Digest,
    produced: ProducedHostedTree,
    components: &mut BTreeMap<ComponentId, Vec<u8>>,
    kinds: &mut BTreeMap<ComponentId, TargetPackComponentKind>,
    dependencies: &mut BTreeMap<ComponentId, Vec<ComponentId>>,
) -> AotResult<ComponentId> {
    if produced.digest != expected {
        return Err(invalid(format!(
            "{name} differs from the independently declared tree identity"
        )));
    }
    let tree = produced.validated_tree()?;
    let tree_id = component_id(name)?;
    if components.contains_key(&tree_id) {
        return Err(invalid(
            "build-tree component identity collides with pack data",
        ));
    }
    let refs = tree
        .blobs
        .iter()
        .map(|entry| entry.component.clone())
        .collect::<Vec<_>>();
    for (id, bytes) in produced.blobs {
        if let Some(previous) = components.get(&id) {
            if !kinds.contains_key(&id) || previous != &bytes {
                return Err(invalid("content-addressed build-tree component collision"));
            }
        } else {
            components.insert(id.clone(), bytes);
            kinds.insert(id, TargetPackComponentKind::RuntimeData);
        }
    }
    components.insert(tree_id.clone(), produced.metadata);
    kinds.insert(tree_id.clone(), kind);
    dependencies.insert(tree_id.clone(), refs);
    Ok(tree_id)
}

fn archive_defines_service_factory(
    archive: &[u8],
    probe: &NativeAbiProbe,
    name: &str,
) -> AotResult<bool> {
    let parsed = object::read::archive::ArchiveFile::parse(archive).map_err(invalid)?;
    let mut count = 0usize;
    let expected = if probe.operating_system == NativeOperatingSystem::Macos {
        format!("_{name}")
    } else {
        name.to_string()
    };
    for (index, member) in parsed.members().enumerate() {
        if index >= 65_536 {
            return Err(invalid("runtime archive member count exceeds probe bound"));
        }
        let member = member.map_err(invalid)?;
        let bytes = member.data(archive).map_err(invalid)?;
        let Ok(image) = object::File::parse(bytes) else {
            continue;
        };
        for symbol in image.symbols() {
            if symbol.is_undefined() || symbol.name().map_err(invalid)? != expected {
                continue;
            }
            let section = symbol
                .section_index()
                .ok_or_else(|| invalid("selected service factory has no code section"))
                .and_then(|index| image.section_by_index(index).map_err(invalid))?;
            if !member_matches_probe_target(&image, probe)
                || !symbol.is_definition()
                || symbol.kind() != object::SymbolKind::Text
                || !matches!(
                    symbol.scope(),
                    object::SymbolScope::Linkage | object::SymbolScope::Dynamic
                )
                || symbol.is_weak()
                || section.kind() != object::SectionKind::Text
            {
                return Err(invalid(
                    "selected service factory name is not an externally linkable target-code definition",
                ));
            }
            count += 1;
            if count > 1 {
                return Err(invalid(
                    "runtime archive duplicates its selected service factory",
                ));
            }
        }
    }
    Ok(count == 1)
}

fn component_id(name: &str) -> AotResult<ComponentId> {
    ComponentId::new(name).map_err(invalid)
}

fn invalid(error: impl std::fmt::Display) -> AotError {
    AotError::contract("aot.target_pack.produce", error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_execution::host::{NativeAcceleratorProfile, NativeArchitecture};
    use runmat_execution_artifact::target_pack::{TargetPackProfile, TargetPackRequirements};

    #[test]
    fn selected_runtime_retains_canonical_notice_as_runtime_data() {
        let probe = runmat_runtime::native::native_abi_probe();
        let cpu_baseline = match probe.architecture {
            NativeArchitecture::X86_64 => runmat_execution::host::NativeCpuBaseline::X86_64V1,
            NativeArchitecture::Aarch64 => runmat_execution::host::NativeCpuBaseline::ArmV8A,
        };
        let capabilities = CapabilitySet(BTreeSet::from([
            CapabilityRequirement::HostRuntime,
            CapabilityRequirement::NativeCode,
        ]));
        let produced = build_hosted_target_pack(
            crate::archive::test_archive::archive(NativeAcceleratorProfile::NONE),
            None,
            None,
            HostedPackDeclaration {
                schema_version: 1,
                cpu_baseline,
                numerical_behavior: NativeNumericalBehavior::PlatformAccurate,
                optimization: NativeOptimization::Speed,
                available_capabilities: capabilities.clone(),
                platform_service_abi: Digest::sha256("platform services"),
                services: Vec::new(),
                native_inputs: Vec::new(),
                linker_tree: Digest::sha256("linker"),
                sysroot_tree: Digest::sha256("sysroot"),
            },
        )
        .unwrap();
        let notice_id = component_id(runmat_runtime::third_party_notices::FILE_NAME).unwrap();
        assert_eq!(
            produced.components[&notice_id],
            runmat_runtime::third_party_notices::bytes()
        );
        let notice = produced
            .manifest
            .components
            .iter()
            .find(|component| component.id == notice_id)
            .unwrap();
        assert_eq!(notice.kind, TargetPackComponentKind::RuntimeData);
        assert_eq!(notice.object.media_type, "text/plain;charset=utf-8");
        let runtime = produced
            .manifest
            .components
            .iter()
            .find(|component| component.id == produced.manifest.runtime_component)
            .unwrap();
        assert_eq!(runtime.dependencies, vec![notice_id.clone()]);
        let required = TargetPackRequirements {
            profile: TargetPackProfile::HostedNative,
            target: produced.manifest.target.clone(),
            abi: produced.manifest.abi.clone(),
            runtime: produced.manifest.runtime,
            catalog: produced.manifest.catalog,
            numerical: produced.manifest.numerical.clone(),
            optimization: produced.manifest.optimization,
            reached_capabilities: CapabilitySet::default(),
            authorized_runtime_capabilities: capabilities,
            effects: EffectSet::default(),
            required_services: BTreeSet::new(),
            components: Vec::new(),
            platform_service_abi: produced.manifest.platform_service_abi,
            service_abis: BTreeMap::new(),
        };
        let selected = produced.manifest.select(&required, None).unwrap();
        assert!(selected.runtime_components().contains(&notice_id));
        produced
            .manifest
            .verify_selected_bytes(&selected, &produced.components)
            .unwrap();
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn deterministic_pack_uses_exact_tree_components_without_hosted_inputs() {
        use crate::target_pack::{build_hosted_tree, HostedTreeEntry};
        use crate::tree::NativeStorageGrant;

        let probe = runmat_runtime::native::native_abi_probe();
        let cpu_baseline = match probe.architecture {
            NativeArchitecture::X86_64 => runmat_execution::host::NativeCpuBaseline::X86_64V1,
            NativeArchitecture::Aarch64 => runmat_execution::host::NativeCpuBaseline::ArmV8A,
        };
        let grant = NativeStorageGrant {
            archive: runmat_package_cache::ArchiveLimits::default(),
            metadata_bytes: 1024 * 1024,
        };
        let tree = |path: &str, bytes: &[u8]| {
            build_hosted_tree(
                vec![HostedTreeEntry::File {
                    path: runmat_package::NormalizedRelativePath::new(path).unwrap(),
                    bytes: bytes.to_vec(),
                    executable: true,
                }],
                grant,
            )
            .unwrap()
        };
        let linker = tree("bin/linker", b"qualified linker bytes");
        let linker_digest = linker.digest;
        let sysroot = tree("lib/crt.o", b"qualified sysroot bytes");
        let sysroot_digest = sysroot.digest;
        let capabilities = CapabilitySet(BTreeSet::from([
            CapabilityRequirement::HostRuntime,
            CapabilityRequirement::NativeCode,
        ]));
        let produced = build_content_addressed_target_pack(
            crate::archive::test_archive::archive(NativeAcceleratorProfile::NONE),
            None,
            None,
            HostedPackDeclaration {
                schema_version: 1,
                cpu_baseline,
                numerical_behavior: NativeNumericalBehavior::PlatformAccurate,
                optimization: NativeOptimization::Speed,
                available_capabilities: capabilities.clone(),
                platform_service_abi: Digest::sha256("platform services"),
                services: Vec::new(),
                native_inputs: Vec::new(),
                linker_tree: linker_digest,
                sysroot_tree: sysroot_digest,
            },
            ContentAddressedPackBuildInputs {
                linker,
                sysroot,
                native_archives: vec![ContentAddressedNativeArchive {
                    id: component_id("libc.a").unwrap(),
                    bytes: b"qualified static native implementation".to_vec(),
                }],
            },
        )
        .unwrap();
        assert!(matches!(
            produced.manifest.build_inputs.linker,
            NativeBuildInput::ContentAddressed { .. }
        ));
        assert!(matches!(
            produced.manifest.build_inputs.sysroot,
            NativeBuildInput::ContentAddressed { .. }
        ));
        assert!(produced
            .manifest
            .link_inputs
            .iter()
            .all(|input| matches!(input, NativeLinkInput::Component { .. })));
        produced
            .manifest
            .verify_selected_bytes(
                &produced
                    .manifest
                    .select(
                        &TargetPackRequirements {
                            profile: TargetPackProfile::DeterministicDeployed,
                            target: produced.manifest.target.clone(),
                            abi: produced.manifest.abi.clone(),
                            runtime: produced.manifest.runtime,
                            catalog: produced.manifest.catalog,
                            numerical: produced.manifest.numerical.clone(),
                            optimization: produced.manifest.optimization,
                            reached_capabilities: CapabilitySet::default(),
                            authorized_runtime_capabilities: capabilities,
                            effects: EffectSet::default(),
                            required_services: BTreeSet::new(),
                            components: Vec::new(),
                            platform_service_abi: produced.manifest.platform_service_abi,
                            service_abis: BTreeMap::new(),
                        },
                        None,
                    )
                    .unwrap(),
                &produced.components,
            )
            .unwrap();
    }
}
