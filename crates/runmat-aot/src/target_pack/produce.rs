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

use crate::{
    archive::{member_matches_probe_target, read_native_abi_probe},
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

pub fn build_hosted_target_pack(
    runtime_archive: Vec<u8>,
    numerical_backend: Option<Vec<u8>>,
    numerical_evidence: Option<Vec<u8>>,
    declaration: HostedPackDeclaration,
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
    let runtime_capabilities = BTreeSet::from([
        CapabilityRequirement::HostRuntime,
        CapabilityRequirement::NativeCode,
    ]);
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
    let source_symbol_present = archive_defines_source_factory(&runtime_archive, &probe)?;
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
    let runtime_id = component_id("runtime.a")?;
    let probe_id = component_id("abi.probe")?;
    let numerical_id = component_id("numerical.a")?;
    let evidence_id = component_id("numerical.evidence")?;
    let mut components = BTreeMap::new();
    components.insert(probe_id.clone(), probe_bytes);
    components.insert(runtime_id.clone(), runtime_archive);
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
        let kind = if id == &probe_id {
            TargetPackComponentKind::NativeAbiProbe
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
                "application/octet-stream",
                bytes,
            )
            .map_err(invalid)?,
            dependencies: Vec::new(),
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
        build_inputs: NativeBuildInputs {
            linker: NativeBuildInput::HostedDeclared {
                fingerprint: declaration.linker_tree,
            },
            sysroot: NativeBuildInput::HostedDeclared {
                fingerprint: declaration.sysroot_tree,
            },
        },
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

fn archive_defines_source_factory(archive: &[u8], probe: &NativeAbiProbe) -> AotResult<bool> {
    let parsed = object::read::archive::ArchiveFile::parse(archive).map_err(invalid)?;
    let mut count = 0usize;
    let expected = if probe.operating_system == NativeOperatingSystem::Macos {
        "_runmat_aot_source_evaluation_component_v2"
    } else {
        "runmat_aot_source_evaluation_component_v2"
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
                .ok_or_else(|| invalid("SourceEvaluation factory has no code section"))
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
                    "SourceEvaluation factory name is not an externally linkable target-code definition",
                ));
            }
            count += 1;
            if count > 1 {
                return Err(invalid(
                    "runtime archive duplicates its SourceEvaluation factory",
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
