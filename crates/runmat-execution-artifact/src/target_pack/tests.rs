use super::*;
use crate::{ObjectDescriptor, ObjectNamespace};
use runmat_execution::host::*;
use runmat_types::{CapabilitySet, EffectKind, EffectSet};
use std::collections::{BTreeMap, BTreeSet};
#[path = "tests/product.rs"]
mod product;
#[path = "tests/target_test.rs"]
mod target_test;

fn id(name: &str) -> ComponentId {
    ComponentId::new(name).unwrap()
}

#[test]
fn source_evaluation_requires_exact_service_bytes_and_independent_host_grant() {
    let (mut manifest, mut required, contents) = fixture();
    let kind = TargetPackServiceKind::SourceEvaluation;
    let abi = runmat_execution::host::native_source_evaluation_service_abi();
    required.required_services.insert(kind);
    assert!(manifest.select(&required, None).is_err());

    let host = runmat_types::CapabilityRequirement::HostRuntime;
    required.reached_capabilities.0.insert(host);
    manifest.capabilities.0.insert(host);
    required.authorized_runtime_capabilities.0.insert(host);
    manifest.services.push(TargetPackService {
        kind,
        abi,
        component: id("runtime"),
        behavior: TargetPackServiceBehavior::DeterministicDeclared,
    });
    assert!(
        manifest.select(&required, None).is_err(),
        "ABI is not granted by inventory"
    );
    required.service_abis.insert(kind, abi);
    let mut denied = required.clone();
    denied.authorized_runtime_capabilities.0.remove(&host);
    assert!(
        manifest.select(&denied, None).is_err(),
        "caller grant is mandatory"
    );

    let selected = manifest.select(&required, None).unwrap();
    assert!(selected.runtime_components().contains(&id("runtime")));
    let retained = selected
        .components()
        .iter()
        .map(|id| (id.clone(), contents[id].clone()))
        .collect();
    let binding = manifest
        .verify_selected_bytes(&selected, &retained)
        .unwrap()
        .selected_product_binding()
        .unwrap();
    assert_eq!(binding.required_services, BTreeSet::from([kind]));
    assert_eq!(binding.services.len(), 1);
    assert_eq!(
        binding.services[0].implementation,
        Digest::sha256(b"runtime")
    );
    let mut missing = binding.clone();
    missing.services.clear();
    assert!(
        missing.validate().is_err(),
        "selected binding cannot omit service"
    );
}

#[test]
fn authenticated_dependency_projection_reuses_scc_closure_without_unselected_roots() {
    let (manifest, required, mut contents) = fixture();
    let selected = manifest.select(&required, None).unwrap();
    contents.retain(|id, _| selected.components().contains(id));
    let verified = manifest
        .verify_selected_bytes(&selected, &contents)
        .unwrap();
    let closure = verified
        .component_dependency_closure(&id("runtime"))
        .unwrap();
    assert_eq!(
        closure.into_iter().collect::<Vec<_>>(),
        [id("a"), id("b"), id("runtime")]
    );
    assert_eq!(
        verified
            .component_dependency_closure(&id("linker"))
            .unwrap()
            .into_iter()
            .collect::<Vec<_>>(),
        [id("linker")]
    );
    assert!(verified
        .component_dependency_closure(&id("unused"))
        .is_err());
}
fn fixture() -> (
    TargetPackManifest,
    TargetPackRequirements,
    BTreeMap<ComponentId, Vec<u8>>,
) {
    // Synthetic portable metadata only: this is not an actual target-built
    // runtime probe, provider qualification, or executable admission.
    let mut layouts = [NativeProbeTypeLayout {
        size: 64,
        alignment: 8,
    }; NativeAbiType::ALL.len()];
    // This synthetic aggregate leaves three disjoint pointer fields in the
    // synthetic 64-byte binding owner; it is not a target-layout assertion.
    layouts[NativeAbiType::BuiltinBindingIdentity as usize].size = 32;
    let probe = NativeAbiProbe::from_target_layout(
        NativeProbeTarget {
            architecture: NativeArchitecture::Aarch64,
            operating_system: NativeOperatingSystem::Linux,
            environment: NativeEnvironment::Gnu,
            endianness: NativeEndianness::Little,
            pointer_width: 64,
        },
        NativeProbeContract {
            schema_version: 1,
            encoded_version: 1,
            preimage: b"fixture contract",
        },
        layouts,
        [
            0, 8, 0, 0, 0, 8, 0, 4, 8, 12, 16, 20, 0, 4, 8, 12, 0, 32, 40, 48,
        ],
    );
    let abi = probe.binding().unwrap();
    let target = SelectedNativeTarget {
        schema_version: SELECTED_NATIVE_TARGET_SCHEMA_VERSION,
        target: NativeTargetIdentity::new(
            NativeArchitecture::Aarch64,
            NativeOperatingSystem::Linux,
            64,
            abi.identity().unwrap(),
            NativeObjectFormat::Elf,
        )
        .unwrap(),
        cpu_baseline: NativeCpuBaseline::ArmV8A,
        endianness: NativeEndianness::Little,
        environment: NativeEnvironment::Gnu,
    };
    let mut contents = BTreeMap::new();
    let mut components = Vec::new();
    for (name, kind, dependencies) in [
        ("a", TargetPackComponentKind::StaticArchive, vec![id("b")]),
        ("abi-probe", TargetPackComponentKind::NativeAbiProbe, vec![]),
        ("b", TargetPackComponentKind::StaticArchive, vec![id("a")]),
        ("clock", TargetPackComponentKind::StaticArchive, vec![]),
        (
            "evidence",
            TargetPackComponentKind::NumericalEvidence,
            vec![],
        ),
        ("linker", TargetPackComponentKind::Toolchain, vec![]),
        (
            "runtime",
            TargetPackComponentKind::StaticArchive,
            vec![id("a")],
        ),
        ("sysroot", TargetPackComponentKind::Sysroot, vec![]),
        ("unused", TargetPackComponentKind::StaticArchive, vec![]),
    ] {
        let bytes = if name == "abi-probe" {
            probe.canonical_bytes().to_vec()
        } else {
            name.as_bytes().to_vec()
        };
        let object = ObjectDescriptor::new(
            ObjectNamespace::ProgramArtifact,
            name,
            "application/octet-stream",
            &bytes,
        )
        .unwrap();
        contents.insert(id(name), bytes);
        components.push(TargetPackComponent {
            id: id(name),
            kind,
            object,
            dependencies,
            capabilities: CapabilitySet::default(),
            effects: EffectSet::default(),
        });
    }
    let numerical = NativeNumericalPolicy {
        schema_version: NATIVE_NUMERICAL_POLICY_SCHEMA_VERSION,
        backend: Digest::sha256(b"runtime"),
        behavior: NativeNumericalBehavior::PlatformAccurate,
    };
    let service_abi = Digest::sha256(b"service ABI");
    let manifest = TargetPackManifest {
        schema_version: TARGET_PACK_SCHEMA_VERSION,
        target: target.clone(),
        abi: abi.clone(),
        abi_probe_component: id("abi-probe"),
        runtime: Digest::sha256(b"runtime semantics"),
        catalog: Digest::sha256(b"catalog"),
        numerical: numerical.clone(),
        numerical_backend_component: id("runtime"),
        numerical_evidence_component: None,
        optimization: NativeOptimization::Speed,
        capabilities: CapabilitySet::default(),
        runtime_component: id("runtime"),
        components,
        // Repeated archive order is meaningful for a mutual dependency SCC.
        link_inputs: ["runtime", "a", "b", "a", "clock", "unused"]
            .into_iter()
            .map(|name| NativeLinkInput::Component {
                component: id(name),
                retention: NativeLinkRetention::Referenced,
            })
            .collect(),
        build_inputs: NativeBuildInputs {
            linker: NativeBuildInput::ContentAddressed {
                component: id("linker"),
            },
            sysroot: NativeBuildInput::ContentAddressed {
                component: id("sysroot"),
            },
        },
        platform_service_abi: service_abi,
        services: vec![TargetPackService {
            kind: TargetPackServiceKind::Effect(EffectKind::Clock),
            abi: service_abi,
            component: id("clock"),
            behavior: TargetPackServiceBehavior::DeterministicDeclared,
        }],
    };
    let required = TargetPackRequirements {
        profile: TargetPackProfile::DeterministicDeployed,
        target,
        abi,
        runtime: manifest.runtime,
        catalog: manifest.catalog,
        numerical,
        optimization: NativeOptimization::Speed,
        reached_capabilities: CapabilitySet::default(),
        authorized_runtime_capabilities: CapabilitySet::default(),
        effects: EffectSet::default(),
        required_services: BTreeSet::new(),
        components: vec![id("a")],
        platform_service_abi: service_abi,
        service_abis: [(
            TargetPackServiceKind::Effect(EffectKind::Clock),
            service_abi,
        )]
        .into_iter()
        .collect(),
    };
    (manifest, required, contents)
}

#[test]
fn canonical_pack_identity_reachability_cycles_link_order_and_exact_bytes() {
    let (manifest, required, contents) = fixture();
    assert_eq!(
        manifest.target.triple().unwrap(),
        "aarch64-unknown-linux-gnu"
    );
    let bytes = manifest.canonical_bytes().unwrap();
    assert_eq!(
        TargetPackManifest::from_canonical_bytes(&bytes).unwrap(),
        manifest
    );
    let mut padded = bytes.clone();
    padded.push(b' ');
    assert!(TargetPackManifest::from_canonical_bytes(&padded).is_err());
    let selected = manifest.select(&required, None).unwrap();
    assert_eq!(
        selected.runtime_components(),
        &[id("a"), id("b"), id("runtime")]
    );
    assert_eq!(
        selected.components(),
        &[
            id("a"),
            id("abi-probe"),
            id("b"),
            id("linker"),
            id("runtime"),
            id("sysroot")
        ]
    );
    let ordered = selected
        .link_inputs()
        .iter()
        .map(|entry| entry.owner().clone())
        .collect::<Vec<_>>();
    assert_eq!(ordered, [id("runtime"), id("a"), id("b"), id("a")]);
    let mut retained = selected
        .components()
        .iter()
        .map(|id| (id.clone(), contents[id].clone()))
        .collect::<BTreeMap<_, _>>();
    assert_eq!(
        manifest
            .verify_selected_bytes(&selected, &retained)
            .unwrap()
            .component_bytes(&id("runtime")),
        Some(&b"runtime"[..])
    );
    product::check_selected_binding(&manifest, &required, &selected, &retained);
    retained.get_mut(&id("a")).unwrap().push(0);
    assert!(manifest
        .verify_selected_bytes(&selected, &retained)
        .is_err());
    for mutation in 0..4 {
        let mut changed = manifest.clone();
        match mutation {
            0 => changed.runtime = Digest::sha256(b"other runtime"),
            1 => changed.optimization = NativeOptimization::SpeedAndSize,
            2 => changed.target.environment = NativeEnvironment::Musl,
            _ => changed.link_inputs.swap(0, 1),
        }
        assert_ne!(manifest.id().unwrap(), changed.id().unwrap());
        if mutation != 3 {
            assert!(changed.select(&required, None).is_err());
        }
    }
    let mut missing = manifest.clone();
    missing.components[0].dependencies = vec![id("absent")];
    assert!(missing.validate().is_err());
    let mut unsorted = manifest.clone();
    unsorted.components.swap(0, 1);
    assert!(unsorted.validate().is_err());
    let mut unknown = manifest.clone();
    unknown.schema_version += 1;
    assert!(unknown.canonical_bytes().is_err());

    product::check_probe_rejection();
}

#[test]
fn profiles_require_explicit_reached_services_and_reject_ambient_or_unknown_inputs() {
    // Suspension/throwing are driver control, not external service factories.
    // Neither can stand in for an independently reached clock service.
    let (mut control_manifest, mut control_required, _) = fixture();
    control_required
        .effects
        .0
        .extend([EffectKind::MaySuspend, EffectKind::MayThrow]);
    control_required.service_abis.clear();
    control_manifest.services.clear();
    control_manifest.select(&control_required, None).unwrap();
    for effect in [EffectKind::MaySuspend, EffectKind::MayThrow] {
        assert!(TargetPackServiceKind::Effect(effect).validate().is_err());
    }
    control_required.effects.0.insert(EffectKind::Clock);
    assert!(control_manifest.select(&control_required, None).is_err());

    let (mut manifest, mut required, contents) = fixture();
    required.effects.0.insert(EffectKind::Clock);
    let selected = manifest.select(&required, None).unwrap();
    assert!(selected.components().contains(&id("clock")));
    let retained = selected
        .components()
        .iter()
        .map(|id| (id.clone(), contents[id].clone()))
        .collect();
    let binding = manifest
        .verify_selected_bytes(&selected, &retained)
        .unwrap()
        .selected_product_binding()
        .unwrap();
    assert_eq!(binding.services.len(), 1);
    assert_eq!(
        binding.services[0].kind,
        TargetPackServiceKind::Effect(EffectKind::Clock)
    );
    assert_eq!(binding.services[0].implementation, Digest::sha256(b"clock"));
    manifest.services[0].behavior = TargetPackServiceBehavior::HostedAmbient;
    assert!(manifest.select(&required, None).is_err());
    required.profile = TargetPackProfile::HostedNative;
    manifest.select(&required, None).unwrap();
    manifest.build_inputs.sysroot = NativeBuildInput::HostedDeclared {
        fingerprint: Digest::sha256(b"nonhermetic host sysroot"),
    };
    manifest.link_inputs.push(NativeLinkInput::HostedLibrary {
        required_by: id("runtime"),
        name: "m".into(),
    });
    manifest.select(&required, None).unwrap();
    required.profile = TargetPackProfile::DeterministicDeployed;
    assert!(manifest.select(&required, None).is_err());
    required.profile = TargetPackProfile::HostedNative;
    required.service_abis.clear();
    assert!(manifest.select(&required, None).is_err());
    required.effects.0.insert(EffectKind::Unknown);
    assert!(manifest.select(&required, None).is_err());

    // Hosted unknown calls retain a specific authenticated dispatch service,
    // not an effect wildcard or a grant inferred from pack availability.
    let (mut manifest, mut required, _) = fixture();
    required.profile = TargetPackProfile::HostedNative;
    required.effects.0.insert(EffectKind::Unknown);
    let dispatch = TargetPackServiceKind::HostedDynamicDispatch;
    let abi = Digest::sha256(b"hosted dynamic dispatch ABI");
    manifest.services.push(TargetPackService {
        kind: dispatch,
        abi,
        component: id("clock"),
        behavior: TargetPackServiceBehavior::HostedAmbient,
    });
    let runtime_capabilities = CapabilitySet(
        [
            runmat_types::CapabilityRequirement::HostRuntime,
            runmat_types::CapabilityRequirement::NativeCode,
        ]
        .into_iter()
        .collect(),
    );
    let component = manifest
        .components
        .iter_mut()
        .find(|entry| entry.id == id("clock"))
        .unwrap();
    component.dependencies = vec![id("unused")];
    component.capabilities = runtime_capabilities.clone();
    manifest.capabilities = runtime_capabilities.clone();
    required.authorized_runtime_capabilities = runtime_capabilities;
    assert!(
        manifest.select(&required, None).is_err(),
        "missing trusted dispatch ABI"
    );
    required.service_abis.insert(dispatch, abi);
    let selected = manifest.select(&required, None).unwrap();
    assert!(selected.runtime_components().contains(&id("clock")));
    assert!(selected.runtime_components().contains(&id("unused")));
    assert_eq!(
        selected.authorized_runtime_capabilities(),
        &required.authorized_runtime_capabilities
    );
    assert_eq!(
        TargetPackManifest::from_canonical_bytes(&manifest.canonical_bytes().unwrap()).unwrap(),
        manifest
    );
    let mut denied = required.clone();
    denied.authorized_runtime_capabilities = CapabilitySet::default();
    assert!(
        manifest.select(&denied, None).is_err(),
        "inventory must not grant runtime capabilities"
    );
    denied = required.clone();
    denied.profile = TargetPackProfile::DeterministicDeployed;
    assert!(
        manifest.select(&denied, None).is_err(),
        "unknown remains non-deterministic even with a dispatch ABI"
    );
    let mut missing = manifest.clone();
    missing
        .link_inputs
        .retain(|input| input.owner() != &id("unused"));
    assert!(
        missing.select(&required, None).is_err(),
        "dispatch dependency closure requires its native link input"
    );
    required
        .service_abis
        .insert(dispatch, Digest::sha256(b"wrong ABI"));
    assert!(manifest.select(&required, None).is_err());
}

#[test]
fn numerical_evidence_bytes_cannot_self_authorize_strict_or_tolerance_selection() {
    for behavior in [
        NativeNumericalBehavior::StrictReproducible {
            qualification: Digest::sha256(b"evidence"),
        },
        NativeNumericalBehavior::ToleranceBased {
            target_test_tolerances: Digest::sha256(b"evidence"),
        },
    ] {
        let (mut manifest, mut required, _) = fixture();
        manifest.numerical.behavior = behavior;
        manifest.numerical_evidence_component = Some(id("evidence"));
        required.numerical = manifest.numerical.clone();
        assert!(manifest.select(&required, None).is_err());
        let trusted = ExternalNumericalAdmission {
            pack: manifest.id().unwrap(),
            target: manifest.target.fingerprint().unwrap(),
            backend: manifest.numerical.backend,
            policy: manifest.numerical.fingerprint().unwrap(),
            evidence: Digest::sha256(b"evidence"),
        };
        let selected = manifest.select(&required, Some(&trusted)).unwrap();
        assert!(selected.components().contains(&id("evidence")));
        let mut stale = trusted.clone();
        stale.backend = Digest::sha256(b"wrong backend");
        assert!(manifest.select(&required, Some(&stale)).is_err());
        manifest.link_inputs.swap(0, 1);
        assert!(
            manifest.select(&required, Some(&trusted)).is_err(),
            "a different dependency/link contract needs fresh external qualification admission"
        );
    }
}

#[test]
fn target_probe_layout_and_program_capability_requirements_are_not_host_defaults() {
    let (manifest, mut required, _) = fixture();
    required.abi.layout_fingerprint = Digest::sha256(b"another target layout");
    assert!(manifest.select(&required, None).is_err());
    let mut changed = manifest.clone();
    changed.target.target.pointer_width = 32;
    assert!(changed.validate().is_err());
    let mut changed = manifest.clone();
    changed.target.cpu_baseline = NativeCpuBaseline::X86_64V1;
    assert!(changed.validate().is_err());
    let mut changed = manifest.clone();
    changed.target.endianness = NativeEndianness::Big;
    assert!(changed.validate().is_err());
    let (manifest, mut required, _) = fixture();
    required
        .reached_capabilities
        .0
        .insert(runmat_types::CapabilityRequirement::Network);
    assert!(manifest.select(&required, None).is_err());
    required.authorized_runtime_capabilities = required.reached_capabilities.clone();
    assert!(
        manifest.select(&required, None).is_err(),
        "caller authorization cannot supply missing pack inventory"
    );
    let mut available = manifest.clone();
    available.capabilities = required.reached_capabilities.clone();
    assert_eq!(
        available
            .select(&required, None)
            .unwrap()
            .reached_capabilities(),
        &required.reached_capabilities
    );
    assert!(ComponentId::new("../escape").is_err());
    let (mut manifest, required, _) = fixture();
    let linker = manifest
        .components
        .iter_mut()
        .find(|component| component.id == id("linker"))
        .unwrap();
    linker.effects.0.insert(EffectKind::Unknown);
    linker
        .capabilities
        .0
        .insert(runmat_types::CapabilityRequirement::Network);
    let selected = manifest.select(&required, None).unwrap();
    assert!(
        !selected.runtime_components().contains(&id("linker")),
        "build-tool bytes are retained without pretending they execute in the deployed runtime"
    );
}
