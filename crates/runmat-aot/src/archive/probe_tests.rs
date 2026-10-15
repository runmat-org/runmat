use super::*;
use ar_archive_writer::{
    write_archive_to_stream, ArchiveKind, NewArchiveMember, DEFAULT_OBJECT_READER,
};
use object::write::{Object as WritableObject, Symbol, SymbolSection};
use object::{
    Architecture, BinaryFormat, Endianness, SectionKind, SymbolFlags, SymbolKind, SymbolScope,
};
use runmat_execution::host::{NativeAbiField, NativeEnvironment};
use std::io::Cursor;

fn archive(objects: &[Vec<u8>]) -> Vec<u8> {
    let members = objects
        .iter()
        .enumerate()
        .map(|(i, bytes)| {
            NewArchiveMember::new(
                bytes.as_slice(),
                &DEFAULT_OBJECT_READER,
                format!("probe{i}.o"),
            )
        })
        .collect::<Vec<_>>();
    let mut output = Cursor::new(Vec::new());
    write_archive_to_stream(&mut output, &members, ArchiveKind::Gnu, false, None).unwrap();
    output.into_inner()
}
fn object(
    probe: &NativeAbiProbe,
    format: BinaryFormat,
    architecture: Architecture,
    bytes: &[u8],
) -> Vec<u8> {
    object_with_relocation(probe, format, architecture, bytes, false)
}
fn object_with_relocation(
    probe: &NativeAbiProbe,
    format: BinaryFormat,
    architecture: Architecture,
    bytes: &[u8],
    relocated: bool,
) -> Vec<u8> {
    let mut object = WritableObject::new(format, architecture, Endianness::Little);
    let section = object.add_section(
        Vec::new(),
        b".runmat_abi".to_vec(),
        SectionKind::ReadOnlyData,
    );
    let offset = object.append_section_data(section, bytes, 1);
    let name = if format == BinaryFormat::MachO {
        format!("_{NATIVE_ABI_PROBE_SYMBOL}")
    } else {
        NATIVE_ABI_PROBE_SYMBOL.to_string()
    };
    let symbol = object.add_symbol(Symbol {
        name: name.into_bytes(),
        value: offset,
        size: NATIVE_ABI_PROBE_BYTES as u64,
        kind: SymbolKind::Data,
        scope: SymbolScope::Linkage,
        weak: false,
        section: SymbolSection::Section(section),
        flags: SymbolFlags::None,
    });
    if relocated {
        object
            .add_relocation(
                section,
                object::write::Relocation {
                    offset: offset + 450,
                    symbol,
                    addend: 0,
                    flags: object::RelocationFlags::Generic {
                        kind: object::RelocationKind::Absolute,
                        encoding: object::RelocationEncoding::Generic,
                        size: 64,
                    },
                },
            )
            .unwrap();
    }
    assert_eq!(probe.canonical_bytes().len(), NATIVE_ABI_PROBE_BYTES);
    object.write().unwrap()
}

#[test]
fn fixed_probe_is_read_without_foreign_execution_and_rejects_container_or_byte_substitution() {
    // Container fixtures are intentionally synthetic metadata, not evidence
    // that this host has executed or ABI-probed either foreign target.
    for (format, operating_system, environment) in [
        (
            BinaryFormat::Elf,
            NativeOperatingSystem::Linux,
            NativeEnvironment::Gnu,
        ),
        (
            BinaryFormat::MachO,
            NativeOperatingSystem::Macos,
            NativeEnvironment::Darwin,
        ),
        (
            BinaryFormat::Coff,
            NativeOperatingSystem::Windows,
            NativeEnvironment::Msvc,
        ),
    ] {
        for (architecture, physical_architecture) in [
            (Architecture::X86_64, NativeArchitecture::X86_64),
            (Architecture::Aarch64, NativeArchitecture::Aarch64),
        ] {
            let mut probe = runmat_runtime::native::native_abi_probe();
            probe.operating_system = operating_system;
            probe.environment = environment;
            probe.architecture = physical_architecture;
            let bytes = object(&probe, format, architecture, &probe.canonical_bytes());
            assert_eq!(
                read_native_abi_probe(&archive(std::slice::from_ref(&bytes))).unwrap(),
                probe
            );
            if format == BinaryFormat::Elf && architecture == Architecture::X86_64 {
                let relocated = object_with_relocation(
                    &probe,
                    format,
                    architecture,
                    &probe.canonical_bytes(),
                    true,
                );
                assert!(
                    read_native_abi_probe(&archive(&[relocated])).is_err(),
                    "canonical-looking bytes with unresolved target relocations must reject"
                );
            }
            assert!(read_native_abi_probe(&archive(&[bytes.clone(), bytes])).is_err());
            let other = if architecture == Architecture::X86_64 {
                Architecture::Aarch64
            } else {
                Architecture::X86_64
            };
            assert!(read_native_abi_probe(&archive(&[object(
                &probe,
                format,
                other,
                &probe.canonical_bytes()
            )]))
            .is_err());
            let mut corrupt = probe.canonical_bytes();
            corrupt[8] += 1;
            assert!(read_native_abi_probe(&archive(&[object(
                &probe,
                format,
                architecture,
                &corrupt
            )]))
            .is_err());
            let mut corrupt = probe.clone();
            corrupt.offsets[NativeAbiField::RequestFunction as usize] =
                corrupt.offset(NativeAbiField::RequestBlock);
            assert!(read_native_abi_probe(&archive(&[object(
                &probe,
                format,
                architecture,
                &corrupt.canonical_bytes()
            )]))
            .is_err());
            let mut trailing = probe.canonical_bytes().to_vec();
            trailing.push(0);
            assert!(NativeAbiProbe::from_canonical_bytes(&trailing).is_err());
        }
    }
    assert!(read_native_abi_probe(&archive(&[])).is_err());
}

#[test]
fn selected_archive_binds_verified_bytes_and_target_probe_without_host_target_inference() {
    use runmat_execution::host::*;
    use runmat_execution::{Digest, ProgramEnvironment};
    use runmat_execution_artifact::target_pack::*;
    use runmat_execution_artifact::{ObjectDescriptor, ObjectNamespace, ProgramTarget};
    use runmat_native_codegen::aot::{validate_native_product, SelectedNativeProduct};
    use runmat_types::{CapabilitySet, EffectSet};
    use std::collections::BTreeMap;

    // Synthetic foreign containers qualify only the structural byte/probe join.
    // They are not foreign ABI execution or installed-provider qualifications.
    let mut probe = runmat_runtime::native::native_abi_probe();
    probe.architecture = NativeArchitecture::X86_64;
    probe.operating_system = NativeOperatingSystem::Linux;
    probe.environment = NativeEnvironment::Gnu;
    let target =
        runmat_native_codegen::NativeTarget::from_probe(probe.clone(), NativeCpuBaseline::X86_64V1)
            .unwrap();
    let abi = probe.binding().unwrap();
    let runtime_bytes = archive(&[object(
        &probe,
        BinaryFormat::Elf,
        Architecture::X86_64,
        &probe.canonical_bytes(),
    )]);
    let id = |name: &str| ComponentId::new(name).unwrap();
    let contents = [
        (id("probe"), probe.canonical_bytes().to_vec()),
        (id("runtime"), runtime_bytes.clone()),
    ]
    .into_iter()
    .collect::<BTreeMap<_, _>>();
    let components = contents
        .iter()
        .map(|(id, bytes)| TargetPackComponent {
            id: id.clone(),
            kind: if id.as_str() == "probe" {
                TargetPackComponentKind::NativeAbiProbe
            } else {
                TargetPackComponentKind::StaticArchive
            },
            object: ObjectDescriptor::new(
                ObjectNamespace::ProgramArtifact,
                id.as_str(),
                "application/octet-stream",
                bytes,
            )
            .unwrap(),
            dependencies: Vec::new(),
            capabilities: CapabilitySet::default(),
            effects: EffectSet::default(),
        })
        .collect();
    let numerical = NativeNumericalPolicy {
        schema_version: NATIVE_NUMERICAL_POLICY_SCHEMA_VERSION,
        backend: Digest::sha256(&runtime_bytes),
        behavior: NativeNumericalBehavior::PlatformAccurate,
    };
    let manifest = TargetPackManifest {
        schema_version: TARGET_PACK_SCHEMA_VERSION,
        target: target.native_parts().unwrap().0.clone(),
        abi: abi.clone(),
        abi_probe_component: id("probe"),
        runtime: Digest::sha256("runtime semantics"),
        catalog: Digest::sha256("catalog"),
        numerical: numerical.clone(),
        numerical_backend_component: id("runtime"),
        numerical_evidence_component: None,
        optimization: NativeOptimization::Speed,
        capabilities: CapabilitySet::default(),
        runtime_component: id("runtime"),
        components,
        link_inputs: vec![NativeLinkInput::Component {
            component: id("runtime"),
            retention: NativeLinkRetention::Referenced,
        }],
        build_inputs: NativeBuildInputs {
            linker: NativeBuildInput::HostedDeclared {
                fingerprint: Digest::sha256("linker"),
            },
            sysroot: NativeBuildInput::HostedDeclared {
                fingerprint: Digest::sha256("sysroot"),
            },
        },
        platform_service_abi: Digest::sha256("ports"),
        services: Vec::new(),
    };
    let required = TargetPackRequirements {
        profile: TargetPackProfile::HostedNative,
        target: manifest.target.clone(),
        abi,
        runtime: manifest.runtime,
        catalog: manifest.catalog,
        numerical,
        optimization: manifest.optimization,
        reached_capabilities: CapabilitySet::default(),
        authorized_runtime_capabilities: CapabilitySet::default(),
        effects: EffectSet::default(),
        components: Vec::new(),
        platform_service_abi: manifest.platform_service_abi,
        service_abis: BTreeMap::new(),
    };
    let selection = manifest.select(&required, None).unwrap();
    let verified = manifest
        .verify_selected_bytes(&selection, &contents)
        .unwrap();
    let selected = SelectedNativeProduct::from_verified(&verified).unwrap();
    let environment = ProgramEnvironment::new(
        runmat_execution::schema::PROGRAM_SEMANTIC_SCHEMA_V2,
        runmat_execution::schema::PROGRAM_COMPILER_SCHEMA_V2,
        manifest.runtime,
        manifest.catalog,
        "matlab",
    )
    .unwrap();
    for encoding in [
        super::super::RuntimeArchiveEncoding::Raw,
        super::super::RuntimeArchiveEncoding::Zstd,
    ] {
        let packed = super::super::build_selected_runtime_archive(
            &selected,
            &environment,
            encoding,
            super::super::RuntimeArchiveCapabilities::standalone_host(),
        )
        .unwrap();
        assert_eq!(packed.decode().unwrap(), runtime_bytes);
        assert_eq!(&packed.manifest.product, selected.product());
        assert_eq!(&packed.manifest.native_target, selected.target());
        assert!(packed.manifest.native_link_tokens.is_empty());
        let recipe_target = ProgramTarget::native_product(
            "selected",
            manifest.target.target.clone(),
            packed.manifest.product.clone(),
        );
        recipe_target.validate().unwrap();
        assert_eq!(
            recipe_target.native_product_mode(),
            Some(selected.product())
        );
        let mut changed = packed.manifest.clone();
        changed.archive_digest = Digest::sha256("different runtime");
        assert!(changed.validate().is_err());
        changed = packed.manifest.clone();
        changed.native_link_tokens.push("-lambient".into());
        assert!(changed.validate().is_err());
    }
    assert!(validate_native_product(
        selected.product(),
        selected.target(),
        manifest.runtime,
        manifest.catalog,
        NativeOptimization::None
    )
    .is_err());
    let mut changed = contents.clone();
    changed.get_mut(&id("runtime")).unwrap().push(0);
    assert!(manifest
        .verify_selected_bytes(&selection, &changed)
        .is_err());
}
