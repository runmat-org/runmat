use std::path::{Path, PathBuf};

use runmat_native_codegen::aot::RelocatableNativeObject;

use crate::{archive::RuntimeArchive, AotError, AotResult};

use super::{LinkerDriver, LinkerFamily};

pub const NATIVE_LINK_PLAN_SCHEMA_VERSION: u16 = 3;

/// Exact authored object and runtime bytes, independent of the shared pack
/// selection/tool qualification. Different programs cannot share this record.
#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize)]
pub struct NativeProgramLinkIdentity {
    pub object: runmat_execution::Digest,
    pub runtime_archive: runmat_execution::Digest,
    pub executable_cache_key: runmat_execution::Digest,
    pub native_cache_key: runmat_execution::Digest,
}
impl NativeProgramLinkIdentity {
    pub(super) fn from_products(
        object: &RelocatableNativeObject,
        runtime: &RuntimeArchive,
    ) -> Self {
        Self {
            object: object.manifest.object_digest,
            runtime_archive: runtime.manifest.archive_digest,
            executable_cache_key: object.manifest.executable_cache_key,
            native_cache_key: object.manifest.native_cache_key,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize)]
pub struct LinkPlan {
    pub schema_version: u16,
    pub product: runmat_execution::host::NativeProductMode,
    pub selected_build: Option<super::NativeLinkBuildIdentity>,
    pub inputs: NativeProgramLinkIdentity,
    pub target_triple: String,
    pub driver: LinkerDriver,
    pub object: PathBuf,
    pub runtime_archive: PathBuf,
    pub output: PathBuf,
    pub response_tokens: Vec<String>,
}

pub fn build_link_plan(
    object_product: &RelocatableNativeObject,
    runtime: &RuntimeArchive,
    driver: LinkerDriver,
    object: &Path,
    archive: &Path,
    output: &Path,
) -> AotResult<LinkPlan> {
    validate_pair(object_product, runtime)?;
    if object_product.manifest.product.selected().is_some()
        || runtime.manifest.product.selected().is_some()
    {
        return Err(AotError::contract("aot.link.selected_admission", "selected link planning requires the complete verified ordered component/toolchain closure"));
    }
    let closed_world = object_product.manifest.runtime_binding_mode
        == runmat_native_codegen::aot::AotRuntimeBindingMode::ClosedWorld;
    if closed_world && !runtime.manifest.capabilities.closed_world_linking {
        return Err(AotError::contract(
            "aot.link.closed_world",
            "runtime archive does not support closed-world extraction",
        ));
    }
    let mut response_tokens = Vec::new();
    let macos = object_product
        .manifest
        .target
        .native_parts()
        .map_err(|error| AotError::contract("aot.link.target", error.to_string()))?
        .0
        .target
        .operating_system
        == runmat_execution::host::NativeOperatingSystem::Macos;
    match driver.family {
        LinkerFamily::UnixCc => {
            response_tokens.push(object.display().to_string());
            if closed_world {
                response_tokens.push(archive.display().to_string());
                response_tokens.push(if macos {
                    "-Wl,-dead_strip".into()
                } else {
                    "-Wl,--gc-sections".into()
                });
            } else {
                push_native_input(&mut response_tokens, driver.family, macos, archive, true);
            }
            response_tokens.push("-o".into());
            response_tokens.push(output.display().to_string());
        }
        LinkerFamily::Msvc => {
            response_tokens.push(object.display().to_string());
            if closed_world {
                response_tokens.push(archive.display().to_string());
                response_tokens.push("/OPT:REF".into());
            } else {
                push_native_input(&mut response_tokens, driver.family, macos, archive, true);
            }
            response_tokens.push(format!("/OUT:{}", output.display()));
        }
    }
    response_tokens.extend(runtime.manifest.native_link_tokens.iter().cloned());
    if response_tokens.iter().any(|token| {
        token.is_empty()
            || token.len() > 4096
            || token.bytes().any(|byte| matches!(byte, 0 | b'\r' | b'\n'))
    }) {
        return Err(AotError::contract(
            "aot.link.response",
            "link response contains an invalid or oversized token",
        ));
    }
    Ok(LinkPlan {
        schema_version: NATIVE_LINK_PLAN_SCHEMA_VERSION,
        product: object_product.manifest.product.clone(),
        selected_build: None,
        inputs: NativeProgramLinkIdentity::from_products(object_product, runtime),
        target_triple: runtime.manifest.target_triple.clone(),
        driver,
        object: object.to_path_buf(),
        runtime_archive: archive.to_path_buf(),
        output: output.to_path_buf(),
        response_tokens,
    })
}

pub(super) fn validate_pair(
    object: &RelocatableNativeObject,
    runtime: &RuntimeArchive,
) -> AotResult<()> {
    object
        .validate()
        .map_err(|error| AotError::contract("aot.link.object", error.to_string()))?;
    runtime.manifest.validate()?;
    if object.manifest.product != runtime.manifest.product {
        return Err(AotError::contract(
            "aot.link.product",
            "object and runtime selected product modes differ",
        ));
    }
    if object.manifest.target != runtime.manifest.native_target {
        return Err(AotError::contract(
            "aot.link.target",
            "user object and runtime archive target bindings do not match",
        ));
    }
    if object.manifest.runtime_fingerprint != runtime.manifest.runtime_fingerprint
        || object.manifest.catalog_fingerprint != runtime.manifest.catalog_fingerprint
    {
        return Err(AotError::contract(
            "aot.link.environment",
            "user object and runtime archive runtime/catalog identities do not match",
        ));
    }
    if object
        .manifest
        .required_capabilities
        .0
        .contains(&runmat_types::CapabilityRequirement::Accelerator)
    {
        // Manifest inventory alone cannot grant hardware support. The
        // unselected planner authenticates the physical archive here; the
        // selected link stages and authenticates those bytes immediately
        // after this common pair check.
        if object.manifest.product.selected().is_none() {
            runtime.decode()?;
        }
        if !runtime.manifest.capabilities.accelerators.has_backend() {
            return Err(AotError::contract(
                "aot.link.accelerator",
                "native object requires an accelerator but the target runtime has no backend",
            ));
        }
    }
    Ok(())
}

pub(super) fn push_native_input(
    tokens: &mut Vec<String>,
    family: LinkerFamily,
    macos: bool,
    path: &Path,
    whole: bool,
) {
    if !whole {
        tokens.push(path.display().to_string());
        return;
    }
    match family {
        LinkerFamily::UnixCc if macos => tokens.extend([
            "-Xlinker".into(),
            "-force_load".into(),
            "-Xlinker".into(),
            path.display().to_string(),
        ]),
        LinkerFamily::UnixCc => tokens.extend([
            "-Wl,--whole-archive".into(),
            path.display().to_string(),
            "-Wl,--no-whole-archive".into(),
        ]),
        LinkerFamily::Msvc => tokens.push(format!("/WHOLEARCHIVE:{}", path.display())),
    }
}

#[cfg(test)]
mod tests {
    use runmat_execution::{Digest, ProgramEnvironment};
    use runmat_native_codegen::{
        aot::{
            native_object_format_for_target, AotRuntimeBindingMode, NativeObjectFunction,
            NativeObjectManifest, NativeOptimization, RelocatableNativeObject, AOT_ENTRY_SYMBOL,
            NATIVE_OBJECT_SCHEMA_VERSION,
        },
        NativeTarget,
    };
    use runmat_types::ProgramFunctionId;

    use crate::{
        archive::{build_runtime_archive, RuntimeArchiveCapabilities, RuntimeArchiveEncoding},
        link::{LinkerDriver, LinkerFamily},
    };

    use super::build_link_plan;

    #[test]
    fn ordered_repeated_inputs_and_macos_comma_paths_remain_individual_arguments() {
        let mut tokens = Vec::new();
        for path in ["a.a", "b.a", "a.a"] {
            super::push_native_input(
                &mut tokens,
                LinkerFamily::UnixCc,
                false,
                path.as_ref(),
                false,
            );
        }
        assert_eq!(tokens, ["a.a", "b.a", "a.a"]);
        tokens.clear();
        super::push_native_input(
            &mut tokens,
            LinkerFamily::UnixCc,
            true,
            "directory,with comma/a.a".as_ref(),
            true,
        );
        assert_eq!(
            tokens,
            [
                "-Xlinker",
                "-force_load",
                "-Xlinker",
                "directory,with comma/a.a"
            ]
        );
    }

    fn products(
        mode: AotRuntimeBindingMode,
    ) -> (RelocatableNativeObject, crate::archive::RuntimeArchive) {
        let runtime = Digest::sha256(b"runtime");
        let catalog = Digest::sha256(b"catalog");
        let environment = ProgramEnvironment::new(
            runmat_execution::schema::PROGRAM_SEMANTIC_SCHEMA_V2,
            runmat_execution::schema::PROGRAM_COMPILER_SCHEMA_V2,
            runtime,
            catalog,
            "matlab",
        )
        .unwrap();
        let bytes = crate::archive::test_archive::archive(
            runmat_execution::host::NativeAcceleratorProfile::NONE,
        );
        let archive = build_runtime_archive(
            &bytes,
            &environment,
            Vec::new(),
            RuntimeArchiveEncoding::Raw,
            RuntimeArchiveCapabilities::standalone_host(),
        )
        .unwrap();
        let bytes = b"object".to_vec();
        let target = NativeTarget::current();
        let function = ProgramFunctionId(1);
        let object = RelocatableNativeObject {
            manifest: NativeObjectManifest {
                schema_version: NATIVE_OBJECT_SCHEMA_VERSION,
                product: runmat_execution::host::NativeProductMode::HostUnselected,
                object_format: native_object_format_for_target(&target).unwrap(),
                target,
                executable_cache_key: Digest::sha256(b"executable"),
                native_cache_key: Digest::sha256(b"native"),
                runtime_fingerprint: runtime,
                catalog_fingerprint: catalog,
                optimization: NativeOptimization::Speed,
                runtime_binding_mode: mode,
                required_capabilities: runmat_types::CapabilitySet::default(),
                foreign_policy: None,
                object_digest: Digest::sha256(&bytes),
                object_bytes: bytes.len() as u64,
                entrypoint: function,
                selected_entrypoint_body: Digest::sha256(b"entrypoint-body"),
                functions: vec![NativeObjectFunction {
                    function,
                    symbol: AOT_ENTRY_SYMBOL.into(),
                }],
                data: Vec::new(),
            },
            bytes,
        };
        (object, archive)
    }

    #[test]
    fn closed_world_uses_selective_archive_extraction_on_unix_and_msvc() {
        let (object, runtime) = products(AotRuntimeBindingMode::ClosedWorld);
        for family in [LinkerFamily::UnixCc, LinkerFamily::Msvc] {
            let plan = build_link_plan(
                &object,
                &runtime,
                LinkerDriver {
                    path: "linker".into(),
                    family,
                    environment: Vec::new(),
                },
                "program.o".as_ref(),
                "runtime.a".as_ref(),
                "program".as_ref(),
            )
            .unwrap();
            assert!(plan
                .response_tokens
                .iter()
                .any(|token| token == "runtime.a"));
            assert!(!plan
                .response_tokens
                .iter()
                .any(|token| token.contains("whole-archive") || token.contains("WHOLEARCHIVE")));
        }
    }

    #[test]
    fn dynamic_runtime_preserves_whole_archive_discovery() {
        let (object, runtime) = products(AotRuntimeBindingMode::Dynamic);
        let plan = build_link_plan(
            &object,
            &runtime,
            LinkerDriver {
                path: "linker".into(),
                family: LinkerFamily::Msvc,
                environment: Vec::new(),
            },
            "program.obj".as_ref(),
            "runtime.lib".as_ref(),
            "program.exe".as_ref(),
        )
        .unwrap();
        assert!(plan
            .response_tokens
            .iter()
            .any(|token| token == "/WHOLEARCHIVE:runtime.lib"));
    }

    #[test]
    fn hard_accelerator_call_rejects_cpu_only_runtime_before_link() {
        let (mut object, runtime) = products(AotRuntimeBindingMode::Dynamic);
        object
            .manifest
            .required_capabilities
            .0
            .insert(runmat_types::CapabilityRequirement::Accelerator);
        let error = build_link_plan(
            &object,
            &runtime,
            LinkerDriver {
                path: "linker".into(),
                family: LinkerFamily::Msvc,
            },
            "program.obj".as_ref(),
            "runtime.lib".as_ref(),
            "program.exe".as_ref(),
        )
        .expect_err("CPU-only target must not satisfy a required GPU call");
        assert!(error.to_string().contains("accelerator"));
    }
}
