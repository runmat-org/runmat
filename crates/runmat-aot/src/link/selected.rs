//! Real selected linking through a qualified host tool and one leased closure.
use std::collections::BTreeSet;

use runmat_execution::host::{NativeArchitecture, NativeOperatingSystem};
use runmat_execution_artifact::target_pack::{NativeLinkInput, NativeLinkRetention};
use runmat_native_codegen::aot::{RelocatableNativeObject, SelectedNativeProduct};

use super::{
    inputs::{HostedNativeBuildTrees, SelectedLinkInputs},
    toolchain::{ExternalNativeLinkAdmission, NativeHostedDependency, NativeLinkAdapter},
    LinkPlan, LinkRequest, LinkedProgram, LinkerDriver, LinkerFamily,
};
use crate::{archive::RuntimeArchive, AotError, AotResult};

/// Complete and publish linking, not selected-runtime execution qualification.
/// Hosted dependency runtime installation (including framework resources and
/// install-name/rpath availability) remains part of the deployment admission;
/// the private build tree is not advertised as a durable runtime installation.
/// Unix install modes are checked only where the build host represents them;
/// cross-target installation is never certified by a host filesystem bit.
pub fn link_selected_standalone(
    selected: &SelectedNativeProduct<'_>,
    object: &RelocatableNativeObject,
    runtime: &RuntimeArchive,
    admission: &ExternalNativeLinkAdmission,
    hosted: HostedNativeBuildTrees<'_>,
    request: &LinkRequest,
) -> AotResult<LinkedProgram> {
    if request.linker.is_some() {
        return Err(invalid(
            "selected linking cannot override its qualified tool through an ambient path",
        ));
    }
    let parent = request
        .output
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| std::path::Path::new("."));
    let inputs = SelectedLinkInputs::stage(selected, object, runtime, admission, hosted, parent)?;
    let plan = plan(
        selected,
        &inputs,
        admission,
        super::NativeProgramLinkIdentity::from_products(object, runtime),
        object.manifest.runtime_binding_mode,
    )?;
    // Validate the actual staged physical bytes again at the handoff, not just
    // a previous path hash or a detached metadata success token.
    inputs.verify(selected, object)?;
    super::invoke::invoke_prepared(
        plan,
        inputs.lease,
        request,
        super::invoke::LinkProcessEnvironment::QualifiedTree,
    )
}

fn plan(
    selected: &SelectedNativeProduct<'_>,
    inputs: &SelectedLinkInputs<'_>,
    admission: &ExternalNativeLinkAdmission,
    identity: super::NativeProgramLinkIdentity,
    binding_mode: runmat_native_codegen::aot::AotRuntimeBindingMode,
) -> AotResult<LinkPlan> {
    let manifest = selected.contents().manifest();
    let target = &manifest.target.target;
    let canonical =
        super::canonical::selected_link_plan(selected, admission, identity.clone(), binding_mode)?;
    let triple = canonical.target_triple.clone();
    let family = canonical.family;
    let macos = target.operating_system == NativeOperatingSystem::Macos;
    let driver_path = inputs.toolchain_root.join(admission.entrypoint.as_str());
    let mut tokens = Vec::new();
    match &admission.adapter {
        NativeLinkAdapter::ClangDriver { linker } => {
            if target.operating_system == NativeOperatingSystem::Windows {
                return Err(invalid(
                    "MSVC targets require the typed MSVC linker adapter",
                ));
            }
            tokens.extend([
                format!("--target={triple}"),
                format!("--sysroot={}", inputs.sysroot_root.display()),
                format!(
                    "-fuse-ld={}",
                    inputs.toolchain_root.join(linker.as_str()).display()
                ),
                "-nostdlib".into(),
                "-nostartfiles".into(),
                "-nodefaultlibs".into(),
            ]);
        }
        NativeLinkAdapter::MsvcLinker => {
            if target.operating_system != NativeOperatingSystem::Windows {
                return Err(invalid(
                    "MSVC linker adapter requires a Windows MSVC selected target",
                ));
            }
            let machine = match target.architecture {
                NativeArchitecture::X86_64 => "X64",
                NativeArchitecture::Aarch64 => "ARM64",
            };
            tokens.extend([
                format!("/MACHINE:{machine}"),
                "/NODEFAULTLIB".into(),
                super::plan::windows_stack_argument(),
            ]);
        }
    }
    tokens.push(inputs.object.display().to_string());
    let mut reached_hosted = BTreeSet::new();
    for input in selected.contents().selection().link_inputs() {
        match input {
            NativeLinkInput::Component {
                component,
                retention,
            } => {
                let path = inputs
                    .components
                    .get(component)
                    .ok_or_else(|| invalid("ordered link member has no exact leased bytes"))?;
                super::plan::push_native_input(
                    &mut tokens,
                    family,
                    macos,
                    path,
                    *retention == NativeLinkRetention::WholeArchive,
                );
            }
            NativeLinkInput::HostedLibrary { name, .. }
            | NativeLinkInput::HostedFramework { name, .. } => {
                let key = if matches!(input, NativeLinkInput::HostedLibrary { .. }) {
                    NativeHostedDependency::Library(name.clone())
                } else {
                    NativeHostedDependency::Framework(name.clone())
                };
                let path = admission.hosted_dependencies.get(&key).ok_or_else(|| {
                    invalid("hosted native input lacks an explicit qualified sysroot locator")
                })?;
                super::native_input::validate_qualified_hosted_dependency(
                    inputs.sysroot_file(path)?,
                    target,
                    &key,
                    path,
                )?;
                reached_hosted.insert(key);
                tokens.push(
                    inputs
                        .sysroot_root
                        .join(path.as_str())
                        .display()
                        .to_string(),
                );
            }
        }
    }
    if !reached_hosted
        .iter()
        .eq(admission.hosted_dependencies.keys())
    {
        return Err(invalid(
            "trusted hosted locator set differs from exact reached native dependencies",
        ));
    }
    if canonical
        .options
        .contains(&super::SelectedLinkOption::RemoveUnusedSections)
    {
        tokens.push(super::section_gc::argument(family, target.operating_system).into());
    }
    match family {
        LinkerFamily::UnixCc => tokens.extend(["-o".into(), inputs.output.display().to_string()]),
        LinkerFamily::Msvc => tokens.push(format!("/OUT:{}", inputs.output.display())),
    }
    if tokens.iter().any(|token| {
        token.is_empty()
            || token.len() > 4096
            || token.bytes().any(|byte| matches!(byte, 0 | b'\r' | b'\n'))
    }) {
        return Err(invalid("selected link token exceeds response bounds"));
    }
    LinkPlan {
        schema_version: super::NATIVE_LINK_PLAN_SCHEMA_VERSION,
        product: selected.product().clone(),
        selected_build: Some(admission.identity()),
        inputs: identity,
        target_triple: triple,
        driver: LinkerDriver {
            path: driver_path,
            family,
            environment: Vec::new(),
        },
        object: inputs.object.clone(),
        runtime_archive: inputs.runtime.clone(),
        output: inputs.output.clone(),
        response_tokens: tokens,
        selected_proof: None,
    }
    .bind_selected(canonical)
}

fn invalid(error: impl std::fmt::Display) -> AotError {
    AotError::contract("aot.link.selected", error.to_string())
}
