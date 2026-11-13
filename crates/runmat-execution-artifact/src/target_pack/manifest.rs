use super::*;
use runmat_execution::host::{
    NativeAbiBinding, NativeNumericalPolicy, NativeOptimization, SelectedNativeTarget,
};
use runmat_types::CapabilitySet;
use serde::{Deserialize, Serialize};

pub const TARGET_PACK_SCHEMA_VERSION: u16 = 3;
pub use runmat_execution::host::{
    NativeDeploymentServiceBehavior as TargetPackServiceBehavior,
    NativeDeploymentServiceKind as TargetPackServiceKind,
};

/// The typed service ID is policy; the component label only locates bytes.
/// Behavior is an implementation declaration, not a replay qualification.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TargetPackService {
    pub kind: TargetPackServiceKind,
    pub abi: Digest,
    pub component: ComponentId,
    pub behavior: TargetPackServiceBehavior,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TargetPackManifest {
    pub schema_version: u16,
    pub target: SelectedNativeTarget,
    pub abi: NativeAbiBinding,
    pub abi_probe_component: ComponentId,
    pub runtime: Digest,
    pub catalog: Digest,
    pub numerical: NativeNumericalPolicy,
    pub numerical_backend_component: ComponentId,
    pub numerical_evidence_component: Option<ComponentId>,
    pub optimization: NativeOptimization,
    /// Available runtime capability inventory, not an implicit caller grant.
    pub capabilities: CapabilitySet,
    pub runtime_component: ComponentId,
    pub components: Vec<TargetPackComponent>,
    pub link_inputs: Vec<NativeLinkInput>,
    pub build_inputs: NativeBuildInputs,
    pub platform_service_abi: Digest,
    pub services: Vec<TargetPackService>,
}

impl TargetPackManifest {
    pub fn validate(&self) -> ArtifactResult<()> {
        if self.schema_version != TARGET_PACK_SCHEMA_VERSION {
            return Err(invalid("unsupported target runtime pack schema"));
        }
        self.target.validate().map_err(contract)?;
        self.abi.validate().map_err(contract)?;
        if self.target.target.abi != self.abi.identity().map_err(contract)? {
            return Err(invalid("pack target and target-probed ABI disagree"));
        }
        nonzero(self.runtime)?;
        nonzero(self.catalog)?;
        nonzero(self.platform_service_abi)?;
        self.numerical.validate().map_err(contract)?;
        let index = components::index(&self.components)?;
        if !index.get(&self.abi_probe_component).is_some_and(|entry| {
            entry.kind == TargetPackComponentKind::NativeAbiProbe
                && entry.object.digest == self.abi.layout_fingerprint
                && entry.object.encoded_length
                    == runmat_execution::host::NATIVE_ABI_PROBE_BYTES as u64
        }) {
            return Err(invalid(
                "target pack ABI probe descriptor is missing or mismatched",
            ));
        }
        if !index
            .get(&self.numerical_backend_component)
            .is_some_and(|entry| {
                entry.object.digest == self.numerical.backend
                    && matches!(
                        entry.kind,
                        TargetPackComponentKind::NativeObject
                            | TargetPackComponentKind::StaticArchive
                    )
            })
        {
            return Err(invalid(
                "target pack numerical backend component is missing or mismatched",
            ));
        }
        match (
            self.numerical.required_evidence(),
            self.numerical_evidence_component.as_ref(),
        ) {
            (None, None) => {}
            (Some(digest), Some(id))
                if index.get(id).is_some_and(|entry| {
                    entry.kind == TargetPackComponentKind::NumericalEvidence
                        && entry.object.digest == digest
                }) => {}
            _ => {
                return Err(invalid(
                    "target pack numerical evidence component is missing or mismatched",
                ))
            }
        }
        if !index.get(&self.runtime_component).is_some_and(|entry| {
            matches!(
                entry.kind,
                TargetPackComponentKind::NativeObject | TargetPackComponentKind::StaticArchive
            )
        }) || self.link_inputs.len() > MAX_LINK_INPUTS
            || self
                .services
                .windows(2)
                .any(|pair| pair[0].kind >= pair[1].kind)
        {
            return Err(invalid(
                "target pack roots/link/service inventory is invalid",
            ));
        }
        for input in &self.link_inputs {
            let owner = index
                .get(input.owner())
                .ok_or_else(|| invalid("link input component is missing"))?;
            match input {
                NativeLinkInput::Component { retention, .. } => {
                    if !matches!(
                        owner.kind,
                        TargetPackComponentKind::NativeObject
                            | TargetPackComponentKind::StaticArchive
                    ) || (*retention == NativeLinkRetention::WholeArchive
                        && owner.kind != TargetPackComponentKind::StaticArchive)
                    {
                        return Err(invalid("native link input kind/retention is invalid"));
                    }
                }
                NativeLinkInput::HostedLibrary { name, .. }
                | NativeLinkInput::HostedFramework { name, .. } => {
                    if matches!(input, NativeLinkInput::HostedFramework { .. })
                        && self.target.target.operating_system
                            != runmat_execution::host::NativeOperatingSystem::Macos
                    {
                        return Err(invalid(
                            "native framework input is incompatible with the selected platform",
                        ));
                    }
                    if name.is_empty()
                        || name.len() > 128
                        || !name.is_ascii()
                        || !name
                            .bytes()
                            .all(|byte| byte.is_ascii_alphanumeric() || b"_+.-".contains(&byte))
                    {
                        return Err(invalid("hosted native dependency label is invalid"));
                    }
                }
            }
        }
        for (input, kind) in [
            (
                &self.build_inputs.linker,
                TargetPackComponentKind::Toolchain,
            ),
            (&self.build_inputs.sysroot, TargetPackComponentKind::Sysroot),
        ] {
            match input {
                NativeBuildInput::ContentAddressed { component }
                    if index.get(component).is_some_and(|entry| entry.kind == kind) => {}
                NativeBuildInput::HostedDeclared { fingerprint } => nonzero(*fingerprint)?,
                _ => {
                    return Err(invalid(
                        "target pack build input is missing or has the wrong role",
                    ))
                }
            }
        }
        for service in &self.services {
            nonzero(service.abi)?;
            service.kind.validate().map_err(contract)?;
            if !index.get(&service.component).is_some_and(|entry| {
                matches!(
                    entry.kind,
                    TargetPackComponentKind::NativeObject | TargetPackComponentKind::StaticArchive
                )
            }) {
                return Err(invalid("target pack effect service is invalid or missing"));
            }
        }
        Ok(())
    }
}
