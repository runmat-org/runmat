use super::*;
use runmat_execution::host::{
    NativeAbiBinding, NativeNumericalPolicy, NativeOptimization, SelectedNativeTarget,
};
use runmat_types::{CapabilityRequirement, CapabilitySet, EffectKind, EffectSet};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};

pub use runmat_execution::host::NativeDeploymentProfile as TargetPackProfile;

/// Supplied by the selected compiler/runtime contracts, not recovered from a
/// pack's self-description. This is link admission, not executable admission.
#[derive(Clone, Debug)]
pub struct TargetPackRequirements {
    pub profile: TargetPackProfile,
    pub target: SelectedNativeTarget,
    pub abi: NativeAbiBinding,
    pub runtime: Digest,
    pub catalog: Digest,
    pub numerical: NativeNumericalPolicy,
    pub optimization: NativeOptimization,
    /// Actual program requirements from canonical compiler contracts.
    pub reached_capabilities: CapabilitySet,
    /// Trusted caller grant for the retained runtime, including runtime service
    /// capabilities where authorized. Later execution must enforce this grant;
    /// pack selection and byte verification do not execute or qualify it.
    pub authorized_runtime_capabilities: CapabilitySet,
    pub effects: EffectSet,
    /// Exact non-effect service demands from the selected canonical program.
    pub required_services: BTreeSet<TargetPackServiceKind>,
    pub components: Vec<ComponentId>,
    pub platform_service_abi: Digest,
    /// Trusted expected ABIs; this map is not serialized into pack JSON.
    pub service_abis: BTreeMap<TargetPackServiceKind, Digest>,
}

/// Explicit EXTERNAL trust input: the caller must obtain this expected record
/// from its backend qualification/target-test admission owner. It cannot be
/// deserialized from an untrusted pack, and retained evidence bytes do not
/// create one. Matching it still does not constitute an execution receipt.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExternalNumericalAdmission {
    /// Pins dependency/link/build-input declarations as well as the backend.
    pub pack: TargetPackId,
    pub target: Digest,
    pub backend: Digest,
    pub policy: Digest,
    pub evidence: Digest,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct TargetPackSelection {
    pub(super) pack: TargetPackId,
    pub(super) profile: TargetPackProfile,
    pub(super) components: Vec<ComponentId>,
    pub(super) runtime_components: Vec<ComponentId>,
    pub(super) roots: Vec<ComponentId>,
    pub(super) link_inputs: Vec<NativeLinkInput>,
    pub(super) reached_capabilities: CapabilitySet,
    pub(super) authorized_runtime_capabilities: CapabilitySet,
    pub(super) reached_effects: EffectSet,
    pub(super) required_services: BTreeSet<TargetPackServiceKind>,
}
impl TargetPackSelection {
    pub fn pack(&self) -> TargetPackId {
        self.pack
    }
    pub fn profile(&self) -> TargetPackProfile {
        self.profile
    }
    pub fn components(&self) -> &[ComponentId] {
        &self.components
    }
    /// Runtime closure excludes build-tool/sysroot/evidence-only retention.
    pub fn runtime_components(&self) -> &[ComponentId] {
        &self.runtime_components
    }
    /// Every retained member is a root or follows exact declared dependencies
    /// from one of these roots; cycles do not discard any member.
    pub fn roots(&self) -> &[ComponentId] {
        &self.roots
    }
    pub fn link_inputs(&self) -> &[NativeLinkInput] {
        &self.link_inputs
    }
    pub fn reached_capabilities(&self) -> &CapabilitySet {
        &self.reached_capabilities
    }
    /// Expected runtime grant for a later execution admission/enforcement join.
    /// Retaining this record does not itself implement that enforcement.
    pub fn authorized_runtime_capabilities(&self) -> &CapabilitySet {
        &self.authorized_runtime_capabilities
    }
    pub fn reached_effects(&self) -> &EffectSet {
        &self.reached_effects
    }
    pub fn required_services(&self) -> &BTreeSet<TargetPackServiceKind> {
        &self.required_services
    }
    pub fn fingerprint(&self) -> Digest {
        Digest::sha256_serialized(&("runmat-target-pack-selection-v3", self))
            .expect("selected target pack serializes")
    }
}

impl TargetPackManifest {
    pub fn select(
        &self,
        required: &TargetPackRequirements,
        trust: Option<&ExternalNumericalAdmission>,
    ) -> ArtifactResult<TargetPackSelection> {
        self.validate()?;
        required.target.validate().map_err(contract)?;
        self.abi.require_exact(&required.abi).map_err(contract)?;
        required.numerical.validate().map_err(contract)?;
        if self.target != required.target
            || self.runtime != required.runtime
            || self.catalog != required.catalog
            || self.numerical != required.numerical
            || self.optimization != required.optimization
            || self.platform_service_abi != required.platform_service_abi
            || !required
                .reached_capabilities
                .0
                .is_subset(&self.capabilities.0)
            || !required
                .reached_capabilities
                .0
                .is_subset(&required.authorized_runtime_capabilities.0)
            || required.components.len() > MAX_COMPONENTS
            || required
                .components
                .windows(2)
                .any(|pair| pair[0] >= pair[1])
        {
            return Err(invalid(
                "selected target pack does not satisfy exact program requirements",
            ));
        }
        if let Some(evidence) = self.numerical.required_evidence() {
            let expected = ExternalNumericalAdmission {
                pack: self.id()?,
                target: self.target.fingerprint().map_err(contract)?,
                backend: self.numerical.backend,
                policy: self.numerical.fingerprint().map_err(contract)?,
                evidence,
            };
            if trust != Some(&expected) {
                return Err(invalid("numerical qualification requires exact external target/backend/policy admission"));
            }
        }
        let index = components::index(&self.components)?;
        for kind in &required.required_services {
            kind.validate().map_err(contract)?;
            if *kind == TargetPackServiceKind::SourceEvaluation
                && required.service_abis.get(kind)
                    != Some(&runmat_execution::host::native_source_evaluation_service_abi())
            {
                return Err(invalid(
                    "selected source evaluation requires the exact linked factory ABI",
                ));
            }
            if *kind == TargetPackServiceKind::SourceEvaluation
                && ![
                    &required.reached_capabilities,
                    &required.authorized_runtime_capabilities,
                    &self.capabilities,
                ]
                .into_iter()
                .all(|capabilities| capabilities.0.contains(&CapabilityRequirement::HostRuntime))
            {
                return Err(invalid(
                    "source evaluation lacks an exact host-runtime grant",
                ));
            }
        }
        let mut roots = required.components.iter().cloned().collect::<BTreeSet<_>>();
        roots.insert(self.runtime_component.clone());
        roots.insert(self.numerical_backend_component.clone());
        let mut build_roots = self
            .numerical_evidence_component
            .iter()
            .cloned()
            .collect::<BTreeSet<_>>();
        build_roots.insert(self.abi_probe_component.clone());
        for input in [&self.build_inputs.linker, &self.build_inputs.sysroot] {
            match input {
                NativeBuildInput::ContentAddressed { component } => {
                    build_roots.insert(component.clone());
                }
                NativeBuildInput::HostedDeclared { .. }
                    if required.profile == TargetPackProfile::DeterministicDeployed =>
                {
                    return Err(invalid(
                        "deterministic pack requires content-addressed linker/sysroot inputs",
                    ))
                }
                NativeBuildInput::HostedDeclared { .. } => {}
            }
        }
        let mut reached = BTreeSet::new();
        let reached_effects = loop {
            components::retain(&index, roots.iter().cloned(), &mut reached)?;
            let before = roots.len();
            let effects = required
                .effects
                .0
                .iter()
                .copied()
                .chain(
                    reached
                        .iter()
                        .flat_map(|id| index[id].effects.0.iter().copied()),
                )
                .collect::<BTreeSet<_>>();
            if effects.contains(&EffectKind::Unknown)
                && required.profile == TargetPackProfile::DeterministicDeployed
            {
                return Err(invalid(
                    "deterministic target pack cannot admit unknown effects",
                ));
            }
            for kind in effects
                .iter()
                .copied()
                .filter_map(TargetPackServiceKind::for_effect)
                .chain(required.required_services.iter().copied())
                .collect::<BTreeSet<_>>()
            {
                let service = self
                    .services
                    .iter()
                    .find(|service| service.kind == kind)
                    .ok_or_else(|| {
                        invalid("reached effect has no target service implementation")
                    })?;
                if kind == TargetPackServiceKind::SourceEvaluation
                    && service.component != self.runtime_component
                {
                    return Err(invalid(
                        "linked source compiler must share the selected runtime archive",
                    ));
                }
                if required.service_abis.get(&kind) != Some(&service.abi)
                    || (required.profile == TargetPackProfile::DeterministicDeployed
                        && service.behavior != TargetPackServiceBehavior::DeterministicDeclared)
                {
                    return Err(invalid(
                        "target service ABI/behavior does not satisfy the selected profile",
                    ));
                }
                roots.insert(service.component.clone());
            }
            if roots.len() == before {
                break EffectSet(effects);
            }
        };
        for id in &reached {
            if !matches!(
                index[id].kind,
                TargetPackComponentKind::NativeObject
                    | TargetPackComponentKind::StaticArchive
                    | TargetPackComponentKind::RuntimeData
            ) || !index[id]
                .capabilities
                .0
                .is_subset(&required.authorized_runtime_capabilities.0)
                || !index[id].capabilities.0.is_subset(&self.capabilities.0)
            {
                return Err(invalid(
                    "retained component introduces undeclared capabilities",
                ));
            }
        }
        let link_inputs = self
            .link_inputs
            .iter()
            .filter(|input| reached.contains(input.owner()))
            .cloned()
            .collect::<Vec<_>>();
        if required.profile == TargetPackProfile::DeterministicDeployed
            && link_inputs
                .iter()
                .any(|input| !matches!(input, NativeLinkInput::Component { .. }))
        {
            return Err(invalid(
                "deterministic pack cannot use ambient native dependencies",
            ));
        }
        for id in &reached {
            if matches!(index[id].kind, TargetPackComponentKind::NativeObject | TargetPackComponentKind::StaticArchive)
                && !link_inputs.iter().any(|input| matches!(input, NativeLinkInput::Component { component, .. } if component == id)) {
                return Err(invalid("retained native component has no ordered link input"));
            }
        }
        let runtime_components = reached.iter().cloned().collect();
        // Build inputs and evidence are authenticated and retained, but their
        // host-side capabilities/dependencies are not runtime link roots.
        components::retain(&index, build_roots.iter().cloned(), &mut reached)?;
        roots.extend(build_roots);
        Ok(TargetPackSelection {
            pack: self.id()?,
            profile: required.profile,
            components: reached.into_iter().collect(),
            runtime_components,
            roots: roots.into_iter().collect(),
            link_inputs,
            reached_capabilities: required.reached_capabilities.clone(),
            authorized_runtime_capabilities: required.authorized_runtime_capabilities.clone(),
            reached_effects,
            required_services: required.required_services.clone(),
        })
    }
}
