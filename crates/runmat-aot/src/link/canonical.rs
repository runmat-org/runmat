//! Path-independent evidence for one selected native link recipe.
//!
//! This record is derived from verified selected inputs and trusted build
//! admission. It is not a linker permission or evidence that the output ran.

use std::io::{self, Write};

use runmat_execution::host::NativeSelectedProductBinding;
use runmat_execution::{Digest, DigestStream};
use runmat_execution_artifact::target_pack::NativeLinkInput;
use runmat_native_codegen::aot::SelectedNativeProduct;

use super::{LinkerFamily, NativeLinkBuildIdentity, NativeProgramLinkIdentity};
use crate::{AotError, AotResult};

const SCHEMA_VERSION: u16 = 1;
const MAX_CANONICAL_BYTES: usize = 16 * 1024 * 1024;

/// Typed fixed options emitted by the selected linker adapter. Physical
/// sysroot, tool and output locations remain only in the invocation record.
#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SelectedLinkOption {
    TargetTriple,
    Sysroot,
    CompanionLinker,
    NoStandardLibrary,
    NoStartFiles,
    NoDefaultLibraries,
    MsvcMachine,
    MsvcNoDefaultLibraries,
    MsvcStackReserve { bytes: u64 },
    RemoveUnusedSections,
    Output,
}

#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CanonicalSelectedLinkPlan {
    schema_version: u16,
    pub product: NativeSelectedProductBinding,
    pub inputs: NativeProgramLinkIdentity,
    pub build: NativeLinkBuildIdentity,
    pub target_triple: String,
    pub family: LinkerFamily,
    pub options: Vec<SelectedLinkOption>,
    /// Exact verified selection order, including repeated members/retention.
    pub native_inputs: Vec<NativeLinkInput>,
}

impl CanonicalSelectedLinkPlan {
    pub(super) fn new(
        product: NativeSelectedProductBinding,
        inputs: NativeProgramLinkIdentity,
        build: NativeLinkBuildIdentity,
        target_triple: String,
        family: LinkerFamily,
        options: Vec<SelectedLinkOption>,
        native_inputs: Vec<NativeLinkInput>,
    ) -> Self {
        Self {
            schema_version: SCHEMA_VERSION,
            product,
            inputs,
            build,
            target_triple,
            family,
            options,
            native_inputs,
        }
    }

    pub fn identity(&self) -> AotResult<Digest> {
        if self.schema_version != SCHEMA_VERSION {
            return Err(AotError::contract(
                "aot.link.canonical",
                "unsupported selected link plan schema",
            ));
        }
        self.product
            .validate()
            .map_err(|error| AotError::contract("aot.link.canonical", error.to_string()))?;
        let mut writer = BoundedDigest::new();
        writer
            .digest
            .update(b"runmat-selected-native-link-plan-v1\0");
        serde_json::to_writer(&mut writer, self)
            .map_err(|error| AotError::contract("aot.link.canonical", error.to_string()))?;
        Ok(writer.digest.finish())
    }
}

pub(super) fn selected_link_plan(
    selected: &SelectedNativeProduct<'_>,
    admission: &super::ExternalNativeLinkAdmission,
    inputs: NativeProgramLinkIdentity,
    binding_mode: runmat_native_codegen::aot::AotRuntimeBindingMode,
) -> AotResult<CanonicalSelectedLinkPlan> {
    use super::NativeHostedDependency;
    use runmat_execution::host::{NativeArchitecture, NativeOperatingSystem};

    let manifest = selected.contents().manifest();
    let target = &manifest.target.target;
    let target_triple = manifest
        .target
        .triple()
        .map_err(|error| AotError::contract("aot.link.canonical", error.to_string()))?;
    let (family, mut options) = match &admission.adapter {
        super::NativeLinkAdapter::ClangDriver { .. } => {
            if target.operating_system == NativeOperatingSystem::Windows {
                return Err(AotError::contract(
                    "aot.link.canonical",
                    "MSVC targets require the typed MSVC linker adapter",
                ));
            }
            (
                LinkerFamily::UnixCc,
                vec![
                    SelectedLinkOption::TargetTriple,
                    SelectedLinkOption::Sysroot,
                    SelectedLinkOption::CompanionLinker,
                    SelectedLinkOption::NoStandardLibrary,
                    SelectedLinkOption::NoStartFiles,
                    SelectedLinkOption::NoDefaultLibraries,
                ],
            )
        }
        super::NativeLinkAdapter::MsvcLinker => {
            if target.operating_system != NativeOperatingSystem::Windows
                || !matches!(
                    target.architecture,
                    NativeArchitecture::X86_64 | NativeArchitecture::Aarch64
                )
            {
                return Err(AotError::contract(
                    "aot.link.canonical",
                    "MSVC linker adapter requires a supported Windows target",
                ));
            }
            (
                LinkerFamily::Msvc,
                vec![
                    SelectedLinkOption::MsvcMachine,
                    SelectedLinkOption::MsvcNoDefaultLibraries,
                    SelectedLinkOption::MsvcStackReserve {
                        bytes: super::plan::WINDOWS_STACK_RESERVE_BYTES,
                    },
                ],
            )
        }
    };
    let reached_hosted = selected
        .contents()
        .selection()
        .link_inputs()
        .iter()
        .filter_map(|input| match input {
            NativeLinkInput::HostedLibrary { name, .. } => {
                Some(NativeHostedDependency::Library(name.clone()))
            }
            NativeLinkInput::HostedFramework { name, .. } => {
                Some(NativeHostedDependency::Framework(name.clone()))
            }
            NativeLinkInput::Component { .. } => None,
        })
        .collect::<std::collections::BTreeSet<_>>();
    if !reached_hosted
        .iter()
        .eq(admission.hosted_dependencies.keys())
    {
        return Err(AotError::contract(
            "aot.link.canonical",
            "trusted hosted locator set differs from exact reached native dependencies",
        ));
    }
    if binding_mode == runmat_native_codegen::aot::AotRuntimeBindingMode::ClosedWorld {
        options.push(SelectedLinkOption::RemoveUnusedSections);
    }
    options.push(SelectedLinkOption::Output);
    Ok(CanonicalSelectedLinkPlan::new(
        selected
            .product()
            .selected()
            .expect("selected constructor")
            .clone(),
        inputs,
        admission.identity(),
        target_triple,
        family,
        options,
        selected.contents().selection().link_inputs().to_vec(),
    ))
}

/// A checked recipe record and its identity. The private fields keep the
/// typed evidence and digest together; neither grants linker execution.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CanonicalSelectedLinkReceipt {
    record: CanonicalSelectedLinkPlan,
    identity: Digest,
}

impl CanonicalSelectedLinkReceipt {
    pub(super) fn new(record: CanonicalSelectedLinkPlan) -> AotResult<Self> {
        let identity = record.identity()?;
        Ok(Self { record, identity })
    }

    pub fn record(&self) -> &CanonicalSelectedLinkPlan {
        &self.record
    }

    pub fn identity(&self) -> Digest {
        self.identity
    }
}

struct BoundedDigest {
    digest: DigestStream,
    bytes: usize,
}

impl BoundedDigest {
    fn new() -> Self {
        Self {
            digest: DigestStream::new(),
            bytes: 0,
        }
    }
}

impl Write for BoundedDigest {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.bytes = self
            .bytes
            .checked_add(bytes.len())
            .filter(|size| *size <= MAX_CANONICAL_BYTES)
            .ok_or_else(|| io::Error::other("selected link plan exceeds canonical bound"))?;
        self.digest.update(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_execution::host::{
        NativeDeploymentProfile, NativeNumericalBehavior, NativeNumericalPolicy,
        NativeOptimization, NATIVE_NUMERICAL_POLICY_SCHEMA_VERSION,
        NATIVE_SELECTED_PRODUCT_SCHEMA_VERSION,
    };
    use runmat_execution_artifact::target_pack::{ComponentId, NativeLinkRetention, TargetPackId};
    use runmat_package::NormalizedRelativePath;
    use runmat_types::{CapabilitySet, EffectSet};

    fn digest(name: &str) -> Digest {
        Digest::sha256(name)
    }

    fn fixture() -> CanonicalSelectedLinkPlan {
        let target = runmat_native_codegen::NativeTarget::current();
        let (target, probe) = target.native_parts().expect("native AOT test host");
        let product = NativeSelectedProductBinding {
            schema_version: NATIVE_SELECTED_PRODUCT_SCHEMA_VERSION,
            pack: TargetPackId::from_digest(digest("pack")).unwrap(),
            selection: digest("selection"),
            target: target.clone(),
            abi: probe.binding().unwrap(),
            runtime: digest("runtime"),
            catalog: digest("catalog"),
            runtime_component: digest("runtime component"),
            platform_service_abi: digest("platform services"),
            numerical: NativeNumericalPolicy {
                schema_version: NATIVE_NUMERICAL_POLICY_SCHEMA_VERSION,
                backend: digest("backend"),
                behavior: NativeNumericalBehavior::PlatformAccurate,
            },
            optimization: NativeOptimization::Speed,
            profile: NativeDeploymentProfile::HostedNative,
            reached_capabilities: CapabilitySet::default(),
            available_runtime_capabilities: CapabilitySet::default(),
            authorized_runtime_capabilities: CapabilitySet::default(),
            reached_effects: EffectSet::default(),
            required_services: Default::default(),
            services: Vec::new(),
        };
        let inputs = NativeProgramLinkIdentity {
            object: digest("object"),
            runtime_archive: digest("archive"),
            executable_cache_key: digest("executable"),
            native_cache_key: digest("native"),
        };
        let build = NativeLinkBuildIdentity {
            product: product.fingerprint().unwrap(),
            execution_host: super::super::NativeLinkExecutionHost::current().unwrap(),
            adapter: super::super::NativeLinkAdapter::MsvcLinker,
            entrypoint: NormalizedRelativePath::new("bin/linker").unwrap(),
            toolchain: digest("toolchain"),
            sysroot: digest("sysroot"),
            qualification: digest("qualification"),
            storage: super::super::NativeStorageGrant {
                archive: runmat_package_cache::ArchiveLimits::default(),
                metadata_bytes: 1024 * 1024,
            },
            hosted_dependencies: Vec::new(),
        };
        CanonicalSelectedLinkPlan::new(
            product,
            inputs,
            build,
            target.triple().unwrap(),
            super::super::LinkerFamily::Msvc,
            vec![
                SelectedLinkOption::MsvcMachine,
                SelectedLinkOption::MsvcNoDefaultLibraries,
                SelectedLinkOption::MsvcStackReserve {
                    bytes: super::super::plan::WINDOWS_STACK_RESERVE_BYTES,
                },
                SelectedLinkOption::RemoveUnusedSections,
                SelectedLinkOption::Output,
            ],
            vec![
                NativeLinkInput::Component {
                    component: ComponentId::new("runtime").unwrap(),
                    retention: NativeLinkRetention::Referenced,
                },
                NativeLinkInput::Component {
                    component: ComponentId::new("numerical").unwrap(),
                    retention: NativeLinkRetention::WholeArchive,
                },
            ],
        )
    }

    #[test]
    fn selected_recipe_identity_ignores_staging_paths_but_binds_order_and_bytes() {
        let first = fixture();
        let second = fixture();
        assert_eq!(first.identity().unwrap(), second.identity().unwrap());
        let mut changed = first.clone();
        changed.inputs.object = digest("other object");
        assert_ne!(first.identity().unwrap(), changed.identity().unwrap());
        changed = first.clone();
        changed.build.toolchain = digest("other toolchain");
        assert_ne!(first.identity().unwrap(), changed.identity().unwrap());
        changed = first.clone();
        changed.native_inputs.reverse();
        assert_ne!(first.identity().unwrap(), changed.identity().unwrap());
        changed = first.clone();
        let stack = changed
            .options
            .iter_mut()
            .find_map(|option| match option {
                SelectedLinkOption::MsvcStackReserve { bytes } => Some(bytes),
                _ => None,
            })
            .unwrap();
        *stack /= 2;
        assert_ne!(first.identity().unwrap(), changed.identity().unwrap());
        changed = first.clone();
        changed
            .options
            .retain(|option| *option != SelectedLinkOption::RemoveUnusedSections);
        assert_ne!(first.identity().unwrap(), changed.identity().unwrap());
    }

    #[test]
    fn canonical_plan_rejects_unknown_wire_fields() {
        let mut encoded = serde_json::to_value(fixture()).unwrap();
        encoded
            .as_object_mut()
            .unwrap()
            .insert("future_authority".into(), serde_json::json!(true));
        assert!(serde_json::from_value::<CanonicalSelectedLinkPlan>(encoded).is_err());
    }

    #[test]
    fn canonical_plan_rejects_future_schema() {
        let mut plan = fixture();
        plan.schema_version += 1;
        assert!(plan.identity().is_err());
    }

    #[test]
    fn selected_physical_mutation_retires_private_proof() {
        let canonical = fixture();
        let mut plan = super::super::LinkPlan {
            schema_version: super::super::NATIVE_LINK_PLAN_SCHEMA_VERSION,
            product: runmat_execution::host::NativeProductMode::Selected {
                binding: Box::new(canonical.product.clone()),
            },
            selected_build: Some(canonical.build.clone()),
            inputs: canonical.inputs.clone(),
            target_triple: canonical.target_triple.clone(),
            driver: super::super::LinkerDriver {
                path: "/tmp/one/linker".into(),
                family: canonical.family,
                environment: Vec::new(),
            },
            object: "/tmp/one/object.o".into(),
            runtime_archive: "/tmp/one/runtime.a".into(),
            output: "/tmp/one/output".into(),
            response_tokens: vec!["/tmp/one/object.o".into()],
            selected_proof: None,
        }
        .bind_selected(canonical.clone())
        .unwrap();
        assert_eq!(
            plan.verify_selected_proof().unwrap(),
            Some(canonical.identity().unwrap())
        );
        let receipt = plan.selected_proof.clone().unwrap().into_receipt();
        assert_eq!(receipt.record(), &canonical);
        assert_eq!(receipt.identity(), canonical.identity().unwrap());
        let mut second_root = plan.clone();
        second_root.driver.path = "/tmp/two/linker".into();
        second_root.object = "/tmp/two/object.o".into();
        second_root.runtime_archive = "/tmp/two/runtime.a".into();
        second_root.output = "/tmp/two/output".into();
        second_root.response_tokens = vec!["/tmp/two/object.o".into()];
        second_root.selected_proof = None;
        let second_root = second_root.bind_selected(canonical.clone()).unwrap();
        assert_eq!(
            second_root.verify_selected_proof().unwrap(),
            plan.verify_selected_proof().unwrap(),
            "staging root must not enter semantic link identity"
        );
        plan.response_tokens.push("unexpected".into());
        assert!(plan.verify_selected_proof().is_err());
    }
}
