//! Selected compilation metadata derives from authenticated retained bytes.
//! This owner does not qualify providers, execute code, or admit a linker.
use runmat_execution::host::{NativeProductMode, NativeSelectedProductBinding};
use runmat_execution::Digest;
use runmat_execution_artifact::target_pack::VerifiedTargetPackContents;

use crate::{NativeAssembly, NativeCodegenError, NativeCodegenResult, NativeTarget};

pub struct SelectedNativeProduct<'a> {
    contents: &'a VerifiedTargetPackContents<'a>,
    target: NativeTarget,
    product: NativeProductMode,
}

/// Selected authority from the qualified linker and generated linked-image
/// entry, as distinct from a caller-owned decoded product declaration.
pub struct LinkedNativeProduct {
    binding: NativeSelectedProductBinding,
}

impl LinkedNativeProduct {
    /// # Safety
    /// Only the generated launcher linked with exact verified object/runtime
    /// bytes may issue this. The binding must come from its immutable program
    /// symbol; arbitrary decoded manifests do not establish this authority.
    pub unsafe fn from_verified_link(
        binding: NativeSelectedProductBinding,
    ) -> NativeCodegenResult<Self> {
        binding.validate().map_err(error)?;
        Ok(Self { binding })
    }

    pub fn validate_assembly(&self, assembly: &NativeAssembly) -> NativeCodegenResult<()> {
        validate_selected_assembly(&self.binding, assembly)
    }

    pub fn binding(&self) -> &NativeSelectedProductBinding {
        &self.binding
    }
}

impl<'a> SelectedNativeProduct<'a> {
    pub fn from_verified(
        contents: &'a VerifiedTargetPackContents<'a>,
    ) -> NativeCodegenResult<Self> {
        let binding = contents.selected_product_binding().map_err(error)?;
        let target =
            NativeTarget::from_probe(contents.abi_probe().clone(), binding.target.cpu_baseline)?;
        if target.native_parts()?.0 != &binding.target {
            return Err(error("verified pack probe differs from selected target"));
        }
        Ok(Self {
            contents,
            target,
            product: NativeProductMode::Selected {
                binding: Box::new(binding),
            },
        })
    }

    pub fn contents(&self) -> &VerifiedTargetPackContents<'a> {
        self.contents
    }
    pub fn target(&self) -> &NativeTarget {
        &self.target
    }
    pub fn product(&self) -> &NativeProductMode {
        &self.product
    }

    pub fn source_evaluation_issuer(
        &self,
    ) -> NativeCodegenResult<Option<runmat_execution::host::VerifiedNativeServiceIssuer>> {
        let binding = self.product.selected().expect("selected constructor");
        if !binding
            .required_services
            .contains(&runmat_execution::host::NativeDeploymentServiceKind::SourceEvaluation)
        {
            return Ok(None);
        }
        self.contents
            .verified_service_issuer(
                runmat_execution::host::NativeDeploymentServiceKind::SourceEvaluation,
            )
            .map(Some)
            .map_err(error)
    }

    pub fn service_issuers(
        &self,
    ) -> NativeCodegenResult<Vec<runmat_execution::host::VerifiedNativeServiceIssuer>> {
        let binding = self.product.selected().expect("selected constructor");
        binding
            .services
            .iter()
            .map(|service| {
                match service.kind {
                    runmat_execution::host::NativeDeploymentServiceKind::SourceEvaluation
                    | runmat_execution::host::NativeDeploymentServiceKind::HostedDynamicDispatch
                    | runmat_execution::host::NativeDeploymentServiceKind::Effect(
                        runmat_types::EffectKind::Clock
                        | runmat_types::EffectKind::Network
                        | runmat_types::EffectKind::UserInterface
                        | runmat_types::EffectKind::HostCallback,
                    ) => {}
                    runmat_execution::host::NativeDeploymentServiceKind::Effect(
                        runmat_types::EffectKind::WorkspaceRead
                        | runmat_types::EffectKind::WorkspaceWrite
                        | runmat_types::EffectKind::EnvironmentRead
                        | runmat_types::EffectKind::EnvironmentWrite
                        | runmat_types::EffectKind::FilesystemRead
                        | runmat_types::EffectKind::FilesystemWrite
                        | runmat_types::EffectKind::Randomness,
                    ) => {}
                    runmat_execution::host::NativeDeploymentServiceKind::Effect(
                        runmat_types::EffectKind::MaySuspend
                        | runmat_types::EffectKind::MayThrow
                        | runmat_types::EffectKind::Unknown,
                    ) => {
                        return Err(error(format!(
                        "selected native executor has no implementation for reached service {:?}",
                        service.kind
                    )))
                    }
                }
                self.contents
                    .verified_service_issuer(service.kind)
                    .map_err(error)
            })
            .collect()
    }

    pub fn validate_assembly(&self, assembly: &NativeAssembly) -> NativeCodegenResult<()> {
        validate_selected_assembly(
            self.product.selected().expect("selected constructor"),
            assembly,
        )?;
        if assembly.target != self.target {
            return Err(error("Native IR target differs from verified pack probe"));
        }
        Ok(())
    }
}

fn validate_selected_assembly(
    binding: &NativeSelectedProductBinding,
    assembly: &NativeAssembly,
) -> NativeCodegenResult<()> {
    assembly.verify()?;
    validate_native_product(
        &NativeProductMode::Selected {
            binding: Box::new(binding.clone()),
        },
        &assembly.target,
        *assembly.program.runtime_fingerprint(),
        *assembly.program.catalog_fingerprint(),
        binding.optimization,
    )?;
    if assembly.target.native_parts()?.0 != &binding.target
        || assembly.required_runtime_services() != binding.required_services
        || !assembly
            .requirements
            .capabilities
            .0
            .is_subset(&binding.reached_capabilities.0)
        || !assembly
            .required_effects()
            .0
            .is_subset(&binding.reached_effects.0)
    {
        return Err(error(
            "Native IR requirements differ from selected pack admission",
        ));
    }
    Ok(())
}

/// Shared cross-envelope consistency, not authentication or execution admission.
pub fn validate_native_product(
    product: &NativeProductMode,
    target: &NativeTarget,
    runtime: Digest,
    catalog: Digest,
    optimization: super::NativeOptimization,
) -> NativeCodegenResult<()> {
    target.validate()?;
    product.validate().map_err(error)?;
    if let Some(binding) = product.selected() {
        let (selected, probe) = target.native_parts()?;
        if selected != &binding.target
            || probe.binding().map_err(error)? != binding.abi
            || runtime != binding.runtime
            || catalog != binding.catalog
            || optimization != binding.optimization
        {
            return Err(error(
                "selected product target, ABI, environment or optimization differs",
            ));
        }
    }
    Ok(())
}

fn error(message: impl std::fmt::Display) -> NativeCodegenError {
    NativeCodegenError::new("native.product.selection", message.to_string())
}
