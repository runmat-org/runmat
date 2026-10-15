//! Selected compilation metadata derives from authenticated retained bytes.
//! This owner does not qualify providers, execute code, or admit a linker.
use runmat_execution::host::NativeProductMode;
use runmat_execution::Digest;
use runmat_execution_artifact::target_pack::VerifiedTargetPackContents;

use crate::{NativeAssembly, NativeCodegenError, NativeCodegenResult, NativeTarget};

pub struct SelectedNativeProduct<'a> {
    contents: &'a VerifiedTargetPackContents<'a>,
    target: NativeTarget,
    product: NativeProductMode,
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

    pub fn validate_assembly(&self, assembly: &NativeAssembly) -> NativeCodegenResult<()> {
        assembly.verify()?;
        let binding = self.product.selected().expect("selected constructor");
        validate_native_product(
            &self.product,
            &assembly.target,
            *assembly.program.runtime_fingerprint(),
            *assembly.program.catalog_fingerprint(),
            binding.optimization,
        )?;
        if assembly.target != self.target
            || !assembly
                .requirements
                .capabilities
                .0
                .is_subset(&binding.reached_capabilities.0)
            || assembly
                .functions
                .iter()
                .flat_map(|function| &function.blocks)
                .flat_map(|block| &block.instructions)
                .any(|instruction| !instruction.effects.0.is_subset(&binding.reached_effects.0))
            || assembly
                .requirements
                .regions
                .iter()
                .any(|region| !region.effects.0.is_subset(&binding.reached_effects.0))
            || assembly
                .requirements
                .parallel
                .parfor_regions
                .iter()
                .any(|region| !region.effects.0.is_subset(&binding.reached_effects.0))
            || assembly
                .requirements
                .parallel
                .spmd_regions
                .iter()
                .any(|region| !region.effects.0.is_subset(&binding.reached_effects.0))
        {
            return Err(error(
                "Native IR requirements differ from selected pack admission",
            ));
        }
        Ok(())
    }
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
