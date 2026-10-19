//! Exact selected binding identities for public moment conversions. These are
//! minted only from the sealed source invocation, not serializable authority.

use super::*;
use runmat_accelerate_api::{moment_precision, ProviderImageMomentPublication, ProviderPrecision};
use runmat_builtins::{BuiltinBindingIdentity, DOUBLE_CATALOG_ENTRY, SINGLE_CATALOG_ENTRY};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ImageSelectedMomentConversion {
    source_input: u64,
    source_output: u64,
    binding: BuiltinBindingIdentity,
    target: ProviderPrecision,
}

impl ImageSelectedMomentConversion {
    pub(crate) fn source_input(&self) -> u64 {
        self.source_input
    }

    pub(crate) fn source_output(&self) -> u64 {
        self.source_output
    }

    pub(crate) fn binding(&self) -> BuiltinBindingIdentity {
        self.binding
    }

    pub(crate) fn target(&self) -> ProviderPrecision {
        self.target
    }

    #[cfg(test)]
    pub(crate) fn fixture(
        source_input: u64,
        source_output: u64,
        binding: BuiltinBindingIdentity,
        target: ProviderPrecision,
    ) -> Self {
        Self {
            source_input,
            source_output,
            binding,
            target,
        }
    }

    #[cfg(test)]
    pub(crate) fn with_binding_for_test(mut self, binding: BuiltinBindingIdentity) -> Self {
        self.binding = binding;
        self
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ImageSelectedMomentBindings {
    semantic_identity: Digest,
    mu: Vec<Vec<ImageSelectedMomentConversion>>,
    variance: Vec<Vec<ImageSelectedMomentConversion>>,
    sigma: Vec<ImageSelectedMomentConversion>,
    sigma_raw: super::sigma_raw::ImageSelectedSigmaRaw,
}

impl ImageSelectedMomentBindings {
    pub(crate) fn semantic_identity(&self) -> Digest {
        self.semantic_identity
    }

    pub(crate) fn mu(&self) -> &[Vec<ImageSelectedMomentConversion>] {
        &self.mu
    }

    pub(crate) fn variance(&self) -> &[Vec<ImageSelectedMomentConversion>] {
        &self.variance
    }

    pub(crate) fn sigma(&self) -> &[ImageSelectedMomentConversion] {
        &self.sigma
    }

    pub(crate) fn sigma_raw(&self) -> super::sigma_raw::ImageSelectedSigmaRaw {
        self.sigma_raw
    }

    #[cfg(test)]
    pub(crate) fn fixture(
        semantic_identity: Digest,
        mu: Vec<Vec<ImageSelectedMomentConversion>>,
        variance: Vec<Vec<ImageSelectedMomentConversion>>,
        sigma: Vec<ImageSelectedMomentConversion>,
        sigma_raw: super::sigma_raw::ImageSelectedSigmaRaw,
    ) -> Self {
        Self {
            semantic_identity,
            mu,
            variance,
            sigma,
            sigma_raw,
        }
    }

    #[cfg(test)]
    pub(crate) fn replace_mu_for_test(
        &mut self,
        index: usize,
        conversions: Vec<ImageSelectedMomentConversion>,
    ) {
        self.mu[index] = conversions;
    }
}

impl ImageSourceInvocation {
    pub(crate) fn selected_moment_bindings(&self) -> Option<ImageSelectedMomentBindings> {
        let moments = self.moment_program.as_ref()?;
        if !moments.validate() || self.workload.as_ref()?.moment_program.as_ref()? != moments {
            return None;
        }
        let route = |calls: &[ProviderImageMomentPublication]| {
            calls
                .iter()
                .map(|call| self.conversion_bindings(call))
                .collect::<Option<Vec<_>>>()
        };
        Some(ImageSelectedMomentBindings {
            semantic_identity: self.semantic_identity,
            mu: route(&moments.mu)?,
            variance: route(&moments.variance)?,
            sigma: self.conversion_bindings(&moments.sigma)?,
            sigma_raw: self.selected_sigma_raw(&moments.sigma)?,
        })
    }

    fn conversion_bindings(
        &self,
        publication: &ProviderImageMomentPublication,
    ) -> Option<Vec<ImageSelectedMomentConversion>> {
        publication
            .conversions
            .iter()
            .map(|call| {
                let step = self
                    .source
                    .steps
                    .iter()
                    .find(|step| u64::from(step.output) == call.source_output)?;
                if step.inputs.as_slice() != [ValueId::try_from(call.source_input).ok()?] {
                    return None;
                }
                let entry = super::super::source_moment_publications::conversion_entry(step)?;
                let binding = step.selected_binding?;
                let target = moment_precision(&call.output_fact)?;
                let target_entry = match target {
                    ProviderPrecision::F32 => &SINGLE_CATALOG_ENTRY,
                    ProviderPrecision::F64 => &DOUBLE_CATALOG_ENTRY,
                };
                if entry.identity != target_entry.identity || binding.builtin != entry.identity {
                    return None;
                }
                let canonical = self.source_fact(step.output)?;
                if canonical.numeric() != call.output_fact.numeric()
                    || canonical.shape != call.output_fact.shape
                {
                    return None;
                }
                Some(ImageSelectedMomentConversion {
                    source_input: call.source_input,
                    source_output: call.source_output,
                    binding,
                    target,
                })
            })
            .collect()
    }
}
