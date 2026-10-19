//! Sealed source identity for the direct public-Sigma SQRT node.

use super::*;
use runmat_accelerate_api::ProviderImageMomentPublication;
use runmat_builtins::{BuiltinBindingIdentity, SQRT_CATALOG_ENTRY};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ImageSelectedSigmaRaw {
    source_input: u64,
    raw_output: u64,
    binding: BuiltinBindingIdentity,
    semantic_identity: Digest,
}

impl ImageSelectedSigmaRaw {
    pub(crate) fn source_input(&self) -> u64 {
        self.source_input
    }

    pub(crate) fn raw_output(&self) -> u64 {
        self.raw_output
    }

    pub(crate) fn semantic_identity(&self) -> Digest {
        self.semantic_identity
    }

    pub(crate) fn matches_publication(&self, publication: &ProviderImageMomentPublication) -> bool {
        self.source_input == publication.source_input
            && self.raw_output == publication.raw_output
            && self.binding.builtin == SQRT_CATALOG_ENTRY.identity
            && SQRT_CATALOG_ENTRY
                .bindings
                .iter()
                .any(|variant| SQRT_CATALOG_ENTRY.binding_identity(variant) == self.binding)
    }

    #[cfg(test)]
    pub(crate) fn fixture(
        source_input: u64,
        raw_output: u64,
        binding: BuiltinBindingIdentity,
        semantic_identity: Digest,
    ) -> Self {
        Self {
            source_input,
            raw_output,
            binding,
            semantic_identity,
        }
    }

    #[cfg(test)]
    pub(crate) fn binding(&self) -> BuiltinBindingIdentity {
        self.binding
    }
}

impl ImageSourceInvocation {
    pub(super) fn selected_sigma_raw(
        &self,
        publication: &ProviderImageMomentPublication,
    ) -> Option<ImageSelectedSigmaRaw> {
        let variance = self.moment_program.as_ref()?.variance.last()?;
        let shifted_id = ValueId::try_from(publication.source_input).ok()?;
        let shifted = self
            .source
            .steps
            .iter()
            .find(|step| step.output == shifted_id)?;
        if !matches!(
            shifted.label,
            crate::graph::AccelNodeLabel::Primitive(crate::graph::PrimitiveOp::Add)
        ) || shifted.inputs.len() != 2
            || u64::from(shifted.inputs[0]) != variance.public_output
            || shifted.selected_binding.is_some()
        {
            return None;
        }
        let step = self
            .source
            .steps
            .iter()
            .find(|step| u64::from(step.output) == publication.raw_output)?;
        if step.inputs.as_slice() != [shifted_id]
            || !matches!(step.label, crate::graph::AccelNodeLabel::Builtin { .. })
        {
            return None;
        }
        let binding = step.selected_binding?;
        if binding.builtin != SQRT_CATALOG_ENTRY.identity
            || !SQRT_CATALOG_ENTRY
                .bindings
                .iter()
                .any(|variant| SQRT_CATALOG_ENTRY.binding_identity(variant) == binding)
        {
            return None;
        }
        let canonical = self.source_fact(step.output)?;
        if canonical.numeric() != publication.raw_fact.numeric()
            || canonical.shape != publication.raw_fact.shape
            || canonical.storage != publication.raw_fact.storage
        {
            return None;
        }
        Some(ImageSelectedSigmaRaw {
            source_input: publication.source_input,
            raw_output: publication.raw_output,
            binding,
            semantic_identity: self.semantic_identity,
        })
    }
}
