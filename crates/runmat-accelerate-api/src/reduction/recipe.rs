use super::*;
use crate::ProviderPrecision;
use anyhow::Result;
use runmat_execution::Digest;
use serde::Serialize;

/// Construction input, not an admitted recipe. No unchecked deserialization
/// path exists for the checked recipe or a selected submission.
#[derive(Debug, Clone, Serialize)]
pub struct ReductionRecipeDescriptor {
    pub config: ReductionRecipeConfig,
    pub inputs: Vec<ReductionInputBinding>,
    pub iteration_shape: Vec<usize>,
    pub output_shape: Vec<usize>,
    pub output_precision: ProviderPrecision,
    pub geometry: ReductionGeometry,
    pub partial: ReductionPartialLayout,
    pub work: ReductionRecipeWork,
}

#[derive(Debug, Clone, Serialize)]
pub struct ReductionShaderSources {
    pub single: String,
    pub pass1: String,
    pub pass2: String,
}

/// One immutable physical recipe for both strategies. Shader validation and
/// authentication of actual prepared buffers remain the executing provider's
/// responsibility; constructing this value is not an execution permission.
pub struct ProviderReductionRecipe {
    identity: Digest,
    descriptor: ReductionRecipeDescriptor,
    sources: ReductionShaderSources,
}

impl ProviderReductionRecipe {
    pub fn new(
        semantic_identity: Digest,
        descriptor: ReductionRecipeDescriptor,
        sources: ReductionShaderSources,
    ) -> Result<Self> {
        validation::recipe(&descriptor, &sources)?;
        let identity = Digest::sha256(serde_json::to_vec(&(
            "typed-reduction-physical-v2",
            semantic_identity,
            &descriptor,
            &sources,
        ))?);
        Ok(Self {
            identity,
            descriptor,
            sources,
        })
    }

    pub fn identity(&self) -> Digest {
        self.identity
    }
    pub fn descriptor(&self) -> &ReductionRecipeDescriptor {
        &self.descriptor
    }
    pub fn sources(&self) -> &ReductionShaderSources {
        &self.sources
    }
}
