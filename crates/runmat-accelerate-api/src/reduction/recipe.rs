use super::*;
use crate::scalar_program::{ScalarNode, ScalarProgram};
use crate::ProviderPrecision;
use anyhow::Result;
use runmat_execution::Digest;
use runmat_types::FusionReductionContract;
use serde::Serialize;

/// Construction input, not an admitted recipe. No unchecked deserialization
/// path exists for the checked recipe or a selected submission.
#[derive(Debug, Clone, Serialize)]
pub struct ReductionRecipeDescriptor {
    pub contract: FusionReductionContract,
    /// Scalar producer/data lowering precision, independent of partial
    /// accumulator precision, public output precision and provider default.
    pub execution_precision: ProviderPrecision,
    pub producer_transport: crate::scalar_program::ScalarTransport,
    pub producer_implementations: Vec<crate::scalar_program::EmittedScalarImplementation>,
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
    semantic_identity: Digest,
    descriptor: ReductionRecipeDescriptor,
    sources: ReductionShaderSources,
    checked_scalar: Option<ScalarProgram>,
    scalar_inputs: Vec<u32>,
}

impl ProviderReductionRecipe {
    pub fn new(
        semantic_identity: Digest,
        descriptor: ReductionRecipeDescriptor,
        sources: ReductionShaderSources,
    ) -> Result<Self> {
        Self::construct(semantic_identity, descriptor, sources, None, Vec::new())
    }

    /// Bind one rederived scalar program to the physical recipe and the exact
    /// ordered input slots. Providers may submit only this checked form; the
    /// legacy constructor remains a non-executable geometry/source receipt.
    pub fn new_checked(
        semantic_identity: Digest,
        descriptor: ReductionRecipeDescriptor,
        sources: ReductionShaderSources,
        checked_scalar: ScalarProgram,
        scalar_inputs: Vec<u32>,
    ) -> Result<Self> {
        Self::construct(
            semantic_identity,
            descriptor,
            sources,
            Some(checked_scalar),
            scalar_inputs,
        )
    }

    fn construct(
        semantic_identity: Digest,
        descriptor: ReductionRecipeDescriptor,
        sources: ReductionShaderSources,
        checked_scalar: Option<ScalarProgram>,
        scalar_inputs: Vec<u32>,
    ) -> Result<Self> {
        validation::recipe(&descriptor, &sources)?;
        if let Some(program) = &checked_scalar {
            let root_class = program
                .wire()
                .nodes
                .iter()
                .find(|node| node.id() == program.wire().root)
                .map(ScalarNode::class)
                .ok_or_else(|| anyhow::anyhow!("checked reduction root is missing"))?;
            anyhow::ensure!(
                root_class == descriptor.contract.input_class,
                "checked scalar root class differs from reduction owner input"
            );
            let declared = || {
                program.wire().nodes.iter().filter_map(|node| match node {
                    ScalarNode::Input { id, class } => Some((*id, *class)),
                    _ => None,
                })
            };
            anyhow::ensure!(
                scalar_inputs.len() == descriptor.inputs.len()
                    && scalar_inputs.len() == declared().count(),
                "checked reduction scalar inputs differ from physical bindings"
            );
            anyhow::ensure!(
                scalar_inputs.iter().all(|id| scalar_inputs
                    .iter()
                    .filter(|candidate| *candidate == id)
                    .count()
                    == declared().filter(|(candidate, _)| candidate == id).count()),
                "checked reduction scalar input set differs from physical bindings"
            );
            for (id, binding) in scalar_inputs.iter().zip(&descriptor.inputs) {
                let class = declared()
                    .find_map(|(candidate, class)| (candidate == *id).then_some(class))
                    .ok_or_else(|| anyhow::anyhow!("missing checked scalar input binding"))?;
                let precision = match class {
                    runmat_types::NumericClass::Single => ProviderPrecision::F32,
                    runmat_types::NumericClass::Double => ProviderPrecision::F64,
                    _ => anyhow::bail!("nonfloating checked scalar input"),
                };
                anyhow::ensure!(
                    precision == binding.precision,
                    "checked scalar class differs from physical binding"
                );
            }
        } else {
            anyhow::ensure!(scalar_inputs.is_empty(), "unbound reduction scalar inputs");
        }
        let identity = Digest::sha256_serialized(&(
            "typed-reduction-physical-v4",
            semantic_identity,
            &descriptor,
            &sources,
            checked_scalar.as_ref().map(ScalarProgram::identity),
            &scalar_inputs,
        ))?;
        Ok(Self {
            identity,
            semantic_identity,
            descriptor,
            sources,
            checked_scalar,
            scalar_inputs,
        })
    }

    pub fn identity(&self) -> Digest {
        self.identity
    }
    pub fn semantic_identity(&self) -> Digest {
        self.semantic_identity
    }
    pub fn descriptor(&self) -> &ReductionRecipeDescriptor {
        &self.descriptor
    }
    pub fn sources(&self) -> &ReductionShaderSources {
        &self.sources
    }
    pub fn checked_scalar(&self) -> Option<&ScalarProgram> {
        self.checked_scalar.as_ref()
    }
    pub fn scalar_inputs(&self) -> &[u32] {
        &self.scalar_inputs
    }
}
