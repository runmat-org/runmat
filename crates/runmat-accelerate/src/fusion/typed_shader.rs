use std::collections::HashMap;

use anyhow::Result;
use runmat_accelerate_api::ProviderShaderRequirements;
use runmat_runtime::builtins::common::spec::ScalarType;

use super::semantics::ResolvedProgram;
use super::typed_producer::{producer_indices, scalar_name, TypedProducer};
use super::FusionGroupPlan;
use crate::graph::ValueId;

/// Selected source and its shader feature requirement share one producer.
pub(crate) struct GeneratedElementwiseShader {
    source: String,
    requirements: ProviderShaderRequirements,
}

impl GeneratedElementwiseShader {
    pub(crate) fn source(&self) -> &str {
        &self.source
    }

    pub(crate) fn requirements(&self) -> ProviderShaderRequirements {
        self.requirements
    }

    pub(crate) fn into_parts(self) -> (String, ProviderShaderRequirements) {
        (self.source, self.requirements)
    }
}

impl FusionGroupPlan {
    /// Buffer precision is a provider ABI choice; each source operation retains
    /// its own language precision. In particular, widening a Single temporary
    /// into an F64 output buffer cannot undo its intermediate rounding.
    #[cfg(test)]
    pub(crate) fn generate_typed_wgsl(
        &self,
        outputs: &[ValueId],
        program: &ResolvedProgram,
        buffer_type: ScalarType,
    ) -> Result<String> {
        Ok(self
            .prepare_typed_shader(outputs, program, buffer_type)?
            .source)
    }

    pub(crate) fn prepare_typed_shader(
        &self,
        outputs: &[ValueId],
        program: &ResolvedProgram,
        buffer_type: ScalarType,
    ) -> Result<GeneratedElementwiseShader> {
        let buffer_name = scalar_name(buffer_type)?;
        let mut expressions = HashMap::new();
        for (index, id) in self.inputs.iter().enumerate() {
            expressions.insert(*id, format!("input{index}.data[i{index}]"));
        }
        // A single public result may use checked lowering only when it covers
        // every source operation. A partial slice could hide an effect or an
        // invalid operation that complete elementwise emission must report.
        let checked_root = self.complete_single_output_root(outputs);
        let roots = checked_root.as_ref().map(std::slice::from_ref);
        let producer = TypedProducer::emit(self, program, buffer_type, expressions, roots)?;
        // Checked scalar programs lower only arithmetic operators and casts;
        // legacy templates conservatively retain the IEEE/hypot helpers.
        let include_ieee_helpers = producer.checked_program().is_none();
        let requirements = match buffer_type {
            ScalarType::F32 => ProviderShaderRequirements::Basic,
            ScalarType::F64 if include_ieee_helpers => ProviderShaderRequirements::F64IeeeWords,
            ScalarType::F64 => ProviderShaderRequirements::Basic,
            ScalarType::I32 | ScalarType::Bool => {
                return Err(anyhow::anyhow!("typed fusion has no floating buffer class"));
            }
        };
        let mut bindings = String::new();
        let mut writes = String::new();
        for (index, id) in outputs.iter().enumerate() {
            let expression = producer.expression(*id)?;
            let binding = self.inputs.len() + index;
            bindings.push_str(&format!(
                "@group(0) @binding({binding}) var<storage, read_write> output{index}: Tensor;\n"
            ));
            writes.push_str(&format!(
                "    output{index}.data[g] = {buffer_name}({expression});\n"
            ));
        }
        let preamble = producer.preamble(program)?;
        let source = format!(
            "{preamble}{}",
            self.build_wgsl_shader(
                buffer_name,
                &bindings,
                self.inputs.len() + outputs.len(),
                &producer.body,
                &writes,
                include_ieee_helpers,
            )
        );
        Ok(GeneratedElementwiseShader {
            source,
            requirements,
        })
    }

    pub(crate) fn complete_single_output_root(&self, outputs: &[ValueId]) -> Option<ValueId> {
        let [root] = outputs else { return None };
        let leaves: HashMap<_, _> = self
            .inputs
            .iter()
            .chain(self.const_values.keys())
            .map(|&id| (id, String::new()))
            .collect();
        (producer_indices(self, &leaves, outputs).ok()?.len() == self.operations.len())
            .then_some(*root)
    }
}
