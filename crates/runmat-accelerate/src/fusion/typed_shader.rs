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
    input_indices: Vec<usize>,
    abi: runmat_accelerate_api::ProviderFusedElementwiseAbi,
    exact_product_words: bool,
    implementations: Vec<runmat_accelerate_api::scalar_program::EmittedScalarImplementation>,
}

impl GeneratedElementwiseShader {
    #[cfg(test)]
    pub(crate) fn input_indices(&self) -> &[usize] {
        &self.input_indices
    }
    pub(crate) fn abi(&self) -> runmat_accelerate_api::ProviderFusedElementwiseAbi {
        self.abi
    }
    pub(crate) fn uses_exact_product_words(&self) -> bool {
        self.exact_product_words
    }
    pub(crate) fn implementations(
        &self,
    ) -> &[runmat_accelerate_api::scalar_program::EmittedScalarImplementation] {
        &self.implementations
    }
    pub(crate) fn source(&self) -> &str {
        &self.source
    }

    pub(crate) fn requirements(&self) -> ProviderShaderRequirements {
        self.requirements
    }

    pub(crate) fn into_parts(self) -> (String, ProviderShaderRequirements, Vec<usize>) {
        (self.source, self.requirements, self.input_indices)
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
        let controls = program.embedded_controls_for_precision(
            self,
            outputs,
            if buffer_type == ScalarType::F32 {
                runmat_accelerate_api::ProviderPrecision::F32
            } else {
                runmat_accelerate_api::ProviderPrecision::F64
            },
        );
        let input_indices = self
            .inputs
            .iter()
            .enumerate()
            .filter_map(|(index, id)| (!controls.contains(id)).then_some(index))
            .collect::<Vec<_>>();
        self.prepare_typed_shader_with_inputs(outputs, program, buffer_type, &input_indices)
    }

    /// Reuse one already prepared physical input projection across dispatch
    /// variants. Extra data bindings retain their original order; an omitted
    /// leaf must still be a receipt-authenticated control for these outputs.
    pub(crate) fn prepare_typed_shader_with_inputs(
        &self,
        outputs: &[ValueId],
        program: &ResolvedProgram,
        buffer_type: ScalarType,
        input_indices: &[usize],
    ) -> Result<GeneratedElementwiseShader> {
        let buffer_name = scalar_name(buffer_type)?;
        let checked_root = self.complete_single_output_root(outputs);
        let roots = checked_root.as_ref().map(std::slice::from_ref);
        let word_storage = buffer_type == ScalarType::F32
            && TypedProducer::requires_binary32_word_transport(self, program, roots)?;
        let storage_name = if word_storage { "u32" } else { buffer_name };
        let mut expressions = HashMap::new();
        let mut words = HashMap::new();
        let allowed_controls = program.embedded_controls_for_precision(
            self,
            outputs,
            if buffer_type == ScalarType::F32 {
                runmat_accelerate_api::ProviderPrecision::F32
            } else {
                runmat_accelerate_api::ProviderPrecision::F64
            },
        );
        if input_indices
            .iter()
            .any(|&index| index >= self.inputs.len())
            || input_indices.windows(2).any(|pair| pair[0] >= pair[1])
        {
            return Err(anyhow::anyhow!("invalid fusion physical input projection"));
        }
        let controls = self
            .inputs
            .iter()
            .enumerate()
            .filter_map(|(index, id)| (!input_indices.contains(&index)).then_some(*id))
            .collect::<Vec<_>>();
        if controls.iter().any(|id| !allowed_controls.contains(id)) {
            return Err(anyhow::anyhow!(
                "fusion physical projection omits a data input"
            ));
        }
        for (binding, &index) in input_indices.iter().enumerate() {
            let load = format!("input{binding}.data[i{binding}]");
            if word_storage {
                words.insert(self.inputs[index], load.clone());
            }
            expressions.insert(
                self.inputs[index],
                if word_storage {
                    format!("bitcast<f32>({load})")
                } else {
                    load
                },
            );
        }
        for id in controls {
            expressions.insert(id, String::new());
        }
        for id in allowed_controls {
            expressions.entry(id).or_default();
        }
        for id in outputs {
            if self.const_values.contains_key(id) && !program.is_invocation_input(*id) {
                expressions.entry(*id).or_default();
            }
        }
        // A single public result may use checked lowering only when it covers
        // every source operation. A partial slice could hide an effect or an
        // invalid operation that complete elementwise emission must report.
        let producer =
            TypedProducer::emit_with_words(self, program, buffer_type, expressions, roots, words)?;
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
            let binding = input_indices.len() + index;
            bindings.push_str(&format!(
                "@group(0) @binding({binding}) var<storage, read_write> output{index}: Tensor;\n"
            ));
            let stored = if word_storage {
                producer
                    .word_expression(*id)
                    .map(str::to_owned)
                    .unwrap_or_else(|| format!("bitcast<u32>({expression})"))
            } else {
                format!("{buffer_name}({expression})")
            };
            writes.push_str(&format!("    output{index}.data[g] = {stored};\n"));
        }
        let preamble = producer.preamble(program)?;
        let source = format!(
            "{preamble}{}",
            self.build_wgsl_shader_with_inputs(
                (input_indices.len(), buffer_name, storage_name),
                &bindings,
                input_indices.len() + outputs.len(),
                &producer.body,
                &writes,
                include_ieee_helpers,
            )
        );
        Ok(GeneratedElementwiseShader {
            abi: runmat_accelerate_api::ProviderFusedElementwiseAbi::BroadcastReal {
                precision: match buffer_type {
                    ScalarType::F32 => runmat_accelerate_api::ProviderPrecision::F32,
                    ScalarType::F64 => runmat_accelerate_api::ProviderPrecision::F64,
                    _ => unreachable!("floating buffer validated"),
                },
            },
            source,
            requirements,
            input_indices: input_indices.to_vec(),
            exact_product_words: producer.uses_exact_product_words(),
            implementations: producer.implementations().to_vec(),
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
