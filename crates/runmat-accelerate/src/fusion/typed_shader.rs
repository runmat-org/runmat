use std::collections::HashMap;

use anyhow::Result;
use runmat_runtime::builtins::common::spec::ScalarType;

use super::semantics::ResolvedProgram;
use super::typed_producer::{scalar_name, TypedProducer};
use super::FusionGroupPlan;
use crate::graph::ValueId;

impl FusionGroupPlan {
    /// Buffer precision is a provider ABI choice; each source operation retains
    /// its own language precision. In particular, widening a Single temporary
    /// into an F64 output buffer cannot undo its intermediate rounding.
    pub(crate) fn generate_typed_wgsl(
        &self,
        outputs: &[ValueId],
        program: &ResolvedProgram,
        buffer_type: ScalarType,
    ) -> Result<String> {
        let buffer_name = scalar_name(buffer_type)?;
        let mut expressions = HashMap::new();
        for (index, id) in self.inputs.iter().enumerate() {
            expressions.insert(*id, format!("input{index}.data[i{index}]"));
        }
        let producer = TypedProducer::emit(self, program, buffer_type, expressions, None)?;
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
        Ok(format!(
            "{preamble}{}",
            self.build_wgsl_shader(
                buffer_name,
                &bindings,
                self.inputs.len() + outputs.len(),
                &producer.body,
                &writes,
            )
        ))
    }
}
