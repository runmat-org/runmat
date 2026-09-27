use std::collections::HashMap;

use anyhow::{anyhow, Result};
use runmat_runtime::builtins::common::spec::{FusionMathHelper, ScalarType};
use runmat_types::NumericClass;

use super::{expression::admitted_expression, primitive_expr, semantics::ResolvedProgram};
use super::{FusionGroupPlan, FusionOp};
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
        let mut body = String::new();
        let mut helpers = Vec::new();
        for (index, operation) in self.operations.iter().enumerate() {
            let (inputs, output) = match operation {
                FusionOp::Primitive { inputs, output, .. }
                | FusionOp::Builtin { inputs, output, .. } => (inputs, output),
            };
            let output = output.ok_or_else(|| anyhow!("fusion operation has no output"))?;
            let operation_type = match program.floating_class(output)? {
                NumericClass::Single => ScalarType::F32,
                NumericClass::Double => ScalarType::F64,
                _ => return Err(anyhow!("unsupported fusion arithmetic class")),
            };
            // An F32-only provider cannot execute an F64 source operation by
            // silently changing its class, including a Double interior node.
            if buffer_type == ScalarType::F32 && operation_type == ScalarType::F64 {
                return Err(anyhow!(
                    "fusion provider cannot represent a Double operation"
                ));
            }
            let operation_name = scalar_name(operation_type)?;
            let operands = inputs
                .iter()
                .map(|id| {
                    let expression = expressions
                        .get(id)
                        .ok_or_else(|| anyhow!("missing typed fusion operand"))?;
                    Ok((*id, format!("{operation_name}({expression})")))
                })
                .collect::<Result<HashMap<_, _>>>()?;
            let expression = if let Some(declaration) = program.declaration(output) {
                for helper in declaration
                    .semantics
                    .into_iter()
                    .flat_map(|semantics| semantics.helpers)
                {
                    if !helpers.contains(&(*helper, operation_type)) {
                        helpers.push((*helper, operation_type));
                    }
                }
                admitted_expression(declaration, inputs, &operands, operation_type)
            } else {
                match operation {
                    FusionOp::Primitive { op, .. } => primitive_expr(*op, inputs, &operands),
                    FusionOp::Builtin { .. } => {
                        return Err(anyhow!("fusion builtin was not admitted"))
                    }
                }
            }
            .ok_or_else(|| anyhow!("no typed fusion expression for the selected operation"))?;
            let temporary = format!("typed{index}");
            body.push_str(&format!(
                "    let {temporary}: {operation_name} = {expression};\n"
            ));
            expressions.insert(output, temporary);
        }
        let mut bindings = String::new();
        let mut writes = String::new();
        for (index, id) in outputs.iter().enumerate() {
            let expression = expressions
                .get(id)
                .ok_or_else(|| anyhow!("missing typed fusion output"))?;
            let binding = self.inputs.len() + index;
            bindings.push_str(&format!(
                "@group(0) @binding({binding}) var<storage, read_write> output{index}: Tensor;\n"
            ));
            writes.push_str(&format!(
                "    output{index}.data[g] = {buffer_name}({expression});\n"
            ));
        }
        // Shader text is itself the backend cache key. Include resolved facts
        // even when two otherwise identical programs happen to emit the same
        // arithmetic, so representation contracts do not share a stale key.
        let identity = program.cache_identity()?;
        let mut math = String::new();
        for (helper, precision) in helpers {
            let precision = match precision {
                ScalarType::F32 => runmat_accelerate_api::ProviderPrecision::F32,
                ScalarType::F64 => runmat_accelerate_api::ProviderPrecision::F64,
                _ => return Err(anyhow!("unsupported fusion math helper precision")),
            };
            match helper {
                FusionMathHelper::RealPower => {
                    math.push_str(&crate::kernel_math::power::functions(precision))
                }
            }
        }
        Ok(format!(
            "// {identity}\n{math}\n{}",
            self.build_wgsl_shader(
                buffer_name,
                &bindings,
                self.inputs.len() + outputs.len(),
                &body,
                &writes,
            )
        ))
    }
}

fn scalar_name(scalar: ScalarType) -> Result<&'static str> {
    match scalar {
        ScalarType::F32 => Ok("f32"),
        ScalarType::F64 => Ok("f64"),
        _ => Err(anyhow!("unsupported floating fusion buffer precision")),
    }
}
