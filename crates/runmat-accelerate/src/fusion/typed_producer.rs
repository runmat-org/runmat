//! Typed scalar producer emission, independent of dispatch and physical indexing.
//! Root slicing does not grant permission to prune execution effects or bypass
//! selected-binding admission. Callers must establish both before execution.
use super::{expression::admitted_expression, primitive_expr, semantics::ResolvedProgram};
use super::{FusionGroupPlan, FusionOp};
use crate::graph::ValueId;
use anyhow::{anyhow, Result};
use runmat_runtime::builtins::common::spec::{FusionMathHelper, ScalarType};
use runmat_types::NumericClass;
use std::collections::{HashMap, HashSet};

pub(crate) struct TypedProducer {
    pub body: String,
    expressions: HashMap<ValueId, String>,
    helpers: Vec<(FusionMathHelper, ScalarType)>,
}
impl TypedProducer {
    /// None preserves complete elementwise emission and its error order.
    /// Some emits only admitted pure producers of the roots, retaining original
    /// operation order and temporary names. Leaves come from the indexing owner.
    pub fn emit(
        plan: &FusionGroupPlan,
        program: &ResolvedProgram,
        buffer_type: ScalarType,
        mut expressions: HashMap<ValueId, String>,
        roots: Option<&[ValueId]>,
    ) -> Result<Self> {
        let selected = roots
            .map(|roots| producer_indices(plan, &expressions, roots))
            .transpose()?;
        let mut body = String::new();
        let mut helpers = Vec::new();
        for (index, operation) in plan.operations.iter().enumerate() {
            if selected
                .as_ref()
                .is_some_and(|indices| !indices.contains(&index))
            {
                continue;
            }
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
                if selected.is_some() {
                    let entry = runmat_builtins::builtin_catalog_entry_by_name(
                        declaration.binding.builtin.name,
                    )
                    .ok_or_else(|| anyhow!("missing typed producer contract"))?;
                    if entry.contract.purity != runmat_builtins::BuiltinPurity::Pure
                        || entry.contract.workspace_effect.is_some()
                        || entry.contract.environment_effect.is_some()
                    {
                        return Err(anyhow!("typed producer slice requires a pure contract"));
                    }
                }
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

        if let Some(roots) = roots {
            for &root in roots {
                // An input-only producer must satisfy the same representation
                // contract as a computed root; absence of arithmetic is not
                // permission to downcast a Double leaf on an F32 provider.
                if program.floating_class(root)? == NumericClass::Double
                    && buffer_type == ScalarType::F32
                {
                    return Err(anyhow!(
                        "fusion provider cannot represent a Double operation"
                    ));
                }
            }
        }
        Ok(Self {
            body,
            expressions,
            helpers,
        })
    }
    pub fn expression(&self, id: ValueId) -> Result<&str> {
        self.expressions
            .get(&id)
            .map(String::as_str)
            .ok_or_else(|| anyhow!("missing typed fusion output"))
    }
    pub fn preamble(&self, program: &ResolvedProgram) -> Result<String> {
        // Shader text is itself the backend cache key. Include resolved facts
        // even when two otherwise identical programs happen to emit the same
        // arithmetic, so representation contracts do not share a stale key.
        let identity = program.cache_identity()?;
        let mut math = String::new();
        for &(helper, precision) in &self.helpers {
            let precision = match precision {
                ScalarType::F32 => runmat_accelerate_api::ProviderPrecision::F32,
                ScalarType::F64 => runmat_accelerate_api::ProviderPrecision::F64,
                _ => return Err(anyhow!("unsupported fusion math helper precision")),
            };
            match helper {
                FusionMathHelper::RealPower => {
                    math.push_str(&crate::kernel_math::power::functions(precision))
                }
                FusionMathHelper::Round => {
                    math.push_str(&crate::kernel_math::rounding::functions(precision))
                }
                FusionMathHelper::FloatingRemainder => math.push_str(
                    &crate::kernel_math::floating_remainder::functions(precision),
                ),
            }
        }

        Ok(format!("// {identity}\n{math}\n"))
    }
}
pub(crate) fn producer_indices(
    plan: &FusionGroupPlan,
    leaves: &HashMap<ValueId, String>,
    roots: &[ValueId],
) -> Result<HashSet<usize>> {
    let mut producers = HashMap::new();
    for (index, operation) in plan.operations.iter().enumerate() {
        let (_, output) = operands(operation);
        let output = output.ok_or_else(|| anyhow!("fusion operation has no output"))?;
        if leaves.contains_key(&output) || producers.insert(output, index).is_some() {
            return Err(anyhow!("ambiguous typed fusion producer"));
        }
    }
    let mut selected = HashSet::new();
    let mut pending: Vec<_> = roots
        .iter()
        .map(|&id| (id, plan.operations.len()))
        .collect();
    while let Some((id, consumer)) = pending.pop() {
        if leaves.contains_key(&id) {
            continue;
        }
        let index = *producers
            .get(&id)
            .ok_or_else(|| anyhow!("missing typed fusion producer"))?;
        if index >= consumer {
            return Err(anyhow!("cyclic or unordered typed fusion producer"));
        }
        if selected.insert(index) {
            let (inputs, _) = operands(&plan.operations[index]);
            pending.extend(inputs.iter().map(|&input| (input, index)));
        }
    }
    Ok(selected)
}
fn operands(operation: &FusionOp) -> (&[ValueId], Option<ValueId>) {
    match operation {
        FusionOp::Primitive { inputs, output, .. } | FusionOp::Builtin { inputs, output, .. } => {
            (inputs, *output)
        }
    }
}
pub(super) fn scalar_name(scalar: ScalarType) -> Result<&'static str> {
    match scalar {
        ScalarType::F32 => Ok("f32"),
        ScalarType::F64 => Ok("f64"),
        _ => Err(anyhow!("unsupported floating fusion buffer precision")),
    }
}
#[cfg(test)]
#[path = "typed_producer_tests.rs"]
mod tests;
