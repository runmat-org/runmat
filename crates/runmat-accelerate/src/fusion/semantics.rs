//! Request-local type resolution. Cached structural plans never own runtime
//! facts or evidence about the contents of a different invocation's operands.

use std::collections::{BTreeMap, HashMap};

use anyhow::{anyhow, Result};
use runmat_builtins::{builtin_catalog_entry_by_name, infer_catalog_call, BuiltinBindingIdentity};
use runmat_execution::Digest;
use runmat_runtime::value_fact::value_fact;
use runmat_types::{
    infer_binary, infer_unary, CallRequest, LiteralContext, NumericClass, NumericDomain,
    OperatorKind, OutputSelection, ValueFact,
};
use runmat_value::Value;

use super::{FusionGroupPlan, FusionOp};
use crate::graph::{PrimitiveOp, ValueId};

#[derive(Debug)]
pub(crate) struct ResolvedProgram {
    facts: HashMap<ValueId, ValueFact>,
    /// Selected bindings are part of request identity, not inferred from output
    /// class or a spelling-based expression dispatch table.
    pub bindings: Vec<BuiltinBindingIdentity>,
}

impl ResolvedProgram {
    pub fn resolve(plan: &FusionGroupPlan, inputs: &[Value]) -> Result<Self> {
        if inputs.len() != plan.inputs.len() {
            return Err(anyhow!("fusion operand count does not match the plan"));
        }
        let mut facts: HashMap<_, _> = plan
            .inputs
            .iter()
            .copied()
            .zip(inputs.iter().map(value_fact))
            .collect();
        for (id, value) in &plan.const_values {
            facts.insert(*id, value_fact(value));
        }
        let mut bindings = Vec::new();
        for operation in &plan.operations {
            let (input_ids, output) = match operation {
                FusionOp::Primitive { inputs, output, .. }
                | FusionOp::Builtin { inputs, output, .. } => (inputs, output),
            };
            let arguments = input_ids
                .iter()
                .map(|id| {
                    facts
                        .get(id)
                        .cloned()
                        .ok_or_else(|| anyhow!("missing fusion operand fact {id}"))
                })
                .collect::<Result<Vec<_>>>()?;
            let fact = match operation {
                FusionOp::Primitive { op, .. } => primitive_fact(*op, &arguments)?,
                FusionOp::Builtin { name, .. } => {
                    let entry = builtin_catalog_entry_by_name(name)
                        .ok_or_else(|| anyhow!("fusion builtin has no canonical type authority"))?;
                    let [binding] = entry.bindings else {
                        return Err(anyhow!("fusion builtin binding is ambiguous"));
                    };
                    bindings.push(entry.binding_identity(binding));
                    let inference = infer_catalog_call(
                        entry,
                        &CallRequest {
                            arguments,
                            // Do not manufacture literal proofs from computed
                            // intermediates. Unknown controls stay unknown.
                            literals: LiteralContext::default(),
                            outputs: OutputSelection::default(),
                        },
                    );
                    if !inference.diagnostics.is_empty() || inference.outputs.len() != 1 {
                        return Err(anyhow!("fusion call does not have one admitted output"));
                    }
                    inference.outputs.into_iter().next().unwrap()
                }
            };
            if let Some(output) = output {
                facts.insert(*output, fact);
            }
        }
        Ok(Self { facts, bindings })
    }

    pub fn fact(&self, id: ValueId) -> Result<&ValueFact> {
        self.facts
            .get(&id)
            .ok_or_else(|| anyhow!("missing fusion value fact {id}"))
    }

    pub fn floating_class(&self, id: ValueId) -> Result<NumericClass> {
        let numeric = self
            .fact(id)?
            .numeric()
            .ok_or_else(|| anyhow!("fusion output is not proven numeric"))?;
        if numeric.domain != NumericDomain::Real
            || !matches!(numeric.class, NumericClass::Single | NumericClass::Double)
        {
            return Err(anyhow!(
                "fusion representation requires a non-floating materializer"
            ));
        }
        Ok(numeric.class)
    }

    /// Versioned canonical encoding of typed identities and full facts. The
    /// shader text is additionally hashed by the placement caller, capturing
    /// the selected binding's actual expression lowering, not a function address.
    pub fn cache_identity(&self) -> Result<Digest> {
        let facts: BTreeMap<_, _> = self.facts.iter().collect();
        let declarations = self
            .bindings
            .iter()
            .map(|identity| {
                let entry = builtin_catalog_entry_by_name(identity.builtin.name)
                    .ok_or_else(|| anyhow!("selected fusion binding is no longer declared"))?;
                Ok((identity, entry.contract, entry.placement))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Digest::sha256(serde_json::to_vec(&(
            "typed-fusion-v1",
            runmat_builtins::BUILTIN_CATALOG_SCHEMA_VERSION,
            declarations,
            facts,
        ))?))
    }
}

fn primitive_fact(operation: PrimitiveOp, arguments: &[ValueFact]) -> Result<ValueFact> {
    let operator = match operation {
        PrimitiveOp::Add => OperatorKind::Add,
        PrimitiveOp::Sub => OperatorKind::Subtract,
        PrimitiveOp::Mul => OperatorKind::MatrixMultiply,
        PrimitiveOp::ElemMul => OperatorKind::ElementwiseMultiply,
        PrimitiveOp::ElemDiv => OperatorKind::ElementwiseDivide,
        PrimitiveOp::ElemLeftDiv => OperatorKind::ElementwiseLeftDivide,
        PrimitiveOp::Neg => OperatorKind::UnaryMinus,
        PrimitiveOp::UPlus => OperatorKind::UnaryPlus,
        PrimitiveOp::LessEqual => OperatorKind::LessEqual,
        PrimitiveOp::Less => OperatorKind::Less,
        PrimitiveOp::Greater => OperatorKind::Greater,
        PrimitiveOp::GreaterEqual => OperatorKind::GreaterEqual,
        PrimitiveOp::Equal => OperatorKind::Equal,
        PrimitiveOp::NotEqual => OperatorKind::NotEqual,
        PrimitiveOp::Transpose => OperatorKind::Transpose,
        // Real operands alone do not establish power's result domain or its
        // exceptional-value semantics. The operation owner must supply proof.
        PrimitiveOp::Pow | PrimitiveOp::ElemPow => {
            return Err(anyhow!(
                "fusion power requires operation-owned domain evidence"
            ));
        }
    };
    let inference = match arguments {
        [operand] => infer_unary(operator, operand),
        [left, right] => infer_binary(operator, left, right),
        _ => return Err(anyhow!("invalid fusion primitive operand count")),
    };
    if !inference.diagnostics.is_empty() {
        return Err(anyhow!("fusion primitive operands are not admitted"));
    }
    Ok(inference.fact)
}

#[cfg(test)]
#[path = "semantics_tests.rs"]
mod tests;
