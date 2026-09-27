//! Request-local type resolution. Cached structural plans never own runtime
//! facts or evidence about the contents of a different invocation's operands.

use std::collections::{BTreeMap, HashMap};

use crate::precision::PrecisionPolicy;
use anyhow::{anyhow, Result};
use runmat_accelerate_api::ProviderPrecision;
use runmat_builtins::{builtin_catalog_entry_by_name, infer_catalog_call, BuiltinBindingIdentity};
use runmat_execution::Digest;
use runmat_runtime::builtins::common::spec::{
    canonical_fusion_declaration, CanonicalFusionDeclaration, FusionAdmissionContext,
};
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
    source_inputs: Vec<ValueFact>,
    precision_policy: PrecisionPolicy,
    provider_precision: ProviderPrecision,
    /// Selected bindings are part of request identity, not inferred from output
    /// class or a spelling-based expression dispatch table.
    pub bindings: Vec<BuiltinBindingIdentity>,
    declarations: HashMap<ValueId, CanonicalFusionDeclaration>,
    evidence: BTreeMap<ValueId, runmat_accelerate_api::NumericalDomainEvidence>,
    admission_work: BTreeMap<ValueId, runmat_runtime::builtins::common::spec::FusionAdmissionWork>,
}

#[derive(Debug, thiserror::Error)]
#[error("fusion operation admission failed: {0}")]
pub(crate) struct FusionAdmissionFailure(pub anyhow::Error);

impl ResolvedProgram {
    pub async fn resolve(
        plan: &FusionGroupPlan,
        inputs: &[Value],
        provider_precision: ProviderPrecision,
        correlation: Option<crate::placement::PlacementCorrelationId>,
    ) -> Result<Self> {
        Self::resolve_observed(
            plan,
            inputs,
            provider_precision,
            PrecisionPolicy::current(),
            correlation,
        )
        .await
    }

    #[cfg(test)]
    pub async fn resolve_with_policy(
        plan: &FusionGroupPlan,
        inputs: &[Value],
        provider_precision: ProviderPrecision,
        precision_policy: PrecisionPolicy,
    ) -> Result<Self> {
        Self::resolve_observed(plan, inputs, provider_precision, precision_policy, None).await
    }

    async fn resolve_observed(
        plan: &FusionGroupPlan,
        inputs: &[Value],
        provider_precision: ProviderPrecision,
        precision_policy: PrecisionPolicy,
        correlation: Option<crate::placement::PlacementCorrelationId>,
    ) -> Result<Self> {
        if inputs.len() != plan.inputs.len() {
            return Err(anyhow!("fusion operand count does not match the plan"));
        }
        let source_inputs: Vec<_> = inputs.iter().map(value_fact).collect();
        let effective_input = |mut fact: ValueFact| {
            if precision_policy.permits_double_to_single()
                && provider_precision == ProviderPrecision::F32
            {
                if let runmat_types::ValueKindFact::Numeric(ref mut numeric) = fact.kind {
                    if numeric.class == NumericClass::Double {
                        numeric.class = NumericClass::Single;
                    }
                }
            }
            fact
        };
        let mut facts: HashMap<_, _> = plan
            .inputs
            .iter()
            .copied()
            .zip(source_inputs.iter().cloned().map(effective_input))
            .collect();
        for (id, value) in &plan.const_values {
            facts.insert(*id, effective_input(value_fact(value)));
        }
        let mut bindings = Vec::new();
        let mut declarations = HashMap::new();
        let mut evidence = BTreeMap::new();
        let mut admission_work = BTreeMap::new();
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
            let entry = match operation {
                FusionOp::Builtin { name, .. } => Some(
                    builtin_catalog_entry_by_name(name)
                        .ok_or_else(|| anyhow!("fusion builtin has no canonical type authority"))?,
                ),
                FusionOp::Primitive {
                    op: PrimitiveOp::ElemPow,
                    ..
                } => Some(&runmat_builtins::POWER_CATALOG_ENTRY),
                FusionOp::Primitive { .. } => None,
            };
            let fact = if let Some(entry) = entry {
                let [binding] = entry.bindings else {
                    return Err(anyhow!("fusion builtin binding is ambiguous"));
                };
                bindings.push(entry.binding_identity(binding));
                let declaration = canonical_fusion_declaration(entry.binding_identity(binding))
                    .ok_or_else(|| anyhow!("missing canonical fusion declaration"))?;
                let inference = infer_catalog_call(
                    entry,
                    &CallRequest {
                        arguments: arguments.clone(),
                        // Do not manufacture literal proofs from computed
                        // intermediates. Unknown controls stay unknown.
                        literals: LiteralContext::default(),
                        outputs: OutputSelection::default(),
                    },
                );
                if !inference.diagnostics.is_empty() || inference.outputs.len() != 1 {
                    return Err(anyhow!("fusion call does not have one admitted output"));
                }
                let mut fact = inference.outputs.into_iter().next().unwrap();
                if let Some(output) = output {
                    if let Some(admit) = declaration.semantics.and_then(|semantics| semantics.admit)
                    {
                        let values: Vec<_> = input_ids
                            .iter()
                            .map(|id| {
                                plan.inputs
                                    .iter()
                                    .position(|input| input == id)
                                    .and_then(|index| inputs.get(index))
                                    .or_else(|| plan.const_values.get(id))
                            })
                            .collect();
                        let mut observed = super::admission::AdmissionObservation::new(correlation);
                        let admitted = admit(FusionAdmissionContext {
                            operands: &values,
                            facts: &arguments,
                            work: &mut observed.work,
                        })
                        .await
                        .map_err(FusionAdmissionFailure)?;
                        admission_work.insert(*output, observed.completed());
                        let admitted = admitted.ok_or_else(|| {
                            anyhow!("fusion operation requires unavailable domain evidence")
                        })?;
                        fact.kind = admitted.output_kind;
                        evidence.insert(*output, admitted.numerical_domain);
                    }
                    declarations.insert(*output, declaration);
                }
                fact
            } else {
                let FusionOp::Primitive { op, .. } = operation else {
                    return Err(anyhow!("fusion builtin has no selected binding"));
                };
                primitive_fact(*op, &arguments)?
            };
            if let Some(output) = output {
                facts.insert(*output, fact);
            }
        }
        Ok(Self {
            facts,
            bindings,
            source_inputs,
            precision_policy,
            provider_precision,
            declarations,
            evidence,
            admission_work,
        })
    }

    pub fn fact(&self, id: ValueId) -> Result<&ValueFact> {
        self.facts
            .get(&id)
            .ok_or_else(|| anyhow!("missing fusion value fact {id}"))
    }

    pub fn uses_implicit_downcast(&self) -> bool {
        self.precision_policy.permits_double_to_single()
            && self.provider_precision == ProviderPrecision::F32
            && self.source_inputs.iter().any(|fact| {
                fact.numeric()
                    .is_some_and(|numeric| numeric.class == NumericClass::Double)
            })
    }

    pub fn declaration(&self, id: ValueId) -> Option<CanonicalFusionDeclaration> {
        self.declarations.get(&id).copied()
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
        let revisions: BTreeMap<_, _> = self
            .declarations
            .iter()
            .map(|(id, declaration)| {
                (
                    *id,
                    declaration
                        .semantics
                        .map(|semantics| (semantics.revision, semantics.helpers)),
                )
            })
            .collect();
        Ok(Digest::sha256(serde_json::to_vec(&(
            "typed-fusion-v1",
            runmat_builtins::BUILTIN_CATALOG_SCHEMA_VERSION,
            declarations,
            facts,
            &self.source_inputs,
            self.precision_policy,
            self.provider_precision,
            revisions,
            &self.evidence,
            &self.admission_work,
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
