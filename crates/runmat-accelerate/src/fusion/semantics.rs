//! Request-local type resolution. Cached structural plans never own runtime
//! facts or evidence about the contents of a different invocation's operands.

use std::collections::{BTreeMap, HashMap};

use crate::precision::PrecisionPolicy;
use anyhow::{anyhow, Result};
use runmat_accelerate_api::ProviderPrecision;
use runmat_builtins::infer_catalog_call;
use runmat_execution::Digest;
use runmat_runtime::builtins::common::spec::{CanonicalFusionDeclaration, FusionAdmissionContext};
use runmat_runtime::value_fact::value_fact;
use runmat_types::{
    infer_binary, infer_unary, CallRequest, LiteralContext, NumericClass, NumericDomain,
    OperatorKind, OutputSelection, ValueFact,
};
use runmat_value::Value;

use super::semantic_bindings::{self, SemanticBinding};
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
    binding_contracts: Vec<SemanticBinding>,
    declarations: HashMap<ValueId, SemanticBinding>,
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
        let inputs: Vec<_> = inputs.iter().collect();
        Self::resolve_borrowed(plan, &inputs, provider_precision, correlation).await
    }

    pub async fn resolve_borrowed(
        plan: &FusionGroupPlan,
        inputs: &[&Value],
        provider_precision: ProviderPrecision,
        correlation: Option<crate::placement::PlacementCorrelationId>,
    ) -> Result<Self> {
        let bindings = semantic_bindings::capture(plan)?;
        Self::resolve_captured(plan, inputs, provider_precision, correlation, &bindings).await
    }

    pub async fn resolve_captured(
        plan: &FusionGroupPlan,
        inputs: &[&Value],
        provider_precision: ProviderPrecision,
        correlation: Option<crate::placement::PlacementCorrelationId>,
        bindings: &[Option<SemanticBinding>],
    ) -> Result<Self> {
        Self::resolve_observed(
            plan,
            inputs,
            provider_precision,
            PrecisionPolicy::current(),
            correlation,
            bindings,
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
        let inputs: Vec<_> = inputs.iter().collect();
        let bindings = semantic_bindings::capture(plan)?;
        Self::resolve_observed(
            plan,
            &inputs,
            provider_precision,
            precision_policy,
            None,
            &bindings,
        )
        .await
    }

    async fn resolve_observed(
        plan: &FusionGroupPlan,
        inputs: &[&Value],
        provider_precision: ProviderPrecision,
        precision_policy: PrecisionPolicy,
        correlation: Option<crate::placement::PlacementCorrelationId>,
        snapshots: &[Option<SemanticBinding>],
    ) -> Result<Self> {
        if inputs.len() != plan.inputs.len() || snapshots.len() != plan.operations.len() {
            return Err(anyhow!("fusion operand count does not match the plan"));
        }
        if !plan
            .operations
            .iter()
            .zip(snapshots)
            .all(|(operation, snapshot)| semantic_bindings::matches(operation, *snapshot))
        {
            return Err(anyhow!(
                "captured fusion binding does not match the operation"
            ));
        }
        let source_inputs: Vec<_> = inputs.iter().map(|value| value_fact(value)).collect();
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
        let mut binding_contracts = Vec::new();
        let mut declarations = HashMap::new();
        let mut evidence = BTreeMap::new();
        let mut admission_work = BTreeMap::new();
        for (operation, snapshot) in plan.operations.iter().zip(snapshots) {
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
            let fact = if let Some(snapshot) = snapshot {
                let entry = snapshot.entry;
                let declaration = snapshot.declaration;
                binding_contracts.push(*snapshot);
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
                                    .and_then(|index| inputs.get(index).copied())
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
                    declarations.insert(*output, *snapshot);
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
            binding_contracts,
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
        self.declarations
            .get(&id)
            .map(|snapshot| snapshot.declaration)
    }
    pub fn catalog_entry(
        &self,
        id: ValueId,
    ) -> Option<&'static runmat_builtins::BuiltinCatalogEntry> {
        self.declarations.get(&id).map(|snapshot| snapshot.entry)
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
            .binding_contracts
            .iter()
            .map(|snapshot| {
                (
                    snapshot.declaration.binding,
                    snapshot.entry.contract,
                    snapshot.entry.placement,
                )
            })
            .collect::<Vec<_>>();
        let revisions: BTreeMap<_, _> = self
            .declarations
            .iter()
            .map(|(id, declaration)| {
                (
                    *id,
                    declaration
                        .declaration
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
    // The host matrix-product owner delegates exactly to elementwise times
    // when either operand is scalar. Other matrix products require their own
    // contraction route: even an outer product adds the term to +0, unlike a
    // bare multiplication for signed zero, and Single uses a wider host sum.
    if operation == PrimitiveOp::Mul
        && !arguments
            .iter()
            .any(|argument| argument.shape.element_count() == Some(1))
    {
        return Err(anyhow!(
            "typed scalar fusion requires a proven scalar matrix-product operand"
        ));
    }
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
