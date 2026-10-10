//! Shared deterministic source inference and receipt assembly. Admission may
//! suspend, but neither driver has a separate source or cache authority.

use super::*;
use runmat_runtime::builtins::common::spec::{FusionAdmissionEvidence, FusionAdmissionWork};

pub(super) struct InferredOperation<'a> {
    pub input_ids: &'a [ValueId],
    pub output: Option<ValueId>,
    pub arguments: Vec<ValueFact>,
    pub fact: ValueFact,
}

pub(super) fn source_assembly<'a>(
    plan: &'a FusionGroupPlan,
    inputs: &[&'a Value],
    provider_precision: ProviderPrecision,
    precision_policy: PrecisionPolicy,
    correlation: Option<crate::placement::PlacementCorrelationId>,
    snapshots: &[Option<SemanticBinding>],
) -> Result<(ResolvedProgram, FusionInvocationLeaves<'a>)> {
    if snapshots.len() != plan.operations.len() {
        return Err(anyhow!("fusion binding count differs from source"));
    }
    let leaves = FusionInvocationLeaves::new_observed(plan, inputs, correlation)?;
    if !plan
        .operations
        .iter()
        .zip(snapshots)
        .all(|(operation, snapshot)| semantic_bindings::matches(operation, *snapshot))
    {
        return Err(anyhow!("selected fusion binding differs from source"));
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
    let facts = leaves
        .values()
        .iter()
        .map(|(&id, &value)| {
            let mut fact = value_fact(value);
            // Authenticate the actual physical class first. An opted class is
            // semantic precision policy, never physical representation proof.
            leaves.refine_input_storage(id, value, &mut fact);
            (id, effective_input(fact))
        })
        .collect();
    Ok((
        ResolvedProgram {
            source_operations: source_operations(plan, snapshots)?,
            source_input_ids: plan.inputs.clone(),
            source_input_set: leaves.input_ids().clone(),
            source_constants: source_constants(plan)?,
            source_slot_constants: source_slot_constants(plan)?,
            source_root: plan.output,
            facts,
            source_inputs,
            precision_policy,
            provider_precision,
            binding_contracts: Vec::new(),
            declarations: HashMap::new(),
            evidence: BTreeMap::new(),
            sign: BTreeMap::new(),
            arithmetic: BTreeMap::new(),
            admission_work: BTreeMap::new(),
        },
        leaves,
    ))
}

pub(super) fn inferred_operation<'a>(
    program: &ResolvedProgram,
    operation: &'a FusionOp,
    snapshot: Option<SemanticBinding>,
    leaves: &FusionInvocationLeaves<'_>,
) -> Result<InferredOperation<'a>> {
    let (input_ids, output) = match operation {
        FusionOp::Primitive { inputs, output, .. } | FusionOp::Builtin { inputs, output, .. } => {
            (inputs.as_slice(), *output)
        }
    };
    let arguments = input_ids
        .iter()
        .map(|id| {
            program
                .facts
                .get(id)
                .cloned()
                .ok_or_else(|| anyhow!("missing fusion operand fact {id}"))
        })
        .collect::<Result<Vec<_>>>()?;
    let fact = if let Some(snapshot) = snapshot {
        let inference = infer_catalog_call(
            snapshot.entry,
            &CallRequest {
                arguments: arguments.clone(),
                literals: request_literals::for_operands(input_ids, leaves.values()),
                outputs: OutputSelection::default(),
            },
        );
        if !inference.diagnostics.is_empty() || inference.outputs.len() != 1 {
            return Err(anyhow!("fusion call has no admitted output"));
        }
        inference.outputs.into_iter().next().unwrap()
    } else {
        let FusionOp::Primitive { op, .. } = operation else {
            return Err(anyhow!("fusion builtin has no selected binding"));
        };
        primitive_fact(*op, &arguments)?
    };
    Ok(InferredOperation {
        input_ids,
        output,
        arguments,
        fact,
    })
}

pub(super) fn selected_values<'a>(
    leaves: &FusionInvocationLeaves<'a>,
    input_ids: &[ValueId],
) -> Vec<Option<&'a Value>> {
    input_ids
        .iter()
        .map(|id| leaves.values().get(id).copied())
        .collect()
}

pub(super) fn record_operation(
    program: &mut ResolvedProgram,
    mut inferred: InferredOperation<'_>,
    snapshot: Option<SemanticBinding>,
    values: &[Option<&Value>],
    admitted: Option<(FusionAdmissionEvidence, FusionAdmissionWork)>,
) -> Result<()> {
    if let Some(snapshot) = snapshot {
        program.binding_contracts.push(snapshot);
    }
    let Some(output) = inferred.output else {
        return Ok(());
    };
    if let Some((admitted, work)) = admitted {
        let snapshot = snapshot.ok_or_else(|| anyhow!("unbound fusion admission"))?;
        inferred.fact.kind = admitted.output_kind;
        if let Some(storage) = admitted.output_storage {
            let numeric = inferred
                .fact
                .numeric()
                .ok_or_else(|| anyhow!("publication carrier has no numeric output"))?;
            let elements = inferred
                .fact
                .shape
                .element_count()
                .ok_or_else(|| anyhow!("publication carrier has no exact geometry"))?;
            if numeric.domain != NumericDomain::Real
                || !matches!(numeric.class, NumericClass::Single | NumericClass::Double)
                || elements == 0
                || !matches!(
                    storage,
                    runmat_types::StorageFact::Dense | runmat_types::StorageFact::Scalar
                )
                || (storage == runmat_types::StorageFact::Scalar
                    && (elements != 1 || numeric.class != NumericClass::Double))
                || (inferred.fact.storage != runmat_types::StorageFact::Unknown
                    && inferred.fact.storage != storage)
            {
                return Err(anyhow!(
                    "operation publication carrier contradicts canonical output"
                ));
            }
            inferred.fact.storage = storage;
        }
        if let Some(sign) = admitted.sign {
            let numeric = inferred
                .fact
                .numeric()
                .ok_or_else(|| anyhow!("sign receipt has nonnumeric output"))?;
            if numeric.domain != NumericDomain::Real || numeric.class != sign.class() {
                return Err(anyhow!("sign receipt differs from admitted output class"));
            }
            program.sign.insert(output, sign);
        }
        if let Some(receipt) = validate_arithmetic(
            snapshot.declaration,
            inferred.input_ids,
            &inferred.arguments,
            &inferred.fact,
            values,
            admitted.arithmetic,
        )? {
            program.arithmetic.insert(output, receipt);
        }
        program.evidence.insert(output, admitted.numerical_domain);
        program.admission_work.insert(output, work);
    }
    if let Some(snapshot) = snapshot {
        if matches!(
            snapshot
                .declaration
                .semantics
                .and_then(|semantic| semantic.scalar_operation),
            Some(runmat_types::ScalarOperation::Cast(
                NumericClass::Single | NumericClass::Double
            ))
        ) && inferred.input_ids.len() == 1
            && inferred.fact.numeric().is_some_and(|numeric| {
                numeric.domain == NumericDomain::Real
                    && matches!(numeric.class, NumericClass::Single | NumericClass::Double)
            })
            && program.evidence.get(&inferred.input_ids[0])
                == Some(&runmat_accelerate_api::NumericalDomainEvidence::NonnegativeOrNaN)
        {
            program.evidence.insert(
                output,
                runmat_accelerate_api::NumericalDomainEvidence::NonnegativeOrNaN,
            );
        }
        if matches!(
            snapshot
                .declaration
                .semantics
                .and_then(|semantic| semantic.scalar_operation),
            Some(runmat_types::ScalarOperation::Cast(
                NumericClass::Single | NumericClass::Double
            ))
        ) && inferred.input_ids.len() == 1
        {
            if let (Some(input), Some(output_numeric)) = (
                program.sign.get(&inferred.input_ids[0]),
                inferred.fact.numeric(),
            ) {
                if inferred.arguments[0].numeric().is_some_and(|numeric| {
                    numeric.domain == NumericDomain::Real && numeric.class == input.class()
                }) && output_numeric.domain == NumericDomain::Real
                {
                    program.sign.insert(
                        output,
                        runmat_runtime::builtins::common::spec::FusionSignEvidence::NonnegativeOrNaN {
                            class: output_numeric.class,
                        },
                    );
                }
            }
        }
        program.declarations.insert(output, snapshot);
    }
    program.facts.insert(output, inferred.fact);
    Ok(())
}

pub(super) fn record_sign_refinement(
    program: &mut ResolvedProgram,
    output: ValueId,
    sign: runmat_runtime::builtins::common::spec::FusionSignEvidence,
    kind: runmat_types::ValueKindFact,
    storage: Option<runmat_types::StorageFact>,
    work: FusionAdmissionWork,
) -> Result<()> {
    let runmat_types::ValueKindFact::Numeric(numeric) = kind else {
        return Err(anyhow!("sign refinement requires a numeric output"));
    };
    if numeric.domain != NumericDomain::Real || numeric.class != sign.class() {
        return Err(anyhow!("sign refinement differs from output class"));
    }
    let fact = program
        .facts
        .get_mut(&output)
        .ok_or_else(|| anyhow!("sign refinement has no recorded output"))?;
    if fact.numeric().is_some_and(|existing| existing != numeric) {
        return Err(anyhow!("sign refinement changes a known numeric output"));
    }
    fact.kind = runmat_types::ValueKindFact::Numeric(numeric);
    if let Some(storage) = storage {
        fact.storage = storage;
    }
    program.sign.insert(output, sign);
    program.admission_work.insert(output, work);
    Ok(())
}
