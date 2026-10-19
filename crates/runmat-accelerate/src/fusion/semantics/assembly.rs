//! Shared deterministic source inference and receipt assembly. Admission may
//! suspend, but neither driver has a separate source or cache authority.

use super::*;
use runmat_runtime::builtins::common::spec::{FusionAdmissionEvidence, FusionAdmissionWork};

pub(super) struct InferredOperation<'a> {
    pub operation: &'a FusionOp,
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
    // A selected POWER scalar is source-typed control, not a staging buffer.
    // Keep its original Double fact even under an opted physical downcast
    // policy. Arithmetic admission still owns whether any specialization exists.
    // Most ordinary unary/binary declarations have no POWER control. Leave
    // this vector unallocated for them; the exact control identity remains
    // selected and sorted on the POWER route below.
    let mut power_controls = Vec::new();
    for (operation, snapshot) in plan.operations.iter().zip(snapshots) {
        let control = (|| {
            let snapshot = snapshot.as_ref()?;
            if snapshot.entry.identity != runmat_builtins::POWER_CATALOG_ENTRY.identity {
                return None;
            }
            let inputs = match operation {
                FusionOp::Primitive { inputs, .. } | FusionOp::Builtin { inputs, .. } => inputs,
            };
            let [_, exponent] = inputs.as_slice() else {
                return None;
            };
            matches!(
                exact_scalar_control(leaves.value(*exponent)),
                Some(runmat_accelerate_api::PowerScalarControl::DoubleBits(_))
            )
            .then_some(*exponent)
        })();
        if let Some(control) = control {
            power_controls.push(control);
        }
    }
    power_controls.sort_unstable();
    power_controls.dedup();
    let effective_input = |id: ValueId, mut fact: ValueFact| {
        if precision_policy.permits_double_to_single()
            && provider_precision == ProviderPrecision::F32
            && power_controls.binary_search(&id).is_err()
        {
            if let runmat_types::ValueKindFact::Numeric(ref mut numeric) = fact.kind {
                if numeric.class == NumericClass::Double {
                    numeric.class = NumericClass::Single;
                }
            }
        }
        fact
    };
    let mut facts = FactIndex::new(plan, &leaves);
    for &(id, value) in leaves.values() {
        let mut fact = value_fact(value);
        // Authenticate the actual physical class first. An opted class is
        // semantic precision policy, never physical representation proof.
        leaves.refine_input_storage(id, value, &mut fact);
        facts.insert(id, effective_input(id, fact))?;
    }
    let declarations = SlotMap::new(&facts);
    let evidence = SlotMap::new(&facts);
    let sign = SlotMap::new(&facts);
    let arithmetic = SlotMap::new(&facts);
    let exact_products = SlotMap::new(&facts);
    let admission_work = SlotMap::new(&facts);
    Ok((
        ResolvedProgram {
            source_operations: source_operations(plan, snapshots)?,
            source_input_ids: plan.inputs.clone(),
            source_input_set: leaves.input_ids().to_vec(),
            source_constants: source_constants(plan)?,
            source_slot_constants: source_slot_constants(plan)?,
            source_root: plan.output,
            facts,
            source_inputs,
            precision_policy,
            provider_precision,
            binding_contracts: Vec::with_capacity(plan.operations.len()),
            declarations,
            evidence,
            sign,
            arithmetic,
            exact_products,
            admission_work,
        },
        leaves,
    ))
}

pub(super) fn inferred_operation<'a>(
    program: &ResolvedProgram,
    operation: &'a FusionOp,
    snapshot: Option<SemanticBinding>,
    leaves: &FusionInvocationLeaves<'_>,
    finite_facts_only: bool,
) -> Result<InferredOperation<'a>> {
    let (input_ids, output) = match operation {
        FusionOp::Primitive { inputs, output, .. } | FusionOp::Builtin { inputs, output, .. } => {
            (inputs.as_slice(), *output)
        }
    };
    let mut arguments = Vec::with_capacity(input_ids.len());
    for id in input_ids {
        arguments.push(
            program
                .facts
                .get(id)
                .cloned()
                .ok_or_else(|| anyhow!("missing fusion operand fact {id}"))?,
        );
    }
    let (fact, arguments) = if let Some(snapshot) = snapshot {
        // The selected pure one/two-input real declaration has no literal
        // control operand: binary arithmetic only consults literals for its
        // four-input `like` form, and trigonometric inference only for its
        // three-input form. Keep those other forms on the exact source-literal
        // route rather than projecting a whole tensor control vector here.
        let pure_numeric = snapshot
            .declaration
            .semantics
            .and_then(|semantics| semantics.finite_preparation_class())
            .is_some_and(|class| class.arity() == input_ids.len());
        let request = CallRequest {
            arguments,
            literals: if pure_numeric {
                Default::default()
            } else {
                request_literals::for_operands(input_ids, |id| leaves.value(id))
            },
            outputs: OutputSelection::default(),
        };
        let fact = if let Some(fact) =
            runmat_builtins::infer_fusion_numeric_fact(snapshot.entry, &request)
        {
            fact.map_err(|_| anyhow!("fusion call has no admitted output"))?
        } else if finite_facts_only {
            return Err(anyhow!(
                "selected fusion declaration has no bounded canonical fact route"
            ));
        } else {
            let inference = infer_catalog_call(snapshot.entry, &request);
            if !inference.diagnostics.is_empty() || inference.outputs.len() != 1 {
                return Err(anyhow!("fusion call has no admitted output"));
            }
            inference.outputs.into_iter().next().unwrap()
        };
        (fact, request.arguments)
    } else {
        let FusionOp::Primitive { op, .. } = operation else {
            return Err(anyhow!("fusion builtin has no selected binding"));
        };
        (primitive_fact(*op, &arguments)?, arguments)
    };
    Ok(InferredOperation {
        operation,
        input_ids,
        output,
        arguments,
        fact,
    })
}

pub(super) fn selected_values<'a>(
    leaves: &'a FusionInvocationLeaves<'_>,
    computed: &'a ComputedScalars,
    input_ids: &[ValueId],
) -> Vec<Option<&'a Value>> {
    let mut selected = Vec::with_capacity(input_ids.len());
    selected.extend(
        input_ids
            .iter()
            .map(|id| leaves.value(*id).or_else(|| computed.get(*id))),
    );
    selected
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
            program.sign.insert(output, sign)?;
        }
        if let Some(receipt) = validate_arithmetic(
            snapshot.declaration,
            inferred.input_ids,
            &inferred.arguments,
            &inferred.fact,
            values,
            admitted.arithmetic,
        )? {
            program.arithmetic.insert(output, receipt)?;
        }
        program.evidence.insert(output, admitted.numerical_domain)?;
        program.admission_work.insert(output, work)?;
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
            )?;
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
                    )?;
                }
            }
        }
        program.declarations.insert(output, snapshot)?;
    }
    program.facts.insert(output, inferred.fact)?;
    exact_product::record(
        program,
        inferred.operation,
        output,
        inferred.input_ids,
        values,
    )?;
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
    program.sign.insert(output, sign)?;
    program.admission_work.insert(output, work)?;
    Ok(())
}
