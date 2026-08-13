use std::collections::{BTreeMap, BTreeSet};

use runmat_hir::{FunctionArgDim, FunctionArgValidator, FunctionArgumentValidation, FunctionId};
use runmat_types::{
    CertaintyFact, DynamicReason, FactJoin, NumericClass, NumericDomain, NumericFact, ShapeFact,
    StorageFact, ValueFact, ValueKindFact,
};

use crate::{MirAssembly, MirBody};

use super::{call_graph, flow};
use crate::analysis::inference::FunctionSummary;

const MAX_SUMMARY_UPDATES: usize = 24;

pub(crate) struct ProgramAnalysis {
    pub summaries: BTreeMap<FunctionId, FunctionSummary>,
    pub parameters: BTreeMap<FunctionId, Vec<ValueFact>>,
    pub captures: BTreeMap<FunctionId, Vec<ValueFact>>,
    pub nonconverged: BTreeSet<FunctionId>,
    pub widened: BTreeSet<FunctionId>,
    pub diagnostics: Vec<crate::MirDiagnostic>,
}

pub(crate) fn analyze_program(assembly: &MirAssembly) -> ProgramAnalysis {
    let externally_reachable = call_graph::externally_reachable_functions(assembly);
    let mut parameters = assembly
        .bodies
        .iter()
        .map(|(function, body)| {
            (
                *function,
                parameter_facts(
                    body,
                    assembly
                        .functions
                        .get(function)
                        .map(|value| &value.argument_validations),
                    externally_reachable.contains(function),
                ),
            )
        })
        .collect::<BTreeMap<_, _>>();
    let declared_parameters = parameters.clone();
    let mut captures = assembly
        .bodies
        .iter()
        .map(|(function, body)| {
            (
                *function,
                body.locals
                    .iter()
                    .filter(|local| matches!(local.kind, crate::MirLocalKind::Capture))
                    .map(|_| ValueFact::never())
                    .collect(),
            )
        })
        .collect::<BTreeMap<_, Vec<_>>>();
    let graph = call_graph::call_graph(assembly);
    let components = call_graph::strongly_connected_components(&graph);
    let analysis_order = components.iter().flatten().copied().collect::<Vec<_>>();
    let mut summaries = initial_summaries(assembly);
    let mut nonconverged = BTreeSet::new();
    let mut widened = BTreeSet::new();
    let mut diagnostics = BTreeMap::new();

    let mut stabilized = false;
    for update in 0..MAX_SUMMARY_UPDATES {
        let mut changed = false;
        let mut changed_functions = BTreeSet::new();
        for function in &analysis_order {
            let Some(body) = assembly.bodies.get(function) else {
                continue;
            };
            let Some(program_function) = u32::try_from(function.0)
                .ok()
                .map(runmat_types::ProgramFunctionId)
            else {
                continue;
            };
            let result = flow::analyze_body(
                body,
                program_function,
                parameters.get(function).map_or(&[], Vec::as_slice),
                captures.get(function).map_or(&[], Vec::as_slice),
                &summaries,
            );
            for observation in &result.calls {
                if let Some(expected) = declared_parameters.get(&observation.callee) {
                    collect_argument_diagnostics(observation, expected, &mut diagnostics);
                }
                let Some(target) = parameters.get_mut(&observation.callee) else {
                    continue;
                };
                if join_parameter_observation(target, &observation.arguments, update > 3) {
                    changed = true;
                    changed_functions.insert(observation.callee);
                    if update > 3 {
                        widened.insert(observation.callee);
                    }
                }
            }
            if propagate_capture_observations(
                assembly,
                *function,
                body,
                &result.final_state,
                &mut captures,
                update > 3,
                &mut widened,
            ) {
                changed = true;
            }
            let next_outputs = joined_outputs(&result.return_facts);
            let current = summaries.get(function).expect("summary initialized");
            let outputs = join_output_vectors(&current.outputs, &next_outputs, update > 3);
            let next = FunctionSummary {
                outputs,
                outputs_complete: body.abi.varargout.is_none(),
                variadic_outputs: body.abi.varargout.is_some(),
                effects: union_effects(&current.effects, &result.final_state.effects),
                capabilities: union_capabilities(
                    &current.capabilities,
                    &result.final_state.capabilities,
                ),
            };
            if current != &next {
                summaries.insert(*function, next);
                changed = true;
                changed_functions.insert(*function);
                if update > 3 {
                    widened.insert(*function);
                }
            }
        }
        if !changed {
            stabilized = true;
            break;
        }
        nonconverged = changed_functions;
    }
    if stabilized {
        nonconverged.clear();
    }

    ProgramAnalysis {
        summaries,
        parameters,
        captures,
        nonconverged,
        widened,
        diagnostics: diagnostics.into_values().collect(),
    }
}

fn collect_argument_diagnostics(
    observation: &flow::CallObservation,
    expected: &[ValueFact],
    diagnostics: &mut BTreeMap<(usize, usize, usize), crate::MirDiagnostic>,
) {
    for (index, (argument, expected)) in observation.arguments.iter().zip(expected).enumerate() {
        let kind_mismatch = match (&argument.kind, &expected.kind) {
            (_, ValueKindFact::Unknown | ValueKindFact::Never) => false,
            (ValueKindFact::Unknown | ValueKindFact::Never, _) => false,
            (ValueKindFact::Numeric(left), ValueKindFact::Numeric(right)) => left != right,
            (left, right) => std::mem::discriminant(left) != std::mem::discriminant(right),
        };
        let shape_mismatch = match (argument.shape.known_dims(), expected.shape.known_dims()) {
            (Some(left), Some(right)) if left.iter().chain(&right).all(Option::is_some) => {
                !argument.shape.is_proven_equivalent(&expected.shape)
            }
            _ => false,
        };
        if !kind_mismatch && !shape_mismatch {
            continue;
        }
        let span = observation
            .argument_spans
            .get(index)
            .copied()
            .unwrap_or(observation.span);
        diagnostics
            .entry((span.start, span.end, index))
            .or_insert_with(|| {
                crate::MirDiagnostic::new(
                    "RM-MIR0012",
                    crate::MirDiagnosticSeverity::Error,
                    "argument is incompatible with the callee's arguments block",
                    span,
                )
                .with_primary_label(format!(
                    "argument {} cannot satisfy its declared contract",
                    index + 1
                ))
                .with_help("pass a value with the declared class and shape")
                .with_category("argument-validation")
            });
    }
}

fn initial_summaries(assembly: &MirAssembly) -> BTreeMap<FunctionId, FunctionSummary> {
    assembly
        .bodies
        .iter()
        .map(|(function, body)| {
            (
                *function,
                FunctionSummary {
                    outputs: body
                        .abi
                        .fixed_outputs
                        .iter()
                        .map(|_| ValueFact::never())
                        .collect(),
                    outputs_complete: body.abi.varargout.is_none(),
                    variadic_outputs: body.abi.varargout.is_some(),
                    effects: Default::default(),
                    capabilities: Default::default(),
                },
            )
        })
        .collect()
}

fn parameter_facts(
    body: &MirBody,
    validations: Option<&Vec<FunctionArgumentValidation>>,
    externally_reachable: bool,
) -> Vec<ValueFact> {
    body.locals
        .iter()
        .filter(|local| matches!(local.kind, crate::MirLocalKind::Parameter))
        .map(|local| {
            validations
                .into_iter()
                .flatten()
                .find(|validation| Some(validation.binding) == local.binding)
                .map(validation_fact)
                .unwrap_or_else(|| {
                    if externally_reachable {
                        ValueFact::unknown(DynamicReason::RuntimeValue)
                    } else {
                        ValueFact::never()
                    }
                })
        })
        .collect()
}

fn validation_fact(validation: &FunctionArgumentValidation) -> ValueFact {
    let kind = validation
        .class_name
        .as_deref()
        .and_then(class_fact)
        .or_else(|| {
            validation.validators.iter().find_map(|validator| {
                let FunctionArgValidator::A(classes) = validator else {
                    return None;
                };
                (classes.len() == 1)
                    .then(|| class_fact(&classes[0]))
                    .flatten()
            })
        })
        .unwrap_or(ValueKindFact::Unknown);
    let shape = validation.size.as_ref().map_or_else(
        || validator_shape(&validation.validators),
        |size| {
            let dims = [&size.rows, &size.cols]
                .into_iter()
                .map(|dimension| match dimension {
                    FunctionArgDim::Any => runmat_types::DimensionFact::Unknown,
                    FunctionArgDim::Exact(value) => runmat_types::DimensionFact::Known(*value),
                })
                .collect::<Vec<_>>();
            if dims
                .iter()
                .all(|dimension| matches!(dimension, runmat_types::DimensionFact::Known(1)))
            {
                ShapeFact::Scalar
            } else {
                ShapeFact::Shaped { dims }
            }
        },
    );
    let storage = if validation
        .validators
        .iter()
        .any(|validator| matches!(validator, FunctionArgValidator::Sparse))
    {
        StorageFact::Sparse
    } else {
        match shape {
            ShapeFact::Scalar => StorageFact::Scalar,
            ShapeFact::Unknown => StorageFact::Unknown,
            _ => StorageFact::Dense,
        }
    };
    let mut fact = ValueFact::unknown(DynamicReason::RuntimeValue);
    fact.kind = kind;
    fact.shape = shape;
    fact.storage = storage;
    fact.certainty = CertaintyFact::Symbolic;
    fact
}

fn class_fact(class: &str) -> Option<ValueKindFact> {
    let numeric = match class.to_ascii_lowercase().as_str() {
        "double" => Some(NumericClass::Double),
        "single" => Some(NumericClass::Single),
        "int8" => Some(NumericClass::Int8),
        "uint8" => Some(NumericClass::UInt8),
        "int16" => Some(NumericClass::Int16),
        "uint16" => Some(NumericClass::UInt16),
        "int32" => Some(NumericClass::Int32),
        "uint32" => Some(NumericClass::UInt32),
        "int64" => Some(NumericClass::Int64),
        "uint64" => Some(NumericClass::UInt64),
        _ => None,
    };
    if let Some(class) = numeric {
        return Some(ValueKindFact::Numeric(NumericFact {
            class,
            domain: NumericDomain::Real,
        }));
    }
    match class.to_ascii_lowercase().as_str() {
        "logical" => Some(ValueKindFact::Logical),
        "char" => Some(ValueKindFact::Character),
        "string" => Some(ValueKindFact::String),
        _ => None,
    }
}

fn validator_shape(validators: &[FunctionArgValidator]) -> ShapeFact {
    if validators
        .iter()
        .any(|validator| matches!(validator, FunctionArgValidator::Column))
    {
        ShapeFact::Shaped {
            dims: vec![
                runmat_types::DimensionFact::Unknown,
                runmat_types::DimensionFact::Known(1),
            ],
        }
    } else if validators
        .iter()
        .any(|validator| matches!(validator, FunctionArgValidator::TextScalar))
    {
        ShapeFact::Scalar
    } else if validators
        .iter()
        .any(|validator| matches!(validator, FunctionArgValidator::Vector))
    {
        ShapeFact::Ranked { rank: 2 }
    } else {
        ShapeFact::Unknown
    }
}

fn join_parameter_observation(
    current: &mut [ValueFact],
    observed: &[ValueFact],
    widen: bool,
) -> bool {
    let mut changed = false;
    for (slot, observed) in current.iter_mut().zip(observed) {
        let joined = if widen {
            runmat_types::FactWiden::widen(slot, observed)
        } else {
            slot.join(observed)
        };
        if *slot != joined {
            *slot = joined;
            changed = true;
        }
    }
    changed
}

fn propagate_capture_observations(
    assembly: &MirAssembly,
    parent: FunctionId,
    parent_body: &MirBody,
    state: &super::FlowState,
    captures: &mut BTreeMap<FunctionId, Vec<ValueFact>>,
    widen: bool,
    widened: &mut BTreeSet<FunctionId>,
) -> bool {
    let mut changed = false;
    for (function, metadata) in &assembly.functions {
        let Some(target) = captures.get_mut(function) else {
            continue;
        };
        for (index, capture) in metadata.captures.iter().enumerate() {
            if capture.from_function != parent {
                continue;
            }
            let Some(source) = parent_body
                .locals
                .iter()
                .find(|local| local.binding == Some(capture.binding))
                .and_then(|local| state.locals.get(local.id.0))
                .and_then(|local| local.fact.as_ref())
            else {
                continue;
            };
            let Some(slot) = target.get_mut(index) else {
                continue;
            };
            let joined = if widen {
                runmat_types::FactWiden::widen(slot, source)
            } else {
                slot.join(source)
            };
            if *slot != joined {
                *slot = joined;
                changed = true;
                if widen {
                    widened.insert(*function);
                }
            }
        }
    }
    changed
}

fn joined_outputs(returns: &[Vec<ValueFact>]) -> Vec<ValueFact> {
    let width = returns.iter().map(Vec::len).max().unwrap_or(0);
    (0..width)
        .map(|index| {
            returns
                .iter()
                .filter_map(|outputs| outputs.get(index))
                .cloned()
                .reduce(|left, right| left.join(&right))
                .unwrap_or_else(ValueFact::never)
        })
        .collect()
}

fn join_output_vectors(current: &[ValueFact], next: &[ValueFact], widen: bool) -> Vec<ValueFact> {
    let width = current.len().max(next.len());
    (0..width)
        .map(|index| match (current.get(index), next.get(index)) {
            (Some(left), Some(right)) if widen => runmat_types::FactWiden::widen(left, right),
            (Some(left), Some(right)) => left.join(right),
            (Some(value), None) | (None, Some(value)) => value.clone(),
            (None, None) => unreachable!(),
        })
        .collect()
}

fn union_effects(
    left: &runmat_types::EffectSet,
    right: &runmat_types::EffectSet,
) -> runmat_types::EffectSet {
    runmat_types::EffectSet(left.0.union(&right.0).copied().collect())
}

fn union_capabilities(
    left: &runmat_types::CapabilitySet,
    right: &runmat_types::CapabilitySet,
) -> runmat_types::CapabilitySet {
    runmat_types::CapabilitySet(left.0.union(&right.0).copied().collect())
}
