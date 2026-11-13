use crate::{
    AssignmentCreationPolicy, AssignmentShapePolicy, DynamicReason, FactInference, FactJoin,
    IndexSelectorFact, InferenceDiagnostic, MutationContract, PlaceMutationKind, ShapeFact,
    StorageFact, ValueFact, ValueKindFact,
};

pub fn infer_mutation(
    current: Option<&ValueFact>,
    assigned: &ValueFact,
    contract: MutationContract,
) -> FactInference {
    if current.is_none() && matches!(contract.creation, AssignmentCreationPolicy::ExistingOnly) {
        return failure(
            "RM-TYPE-MUTATION-CREATION",
            "mutation requires an existing target",
        );
    }
    if matches!(contract.kind, PlaceMutationKind::Delete) {
        let Some(current) = current else {
            return failure(
                "RM-TYPE-MUTATION-DELETE",
                "deletion requires an existing target",
            );
        };
        let mut result = current.clone();
        result.shape = crate::ShapeFact::Unknown;
        result.certainty = crate::CertaintyFact::Dynamic(DynamicReason::RuntimeValue);
        return FactInference::exact(result);
    }
    if let Some(current) = current {
        match contract.shape {
            AssignmentShapePolicy::Exact
                if !current.shape.is_proven_equivalent(&assigned.shape) =>
            {
                return failure(
                    "RM-TYPE-MUTATION-SHAPE",
                    "exact assignment requires matching shapes",
                )
            }
            AssignmentShapePolicy::ScalarExpansion
                if !assigned.is_scalar()
                    && !current.shape.is_proven_equivalent(&assigned.shape) =>
            {
                return failure(
                    "RM-TYPE-MUTATION-SHAPE",
                    "scalar-expansion assignment requires a scalar or matching shape",
                )
            }
            _ => {}
        }
    }
    FactInference::exact(assigned.clone())
}

pub fn infer_index_mutation(
    base: &ValueFact,
    selectors: &[IndexSelectorFact],
    assigned: &ValueFact,
    contract: MutationContract,
) -> FactInference {
    if selectors
        .iter()
        .any(|selector| matches!(selector, IndexSelectorFact::KnownOneBasedIndex(0)))
    {
        return failure(
            "RM-TYPE-MUTATION-INDEX",
            "assignment index must resolve to a positive one-based position",
        );
    }
    if matches!(contract.kind, PlaceMutationKind::Delete) {
        let mut output = base.clone();
        output.shape = ShapeFact::Unknown;
        output.certainty = crate::CertaintyFact::Dynamic(DynamicReason::RuntimeValue);
        return FactInference::exact(output);
    }
    if !matches!(contract.shape, AssignmentShapePolicy::MatlabCompatible) {
        let selected = crate::infer_index(
            base,
            if matches!(contract.kind, PlaceMutationKind::CellAssign) {
                crate::IndexKind::Brace
            } else {
                crate::IndexKind::Paren
            },
            selectors,
            crate::IndexResultContext::AssignmentTarget,
        );
        let validation = infer_mutation(Some(&selected.fact), assigned, contract);
        if !validation.diagnostics.is_empty() {
            return validation;
        }
    }
    // CreateArrayByIndex permits an absent target, but Unknown does not prove
    // absence: it can also describe an existing value supplied at runtime.
    // In particular, an absent numeric VM target starts as empty Double, not
    // as an empty array with the RHS class.
    let mut output = base.clone();
    let (shape, shape_may_change) = written_shape(&output.shape, selectors);
    output.shape = shape;
    let domain_may_change = matches!(
        (&base.kind, &assigned.kind),
        (
            ValueKindFact::Numeric(crate::NumericFact {
                domain: crate::NumericDomain::Real,
                ..
            }),
            ValueKindFact::Numeric(crate::NumericFact {
                domain: crate::NumericDomain::Complex,
                ..
            }) | ValueKindFact::Unknown
        )
    );
    let may_detach = !matches!(base.view, crate::ViewFact::Materialized)
        || !matches!(base.alias, crate::AliasFact::Unique);
    if shape_may_change || domain_may_change || may_detach {
        // Growth can re-stride host storage or restore a new resident handle;
        // complex promotion can allocate new storage even without growth.
        // The old view/alias and physical-layout facts are not receipts for it.
        output.layout = crate::LayoutFact::Unknown;
        output.contiguity = crate::ContiguityFact::Unknown;
        output.view = crate::ViewFact::Unknown;
        output.alias = crate::AliasFact::Unknown;
        if output.shape == ShapeFact::Unknown {
            output.certainty = crate::CertaintyFact::Dynamic(DynamicReason::RuntimeValue);
        }
    }
    if let ValueKindFact::Numeric(base_numeric) = &base.kind {
        if base_numeric.domain == crate::NumericDomain::Real {
            match &assigned.kind {
                ValueKindFact::Numeric(rhs) if rhs.domain == crate::NumericDomain::Complex => {
                    if selectors_guarantee_nonempty(&base.shape, selectors) {
                        output.kind = ValueKindFact::Numeric(crate::NumericFact {
                            class: base_numeric.class,
                            domain: crate::NumericDomain::Complex,
                        });
                    } else {
                        output.kind = ValueKindFact::Unknown;
                        output.certainty =
                            crate::CertaintyFact::Dynamic(DynamicReason::RuntimeValue);
                    }
                }
                ValueKindFact::Unknown => {
                    output.kind = ValueKindFact::Unknown;
                    output.certainty = crate::CertaintyFact::Dynamic(DynamicReason::RuntimeValue);
                }
                _ => {}
            }
        }
    }
    if let ValueKindFact::Cell(cell) = &mut output.kind {
        let payload = if matches!(contract.kind, PlaceMutationKind::CellAssign) {
            assigned
        } else if let ValueKindFact::Cell(assigned_cell) = &assigned.kind {
            assigned_cell.element.as_ref()
        } else {
            return failure(
                "RM-TYPE-CELL-PAREN-ASSIGN",
                "parenthesis assignment into a cell array requires a cell value",
            );
        };
        cell.element = Box::new(FactJoin::join(cell.element.as_ref(), payload));
        if selectors.len() == 1 {
            if let Some(index) =
                crate::rules::indexing::known_linear_index(&output.shape, selectors)
            {
                if cell.elements_complete {
                    if index > cell.elements.len() {
                        cell.element =
                            Box::new(FactJoin::join(cell.element.as_ref(), &empty_cell_payload()));
                    }
                    cell.elements.resize_with(index + 1, empty_cell_payload);
                    cell.elements[index] = payload.clone();
                } else {
                    cell.elements.clear();
                }
            } else {
                cell.elements.clear();
                cell.elements_complete = false;
            }
        } else {
            cell.elements.clear();
            cell.elements_complete = false;
        }
    }
    if matches!(base.kind, ValueKindFact::Numeric(_)) && output.storage == StorageFact::Scalar {
        // Even an in-bounds scalar write publishes a Tensor, not a primitive
        // numeric scalar. This also covers a shape-one result.
        output.storage = StorageFact::Dense;
    }
    FactInference::exact(output)
}

fn empty_cell_payload() -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(crate::NumericFact {
            class: crate::NumericClass::Double,
            domain: crate::NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(0), Some(0)]),
        StorageFact::Dense,
    )
}

// Mirror the runtime assignment plan's *result* shape, including its collapsed
// final subscript for N-D arrays. Unknown numeric selectors may grow: retaining
// an old exact extent would make subsequent static resource facts unsound.
fn written_shape(shape: &ShapeFact, selectors: &[IndexSelectorFact]) -> (ShapeFact, bool) {
    if selectors.is_empty() {
        return (ShapeFact::Unknown, true);
    }
    let Some(mut original) = exact_dimensions(shape) else {
        let bounded = selectors.iter().all(definitely_bounded_without_extent);
        return if bounded {
            (shape.clone(), false)
        } else {
            (ShapeFact::Unknown, true)
        };
    };
    if original.len() < 2 {
        original.resize(2, 1);
    }
    let Some(effective) = effective_dimensions(&original, selectors.len()) else {
        return (ShapeFact::Unknown, true);
    };
    let mut planned = effective.clone();
    for (dimension, selector) in selectors.iter().enumerate() {
        match selector {
            IndexSelectorFact::KnownOneBasedIndex(index) => {
                planned[dimension] = planned[dimension].max(*index);
            }
            IndexSelectorFact::Colon => {}
            IndexSelectorFact::End { offset } if *offset <= 0 => {}
            IndexSelectorFact::End { offset } => {
                let Ok(offset) = usize::try_from(*offset) else {
                    return (ShapeFact::Unknown, true);
                };
                let Some(end) = effective[dimension].checked_add(offset) else {
                    return (ShapeFact::Unknown, true);
                };
                planned[dimension] = end;
            }
            IndexSelectorFact::Logical(_) if effective[dimension] > 0 => {}
            _ => return (ShapeFact::Unknown, true),
        }
    }
    if planned == effective {
        return (shape.clone(), false);
    }
    if selectors.len() == 1 {
        let target = planned[0];
        let mut result = original.clone();
        let current = effective[0];
        if current == 0 {
            result = vec![1, target];
        } else if result.iter().skip(2).all(|extent| *extent == 1) && result[0] == 1 {
            result[1] = target;
            result.truncate(2);
        } else if result.iter().skip(1).all(|extent| *extent == 1) {
            result[0] = target;
            result.truncate(2);
        } else {
            let Some(leading) = result[..result.len() - 1]
                .iter()
                .try_fold(1usize, |count, extent| count.checked_mul(*extent))
            else {
                return (ShapeFact::Unknown, true);
            };
            *result.last_mut().expect("normalized shape") = target.div_ceil(leading.max(1));
        }
        return (known_shape(result), true);
    }
    if planned.len() < 2 {
        planned.resize(2, 1);
    }
    (known_shape(planned), true)
}

fn exact_dimensions(shape: &ShapeFact) -> Option<Vec<usize>> {
    shape.known_dims()?.into_iter().collect::<Option<Vec<_>>>()
}

fn effective_dimensions(shape: &[usize], count: usize) -> Option<Vec<usize>> {
    if count == 1 {
        return Some(vec![shape
            .iter()
            .try_fold(1usize, |n, d| n.checked_mul(*d))?]);
    }
    let mut effective = Vec::with_capacity(count);
    for dimension in 0..count {
        if dimension + 1 == count {
            effective.push(
                shape
                    .iter()
                    .skip(dimension)
                    .try_fold(1usize, |n, d| n.checked_mul(*d))?,
            );
        } else {
            effective.push(shape.get(dimension).copied().unwrap_or(1));
        }
    }
    Some(effective)
}

fn known_shape(dims: Vec<usize>) -> ShapeFact {
    ShapeFact::from(dims.into_iter().map(Some).collect::<Vec<_>>())
}

fn definitely_bounded_without_extent(selector: &IndexSelectorFact) -> bool {
    match selector {
        IndexSelectorFact::Colon => true,
        IndexSelectorFact::End { offset } => *offset <= 0,
        _ => false,
    }
}

fn selectors_guarantee_nonempty(shape: &ShapeFact, selectors: &[IndexSelectorFact]) -> bool {
    if selectors.is_empty() {
        return false;
    }
    if selectors.iter().all(|selector| {
        matches!(selector, IndexSelectorFact::KnownOneBasedIndex(index) if *index > 0)
            || matches!(selector, IndexSelectorFact::End { .. })
    }) {
        return true;
    }
    let Some(mut dims) = exact_dimensions(shape) else {
        return false;
    };
    if dims.len() < 2 {
        dims.resize(2, 1);
    }
    let Some(effective) = effective_dimensions(&dims, selectors.len()) else {
        return false;
    };
    selectors
        .iter()
        .enumerate()
        .all(|(dimension, selector)| match selector {
            IndexSelectorFact::KnownOneBasedIndex(index) => *index > 0,
            IndexSelectorFact::Colon => effective[dimension] > 0,
            IndexSelectorFact::End { .. } => true,
            _ => false,
        })
}

fn failure(code: &str, message: &str) -> FactInference {
    FactInference {
        fact: ValueFact::unknown(DynamicReason::ConflictingControlFlow),
        diagnostics: vec![InferenceDiagnostic::error(code, message)],
    }
}
