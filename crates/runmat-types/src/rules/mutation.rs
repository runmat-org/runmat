use crate::{
    AssignmentCreationPolicy, AssignmentShapePolicy, CellFact, DynamicReason, FactInference,
    FactJoin, IndexSelectorFact, InferenceDiagnostic, MutationContract, PlaceMutationKind,
    ShapeFact, StorageFact, ValueFact, ValueKindFact,
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
    let mut output = if matches!(base.kind, ValueKindFact::Unknown)
        && matches!(
            contract.creation,
            AssignmentCreationPolicy::CreateArrayByIndex
        ) {
        created_index_target(assigned, contract.kind)
    } else {
        base.clone()
    };
    if selectors.len() == 1 {
        grow_linear_shape(&mut output.shape, &selectors[0]);
    } else if let ShapeFact::Shaped { dims } = &mut output.shape {
        for (dimension, selector) in selectors.iter().enumerate() {
            let Some(index) = (match selector {
                IndexSelectorFact::KnownOneBasedIndex(index) => Some(*index),
                _ => None,
            }) else {
                continue;
            };
            if dimension >= dims.len() {
                dims.resize(dimension + 1, crate::DimensionFact::Known(1));
            }
            if let crate::DimensionFact::Known(current) = &mut dims[dimension] {
                *current = (*current).max(index);
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
    FactInference::exact(output)
}

fn created_index_target(assigned: &ValueFact, kind: PlaceMutationKind) -> ValueFact {
    let shape = ShapeFact::from(vec![Some(0), Some(0)]);
    if matches!(kind, PlaceMutationKind::CellAssign) {
        return ValueFact::proven(
            ValueKindFact::Cell(CellFact {
                element: Box::new(assigned.clone()),
                elements: Vec::new(),
                elements_complete: true,
            }),
            shape,
            StorageFact::Dense,
        );
    }
    let mut output = assigned.clone();
    output.shape = shape;
    output.storage = StorageFact::Dense;
    output
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

fn grow_linear_shape(shape: &mut ShapeFact, selector: &IndexSelectorFact) {
    let IndexSelectorFact::KnownOneBasedIndex(index) = selector else {
        return;
    };
    let ShapeFact::Shaped { dims } = shape else {
        return;
    };
    dims.resize(2, crate::DimensionFact::Known(1));
    let known = dims
        .iter()
        .map(|dimension| match dimension {
            crate::DimensionFact::Known(value) => Some(*value),
            crate::DimensionFact::Symbolic(_) | crate::DimensionFact::Unknown => None,
        })
        .collect::<Option<Vec<_>>>();
    let Some(known) = known else { return };
    let current_count = known
        .iter()
        .try_fold(1usize, |count, dimension| count.checked_mul(*dimension));
    if current_count.is_some_and(|count| *index <= count) {
        return;
    }
    if known.len() != 2 {
        *shape = ShapeFact::Unknown;
        return;
    }
    let (rows, columns) = (known[0], known[1]);
    if rows == 1 {
        dims[1] = crate::DimensionFact::Known(*index);
    } else if columns == 1 {
        dims[0] = crate::DimensionFact::Known(*index);
    } else if rows > 0 {
        dims[1] = crate::DimensionFact::Known(index.div_ceil(rows));
    } else {
        dims[0] = crate::DimensionFact::Known(1);
        dims[1] = crate::DimensionFact::Known(*index);
    }
}

fn failure(code: &str, message: &str) -> FactInference {
    FactInference {
        fact: ValueFact::unknown(DynamicReason::ConflictingControlFlow),
        diagnostics: vec![InferenceDiagnostic::error(code, message)],
    }
}
