use crate::{
    CellFact, DynamicReason, FactInference, IndexKind, IndexResultContext, IndexSelectorFact,
    OutputListFact, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

pub fn infer_index(
    base: &ValueFact,
    kind: IndexKind,
    selectors: &[IndexSelectorFact],
    context: IndexResultContext,
) -> FactInference {
    if let Some(diagnostic) = selector_bounds_diagnostic(&base.shape, selectors, context) {
        return FactInference {
            fact: ValueFact::unknown(DynamicReason::ConflictingControlFlow),
            diagnostics: vec![diagnostic],
        };
    }
    match (&base.kind, kind) {
        (ValueKindFact::Cell(cell), IndexKind::Brace) => {
            return infer_cell_contents(cell, &base.shape, selectors, context)
        }
        (ValueKindFact::Cell(cell), IndexKind::Paren) => {
            return FactInference::exact(cell_container_result(base, cell, selectors))
        }
        (ValueKindFact::Struct(structure), IndexKind::Paren) => {
            let mut result = container_result(base, selectors);
            result.kind = ValueKindFact::Struct(structure.clone());
            return FactInference::exact(result);
        }
        (ValueKindFact::Object(object), IndexKind::Paren) => {
            let mut result = container_result(base, selectors);
            result.kind = ValueKindFact::Object(object.clone());
            return FactInference::exact(result);
        }
        (_, IndexKind::Brace) => {
            return FactInference {
                fact: ValueFact::unknown(DynamicReason::UnsupportedRepresentation),
                diagnostics: vec![crate::InferenceDiagnostic::error(
                    "RM-TYPE-BRACE-INDEX",
                    "brace indexing requires a cell-like value",
                )],
            }
        }
        _ => {}
    }

    let scalar = !selectors.is_empty()
        && selectors.iter().all(|selector| {
            matches!(
                selector,
                IndexSelectorFact::Scalar
                    | IndexSelectorFact::KnownOneBasedIndex(_)
                    | IndexSelectorFact::End { .. }
            )
        });
    let shape = if scalar {
        ShapeFact::Scalar
    } else {
        index_shape(&base.shape, selectors)
    };
    let storage = if shape.element_count() == Some(1) {
        StorageFact::Scalar
    } else {
        base.storage
    };
    let mut result = base.clone();
    result.shape = shape;
    result.storage = storage;
    result.view = crate::ViewFact::ReadOnlyView;
    result.alias = crate::AliasFact::Shared;
    FactInference::exact(result)
}

fn infer_cell_contents(
    cell: &CellFact,
    shape: &ShapeFact,
    selectors: &[IndexSelectorFact],
    context: IndexResultContext,
) -> FactInference {
    if let Some(index) = known_linear_index(shape, selectors) {
        if let Some(value) = cell.elements.get(index) {
            return FactInference::exact(value.clone());
        }
    }
    if matches!(
        context,
        IndexResultContext::ReadCommaList | IndexResultContext::FunctionArgumentExpansion
    ) {
        let (outputs, variadic) = if cell.elements_complete {
            match selectors {
                [IndexSelectorFact::Colon] => (cell.elements.clone(), false),
                [IndexSelectorFact::Numeric(selector)] => selector
                    .shape
                    .element_count()
                    .map_or((Vec::new(), true), |count| {
                        (vec![(*cell.element).clone(); count], false)
                    }),
                _ => (Vec::new(), true),
            }
        } else {
            (Vec::new(), true)
        };
        return FactInference::exact(ValueFact::scalar(ValueKindFact::OutputList(
            OutputListFact { outputs, variadic },
        )));
    }
    FactInference::exact((*cell.element).clone())
}

fn cell_container_result(
    base: &ValueFact,
    cell: &CellFact,
    selectors: &[IndexSelectorFact],
) -> ValueFact {
    let mut result = container_result(base, selectors);
    let mut selected = cell.clone();
    if let Some(index) = known_linear_index(&base.shape, selectors) {
        if let Some(value) = cell.elements.get(index) {
            selected.elements = vec![value.clone()];
            selected.elements_complete = true;
        } else {
            selected.elements.clear();
            selected.elements_complete = false;
        }
    } else if !matches!(selectors, [IndexSelectorFact::Colon]) {
        selected.elements.clear();
        selected.elements_complete = false;
    }
    result.kind = ValueKindFact::Cell(selected);
    result
}

pub(crate) fn known_linear_index(
    shape: &ShapeFact,
    selectors: &[IndexSelectorFact],
) -> Option<usize> {
    if selectors.len() == 1 {
        return match selectors[0] {
            IndexSelectorFact::KnownOneBasedIndex(index) => index.checked_sub(1),
            _ => None,
        };
    }
    let dimensions = shape.known_dims()?;
    let mut stride = 1usize;
    let mut offset = 0usize;
    for (dimension, selector) in selectors.iter().enumerate() {
        let index = match selector {
            IndexSelectorFact::KnownOneBasedIndex(index) => index.checked_sub(1)?,
            IndexSelectorFact::End { offset } => {
                let bound = *dimensions.get(dimension)?.as_ref()?;
                let resolved = isize::try_from(bound).ok()?.checked_add(*offset)?;
                usize::try_from(resolved.checked_sub(1)?).ok()?
            }
            _ => return None,
        };
        offset = offset.checked_add(index.checked_mul(stride)?)?;
        stride = stride.checked_mul(*dimensions.get(dimension)?.as_ref()?)?;
    }
    Some(offset)
}

fn container_result(base: &ValueFact, selectors: &[IndexSelectorFact]) -> ValueFact {
    let mut result = base.clone();
    result.shape = index_shape(&base.shape, selectors);
    result.view = crate::ViewFact::ReadOnlyView;
    result.alias = crate::AliasFact::Shared;
    result
}

fn index_shape(base: &ShapeFact, selectors: &[IndexSelectorFact]) -> ShapeFact {
    if selectors.is_empty() {
        return base.clone();
    }
    if selectors.len() == 1 {
        return selector_shape(&selectors[0], base);
    }
    let base_dimensions = dimensions(base, selectors.len().max(2));
    let mut output = Vec::with_capacity(selectors.len().max(2));
    for (dimension, selector) in selectors.iter().enumerate() {
        match selector {
            IndexSelectorFact::Scalar
            | IndexSelectorFact::KnownOneBasedIndex(_)
            | IndexSelectorFact::End { .. } => output.push(crate::DimensionFact::Known(1)),
            IndexSelectorFact::Colon => output.push(base_dimensions[dimension].clone()),
            IndexSelectorFact::Numeric(fact) => {
                output.push(
                    fact.shape
                        .element_count()
                        .map_or(crate::DimensionFact::Unknown, crate::DimensionFact::Known),
                );
            }
            IndexSelectorFact::Logical(_) => output.push(crate::DimensionFact::Unknown),
            IndexSelectorFact::Unknown => output.push(crate::DimensionFact::Unknown),
        }
    }
    output.resize(2, crate::DimensionFact::Known(1));
    ShapeFact::Shaped { dims: output }
}

fn selector_shape(selector: &IndexSelectorFact, base: &ShapeFact) -> ShapeFact {
    match selector {
        IndexSelectorFact::Scalar
        | IndexSelectorFact::KnownOneBasedIndex(_)
        | IndexSelectorFact::End { .. } => ShapeFact::Scalar,
        IndexSelectorFact::Colon => ShapeFact::from(vec![base.element_count(), Some(1)]),
        IndexSelectorFact::Numeric(fact) => fact.shape.clone(),
        IndexSelectorFact::Logical(_) => ShapeFact::from(vec![None, Some(1)]),
        IndexSelectorFact::Unknown => ShapeFact::Unknown,
    }
}

fn dimensions(shape: &ShapeFact, rank: usize) -> Vec<crate::DimensionFact> {
    let mut dimensions = match shape {
        ShapeFact::Unknown => vec![crate::DimensionFact::Unknown; rank],
        ShapeFact::Scalar => vec![crate::DimensionFact::Known(1); 2],
        ShapeFact::Ranked { rank } => vec![crate::DimensionFact::Unknown; *rank],
        ShapeFact::Shaped { dims } => dims.clone(),
    };
    dimensions.resize(rank, crate::DimensionFact::Known(1));
    dimensions
}

fn selector_bounds_diagnostic(
    base: &ShapeFact,
    selectors: &[IndexSelectorFact],
    context: IndexResultContext,
) -> Option<crate::InferenceDiagnostic> {
    if let Some(dimension) = selectors
        .iter()
        .position(|selector| matches!(selector, IndexSelectorFact::KnownOneBasedIndex(0)))
    {
        return Some(
            crate::InferenceDiagnostic::error(
                "RM-TYPE-INDEX-BOUNDS",
                "indices must be positive one-based positions",
            )
            .at_dimension(dimension),
        );
    }
    if matches!(
        context,
        IndexResultContext::AssignmentTarget | IndexResultContext::DeletionTarget
    ) {
        return None;
    }
    let bounds = if selectors.len() == 1 {
        vec![base.element_count()]
    } else {
        let dimensions = dimensions(base, selectors.len());
        dimensions
            .into_iter()
            .map(|dimension| match dimension {
                crate::DimensionFact::Known(value) => Some(value),
                crate::DimensionFact::Symbolic(_) | crate::DimensionFact::Unknown => None,
            })
            .collect()
    };
    for (dimension, (selector, bound)) in selectors.iter().zip(bounds).enumerate() {
        let Some(bound) = bound else { continue };
        let invalid = match selector {
            IndexSelectorFact::KnownOneBasedIndex(index) => *index == 0 || *index > bound,
            IndexSelectorFact::End { offset } => {
                let resolved = isize::try_from(bound)
                    .ok()
                    .and_then(|bound| bound.checked_add(*offset));
                !resolved.is_some_and(|index| index >= 1 && index <= bound as isize)
            }
            _ => false,
        };
        if invalid {
            return Some(
                crate::InferenceDiagnostic::error(
                    "RM-TYPE-INDEX-BOUNDS",
                    format!(
                        "index for dimension {} is outside the proven bound {bound}",
                        dimension + 1
                    ),
                )
                .at_dimension(dimension),
            );
        }
    }
    None
}
