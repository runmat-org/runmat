//! Position-preserving facts for recursively differentiated cell targets.
use super::*;
use runmat_types::{CellFact, InferenceDiagnostic};

pub(super) fn gradient(target: &ValueFact) -> ValueFact {
    let ValueKindFact::Cell(cell) = &target.kind else {
        return gradient_for_target(target);
    };
    let mut result = ValueFact::proven(
        ValueKindFact::Cell(CellFact {
            element: Box::new(gradient(&cell.element)),
            elements: cell.elements.iter().map(gradient).collect(),
            elements_complete: cell.elements_complete,
        }),
        target.shape.clone(),
        target.storage,
    );
    // Distinct outer containers may retain the same cached leaf adjoints.
    result.alias = AliasFact::Unknown;
    result
}

pub(super) fn check(target: &ValueFact, index: usize, diagnostics: &mut Vec<InferenceDiagnostic>) {
    match &target.kind {
        ValueKindFact::Cell(cell) => {
            // The common fact proves a leaf exists only when the cell is
            // known nonempty. Empty cells never visit their common fact.
            if !cell.elements_complete
                && target.shape.element_count().is_some_and(|count| count > 0)
            {
                check(&cell.element, index, diagnostics);
            }
            for element in &cell.elements {
                check(element, index, diagnostics);
            }
        }
        ValueKindFact::Object(_) => check_dlarray_data(target, index, diagnostics),
        // Struct metadata and network/table projections have their own
        // schemas; this cell traversal does not guess those schemas.
        ValueKindFact::Struct(_) | ValueKindFact::Unknown | ValueKindFact::Never => {}
        _ => diagnostics.push(argument_error(
            "RM-DL-DLGRADIENT-TARGET",
            "targets must be traced dlarray values or supported learnables trees",
            index,
        )),
    }
}
