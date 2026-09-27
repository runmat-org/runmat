use crate::{
    catalog::entries::io::text_admission::{finish, input, reject, Role},
    BuiltinCatalogEntry,
};
use runmat_types::*;
pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if !(1..=2).contains(&request.arguments.len()) {
        reject(
            &mut diagnostics,
            0,
            "expected filename and optional output folder",
        );
    }
    input(request, 0, Role::Paths, &mut diagnostics);
    input(request, 1, Role::Scalar, &mut diagnostics);
    let element = ValueFact::proven(
        ValueKindFact::Character,
        ShapeFact::from(vec![Some(1), None]),
        StorageFact::Dense,
    );
    let output = ValueFact::proven(
        ValueKindFact::Cell(CellFact {
            element: Box::new(element),
            elements: vec![],
            elements_complete: false,
        }),
        ShapeFact::from(vec![None, Some(1)]),
        StorageFact::Dense,
    );
    finish(request, entry, output, diagnostics)
}
