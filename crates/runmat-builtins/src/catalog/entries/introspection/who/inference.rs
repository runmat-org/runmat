use crate::BuiltinCatalogEntry;
use runmat_types::*;

pub(super) fn infer(request: &CallRequest, _entry: &BuiltinCatalogEntry) -> CallInference {
    let analysis = super::super::workspace_query::analyze(request, "who");
    let element = ValueFact::scalar(ValueKindFact::String);
    let output = ValueFact::proven(
        ValueKindFact::Cell(CellFact {
            element: Box::new(element),
            elements: Vec::new(),
            elements_complete: false,
        }),
        ShapeFact::from(vec![None, Some(1)]),
        StorageFact::Dense,
    );
    let mut result = infer_call(&CallContract::fixed(vec![output]), request);
    super::super::workspace_query::apply_form_effects(&mut result, analysis.form, false);
    if !analysis.diagnostics.is_empty() {
        result.outputs.iter_mut().for_each(|output| {
            *output = ValueFact::unknown(DynamicReason::UnsupportedRepresentation)
        });
    }
    result.diagnostics.extend(analysis.diagnostics);
    result
}
