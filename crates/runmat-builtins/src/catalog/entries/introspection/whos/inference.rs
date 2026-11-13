use crate::BuiltinCatalogEntry;
use runmat_types::*;
use std::collections::BTreeMap;

pub(super) fn infer(request: &CallRequest, _entry: &BuiltinCatalogEntry) -> CallInference {
    let analysis = super::super::workspace_query::analyze(request, "whos");
    let text = ValueFact::scalar(ValueKindFact::String);
    let flag = ValueFact::scalar(ValueKindFact::Logical);
    let number = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    let size = ValueFact::proven(
        number.kind.clone(),
        ShapeFact::from(vec![Some(1), None]),
        StorageFact::Dense,
    );
    let fields = BTreeMap::from([
        ("name".to_owned(), text.clone()),
        ("size".to_owned(), size),
        ("bytes".to_owned(), number),
        ("class".to_owned(), text.clone()),
        ("global".to_owned(), flag.clone()),
        ("sparse".to_owned(), flag.clone()),
        ("complex".to_owned(), flag.clone()),
        ("nesting".to_owned(), text),
        ("persistent".to_owned(), flag),
    ]);
    let output = ValueFact::proven(
        ValueKindFact::Struct(StructFact::array(fields, true, Vec::new(), false)),
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
