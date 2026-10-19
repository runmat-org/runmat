use crate::{catalog::inference::argument_error, BuiltinCatalogEntry};
use runmat_types::*;
use std::collections::BTreeMap;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 1 {
        diagnostics.push(argument_error(
            "RM-CATALOG-XMLREAD-ARITY",
            "xmlread requires exactly one filename",
            0,
        ));
    }
    super::super::inference::input(request, 0, &mut diagnostics);
    let mut element_properties = BTreeMap::new();
    element_properties.insert(
        "TagName".into(),
        ValueFact::proven(
            ValueKindFact::Character,
            ShapeFact::from(vec![Some(1), None]),
            StorageFact::Dense,
        ),
    );
    let element = ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(super::super::XML_ELEMENT_CLASS.owned()),
            properties: element_properties,
            properties_complete: false,
            handle_semantics: Some(false),
        }),
        ShapeFact::Scalar,
        StorageFact::Opaque,
    );
    let mut properties = BTreeMap::new();
    properties.insert(
        "Filename".into(),
        ValueFact::proven(
            ValueKindFact::Character,
            ShapeFact::from(vec![Some(1), None]),
            StorageFact::Dense,
        ),
    );
    properties.insert("Text".into(), ValueFact::scalar(ValueKindFact::String));
    properties.insert(
        "DocumentElementName".into(),
        ValueFact::proven(
            ValueKindFact::Character,
            ShapeFact::from(vec![Some(1), None]),
            StorageFact::Dense,
        ),
    );
    properties.insert("DocumentElement".into(), element);
    let document = ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(super::super::XML_DOCUMENT_CLASS.owned()),
            properties,
            properties_complete: true,
            handle_semantics: Some(false),
        }),
        ShapeFact::Scalar,
        StorageFact::Opaque,
    );
    super::super::inference::finish(request, entry, document, diagnostics)
}
