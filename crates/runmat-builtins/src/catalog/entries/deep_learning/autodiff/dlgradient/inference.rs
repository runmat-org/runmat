use crate::{
    catalog::inference::{argument_error, finish_fixed_outputs},
    BuiltinCatalogEntry,
};
use runmat_types::{
    standard, CallInference, CallRequest, DynamicReason, NumericClass, ObjectFact, ResidencyFact,
    StorageFact, ValueFact, ValueKindFact,
};
use std::collections::BTreeMap;

fn traced_dlarray_possible(fact: &ValueFact) -> bool {
    match &fact.kind {
        ValueKindFact::Unknown => true,
        ValueKindFact::Object(object) => object
            .runtime_class
            .as_ref()
            .is_none_or(|class| class == &standard::DL_ARRAY.owned()),
        _ => false,
    }
}

fn gradient_for_target(target: &ValueFact) -> ValueFact {
    let ValueKindFact::Object(object) = &target.kind else {
        return ValueFact::unknown(DynamicReason::RuntimeValue);
    };
    if object.runtime_class.as_ref() != Some(&standard::DL_ARRAY.owned()) {
        return ValueFact::unknown(DynamicReason::RuntimeValue);
    }
    let mut properties = BTreeMap::new();
    if let Some(data) = object.properties.get(crate::DLARRAY_DATA_PROPERTY) {
        properties.insert(crate::DLARRAY_DATA_PROPERTY.into(), data.clone());
    }
    for name in ["Format", "Labels"] {
        if let Some(value) = object.properties.get(name) {
            properties.insert(name.into(), value.clone());
        }
    }
    let mut result = ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(standard::DL_ARRAY.owned()),
            properties,
            properties_complete: false,
            handle_semantics: Some(false),
        }),
        target.shape.clone(),
        StorageFact::Opaque,
    );
    result.residency = ResidencyFact::Host;
    result
}

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() < 2 {
        diagnostics.push(argument_error(
            "RM-DL-DLGRADIENT-ARITY",
            "dlgradient requires a scalar traced loss and at least one target",
            request.arguments.len().saturating_sub(1),
        ));
    }
    let targets = request.arguments.len().saturating_sub(1);
    if request
        .outputs
        .requested
        .known_count()
        .is_some_and(|count| count > targets)
    {
        diagnostics.push(argument_error(
            "RM-DL-DLGRADIENT-OUTPUTS",
            "dlgradient cannot return more gradients than targets",
            0,
        ));
    }
    if let Some(loss) = request.arguments.first() {
        if !traced_dlarray_possible(loss) || loss.shape.element_count().is_some_and(|n| n != 1) {
            diagnostics.push(argument_error(
                "RM-DL-DLGRADIENT-LOSS",
                "loss must be a scalar dlarray traced inside dlfeval",
                0,
            ));
        }
    }
    let mut outputs = Vec::with_capacity(targets);
    for (index, target) in request.arguments.iter().enumerate().skip(1) {
        if matches!(
            target.kind,
            ValueKindFact::Numeric(_) | ValueKindFact::Logical
        ) {
            diagnostics.push(argument_error(
                "RM-DL-DLGRADIENT-TARGET",
                "targets must be traced dlarray values or supported learnables trees",
                index,
            ));
        }
        if let ValueKindFact::Object(object) = &target.kind {
            if object.runtime_class.as_ref() == Some(&standard::DL_ARRAY.owned()) {
                if let Some(data) = object.properties.get(crate::DLARRAY_DATA_PROPERTY) {
                    if data.numeric().is_some_and(|number| {
                        !matches!(number.class, NumericClass::Double | NumericClass::Single)
                    }) {
                        diagnostics.push(argument_error(
                            "RM-DL-DLGRADIENT-CLASS",
                            "dlarray targets must contain double or single differentiable Data",
                            index,
                        ));
                    }
                }
            }
        }
        outputs.push(gradient_for_target(target));
    }
    finish_fixed_outputs(entry, request, outputs, diagnostics)
}
