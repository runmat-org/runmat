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

fn known_unsupported_data(data: &ValueFact) -> bool {
    matches!(data.residency, ResidencyFact::Device { .. })
        || data.storage == StorageFact::Sparse
        || match &data.kind {
            ValueKindFact::Unknown | ValueKindFact::Logical => false,
            ValueKindFact::Numeric(number) => {
                !matches!(number.class, NumericClass::Double | NumericClass::Single)
                    || number.domain != runmat_types::NumericDomain::Real
            }
            _ => true,
        }
}

fn check_dlarray_data(
    fact: &ValueFact,
    index: usize,
    diagnostics: &mut Vec<runmat_types::InferenceDiagnostic>,
) {
    let ValueKindFact::Object(object) = &fact.kind else {
        return;
    };
    if object.runtime_class.as_ref() != Some(&standard::DL_ARRAY.owned()) {
        return;
    }
    let Some(data) = object.properties.get(crate::DLARRAY_DATA_PROPERTY) else {
        if object.properties_complete {
            diagnostics.push(argument_error(
                "RM-DL-DLGRADIENT-DATA",
                "dlarray is missing Data",
                index,
            ));
        }
        return;
    };
    if matches!(data.residency, ResidencyFact::Device { .. }) {
        diagnostics.push(argument_error(
            "RM-DL-DLGRADIENT-RESIDENCY",
            "provider-resident dlarray Data cannot be differentiated by the host tape",
            index,
        ));
    } else if data.storage == StorageFact::Sparse {
        diagnostics.push(argument_error(
            "RM-DL-DLGRADIENT-STORAGE",
            "sparse dlarray Data cannot be differentiated by the host tape",
            index,
        ));
    } else if known_unsupported_data(data) {
        diagnostics.push(argument_error(
            "RM-DL-DLGRADIENT-CLASS",
            "dlarray Data must be real double, single, or logical host data",
            index,
        ));
    }
}

fn gradient_for_target(target: &ValueFact) -> ValueFact {
    let ValueKindFact::Object(object) = &target.kind else {
        return ValueFact::unknown(DynamicReason::RuntimeValue);
    };
    if object.runtime_class.as_ref() != Some(&standard::DL_ARRAY.owned()) {
        return ValueFact::unknown(DynamicReason::RuntimeValue);
    }
    if object.properties_complete && !object.properties.contains_key(crate::DLARRAY_DATA_PROPERTY) {
        return ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
    }
    let mut properties = BTreeMap::new();
    if let Some(data) = object.properties.get(crate::DLARRAY_DATA_PROPERTY) {
        let gradient_data = match &data.kind {
            _ if known_unsupported_data(data) => {
                ValueFact::unknown(DynamicReason::UnsupportedRepresentation)
            }
            ValueKindFact::Numeric(number) => ValueFact::proven(
                ValueKindFact::Numeric(*number),
                data.shape.clone(),
                StorageFact::Dense,
            ),
            ValueKindFact::Logical => ValueFact::proven(
                ValueKindFact::Numeric(runmat_types::NumericFact {
                    class: NumericClass::Double,
                    domain: runmat_types::NumericDomain::Real,
                }),
                data.shape.clone(),
                StorageFact::Dense,
            ),
            _ => ValueFact::unknown(DynamicReason::RuntimeValue),
        };
        properties.insert(crate::DLARRAY_DATA_PROPERTY.into(), gradient_data);
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
        check_dlarray_data(loss, 0, &mut diagnostics);
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
        check_dlarray_data(target, index, &mut diagnostics);
        outputs.push(gradient_for_target(target));
    }
    finish_fixed_outputs(entry, request, outputs, diagnostics)
}
