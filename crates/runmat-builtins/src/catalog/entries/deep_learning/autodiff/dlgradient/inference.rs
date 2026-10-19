use crate::{
    catalog::inference::{argument_error, finish_fixed_outputs},
    BuiltinCatalogEntry,
};
use runmat_types::{
    standard, AliasFact, CallInference, CallRequest, DynamicReason, NumericClass, ObjectFact,
    ResidencyFact, StorageFact, ValueFact, ValueKindFact,
};
use std::collections::BTreeMap;

mod targets;

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
    data.storage == StorageFact::Sparse
        || match &data.kind {
            ValueKindFact::Unknown => false,
            ValueKindFact::Logical => matches!(data.residency, ResidencyFact::Device { .. }),
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
    if data.storage == StorageFact::Sparse {
        diagnostics.push(argument_error(
            "RM-DL-DLGRADIENT-STORAGE",
            "sparse dlarray Data cannot be differentiated by the host tape",
            index,
        ));
    } else if known_unsupported_data(data) {
        diagnostics.push(argument_error(
            "RM-DL-DLGRADIENT-CLASS",
            "dlarray Data must be real double or single, or supported host logical data",
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
        let mut gradient_data = match &data.kind {
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
        // Repeated targets and shared learnables can clone the same CoW tape
        // gradient into more than one output. Its class and allocation layout
        // are known, but exclusive allocation ownership and placement depend
        // on the executed loss graph, not on the target's original Data.
        gradient_data.alias = AliasFact::Unknown;
        gradient_data.residency = ResidencyFact::Unknown;
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
    // The dlarray wrapper remains a host object even when Data is resident.
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
        let consumed = request
            .outputs
            .requested
            .known_count()
            .is_some_and(|count| index <= count);
        if consumed {
            targets::check(target, index, &mut diagnostics);
        }
        outputs.push(targets::gradient(target));
    }
    finish_fixed_outputs(entry, request, outputs, diagnostics)
}
