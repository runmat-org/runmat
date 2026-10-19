use crate::{
    catalog::inference::{argument_error, finish_fixed},
    BuiltinCatalogEntry,
};
use runmat_types::{
    standard, CallInference, CallRequest, DynamicReason, NumericClass, NumericDomain, NumericFact,
    ObjectFact, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};
use std::collections::BTreeMap;

#[derive(Clone, Copy)]
pub(super) enum BinaryGeometry {
    Elementwise,
    Matrix,
}

fn operand_data(value: &ValueFact) -> Option<(&ValueFact, bool)> {
    match &value.kind {
        ValueKindFact::Object(object)
            if object.runtime_class.as_ref() == Some(&standard::DL_ARRAY.owned()) =>
        {
            object
                .properties
                .get(crate::DLARRAY_DATA_PROPERTY)
                .map(|data| (data, true))
        }
        ValueKindFact::Object(object) if object.runtime_class.is_none() => Some((value, false)),
        ValueKindFact::Numeric(_) | ValueKindFact::Logical | ValueKindFact::Unknown => {
            Some((value, false))
        }
        _ => None,
    }
}

fn known_floating_class(value: &ValueFact) -> Option<NumericClass> {
    match &value.kind {
        ValueKindFact::Logical => Some(NumericClass::Double),
        ValueKindFact::Numeric(number) => Some(number.class),
        _ => None,
    }
}

fn result_class(left: &ValueFact, right: &ValueFact) -> Option<NumericClass> {
    match (known_floating_class(left), known_floating_class(right)) {
        (Some(NumericClass::Single), Some(NumericClass::Single | NumericClass::Double))
        | (Some(NumericClass::Double), Some(NumericClass::Single)) => Some(NumericClass::Single),
        (Some(NumericClass::Double), Some(NumericClass::Double)) => Some(NumericClass::Double),
        _ => None,
    }
}

fn output_shape(left: &ShapeFact, right: &ShapeFact, geometry: BinaryGeometry) -> ShapeFact {
    match geometry {
        BinaryGeometry::Elementwise if left.element_count() == Some(1) => right.clone(),
        BinaryGeometry::Elementwise
            if right.element_count() == Some(1) || left.is_proven_equivalent(right) =>
        {
            left.clone()
        }
        BinaryGeometry::Elementwise => ShapeFact::Unknown,
        BinaryGeometry::Matrix if left.element_count() == Some(1) => right.clone(),
        BinaryGeometry::Matrix if right.element_count() == Some(1) => left.clone(),
        BinaryGeometry::Matrix => {
            let (Some(a), Some(b)) = (left.known_dims(), right.known_dims()) else {
                return ShapeFact::Unknown;
            };
            if a.len() != 2 || b.len() != 2 || a[1].zip(b[0]).is_some_and(|(x, y)| x != y) {
                return ShapeFact::Unknown;
            }
            ShapeFact::from(vec![a[0], b[1]])
        }
    }
}

fn known_incompatible(left: &ShapeFact, right: &ShapeFact, geometry: BinaryGeometry) -> bool {
    match geometry {
        BinaryGeometry::Elementwise => {
            if left.element_count() == Some(1) || right.element_count() == Some(1) {
                return false;
            }
            let (Some(left), Some(right)) = (left.known_dims(), right.known_dims()) else {
                return false;
            };
            left.iter().all(Option::is_some)
                && right.iter().all(Option::is_some)
                && !ShapeFact::from(left).is_proven_equivalent(&ShapeFact::from(right))
        }
        BinaryGeometry::Matrix => {
            if left.element_count() == Some(1) || right.element_count() == Some(1) {
                return false;
            }
            let (Some(left), Some(right)) = (left.known_dims(), right.known_dims()) else {
                return false;
            };
            left.get(2..)
                .is_some_and(|tail| tail.iter().any(|dim| dim.is_some_and(|n| n != 1)))
                || right
                    .get(2..)
                    .is_some_and(|tail| tail.iter().any(|dim| dim.is_some_and(|n| n != 1)))
                || left
                    .get(1)
                    .copied()
                    .flatten()
                    .zip(right.first().copied().flatten())
                    .is_some_and(|(a, b)| a != b)
        }
    }
}

fn wrapped(data: ValueFact, metadata_owner: Option<&ObjectFact>) -> ValueFact {
    let mut properties = BTreeMap::new();
    properties.insert(crate::DLARRAY_DATA_PROPERTY.into(), data.clone());
    if let Some(owner) = metadata_owner {
        for key in ["Format", "Labels"] {
            if let Some(property) = owner.properties.get(key) {
                properties.insert(key.into(), property.clone());
            }
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
        data.shape.clone(),
        StorageFact::Opaque,
    );
    result.residency = runmat_types::ResidencyFact::Host;
    result
}

pub(super) fn binary(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    geometry: BinaryGeometry,
) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 2 {
        diagnostics.push(argument_error(
            "RM-DL-ARRAY-ARITY",
            "dlarray binary methods require two operands",
            0,
        ));
    }
    let parsed: Vec<_> = request.arguments.iter().take(2).map(operand_data).collect();
    for (index, operand) in parsed.iter().enumerate() {
        match operand {
            None => diagnostics.push(argument_error(
                "RM-DL-ARRAY-CLASS",
                "operand must be a real floating dlarray or numeric value",
                index,
            )),
            Some((data, _))
                if data.numeric().is_some_and(|number| {
                    !matches!(number.class, NumericClass::Double | NumericClass::Single)
                        || number.domain != NumericDomain::Real
                }) =>
            {
                diagnostics.push(argument_error(
                    "RM-DL-ARRAY-CLASS",
                    "dlarray arithmetic requires real double or single data",
                    index,
                ))
            }
            _ => {}
        }
    }
    let result = if let [Some((left, left_dl)), Some((right, right_dl))] = parsed.as_slice() {
        if known_incompatible(&left.shape, &right.shape, geometry) {
            diagnostics.push(argument_error(
                "RM-DL-ARRAY-SHAPE",
                "dlarray operands have incompatible shapes for this method",
                1,
            ));
        }
        let shape = output_shape(&left.shape, &right.shape, geometry);
        let data = result_class(left, right).map_or_else(
            || ValueFact::unknown(DynamicReason::RuntimeValue),
            |class| {
                ValueFact::proven(
                    ValueKindFact::Numeric(NumericFact {
                        class,
                        domain: NumericDomain::Real,
                    }),
                    shape.clone(),
                    StorageFact::Dense,
                )
            },
        );
        let mut data = data;
        data.shape = shape;
        let owner = if *left_dl {
            request.arguments.first()
        } else if *right_dl {
            request.arguments.get(1)
        } else {
            None
        };
        let owner = owner.and_then(|fact| match &fact.kind {
            ValueKindFact::Object(object) => Some(object),
            _ => None,
        });
        wrapped(data, owner)
    } else {
        ValueFact::unknown(DynamicReason::RuntimeValue)
    };
    finish_fixed(entry, request, result, diagnostics)
}

pub(super) fn sum(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.is_empty() || request.arguments.len() > 2 {
        diagnostics.push(argument_error(
            "RM-DL-SUM-ARITY",
            "dlarray.sum accepts the input and optional 'all' selector",
            0,
        ));
    }
    if let Some(selector) = request.arguments.get(1) {
        if !matches!(
            &selector.kind,
            ValueKindFact::String | ValueKindFact::Character | ValueKindFact::Unknown
        ) {
            diagnostics.push(argument_error(
                "RM-DL-SUM-SELECTOR",
                "dlarray.sum accepts only the optional 'all' selector",
                1,
            ));
        }
        if request
            .literals
            .literal_string_at(1)
            .is_some_and(|literal| literal != "all")
        {
            diagnostics.push(argument_error(
                "RM-DL-SUM-SELECTOR",
                "dlarray.sum accepts only the optional 'all' selector",
                1,
            ));
        }
    }
    let result = request
        .arguments
        .first()
        .and_then(operand_data)
        .map_or_else(
            || ValueFact::unknown(DynamicReason::RuntimeValue),
            |(data, _is_dl)| {
                if data.numeric().is_some_and(|number| {
                    !matches!(number.class, NumericClass::Single | NumericClass::Double)
                        || number.domain != NumericDomain::Real
                }) {
                    diagnostics.push(argument_error(
                        "RM-DL-SUM-CLASS",
                        "dlarray.sum requires real double or single data",
                        0,
                    ));
                }
                let mut result_data = if matches!(&data.kind, ValueKindFact::Logical) {
                    ValueFact::proven(
                        ValueKindFact::Numeric(NumericFact {
                            class: NumericClass::Double,
                            domain: NumericDomain::Real,
                        }),
                        ShapeFact::Scalar,
                        StorageFact::Dense,
                    )
                } else {
                    data.clone()
                };
                result_data.shape = ShapeFact::Scalar;
                let owner = request.arguments.first().and_then(|fact| match &fact.kind {
                    ValueKindFact::Object(object) => Some(object),
                    _ => None,
                });
                wrapped(result_data, owner)
            },
        );
    finish_fixed(entry, request, result, diagnostics)
}
