use crate::{catalog::inference::argument_error, BuiltinCatalogEntry};
use runmat_types::{
    broadcast_shape, infer_call, standard, CallContract, CallInference, CallRequest, DimensionFact,
    DynamicReason, InferenceDiagnostic, NumericClass, NumericDomain, NumericFact, ResidencyFact,
    ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

fn real_double(shape: ShapeFact) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        shape.clone(),
        if shape.element_count() == Some(1) {
            StorageFact::Scalar
        } else {
            StorageFact::Dense
        },
    )
}

fn logical(shape: ShapeFact) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Logical,
        shape.clone(),
        if shape.element_count() == Some(1) {
            StorageFact::Scalar
        } else {
            StorageFact::Dense
        },
    )
}

fn character(rows: DimensionFact) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Character,
        ShapeFact::Shaped {
            dims: vec![rows, DimensionFact::Unknown],
        },
        StorageFact::Dense,
    )
}

fn dynamic_host() -> ValueFact {
    let mut value = ValueFact::unknown(DynamicReason::RuntimeValue);
    value.residency = ResidencyFact::Host;
    value
}

fn datetime_or_number_or_text(value: &ValueFact) -> bool {
    match &value.kind {
        ValueKindFact::Unknown | ValueKindFact::String | ValueKindFact::Character => true,
        ValueKindFact::Numeric(number) => number.domain == NumericDomain::Real,
        ValueKindFact::Object(object) => match object.runtime_class.as_ref() {
            Some(class) => class == &standard::DATETIME.owned(),
            None => object.class.is_none(),
        },
        _ => false,
    }
}

fn real_numeric(value: &ValueFact) -> bool {
    matches!(
        value.kind,
        ValueKindFact::Unknown
            | ValueKindFact::Logical
            | ValueKindFact::Numeric(NumericFact {
                domain: NumericDomain::Real,
                ..
            })
    )
}

fn check_arity(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    valid: bool,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) {
    if !valid {
        diagnostics.push(argument_error(
            "RM-DATETIME-LEGACY-ARITY",
            format!("{}: unsupported input count", entry.identity.name),
            request.arguments.len().saturating_sub(1),
        ));
    }
}

fn check_date(request: &CallRequest, count: usize, diagnostics: &mut Vec<InferenceDiagnostic>) {
    for (index, value) in request.arguments.iter().take(count).enumerate() {
        if !datetime_or_number_or_text(value) {
            diagnostics.push(argument_error(
                "RM-DATETIME-LEGACY-INPUT",
                "expected a real serial date, datetime object, or supported date text",
                index,
            ));
        }
    }
}

fn check_numeric(request: &CallRequest, count: usize, diagnostics: &mut Vec<InferenceDiagnostic>) {
    for (index, value) in request.arguments.iter().take(count).enumerate() {
        if !real_numeric(value) {
            diagnostics.push(argument_error(
                "RM-DATETIME-LEGACY-NUMERIC",
                "expected real numeric date components",
                index,
            ));
        }
    }
}

fn serial_shape(value: &ValueFact, recognize_datevec: bool) -> ShapeFact {
    match &value.kind {
        ValueKindFact::Object(_) => {
            // The host datetime object is scalar even when its __serial payload is an array.
            ShapeFact::Unknown
        }
        ValueKindFact::Character => match value.shape.known_dims() {
            Some(dims) => ShapeFact::from(vec![dims.first().copied().flatten(), Some(1)]),
            None => ShapeFact::Unknown,
        },
        ValueKindFact::String => value.shape.clone(),
        ValueKindFact::Numeric(_) if recognize_datevec => {
            let Some(dims) = value.shape.known_dims() else {
                return ShapeFact::Unknown;
            };
            if dims.len() >= 2 && matches!(dims[1], Some(3 | 6)) {
                return ShapeFact::from(vec![dims[0], Some(1)]);
            }
            if dims.len() >= 2 && dims[1].is_none() {
                // At runtime three or six columns yield one serial per row;
                // every other column count retains the input serial shape.
                return ShapeFact::Unknown;
            }
            if matches!(value.shape.element_count(), Some(3 | 6)) {
                return ShapeFact::Scalar;
            }
            value.shape.clone()
        }
        _ => value.shape.clone(),
    }
}

fn date_array_shape(value: &ValueFact) -> ShapeFact {
    serial_shape(value, false)
}

fn matching_component_shape(
    values: &[ValueFact],
    diagnostics: &mut Vec<InferenceDiagnostic>,
) -> ShapeFact {
    let mut shape = ShapeFact::Scalar;
    for (index, value) in values.iter().enumerate() {
        if value.shape.element_count() == Some(1) {
            continue;
        }
        if shape == ShapeFact::Scalar {
            shape = value.shape.clone();
        } else if shape != value.shape {
            diagnostics.push(argument_error(
                "RM-DATETIME-LEGACY-SHAPE",
                "non-scalar date components must have matching shapes",
                index,
            ));
            return ShapeFact::Unknown;
        }
    }
    shape
}

fn finish(
    entry: &BuiltinCatalogEntry,
    request: &CallRequest,
    outputs: Vec<ValueFact>,
    mut diagnostics: Vec<InferenceDiagnostic>,
) -> CallInference {
    let mut contract = CallContract::fixed(outputs);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    diagnostics.append(&mut result.diagnostics);
    result.diagnostics = diagnostics;
    result
}

pub(super) fn datenum(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    let args = &request.arguments;
    check_arity(
        request,
        entry,
        matches!(args.len(), 0 | 1 | 3..=6),
        &mut diagnostics,
    );
    let shape = match args.len() {
        0 => ShapeFact::Scalar,
        1 => {
            check_date(request, 1, &mut diagnostics);
            serial_shape(&args[0], true)
        }
        3..=6 => {
            check_numeric(request, args.len(), &mut diagnostics);
            matching_component_shape(args, &mut diagnostics)
        }
        _ => ShapeFact::Unknown,
    };
    finish(entry, request, vec![real_double(shape)], diagnostics)
}

pub(super) fn datevec(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    check_arity(
        request,
        entry,
        request.arguments.len() == 1,
        &mut diagnostics,
    );
    check_date(request, 1, &mut diagnostics);
    if request.outputs.requested.known_count().is_none() {
        let mut result = finish(entry, request, vec![dynamic_host()], diagnostics);
        result.dynamic_outputs = true;
        return result;
    }
    let serial_shape = request
        .arguments
        .first()
        .map(date_array_shape)
        .unwrap_or(ShapeFact::Unknown);
    let rows = serial_shape
        .element_count()
        .map_or(DimensionFact::Unknown, DimensionFact::Known);
    let count = request.outputs.requested.known_count().unwrap_or(1);
    if count > 6 {
        diagnostics.push(argument_error(
            "RM-DATEVEC-OUTPUTS",
            "datevec supports at most six outputs",
            0,
        ));
    }
    let outputs = if count <= 1 {
        vec![real_double(ShapeFact::Shaped {
            dims: vec![rows, DimensionFact::Known(6)],
        })]
    } else {
        (0..count.min(6))
            .map(|_| {
                real_double(ShapeFact::Shaped {
                    dims: vec![rows.clone(), DimensionFact::Known(1)],
                })
            })
            .collect()
    };
    finish(entry, request, outputs, diagnostics)
}

pub(super) fn datestr(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    check_arity(
        request,
        entry,
        (1..=2).contains(&request.arguments.len()),
        &mut diagnostics,
    );
    check_date(request, 1, &mut diagnostics);
    if let Some(value) = request.arguments.get(1) {
        let numeric = matches!(value.kind, ValueKindFact::Numeric(_));
        let valid_kind = matches!(
            value.kind,
            ValueKindFact::String
                | ValueKindFact::Character
                | ValueKindFact::Unknown
                | ValueKindFact::Numeric(NumericFact {
                    class: NumericClass::Double,
                    domain: NumericDomain::Real,
                })
        );
        let invalid_code = numeric
            && (value.shape.element_count().is_some_and(|count| count != 1)
                || matches!(value.storage, StorageFact::Sparse | StorageFact::Opaque)
                || request.literals.numeric_at(1).is_some_and(|code| {
                    !code.is_finite() || code.fract() != 0.0 || !(-1.0..=31.0).contains(&code)
                }));
        if !valid_kind
            || invalid_code
            || matches!(
                value.residency,
                ResidencyFact::Device { .. } | ResidencyFact::Remote { .. }
            )
        {
            diagnostics.push(argument_error(
                "RM-DATESTR-FORMAT",
                "format must be host text or a real double scalar integer from -1 through 31",
                1,
            ));
        }
    }
    let rows = request
        .arguments
        .first()
        .and_then(|value| serial_shape(value, true).element_count())
        .map_or(DimensionFact::Unknown, DimensionFact::Known);
    finish(entry, request, vec![character(rows)], diagnostics)
}

pub(super) fn weekday(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    check_arity(
        request,
        entry,
        (1..=2).contains(&request.arguments.len()),
        &mut diagnostics,
    );
    check_date(request, 1, &mut diagnostics);
    if let Some(option) = request.arguments.get(1) {
        if !matches!(
            option.kind,
            ValueKindFact::String | ValueKindFact::Character | ValueKindFact::Unknown
        ) {
            diagnostics.push(argument_error(
                "RM-WEEKDAY-FORM",
                "DayForm must be 'short' or 'long' text",
                1,
            ));
        }
        if request
            .literals
            .literal_string_at(1)
            .is_some_and(|literal| literal != "short" && literal != "long")
        {
            diagnostics.push(argument_error(
                "RM-WEEKDAY-FORM",
                "DayForm must be 'short' or 'long'",
                1,
            ));
        }
    }
    if request.outputs.requested.known_count().is_none() {
        let mut result = finish(entry, request, vec![dynamic_host()], diagnostics);
        result.dynamic_outputs = true;
        return result;
    }
    let shape = request
        .arguments
        .first()
        .map(date_array_shape)
        .unwrap_or(ShapeFact::Unknown);
    let rows = shape
        .element_count()
        .map_or(DimensionFact::Unknown, DimensionFact::Known);
    let count = request.outputs.requested.known_count().unwrap_or(1);
    if count > 2 {
        diagnostics.push(argument_error(
            "RM-WEEKDAY-OUTPUTS",
            "weekday supports at most two outputs",
            0,
        ));
    }
    let outputs = if count > 1 {
        vec![real_double(shape), character(rows)]
    } else {
        vec![real_double(shape)]
    };
    finish(entry, request, outputs, diagnostics)
}

pub(super) fn eomday(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    check_arity(
        request,
        entry,
        request.arguments.len() == 2,
        &mut diagnostics,
    );
    check_numeric(request, 2, &mut diagnostics);
    let shape = if request.arguments.len() >= 2 {
        match scalar_or_matching_shape(&request.arguments[0].shape, &request.arguments[1].shape) {
            Ok(shape) => shape,
            Err(_) => {
                diagnostics.push(argument_error(
                    "RM-EOMDAY-SHAPE",
                    "year and month must be scalar or have matching sizes",
                    1,
                ));
                ShapeFact::Unknown
            }
        }
    } else {
        ShapeFact::Unknown
    };
    finish(entry, request, vec![real_double(shape)], diagnostics)
}

pub(super) fn etime(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    check_arity(
        request,
        entry,
        request.arguments.len() == 2,
        &mut diagnostics,
    );
    check_numeric(request, 2, &mut diagnostics);
    let shape = if request.arguments.len() >= 2 {
        let left = datevec_rows(&request.arguments[0]);
        let right = datevec_rows(&request.arguments[1]);
        match scalar_or_matching_shape(&left, &right) {
            Ok(shape) => shape,
            Err(_) => {
                diagnostics.push(argument_error(
                    "RM-ETIME-SHAPE",
                    "date-vector row counts must be scalar or match",
                    1,
                ));
                ShapeFact::Unknown
            }
        }
    } else {
        ShapeFact::Unknown
    };
    finish(entry, request, vec![real_double(shape)], diagnostics)
}

fn datevec_rows(value: &ValueFact) -> ShapeFact {
    let Some(dims) = value.shape.known_dims() else {
        return ShapeFact::Unknown;
    };
    if dims.len() >= 2 && matches!(dims[1], Some(3 | 6)) {
        ShapeFact::from(vec![dims[0], Some(1)])
    } else if matches!(value.shape.element_count(), Some(3 | 6)) {
        ShapeFact::Scalar
    } else {
        ShapeFact::Unknown
    }
}

pub(super) fn isbetween(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    check_arity(
        request,
        entry,
        request.arguments.len() == 3,
        &mut diagnostics,
    );
    check_date(request, 3, &mut diagnostics);
    let shape = if request.arguments.len() >= 3 {
        let a = date_array_shape(&request.arguments[0]);
        let b = date_array_shape(&request.arguments[1]);
        let c = date_array_shape(&request.arguments[2]);
        match broadcast_shape(&a, &b).and_then(|ab| broadcast_shape(&ab, &c)) {
            Ok(shape) => shape,
            Err(_) => {
                diagnostics.push(argument_error(
                    "RM-ISBETWEEN-SHAPE",
                    "date bounds cannot broadcast",
                    2,
                ));
                ShapeFact::Unknown
            }
        }
    } else {
        ShapeFact::Unknown
    };
    finish(entry, request, vec![logical(shape)], diagnostics)
}

fn scalar_or_matching_shape(left: &ShapeFact, right: &ShapeFact) -> Result<ShapeFact, ()> {
    if left.element_count() == Some(1) {
        return Ok(right.clone());
    }
    if right.element_count() == Some(1) {
        return Ok(left.clone());
    }
    let left_known = left
        .known_dims()
        .is_some_and(|dims| dims.iter().all(Option::is_some));
    let right_known = right
        .known_dims()
        .is_some_and(|dims| dims.iter().all(Option::is_some));
    match (left_known, right_known) {
        (true, true) if left.is_proven_equivalent(right) => Ok(left.clone()),
        (true, true) => Err(()),
        _ => Ok(ShapeFact::Unknown),
    }
}
