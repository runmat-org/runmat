use crate::{
    catalog::inference::{argument_error, finish_fixed},
    BuiltinCatalogEntry,
};
use runmat_types::{
    CallInference, CallRequest, DynamicReason, NumericClass, NumericDomain, NumericFact,
    ResidencyFact, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

#[derive(Clone, Copy)]
enum Scheme {
    Psk,
    Qam,
}

pub(super) fn infer_pskmod(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer(request, entry, Scheme::Psk)
}

pub(super) fn infer_qammod(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer(request, entry, Scheme::Qam)
}

fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry, scheme: Scheme) -> CallInference {
    let name = entry.identity.name;
    let mut diagnostics = Vec::new();
    if request.arguments.len() < 2 {
        diagnostics.push(argument_error(
            "RM-CATALOG-MODULATION-ARITY",
            format!("{name} requires symbol input X and modulation order M"),
            request.arguments.len(),
        ));
    }
    let input = request.arguments.first();
    if let Some(value) = input {
        let admissible = matches!(
            &value.kind,
            ValueKindFact::Unknown
                | ValueKindFact::Logical
                | ValueKindFact::Numeric(NumericFact {
                    domain: NumericDomain::Real,
                    ..
                })
        );
        if !admissible {
            diagnostics.push(argument_error(
                "RM-CATALOG-MODULATION-INPUT",
                "X must contain real numeric symbols or bits",
                0,
            ));
        }
        if let ValueKindFact::Numeric(NumericFact { class, .. }) = &value.kind {
            let allowed = match scheme {
                Scheme::Psk => !matches!(class, NumericClass::Int64 | NumericClass::UInt64),
                Scheme::Qam => !matches!(
                    class,
                    NumericClass::Int32
                        | NumericClass::UInt32
                        | NumericClass::Int64
                        | NumericClass::UInt64
                ),
            };
            if !allowed {
                diagnostics.push(argument_error(
                    "RM-CATALOG-MODULATION-CLASS",
                    format!("{name} does not accept this fixed-width X class"),
                    0,
                ));
            }
        }
    }
    let order = request.literals.numeric_at(1);
    if order.is_some_and(|value| {
        !value.is_finite()
            || value < 2.0
            || value.fract() != 0.0
            || (matches!(scheme, Scheme::Qam)
                && (value >= usize::MAX as f64 || !(value as usize).is_power_of_two()))
    }) {
        diagnostics.push(argument_error(
            "RM-CATALOG-MODULATION-ORDER",
            "M must be an integer greater than one; qammod requires a power of two",
            1,
        ));
    }

    let mapping_index = if matches!(scheme, Scheme::Psk) { 3 } else { 2 };
    if request.arguments.get(mapping_index).is_some_and(|fact| {
        matches!(
            fact.kind,
            ValueKindFact::Numeric(_) | ValueKindFact::Logical
        ) && fact.shape.known_dims().is_some_and(|dims| {
            dims.len() > 2
                || (dims.len() == 2
                    && dims[0].is_some_and(|rows| rows > 1)
                    && dims[1].is_some_and(|cols| cols > 1))
        })
    }) {
        diagnostics.push(argument_error(
            "RM-CATALOG-MODULATION-MAPPING-SHAPE",
            "custom symbol order must be a vector",
            mapping_index,
        ));
    }

    let mut bit_input = false;
    let mut output_class = match scheme {
        Scheme::Psk => NumericClass::Double,
        Scheme::Qam
            if input.is_some_and(|fact| {
                matches!(
                    &fact.kind,
                    ValueKindFact::Numeric(NumericFact {
                        class: NumericClass::Single,
                        ..
                    })
                )
            }) =>
        {
            NumericClass::Single
        }
        Scheme::Qam => NumericClass::Double,
    };
    let mut output_class_known = !matches!(scheme, Scheme::Qam)
        || input.is_some_and(|fact| fact.kind != ValueKindFact::Unknown);
    let mut bit_input_known = true;
    for index in 2..request.arguments.len().saturating_sub(1) {
        let Some(name) = request.literals.literal_string_at(index) else {
            continue;
        };
        let option = name.trim().to_ascii_lowercase();
        let value = request.literals.literal_string_at(index + 1);
        match option.as_str() {
            "inputtype" => match value.map(|text| text.trim().to_ascii_lowercase()) {
                Some(text) if text == "bit" => bit_input = true,
                Some(text) if text == "integer" => bit_input = false,
                Some(_) => diagnostics.push(argument_error(
                    "RM-CATALOG-MODULATION-INPUT-TYPE",
                    "InputType must be 'integer' or 'bit'",
                    index + 1,
                )),
                None => bit_input_known = false,
            },
            "outputdatatype" => match value.map(|text| text.trim().to_ascii_lowercase()) {
                Some(text) if text == "single" => {
                    output_class = NumericClass::Single;
                    output_class_known = true;
                }
                Some(text) if text == "double" => {
                    output_class = NumericClass::Double;
                    output_class_known = true;
                }
                Some(_) => diagnostics.push(argument_error(
                    "RM-CATALOG-MODULATION-DTYPE",
                    "OutputDataType must be 'single' or 'double'",
                    index + 1,
                )),
                None => output_class_known = false,
            },
            "plotconstellation" => {
                if request.literals.literal_bool_at(index + 1) == Some(true)
                    || request.literals.numeric_at(index + 1) == Some(1.0)
                {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-MODULATION-PLOT",
                        "PlotConstellation=true is not implemented",
                        index + 1,
                    ));
                }
            }
            _ => {}
        }
    }
    if !bit_input
        && bit_input_known
        && input.is_some_and(|fact| fact.kind == ValueKindFact::Logical)
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-MODULATION-LOGICAL",
            "logical X requires InputType='bit'",
            0,
        ));
    }
    if bit_input
        && order.is_some_and(|value| {
            value.is_finite()
                && value >= 2.0
                && value.fract() == 0.0
                && value < usize::MAX as f64
                && !(value as usize).is_power_of_two()
        })
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-MODULATION-BITS-ORDER",
            "bit input requires a power-of-two modulation order",
            1,
        ));
    }

    let shape = output_shape(input, bit_input, bit_input_known, order);
    if bit_input {
        if let (Some(rows), Some(bits)) = (
            input
                .and_then(|value| value.shape.known_dims())
                .and_then(|dims| dims.first().copied().flatten()),
            order.and_then(bits_per_symbol),
        ) {
            if rows != 0 && !rows.is_multiple_of(bits) {
                diagnostics.push(argument_error(
                    "RM-CATALOG-MODULATION-BIT-ROWS",
                    "the first dimension of bit input must be divisible by log2(M)",
                    0,
                ));
            }
        }
    }
    let mut output = if output_class_known && diagnostics.is_empty() {
        ValueFact::proven(
            ValueKindFact::Numeric(NumericFact {
                class: output_class,
                domain: NumericDomain::Complex,
            }),
            shape,
            StorageFact::Dense,
        )
    } else {
        let mut fact = ValueFact::unknown(DynamicReason::RuntimeValue);
        fact.shape = shape;
        fact
    };
    output.residency = match input.map(|fact| &fact.residency) {
        Some(ResidencyFact::Host) => ResidencyFact::Host,
        Some(ResidencyFact::Device { provider }) => ResidencyFact::Device {
            provider: provider.clone(),
        },
        _ => ResidencyFact::Unknown,
    };
    finish_fixed(entry, request, output, diagnostics)
}

fn output_shape(
    input: Option<&ValueFact>,
    bit_input: bool,
    bit_input_known: bool,
    order: Option<f64>,
) -> ShapeFact {
    let Some(input) = input else {
        return ShapeFact::Unknown;
    };
    if !bit_input_known {
        let mut dims = match input.shape.known_dims() {
            Some(dims) => dims,
            None => return ShapeFact::Unknown,
        };
        if let Some(first) = dims.first_mut() {
            *first = None;
        }
        return ShapeFact::from(dims);
    }
    if !bit_input {
        return input.shape.clone();
    }
    let mut dims = match input.shape.known_dims() {
        Some(dims) => dims,
        None => return ShapeFact::Unknown,
    };
    let bits_per_symbol = order.and_then(bits_per_symbol);
    if let Some(first) = dims.first_mut() {
        *first = match (*first, bits_per_symbol) {
            (Some(rows), Some(bits)) if bits > 0 && rows.is_multiple_of(bits) => Some(rows / bits),
            _ => None,
        };
    }
    ShapeFact::from(dims)
}

fn bits_per_symbol(value: f64) -> Option<usize> {
    (value.is_finite()
        && value >= 2.0
        && value.fract() == 0.0
        && value < usize::MAX as f64
        && (value as usize).is_power_of_two())
    .then_some((value as usize).trailing_zeros() as usize)
}
