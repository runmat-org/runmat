use crate::catalog::inference::argument_error;
use crate::BuiltinCatalogEntry;
use runmat_types::{
    infer_call, CallContract, CallInference, CallRequest, LiteralContext, LiteralValue,
    NumericClass, NumericDomain, NumericFact, ResidencyFact, ShapeFact, StorageFact, ValueFact,
    ValueKindFact,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ImageIndexMode {
    Intensity,
    Indexed,
}

impl ImageIndexMode {
    pub fn from_option_text(text: &str) -> Option<Self> {
        text.trim()
            .eq_ignore_ascii_case("indexed")
            .then_some(Self::Indexed)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum ImageOutputClass {
    Double,
    UInt8,
    UInt16,
}

impl ImageOutputClass {
    fn numeric_class(self) -> NumericClass {
        match self {
            Self::Double => NumericClass::Double,
            Self::UInt8 => NumericClass::UInt8,
            Self::UInt16 => NumericClass::UInt16,
        }
    }
}

pub(super) fn infer(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    output_class: ImageOutputClass,
) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.is_empty() || request.arguments.len() > 2 {
        diagnostics.push(argument_error(
            "RM-CATALOG-IMAGE-CONVERSION-ARITY",
            "image conversion requires an image and optionally the indexed flag",
            request.arguments.len().saturating_sub(1),
        ));
    }
    let mode = if let Some(option) = request.arguments.get(1) {
        let invalid_geometry = match option.kind {
            ValueKindFact::String => option.shape.is_definitely_nonscalar(),
            // Runtime flattens character payloads, including columns and N-D
            // CharArray values, before decoding the option spelling.
            ValueKindFact::Character => false,
            _ => false,
        };
        if !matches!(
            option.kind,
            ValueKindFact::String | ValueKindFact::Character | ValueKindFact::Unknown
        ) || invalid_geometry
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-IMAGE-CONVERSION-MODE",
                "second input must be the text indexed",
                1,
            ));
        }
        match request.literals.literal_args.get(1) {
            Some(
                LiteralValue::String(text)
                | LiteralValue::Character(text)
                | LiteralValue::Keyword(text),
            ) => match ImageIndexMode::from_option_text(text) {
                Some(mode) => Some(mode),
                None => {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-IMAGE-CONVERSION-MODE",
                        "second input must be the text indexed",
                        1,
                    ));
                    None
                }
            },
            _ => None,
        }
    } else {
        Some(ImageIndexMode::Intensity)
    };
    if let Some(input) = request.arguments.first() {
        let valid_class = match input.kind {
            ValueKindFact::Numeric(NumericFact {
                class,
                domain: NumericDomain::Real,
            }) => match mode {
                Some(ImageIndexMode::Intensity) => matches!(
                    class,
                    NumericClass::Double
                        | NumericClass::Single
                        | NumericClass::Int16
                        | NumericClass::UInt8
                        | NumericClass::UInt16
                ),
                Some(ImageIndexMode::Indexed) => matches!(
                    class,
                    NumericClass::Double | NumericClass::UInt8 | NumericClass::UInt16
                ),
                None => matches!(
                    class,
                    NumericClass::Double
                        | NumericClass::UInt8
                        | NumericClass::UInt16
                        | NumericClass::Single
                        | NumericClass::Int16
                ),
            },
            ValueKindFact::Logical | ValueKindFact::Unknown => true,
            _ => false,
        };
        if !valid_class || input.storage == StorageFact::Sparse {
            diagnostics.push(argument_error(
                "RM-CATALOG-IMAGE-CONVERSION-CLASS",
                "unsupported image storage or class for this conversion mode",
                0,
            ));
        }
        if mode == Some(ImageIndexMode::Indexed) {
            if let ValueKindFact::Numeric(NumericFact {
                class,
                domain: NumericDomain::Real,
            }) = input.kind
            {
                let maximum = match (output_class, class) {
                    (ImageOutputClass::UInt8, NumericClass::UInt16) => Some(255.0),
                    (ImageOutputClass::UInt8, NumericClass::Double) => Some(256.0),
                    (ImageOutputClass::UInt16, NumericClass::Double) => Some(65536.0),
                    _ => None,
                };
                if maximum.is_some_and(|maximum| {
                    request
                        .literals
                        .literal_args
                        .first()
                        .and_then(LiteralContext::numeric_from_literal)
                        .is_some_and(|value| {
                            value
                                .partial_cmp(&maximum)
                                .is_none_or(|ordering| ordering.is_gt())
                        })
                }) {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-IMAGE-CONVERSION-INDEX-BOUND",
                        "indexed image value exceeds the accepted class range",
                        0,
                    ));
                }
            }
        }
    }
    let input_shape = request
        .arguments
        .first()
        .map_or(ShapeFact::Unknown, |input| input.shape.clone());
    let shape = if input_shape.element_count() == Some(1) {
        ShapeFact::Scalar
    } else {
        input_shape
    };
    let storage = match shape.element_count() {
        Some(1) => StorageFact::Scalar,
        Some(_) => StorageFact::Dense,
        None => StorageFact::Unknown,
    };
    let mut output = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: output_class.numeric_class(),
            domain: NumericDomain::Real,
        }),
        shape,
        storage,
    );
    // A resident input may be restored to its exact owner, while an automatic
    // fallback can remain host-side. Inference cannot promise either route.
    if !matches!(
        request.arguments.first().map(|fact| &fact.residency),
        Some(ResidencyFact::Host)
    ) {
        output.residency = ResidencyFact::Unknown;
    }
    let mut contract = CallContract::fixed(vec![output]);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    result.diagnostics.extend(diagnostics);
    result
}

#[cfg(test)]
mod tests;
