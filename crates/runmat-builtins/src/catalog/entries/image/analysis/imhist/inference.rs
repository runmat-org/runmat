use super::{IMHIST_INTEGER_TOLERANCE, IMHIST_MAX_BINS};
use crate::catalog::inference::argument_error;
use crate::BuiltinCatalogEntry;
use runmat_types::*;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if !(1..=2).contains(&request.arguments.len()) {
        diagnostics.push(argument_error(
            "RM-CATALOG-IMHIST-ARITY",
            "imhist expects an image and optional bin count or colormap",
            request.arguments.len().saturating_sub(1),
        ));
    }
    let requested = request.outputs.requested.known_count();
    if requested.is_some_and(|count| count > 2) {
        diagnostics.push(argument_error(
            "RM-CATALOG-IMHIST-OUTPUTS",
            "imhist returns at most counts and bin locations",
            2,
        ));
    }
    let image = request.arguments.first();
    let image_class = image.and_then(|value| image_class(value, 0, &mut diagnostics));
    let bins = match request.arguments.get(1) {
        None => match image_class {
            Some(ImageClass::Logical) => Some(2),
            Some(ImageClass::Numeric) => Some(256),
            None => None,
        },
        Some(control) => {
            if is_known_colormap_shape(control) {
                if image
                    .is_some_and(|value| matches!(value.residency, ResidencyFact::Device { .. }))
                    || matches!(control.residency, ResidencyFact::Device { .. })
                {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-IMHIST-INDEXED-DEVICE",
                        "indexed gpuArray images are not supported",
                        0,
                    ));
                }
                if !matches!(
                    control.kind,
                    ValueKindFact::Unknown
                        | ValueKindFact::Numeric(NumericFact {
                            class: NumericClass::Double,
                            domain: NumericDomain::Real
                        })
                ) || matches!(control.storage, StorageFact::Sparse | StorageFact::Opaque)
                {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-IMHIST-COLORMAP",
                        "colormap must be a dense real double Nx3 array",
                        1,
                    ));
                }
                control
                    .shape
                    .known_dims()
                    .and_then(|dims| dims.first().copied().flatten())
            } else if control.shape.is_definitely_nonscalar() && !could_be_colormap_shape(control) {
                diagnostics.push(argument_error(
                    "RM-CATALOG-IMHIST-CONTROL",
                    "second argument must be a positive scalar bin count or Nx3 colormap",
                    1,
                ));
                None
            } else if could_be_colormap_shape(control) {
                // A partially known Nx3 map is possible; neither its mode nor
                // its row count is proven until the runtime value is available.
                None
            } else {
                infer_bin_count(control, request, &mut diagnostics)
            }
        }
    };
    if bins.is_some_and(|count| count == 0 || count > IMHIST_MAX_BINS) {
        diagnostics.push(argument_error(
            "RM-CATALOG-IMHIST-BINS",
            "histogram bin count must be in 1..=1000000",
            1,
        ));
    }
    let output = if diagnostics.is_empty() {
        let mut fact = ValueFact::proven(
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            }),
            ShapeFact::from(vec![bins, Some(1)]),
            StorageFact::Dense,
        );
        if image.is_some_and(|value| !matches!(value.residency, ResidencyFact::Host)) {
            fact.residency = ResidencyFact::Unknown;
        }
        fact
    } else {
        ValueFact::unknown(DynamicReason::UnsupportedRepresentation)
    };
    let mut contract = CallContract::fixed(vec![output.clone(), output]);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    result.dynamic_outputs = requested.is_none();
    result.diagnostics.extend(diagnostics);
    result
}

#[derive(Clone, Copy)]
enum ImageClass {
    Logical,
    Numeric,
}

fn image_class(
    image: &ValueFact,
    index: usize,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) -> Option<ImageClass> {
    if matches!(image.storage, StorageFact::Sparse | StorageFact::Opaque) {
        diagnostics.push(argument_error(
            "RM-CATALOG-IMHIST-STORAGE",
            "imhist requires dense image storage",
            index,
        ));
        return None;
    }
    match image.kind {
        ValueKindFact::Logical => Some(ImageClass::Logical),
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Int64 | NumericClass::UInt64,
            ..
        }) => {
            diagnostics.push(argument_error(
                "RM-CATALOG-IMHIST-WIDE",
                "int64 and uint64 image classes are unsupported",
                index,
            ));
            None
        }
        ValueKindFact::Numeric(NumericFact {
            domain: NumericDomain::Complex,
            ..
        }) => {
            diagnostics.push(argument_error(
                "RM-CATALOG-IMHIST-COMPLEX",
                "imhist requires real image values",
                index,
            ));
            None
        }
        ValueKindFact::Numeric(_) => Some(ImageClass::Numeric),
        ValueKindFact::Unknown => None,
        _ => {
            diagnostics.push(argument_error(
                "RM-CATALOG-IMHIST-IMAGE",
                "imhist expects a real numeric or logical image",
                index,
            ));
            None
        }
    }
}

fn is_known_colormap_shape(value: &ValueFact) -> bool {
    value
        .shape
        .known_dims()
        .is_some_and(|dims| dims.len() == 2 && dims.get(1) == Some(&Some(3)))
}

fn could_be_colormap_shape(value: &ValueFact) -> bool {
    value.shape.known_dims().is_none_or(|dims| {
        dims.len() == 2
            && dims
                .get(1)
                .is_some_and(|cols| cols.is_none_or(|count| count == 3))
    })
}

fn infer_bin_count(
    control: &ValueFact,
    request: &CallRequest,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) -> Option<usize> {
    if !matches!(
        control.kind,
        ValueKindFact::Numeric(NumericFact {
            domain: NumericDomain::Real,
            ..
        }) | ValueKindFact::Logical
            | ValueKindFact::Unknown
    ) || matches!(control.storage, StorageFact::Sparse | StorageFact::Opaque)
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-IMHIST-BIN-CLASS",
            "bin count must be a real numeric or logical scalar",
            1,
        ));
    }
    if let Some(value) = request.literals.numeric_at(1) {
        let rounded = value.round();
        if !value.is_finite()
            || value < 1.0
            || (rounded - value).abs() > IMHIST_INTEGER_TOLERANCE
            || rounded > IMHIST_MAX_BINS as f64
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-IMHIST-BIN-VALUE",
                "bin count must be a positive integral scalar within the supported range",
                1,
            ));
            return None;
        }
        return Some(rounded as usize);
    }
    None
}
