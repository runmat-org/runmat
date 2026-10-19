use crate::catalog::inference::argument_error;
use crate::BuiltinCatalogEntry;
use runmat_types::{
    infer_call, CallContract, CallInference, CallRequest, DimensionFact, NumericClass,
    NumericDomain, NumericFact, ResidencyFact, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RgbConversion {
    Gray,
    Hsv,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Layout {
    Truecolor,
    Colormap,
    Unknown,
    Invalid,
}

fn layout(shape: &ShapeFact) -> Layout {
    match shape {
        ShapeFact::Shaped { dims } if dims.len() == 3 => {
            if dims
                .iter()
                .take(2)
                .any(|dim| matches!(dim, DimensionFact::Known(0)))
                || matches!(dims[2], DimensionFact::Known(channel) if channel != 3)
            {
                Layout::Invalid
            } else {
                Layout::Truecolor
            }
        }
        ShapeFact::Shaped { dims } if dims.len() == 2 => {
            if matches!(dims[0], DimensionFact::Known(0))
                || matches!(dims[1], DimensionFact::Known(channel) if channel != 3)
            {
                Layout::Invalid
            } else {
                Layout::Colormap
            }
        }
        ShapeFact::Ranked { rank: 3 } => Layout::Truecolor,
        ShapeFact::Ranked { rank: 2 } => Layout::Colormap,
        ShapeFact::Unknown => Layout::Unknown,
        _ => Layout::Invalid,
    }
}

fn output_shape(shape: &ShapeFact, layout: Layout, conversion: RgbConversion) -> ShapeFact {
    match conversion {
        RgbConversion::Hsv => shape.clone(),
        RgbConversion::Gray => match (shape, layout) {
            (ShapeFact::Shaped { dims }, Layout::Truecolor) => {
                if matches!(
                    (&dims[0], &dims[1]),
                    (DimensionFact::Known(1), DimensionFact::Known(1))
                ) {
                    ShapeFact::Scalar
                } else {
                    ShapeFact::Shaped {
                        dims: dims[..2].to_vec(),
                    }
                }
            }
            (ShapeFact::Ranked { rank: 3 }, Layout::Truecolor) => ShapeFact::Ranked { rank: 2 },
            (_, Layout::Colormap) => shape.clone(),
            _ => ShapeFact::Unknown,
        },
    }
}

pub(super) fn infer(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    conversion: RgbConversion,
) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 1 {
        diagnostics.push(argument_error(
            "RM-CATALOG-RGB-ARITY",
            "RGB conversion requires one input",
            0,
        ));
    }
    let input = request.arguments.first();
    let geometry = input.map_or(Layout::Unknown, |fact| layout(&fact.shape));
    if geometry == Layout::Invalid {
        diagnostics.push(argument_error(
            "RM-CATALOG-RGB-SHAPE",
            "expected a nonempty MxNx3 RGB image or Nx3 colormap",
            0,
        ));
    }
    let class = match input.map(|fact| &fact.kind) {
        Some(ValueKindFact::Numeric(NumericFact {
            class,
            domain: NumericDomain::Real,
        })) => Some(*class),
        Some(ValueKindFact::Unknown) | None => None,
        _ => {
            diagnostics.push(argument_error(
                "RM-CATALOG-RGB-CLASS",
                "expected a real numeric RGB image",
                0,
            ));
            None
        }
    };
    if let Some(class) = class {
        let supported = match geometry {
            Layout::Colormap => class == NumericClass::Double,
            Layout::Truecolor | Layout::Unknown => matches!(
                class,
                NumericClass::Single
                    | NumericClass::Double
                    | NumericClass::UInt8
                    | NumericClass::UInt16
            ),
            Layout::Invalid => true,
        };
        if !supported {
            diagnostics.push(argument_error(
                "RM-CATALOG-RGB-CLASS",
                "unsupported RGB image or colormap class",
                0,
            ));
        }
    }
    if input.is_some_and(|fact| fact.storage == StorageFact::Sparse) {
        diagnostics.push(argument_error(
            "RM-CATALOG-RGB-STORAGE",
            "RGB image must use dense storage",
            0,
        ));
    }
    let shape = output_shape(
        &input.map_or(ShapeFact::Unknown, |fact| fact.shape.clone()),
        geometry,
        conversion,
    );
    let output_class = match conversion {
        RgbConversion::Gray if class == Some(NumericClass::Single) => match shape.element_count() {
            Some(1) => Some(NumericClass::Double),
            Some(_) => Some(NumericClass::Single),
            None => None,
        },
        RgbConversion::Gray => class,
        RgbConversion::Hsv => class.map(|class| {
            if class == NumericClass::Single {
                NumericClass::Single
            } else {
                NumericClass::Double
            }
        }),
    };
    let kind = output_class.map_or(ValueKindFact::Unknown, |class| {
        ValueKindFact::Numeric(NumericFact {
            class,
            domain: NumericDomain::Real,
        })
    });
    let storage = match shape.element_count() {
        Some(1) => StorageFact::Scalar,
        Some(_) => StorageFact::Dense,
        None => StorageFact::Unknown,
    };
    let mut output = ValueFact::proven(kind, shape, storage);
    if output.kind == ValueKindFact::Unknown {
        output.certainty =
            runmat_types::CertaintyFact::Dynamic(runmat_types::DynamicReason::RuntimeValue);
    }
    if !matches!(input.map(|fact| &fact.residency), Some(ResidencyFact::Host)) {
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
