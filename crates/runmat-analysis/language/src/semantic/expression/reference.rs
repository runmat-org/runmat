use runmat_parser::fea::FeaReference;

use super::{CheckedExpression, CheckedExpressionKind};
use crate::{FeaFieldType, FeaType, PhysicalDimension};

pub(super) fn quantity_type(value: &FeaType, dimension: PhysicalDimension) -> Option<FeaType> {
    match value {
        FeaType::Integer | FeaType::Real => Some(FeaType::Quantity(dimension)),
        FeaType::Vector { element, length } => Some(FeaType::Vector {
            element: Box::new(quantity_type(element, dimension)?),
            length: *length,
        }),
        _ => None,
    }
}

pub(super) fn contextual_reference_type(path: &[String]) -> Option<FeaType> {
    match path {
        [analysis, _, state, _] if analysis == "analysis" && state == "state" => {
            Some(FeaType::Unknown)
        }
        [analysis, _, field, location, quantity] if analysis == "analysis" && field == "field" => {
            let location = super::super::procedure::field_location(location)?;
            super::super::procedure::field_reference_type(quantity).map(|(quantity, value_type)| {
                FeaType::Field(FeaFieldType {
                    value: Box::new(value_type),
                    location: Some(location),
                    quantity: Some(quantity),
                })
            })
        }
        [model, namespace, _] if model == "model" && namespace == "response_spectrum" => {
            Some(FeaType::ResponseSpectrum)
        }
        [model, namespace, _] if model == "model" && namespace == "random_excitation" => {
            Some(FeaType::RandomExcitation)
        }
        [model, namespace, _]
            if model == "model"
                && matches!(
                    namespace.as_str(),
                    "constraint"
                        | "boundary"
                        | "connector"
                        | "interface"
                        | "excitation"
                        | "load"
                        | "load_case"
                        | "signal"
                        | "initial"
                ) =>
        {
            Some(FeaType::Unknown)
        }
        [case, ..] if case == "case" => Some(FeaType::Unknown),
        [profile, _] if profile == "profile" => Some(FeaType::Enum),
        _ => None,
    }
}

pub(super) fn reference_path(reference: &FeaReference) -> Vec<String> {
    reference
        .segments
        .iter()
        .map(|segment| segment.text.clone())
        .collect()
}

pub(super) fn is_untyped_zero(expression: &CheckedExpression) -> bool {
    matches!(&expression.kind, CheckedExpressionKind::Number(value) if value.parse::<f64>() == Ok(0.0))
}
