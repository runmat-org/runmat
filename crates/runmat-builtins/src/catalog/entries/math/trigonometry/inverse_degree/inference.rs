use crate::catalog::inference::{argument_error, finish_fixed_outputs};
use crate::BuiltinCatalogEntry;
use runmat_types::{
    CallInference, CallRequest, DynamicReason, LiteralValue, NumericClass, NumericDomain,
    NumericFact, ResidencyFact, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

#[derive(Clone, Copy)]
pub(super) enum Function {
    Asin,
    Acos,
    Atan,
    Asec,
    Acsc,
    Acot,
}

impl Function {
    fn real_domain_is_closed(self) -> bool {
        matches!(self, Self::Atan | Self::Acot)
    }
}

pub(super) fn infer(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    function: Function,
) -> CallInference {
    let name = entry.identity.name;
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 1 {
        diagnostics.push(argument_error(
            "RM-CATALOG-INVERSE-DEGREE-ARITY",
            format!("{name} requires exactly one input"),
            request.arguments.len().saturating_sub(1),
        ));
    }
    let input = request.arguments.first();
    let mut output = ValueFact::unknown(DynamicReason::RuntimeValue);
    output.shape = input.map_or(ShapeFact::Unknown, |input| input.shape.clone());
    output.storage = StorageFact::Dense;
    output.residency = input.map_or(ResidencyFact::Unknown, |input| match &input.residency {
        ResidencyFact::Host => ResidencyFact::Host,
        _ => ResidencyFact::Unknown,
    });
    if let Some(input) = input {
        if matches!(input.storage, StorageFact::Sparse) {
            diagnostics.push(argument_error(
                "RM-CATALOG-INVERSE-DEGREE-SPARSE",
                format!("{name} does not accept sparse input"),
                0,
            ));
        }
        let class_and_domain = match &input.kind {
            ValueKindFact::Numeric(NumericFact { class, domain })
                if *domain == NumericDomain::Complex
                    && !matches!(class, NumericClass::Single | NumericClass::Double) =>
            {
                diagnostics.push(argument_error(
                    "RM-CATALOG-INVERSE-DEGREE-COMPLEX-INTEGER",
                    format!("{name} does not accept complex integer input"),
                    0,
                ));
                None
            }
            ValueKindFact::Numeric(NumericFact { class, domain }) => Some((
                if *class == NumericClass::Single {
                    NumericClass::Single
                } else {
                    NumericClass::Double
                },
                *domain,
            )),
            ValueKindFact::Logical | ValueKindFact::Character => {
                Some((NumericClass::Double, NumericDomain::Real))
            }
            ValueKindFact::Unknown => None,
            _ => {
                diagnostics.push(argument_error(
                    "RM-CATALOG-INVERSE-DEGREE-INPUT",
                    format!("{name} requires numeric, logical or character input"),
                    0,
                ));
                None
            }
        };
        if let Some((class, domain)) = class_and_domain {
            let literal_domain = request
                .literals
                .literal_args
                .first()
                .and_then(|literal| literal_domain(literal, function));
            if let Some(output_domain) = if domain == NumericDomain::Complex {
                Some(NumericDomain::Complex)
            } else if function.real_domain_is_closed() {
                Some(NumericDomain::Real)
            } else {
                literal_domain
            } {
                output.kind = ValueKindFact::Numeric(NumericFact {
                    class,
                    domain: output_domain,
                });
            }
        }
    }
    finish_fixed_outputs(entry, request, vec![output], diagnostics)
}

fn literal_domain(literal: &LiteralValue, function: Function) -> Option<NumericDomain> {
    match literal {
        LiteralValue::Number(value) => Some(real_domain(*value, function)),
        LiteralValue::Real { text, .. } | LiteralValue::Integer { text, .. } => text
            .parse::<f64>()
            .ok()
            .map(|value| real_domain(value, function)),
        LiteralValue::Bool(value) => Some(real_domain(f64::from(u8::from(*value)), function)),
        LiteralValue::Character(value) => combine_domains(
            value
                .chars()
                .map(|character| Some(real_domain(f64::from(u32::from(character)), function))),
        ),
        LiteralValue::Complex { .. } => Some(NumericDomain::Complex),
        LiteralValue::Vector(values) => {
            combine_domains(values.iter().map(|value| literal_domain(value, function)))
        }
        LiteralValue::Matrix(rows) => combine_domains(
            rows.iter()
                .flatten()
                .map(|value| literal_domain(value, function)),
        ),
        LiteralValue::Empty => Some(NumericDomain::Real),
        LiteralValue::String(_)
        | LiteralValue::Keyword(_)
        | LiteralValue::Symbolic(_)
        | LiteralValue::Unknown => None,
    }
}

fn combine_domains(
    domains: impl IntoIterator<Item = Option<NumericDomain>>,
) -> Option<NumericDomain> {
    let mut combined = NumericDomain::Real;
    for domain in domains {
        if domain? == NumericDomain::Complex {
            combined = NumericDomain::Complex;
        }
    }
    Some(combined)
}

fn real_domain(value: f64, function: Function) -> NumericDomain {
    let real = match function {
        Function::Asin | Function::Acos => value.is_nan() || (-1.0..=1.0).contains(&value),
        Function::Atan | Function::Acot => true,
        Function::Asec | Function::Acsc => value.is_nan() || value.abs() >= 1.0,
    };
    if real {
        NumericDomain::Real
    } else {
        NumericDomain::Complex
    }
}
