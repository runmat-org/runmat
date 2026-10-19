use crate::catalog::inference::{argument_error, finish_fixed, literal_text};
use crate::BuiltinCatalogEntry;
use runmat_types::*;

pub(super) fn infer_integer(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    class: NumericClass,
) -> CallInference {
    infer(request, entry, Some(class))
}
pub(super) fn infer_global(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer(request, entry, None)
}

fn infer(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    class: Option<NumericClass>,
) -> CallInference {
    let mut output = ValueFact::unknown(DynamicReason::RuntimeValue);
    output.kind = ValueKindFact::Numeric(NumericFact {
        class: class.unwrap_or(NumericClass::Double),
        domain: NumericDomain::Real,
    });
    output.storage = StorageFact::Dense;
    output.residency = ResidencyFact::Host;
    let mut literals = request.literals.literal_args.as_slice();
    let mut count = request.arguments.len();
    if class.is_none() {
        if let Some(text) = literals.last().and_then(literal_text) {
            count = count.saturating_sub(1);
            literals = &literals[..literals.len() - 1];
            output.kind = NumericClass::from_class_name(&text)
                .map(|class| {
                    ValueKindFact::Numeric(NumericFact {
                        class,
                        domain: NumericDomain::Real,
                    })
                })
                .unwrap_or_else(|| match text.to_ascii_lowercase().as_str() {
                    "logical" => ValueKindFact::Logical,
                    "char" => ValueKindFact::Character,
                    "string" => ValueKindFact::String,
                    _ => ValueKindFact::Unknown,
                });
            if text.eq_ignore_ascii_case("gpuarray") {
                output.residency = ResidencyFact::Unknown;
            }
        } else if request.arguments.last().is_some_and(|arg| {
            matches!(
                arg.kind,
                ValueKindFact::Unknown | ValueKindFact::String | ValueKindFact::Character
            )
        }) {
            output.kind = ValueKindFact::Unknown;
            output.storage = StorageFact::Unknown;
            output.residency = ResidencyFact::Unknown;
        }
    }
    let mut diagnostics = Vec::new();
    match dimensions(literals, count) {
        Ok(Some(shape)) => {
            output.shape = ShapeFact::from(shape.into_iter().map(Some).collect::<Vec<_>>())
        }
        Err(()) => diagnostics.push(argument_error(
            "RM-EMPTY-SIZE",
            "empty requires integer dimensions and at least one zero dimension",
            0,
        )),
        Ok(None) => {}
    }
    finish_fixed(entry, request, output, diagnostics)
}

fn dimensions(literals: &[LiteralValue], count: usize) -> Result<Option<Vec<usize>>, ()> {
    if count == 0 {
        return Ok(Some(vec![0, 0]));
    }
    if literals.len() != count {
        return Ok(None);
    }
    let values = if count == 1 {
        match &literals[0] {
            LiteralValue::Empty => return Ok(Some(vec![0, 0])),
            LiteralValue::Vector(values) => values.as_slice(),
            LiteralValue::Matrix(rows) if rows.len() == 1 => rows[0].as_slice(),
            LiteralValue::Matrix(_) => return Err(()),
            _ => literals,
        }
    } else {
        literals
    };
    let mut shape = Vec::with_capacity(values.len());
    for value in values {
        match dimension(value)? {
            Some(value) => shape.push(value),
            None => return Ok(None),
        }
    }
    if shape.is_empty() {
        shape = vec![0, 0];
    }
    if shape.len() == 1 {
        shape.push(shape[0]);
    }
    while shape.len() > 2 && shape.last() == Some(&1) {
        shape.pop();
    }
    if !shape.contains(&0) {
        return Err(());
    }
    Ok(Some(shape))
}

fn dimension(value: &LiteralValue) -> Result<Option<usize>, ()> {
    match value {
        LiteralValue::Integer { text, .. } => {
            let value = text.parse::<i128>().map_err(|_| ())?.max(0);
            usize::try_from(value).map(Some).map_err(|_| ())
        }
        LiteralValue::Number(value) => floating(*value),
        LiteralValue::Real { text, .. } => floating(text.parse().map_err(|_| ())?),
        LiteralValue::Unknown | LiteralValue::Symbolic(_) => Ok(None),
        _ => Err(()),
    }
}
fn floating(value: f64) -> Result<Option<usize>, ()> {
    if !value.is_finite() {
        return Err(());
    }
    if value < 0.0 {
        return Ok(Some(0));
    }
    let fits_usize = value < usize::MAX as f64 || (usize::BITS < 64 && value == usize::MAX as f64);
    if value.fract() != 0.0 || !fits_usize {
        return Err(());
    }
    Ok(Some(value as usize))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn request(literals: Vec<LiteralValue>) -> CallRequest {
        CallRequest {
            arguments: literals
                .iter()
                .map(|_| ValueFact::unknown(DynamicReason::RuntimeValue))
                .collect(),
            literals: LiteralContext::new(literals),
            outputs: OutputSelection::default(),
        }
    }
    #[test]
    fn classes_shapes_unknown_and_outputs() {
        let entry = &super::super::UINT64_EMPTY_CATALOG_ENTRY;
        let mut req = request(vec![LiteralValue::Vector(vec![
            LiteralValue::Number(0.0),
            LiteralValue::Number(3.0),
            LiteralValue::Number(1.0),
        ])]);
        let result = infer_integer(&req, entry, NumericClass::UInt64);
        assert_eq!(
            result.outputs[0].shape,
            ShapeFact::from(vec![Some(0), Some(3)])
        );
        assert_eq!(
            result.outputs[0].numeric().unwrap().class,
            NumericClass::UInt64
        );
        req.literals = LiteralContext::new(vec![LiteralValue::Unknown]);
        req.arguments[0].shape = ShapeFact::from(vec![Some(1), Some(2)]);
        assert_eq!(
            infer_integer(&req, entry, NumericClass::UInt64).outputs[0].shape,
            ShapeFact::Unknown
        );
        req.outputs = OutputSelection::new(RequestedOutputCount::Exactly(2));
        assert!(!infer_integer(&req, entry, NumericClass::UInt64)
            .diagnostics
            .is_empty());
    }
    #[test]
    fn size_errors_and_exact_wide_dimensions() {
        assert_eq!(floating(usize::MAX as f64).is_ok(), usize::BITS < 64);
        assert!(dimensions(&[LiteralValue::Number(2.0)], 1).is_err());
        assert!(dimension(&LiteralValue::Number(0.5)).is_err());
        assert_eq!(dimension(&LiteralValue::Number(-0.5)), Ok(Some(0)));
        assert_eq!(
            dimension(&LiteralValue::Integer {
                text: u64::MAX.to_string(),
                class: NumericClass::UInt64
            }),
            usize::try_from(u64::MAX).map(Some).map_err(|_| ())
        );
    }
    #[test]
    fn global_typename_changes_class_not_shape() {
        let result = infer_global(
            &request(vec![
                LiteralValue::Number(0.0),
                LiteralValue::Number(4.0),
                LiteralValue::String("single".into()),
            ]),
            &super::super::EMPTY_CATALOG_ENTRY,
        );
        assert_eq!(
            result.outputs[0].numeric().unwrap().class,
            NumericClass::Single
        );
        assert_eq!(
            result.outputs[0].shape,
            ShapeFact::from(vec![Some(0), Some(4)])
        );
    }
}
