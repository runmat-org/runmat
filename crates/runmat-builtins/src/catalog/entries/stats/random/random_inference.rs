//! Shared random-family fact construction, not an identity registry.
use crate::{
    catalog::inference::{argument_error, numeric_kind},
    BuiltinCatalogEntry,
};
use runmat_types::*;

pub(super) fn finish(
    entry: &BuiltinCatalogEntry,
    request: &CallRequest,
    outputs: Vec<ValueFact>,
    diagnostics: Vec<InferenceDiagnostic>,
    padded: bool,
) -> CallInference {
    let mut contract = CallContract::fixed(outputs);
    if padded {
        contract.maximum_outputs = None;
        contract.variadic_output = Some(Box::new(numeric(
            ShapeFact::from(vec![Some(0), Some(0)]),
            Some(NumericClass::Double),
            ResidencyFact::Host,
        )));
    }
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    result.diagnostics.splice(0..0, diagnostics);
    result
}

pub(super) fn numeric(
    shape: ShapeFact,
    class: Option<NumericClass>,
    residency: ResidencyFact,
) -> ValueFact {
    let mut fact = ValueFact::unknown(DynamicReason::RuntimeValue);
    if let Some(class) = class {
        fact.kind = numeric_kind(class, NumericDomain::Real);
    }
    fact.shape = shape;
    fact.residency = residency;
    fact.layout = LayoutFact::ColumnMajor;
    fact.contiguity = ContiguityFact::Contiguous;
    fact.view = ViewFact::Materialized;
    fact.alias = AliasFact::Unique;
    fact.mutation = MutationFact::ValueSemantics;
    // Runtime collapses a scalar double tensor to Value::Num, but retains single.
    if class == Some(NumericClass::Single) || fact.shape.element_count().is_some_and(|n| n != 1) {
        fact.storage = StorageFact::Dense;
    }
    fact
}

pub(super) fn precision(args: &[ValueFact]) -> Option<NumericClass> {
    if args.iter().any(|arg| {
        arg.numeric()
            .is_some_and(|n| n.class == NumericClass::Single)
    }) {
        Some(NumericClass::Single)
    } else if args.iter().any(|arg| arg.kind == ValueKindFact::Unknown) {
        None
    } else {
        Some(NumericClass::Double)
    }
}

/// Extents are parsed as integers, without rounding a native integer via f64.
pub(super) fn count(value: &LiteralValue) -> Option<usize> {
    match value {
        LiteralValue::Integer { .. } => value
            .integer_value_i128()
            .and_then(|n| usize::try_from(n).ok()),
        LiteralValue::Number(n)
            if n.is_finite() && *n >= 0.0 && n.fract() == 0.0 && *n < usize::MAX as f64 =>
        {
            Some(*n as usize)
        }
        LiteralValue::Real { text, .. } => text
            .parse::<f64>()
            .ok()
            .and_then(|n| count(&LiteralValue::Number(n))),
        _ => None,
    }
}

pub(super) fn literal_count(request: &CallRequest, index: usize) -> Option<usize> {
    request.literals.literal_args.get(index).and_then(count)
}

pub(super) fn normalized_shape(shape: &ShapeFact) -> ShapeFact {
    if *shape == ShapeFact::Scalar {
        return ShapeFact::Scalar;
    }
    let Some(mut dims) = shape.known_dims() else {
        return ShapeFact::Unknown;
    };
    if dims.is_empty() {
        dims = vec![Some(1), Some(1)];
    }
    if dims.len() == 1 {
        dims.push(Some(1));
    }
    while dims.len() > 2 && dims.last() == Some(&Some(1)) {
        dims.pop();
    }
    if dims.len() > 2 && dims.last() == Some(&None) {
        return ShapeFact::Unknown;
    }
    ShapeFact::from(dims)
}

pub(super) fn dimensions(request: &CallRequest, start: usize) -> ShapeFact {
    if request.arguments.len() == start {
        return ShapeFact::Scalar;
    }
    let mut dims = Vec::new();
    for index in start..request.arguments.len() {
        let Some(value) = request.literals.literal_args.get(index) else {
            return ShapeFact::Unknown;
        };
        let values = match value {
            LiteralValue::Vector(values) => values.iter().collect::<Vec<_>>(),
            LiteralValue::Matrix(rows)
                if rows.len() == 1 || rows.iter().all(|row| row.len() == 1) =>
            {
                rows.iter().flatten().collect()
            }
            LiteralValue::Empty => Vec::new(),
            LiteralValue::Matrix(_) => return ShapeFact::Unknown,
            value => vec![value],
        };
        for value in values {
            let Some(n) = count(value) else {
                return ShapeFact::Unknown;
            };
            dims.push(Some(n));
        }
    }
    if dims.is_empty() {
        dims = vec![Some(0), Some(0)];
    }
    if dims.len() == 1 {
        dims.push(dims[0]);
    }
    while dims.len() > 2 && dims.last() == Some(&Some(1)) {
        dims.pop();
    }
    ShapeFact::from(dims)
}

pub(super) fn arity(
    request: &CallRequest,
    minimum: usize,
    maximum: Option<usize>,
) -> Vec<InferenceDiagnostic> {
    if request.arguments.len() < minimum || maximum.is_some_and(|n| request.arguments.len() > n) {
        vec![argument_error(
            "RM-CATALOG-RANDOM-ARITY",
            "unsupported number of random-function inputs",
            request.arguments.len(),
        )]
    } else {
        Vec::new()
    }
}

/// Scalar-parameter distributions share precision selection, but not placement.
pub(super) fn scalar_distribution(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    single: bool,
    preserve_owner: bool,
) -> CallInference {
    let mut diagnostics = arity(request, 2, None);
    for (index, arg) in request.arguments.iter().take(2).enumerate() {
        if arg.shape.element_count().is_some_and(|n| n != 1) {
            diagnostics.push(argument_error(
                "RM-CATALOG-RANDOM-SCALAR",
                "distribution parameters must be scalar",
                index,
            ));
        }
        if matches!(
            arg.kind,
            ValueKindFact::Numeric(NumericFact {
                domain: NumericDomain::Complex,
                ..
            })
        ) {
            diagnostics.push(argument_error(
                "RM-CATALOG-RANDOM-REAL",
                "distribution parameters must be real",
                index,
            ));
        }
    }
    let data = &request.arguments[..request.arguments.len().min(2)];
    let residency =
        if !preserve_owner || data.iter().all(|arg| arg.residency == ResidencyFact::Host) {
            ResidencyFact::Host
        } else {
            ResidencyFact::Unknown
        };
    let class = if single {
        precision(data)
    } else {
        Some(NumericClass::Double)
    };
    finish(
        entry,
        request,
        vec![numeric(dimensions(request, 2), class, residency)],
        diagnostics,
        false,
    )
}

#[cfg(test)]
mod tests {
    use super::super::{
        BOOTSTRP_CATALOG_ENTRY, DATASAMPLE_CATALOG_ENTRY, LHSDESIGN_CATALOG_ENTRY,
        MVNRND_CATALOG_ENTRY, NORMRND_CATALOG_ENTRY, RANDSAMPLE_CATALOG_ENTRY, RNG_CATALOG_ENTRY,
    };
    use super::*;
    fn request(
        arguments: Vec<ValueFact>,
        literals: Vec<LiteralValue>,
        outputs: usize,
    ) -> CallRequest {
        CallRequest {
            arguments,
            literals: LiteralContext::new(literals),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
        }
    }
    fn infer(entry: &BuiltinCatalogEntry, request: &CallRequest) -> CallInference {
        crate::infer_catalog_call(entry, request)
    }
    #[test]
    fn rng_snapshot_is_structured_but_seed_and_legacy_state_remain_dynamic() {
        let result = infer(&RNG_CATALOG_ENTRY, &request(vec![], vec![], 1));
        assert!(matches!(result.outputs[0].kind, ValueKindFact::Struct(_)));
        assert_eq!(result.outputs[0].shape, ShapeFact::Scalar);
        assert!(result.effects.0.contains(&EffectKind::Randomness));
        assert!(result.effects.0.contains(&EffectKind::Clock));
        assert!(result.effects.0.contains(&EffectKind::EnvironmentRead));
        assert!(!infer(&RNG_CATALOG_ENTRY, &request(vec![], vec![], 2))
            .diagnostics
            .is_empty());
    }
    #[test]
    fn normal_precision_is_selected_by_data_not_structural_extents() {
        let double = numeric(
            ShapeFact::Scalar,
            Some(NumericClass::Double),
            ResidencyFact::Host,
        );
        let single = numeric(
            ShapeFact::Scalar,
            Some(NumericClass::Single),
            ResidencyFact::Host,
        );
        let req = request(
            vec![double.clone(), single.clone(), double.clone(), double],
            vec![
                LiteralValue::Unknown,
                LiteralValue::Unknown,
                LiteralValue::Number(2.0),
                LiteralValue::Number(3.0),
            ],
            1,
        );
        let out = infer(&NORMRND_CATALOG_ENTRY, &req).outputs.remove(0);
        assert_eq!(out.numeric().unwrap().class, NumericClass::Single);
        assert_eq!(out.shape, ShapeFact::from(vec![Some(2), Some(3)]));
        assert_eq!(out.residency, ResidencyFact::Host);
        let mut invalid = req;
        invalid.arguments[0].shape = ShapeFact::from(vec![Some(2), Some(1)]);
        assert!(!infer(&NORMRND_CATALOG_ENTRY, &invalid)
            .diagnostics
            .is_empty());
    }
    #[test]
    fn multivariate_rows_and_single_precision_are_preserved() {
        let mu = numeric(
            ShapeFact::from(vec![Some(1), Some(3)]),
            Some(NumericClass::Single),
            ResidencyFact::Host,
        );
        let sigma = numeric(
            ShapeFact::from(vec![Some(3), Some(3)]),
            Some(NumericClass::Double),
            ResidencyFact::Host,
        );
        let count = numeric(
            ShapeFact::Scalar,
            Some(NumericClass::UInt64),
            ResidencyFact::Host,
        );
        let req = request(
            vec![mu, sigma, count],
            vec![
                LiteralValue::Unknown,
                LiteralValue::Unknown,
                LiteralValue::Integer {
                    text: "7".into(),
                    class: NumericClass::UInt64,
                },
            ],
            1,
        );
        let out = infer(&MVNRND_CATALOG_ENTRY, &req).outputs.remove(0);
        assert_eq!(out.numeric().unwrap().class, NumericClass::Single);
        assert_eq!(out.shape, ShapeFact::from(vec![Some(7), Some(3)]));
    }
    #[test]
    fn sample_and_callback_output_classes_have_distinct_authorities() {
        let data = numeric(
            ShapeFact::from(vec![Some(5), Some(3)]),
            Some(NumericClass::UInt64),
            ResidencyFact::Host,
        );
        let k = numeric(
            ShapeFact::Scalar,
            Some(NumericClass::Double),
            ResidencyFact::Host,
        );
        let req = request(
            vec![data.clone(), k.clone()],
            vec![LiteralValue::Unknown, LiteralValue::Number(2.0)],
            2,
        );
        let result = infer(&DATASAMPLE_CATALOG_ENTRY, &req);
        assert_eq!(
            result.outputs[0].numeric().unwrap().class,
            NumericClass::UInt64
        );
        assert_eq!(
            result.outputs[0].shape,
            ShapeFact::from(vec![Some(2), Some(3)])
        );
        assert_eq!(
            result.outputs[1].numeric().unwrap().class,
            NumericClass::Double
        );
        assert_eq!(
            result.outputs[1].shape,
            ShapeFact::from(vec![Some(2), Some(1)])
        );
        let callback = ValueFact::unknown(DynamicReason::RuntimeValue);
        let req = request(
            vec![k, callback, data],
            vec![
                LiteralValue::Number(2.0),
                LiteralValue::Unknown,
                LiteralValue::Unknown,
            ],
            2,
        );
        let result = infer(&BOOTSTRP_CATALOG_ENTRY, &req);
        assert_eq!(result.outputs[0].kind, ValueKindFact::Unknown);
        assert_eq!(
            result.outputs[1].numeric().unwrap().class,
            NumericClass::Double
        );
    }
    #[test]
    fn lhs_dimensions_are_known_and_zero_outputs_retain_effects() {
        let dimension = numeric(
            ShapeFact::Scalar,
            Some(NumericClass::Double),
            ResidencyFact::Host,
        );
        let req = request(
            vec![dimension.clone(), dimension],
            vec![LiteralValue::Number(4.0), LiteralValue::Number(2.0)],
            0,
        );
        let result = infer(&LHSDESIGN_CATALOG_ENTRY, &req);
        assert!(result.outputs.is_empty());
        assert!(result.effects.0.contains(&EffectKind::Randomness));
    }
    #[test]
    fn dimensions_are_exact_and_normalized() {
        assert_eq!(
            count(&LiteralValue::Integer {
                text: "9007199254740993".into(),
                class: NumericClass::UInt64
            }),
            usize::try_from(9007199254740993_u64).ok()
        );
        assert_eq!(count(&LiteralValue::Number(1.5)), None);
        let request = CallRequest {
            arguments: vec![ValueFact::unknown(DynamicReason::RuntimeValue); 3],
            literals: LiteralContext::new(vec![
                LiteralValue::Number(2.0),
                LiteralValue::Number(3.0),
                LiteralValue::Number(1.0),
            ]),
            outputs: OutputSelection::default(),
        };
        assert_eq!(
            dimensions(&request, 0),
            ShapeFact::from(vec![Some(2), Some(3)])
        );
    }

    #[test]
    fn bootstrap_zero_outputs_skip_argument_validation() {
        let result = infer(&BOOTSTRP_CATALOG_ENTRY, &request(Vec::new(), Vec::new(), 0));
        assert!(result.outputs.is_empty());
        assert!(result.diagnostics.is_empty());
    }

    #[test]
    fn unknown_population_orientation_is_not_guessed() {
        let population = numeric(
            ShapeFact::Ranked { rank: 2 },
            Some(NumericClass::UInt64),
            ResidencyFact::Host,
        );
        let count = numeric(
            ShapeFact::Scalar,
            Some(NumericClass::Double),
            ResidencyFact::Host,
        );
        let req = request(
            vec![population, count],
            vec![LiteralValue::Unknown, LiteralValue::Number(3.0)],
            1,
        );
        let result = infer(&RANDSAMPLE_CATALOG_ENTRY, &req);
        assert_eq!(result.outputs[0].shape, ShapeFact::Unknown);
        assert_eq!(result.outputs[0].kind, ValueKindFact::Unknown);
    }
}
