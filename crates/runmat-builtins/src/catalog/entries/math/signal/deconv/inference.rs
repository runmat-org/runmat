use crate::catalog::inference::argument_error;
use crate::BuiltinCatalogEntry;
use runmat_types::*;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 2 {
        diagnostics.push(argument_error(
            "RM-CATALOG-DECONV-ARITY",
            "deconv expects a numerator and denominator",
            request.arguments.len().saturating_sub(1),
        ));
    }
    let count = request.outputs.requested.known_count();
    if count.is_some_and(|count| count > 2) {
        diagnostics.push(argument_error(
            "RM-CATALOG-DECONV-OUTPUTS",
            "deconv returns at most a quotient and remainder",
            2,
        ));
    }
    let numerator = request
        .arguments
        .first()
        .and_then(|value| vector(value, 0, &mut diagnostics));
    let denominator = request
        .arguments
        .get(1)
        .and_then(|value| vector(value, 1, &mut diagnostics));
    if denominator.is_some_and(|value| value.len == Some(0))
        || request
            .literals
            .literal_args
            .get(1)
            .is_some_and(definitely_all_zero)
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-DECONV-DENOMINATOR",
            "deconv denominator must contain a nonzero coefficient",
            1,
        ));
    }
    for index in 0..request.arguments.len().min(2) {
        if request
            .literals
            .literal_args
            .get(index)
            .is_some_and(known_inexact_integer)
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-DECONV-INTEGER",
                "deconv integer coefficients must convert to double exactly",
                index,
            ));
        }
    }
    let admitted = diagnostics.is_empty();
    let output_count = count.unwrap_or(1).min(2);
    let outputs = (0..output_count)
        .map(|slot| {
            if admitted {
                match (numerator, denominator) {
                    (Some(left), Some(right)) => output(left, right, slot),
                    _ => ValueFact::unknown(DynamicReason::RuntimeValue),
                }
            } else {
                ValueFact::unknown(DynamicReason::UnsupportedRepresentation)
            }
        })
        .collect();
    let mut contract = CallContract::fixed(outputs);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    result.dynamic_outputs = count.is_none();
    result.diagnostics.extend(diagnostics);
    result
}

fn known_inexact_integer(value: &LiteralValue) -> bool {
    match value {
        LiteralValue::Vector(values) => values.iter().any(known_inexact_integer),
        LiteralValue::Matrix(rows) => rows.iter().flatten().any(known_inexact_integer),
        value => value.integer_is_exact_binary64() == Some(false),
    }
}

fn definitely_all_zero(value: &LiteralValue) -> bool {
    match value {
        LiteralValue::Vector(values) => values.is_empty() || values.iter().all(definitely_all_zero),
        LiteralValue::Matrix(rows) => {
            rows.is_empty() || rows.iter().flatten().all(definitely_all_zero)
        }
        LiteralValue::Empty | LiteralValue::Bool(false) => true,
        LiteralValue::Number(number) => *number == 0.0,
        LiteralValue::Real { .. } | LiteralValue::Integer { .. } => {
            LiteralContext::numeric_from_literal(value) == Some(0.0)
        }
        LiteralValue::Complex {
            real, imaginary, ..
        } => real.parse::<f64>().ok() == Some(0.0) && imaginary.parse::<f64>().ok() == Some(0.0),
        _ => false,
    }
}

#[derive(Clone, Copy)]
struct VectorFact {
    class: Option<NumericClass>,
    domain: Option<NumericDomain>,
    integer_or_logical: bool,
    len: Option<usize>,
    orientation: Option<Orientation>,
    host: bool,
}

#[derive(Clone, Copy)]
enum Orientation {
    Row,
    Column,
}

fn vector(
    value: &ValueFact,
    index: usize,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) -> Option<VectorFact> {
    if matches!(value.storage, StorageFact::Sparse | StorageFact::Opaque) {
        diagnostics.push(argument_error(
            "RM-CATALOG-DECONV-STORAGE",
            "deconv requires dense coefficient vectors",
            index,
        ));
        return None;
    }
    let (class, domain, integer_or_logical) = match value.kind {
        ValueKindFact::Numeric(fact) => (
            Some(fact.class),
            Some(fact.domain),
            fact.class.integer_class().is_some(),
        ),
        ValueKindFact::Logical => (Some(NumericClass::Double), Some(NumericDomain::Real), true),
        ValueKindFact::Unknown => (None, None, false),
        _ => {
            diagnostics.push(argument_error(
                "RM-CATALOG-DECONV-INPUT",
                "deconv expects numeric or logical coefficient vectors",
                index,
            ));
            return None;
        }
    };
    if class.is_some_and(|class| class.integer_class().is_some())
        && domain == Some(NumericDomain::Complex)
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-DECONV-COMPLEX-INTEGER",
            "deconv does not admit typed complex-integer coefficients",
            index,
        ));
        return None;
    }
    let (len, orientation) = match &value.shape {
        ShapeFact::Scalar => (Some(1), Some(Orientation::Row)),
        ShapeFact::Shaped { dims } => {
            if dims.len() > 2
                && dims[2..]
                    .iter()
                    .any(|dim| matches!(dim, DimensionFact::Known(n) if *n != 1))
            {
                diagnostics.push(argument_error(
                    "RM-CATALOG-DECONV-VECTOR",
                    "deconv inputs must be vectors",
                    index,
                ));
                return None;
            }
            let known = dims
                .iter()
                .map(|dim| match dim {
                    DimensionFact::Known(n) => Some(*n),
                    _ => None,
                })
                .collect::<Option<Vec<_>>>();
            match known {
                Some(known) => {
                    let len = known
                        .iter()
                        .try_fold(1_usize, |product, dim| product.checked_mul(*dim));
                    let (rows, cols) = match known.as_slice() {
                        [] => (1, 1),
                        [cols] => (1, *cols),
                        [rows, cols, ..] => (*rows, *cols),
                    };
                    if rows > 1 && cols > 1 && len != Some(0) {
                        diagnostics.push(argument_error(
                            "RM-CATALOG-DECONV-VECTOR",
                            "deconv inputs must be vectors",
                            index,
                        ));
                        return None;
                    }
                    (
                        len,
                        Some(if cols == 1 && rows > 1 {
                            Orientation::Column
                        } else {
                            Orientation::Row
                        }),
                    )
                }
                None => (None, None),
            }
        }
        ShapeFact::Unknown | ShapeFact::Ranked { .. } => (None, None),
    };
    Some(VectorFact {
        class,
        domain,
        integer_or_logical,
        len,
        orientation,
        host: value.residency == ResidencyFact::Host,
    })
}

fn output(left: VectorFact, right: VectorFact, slot: usize) -> ValueFact {
    let class = if left.integer_or_logical || right.integer_or_logical {
        NumericClass::Double
    } else if left.class == Some(NumericClass::Single) || right.class == Some(NumericClass::Single)
    {
        if left.class.is_none() || right.class.is_none() {
            return ValueFact::unknown(DynamicReason::RuntimeValue);
        }
        NumericClass::Single
    } else if left.class.is_some() && right.class.is_some() {
        NumericClass::Double
    } else {
        return ValueFact::unknown(DynamicReason::RuntimeValue);
    };
    let domain = if left.domain == Some(NumericDomain::Complex)
        || right.domain == Some(NumericDomain::Complex)
    {
        NumericDomain::Complex
    } else if left.domain.is_some() && right.domain.is_some() {
        NumericDomain::Real
    } else {
        return ValueFact::unknown(DynamicReason::RuntimeValue);
    };
    let shape = if slot == 1 {
        left.len
            .zip(left.orientation)
            .map_or(ShapeFact::Unknown, |(len, orientation)| {
                oriented_shape(Some(len), orientation)
            })
    } else if left.len == Some(1) {
        ShapeFact::Scalar
    } else {
        left.orientation.map_or(ShapeFact::Unknown, |orientation| {
            oriented_shape(None, orientation)
        })
    };
    let host = left.host && right.host;
    let storage = match shape.element_count() {
        _ if !host => StorageFact::Unknown,
        Some(1) if class == NumericClass::Double => StorageFact::Scalar,
        Some(_) => StorageFact::Dense,
        None => StorageFact::Unknown,
    };
    let mut fact = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact { class, domain }),
        shape,
        storage,
    );
    if !host {
        fact.residency = ResidencyFact::Unknown;
        fact.layout = LayoutFact::Unknown;
        fact.contiguity = ContiguityFact::Unknown;
        fact.view = ViewFact::Unknown;
        fact.alias = AliasFact::Unknown;
    }
    fact
}

fn oriented_shape(len: Option<usize>, orientation: Orientation) -> ShapeFact {
    match orientation {
        Orientation::Row => ShapeFact::from(vec![Some(1), len]),
        Orientation::Column => ShapeFact::from(vec![len, Some(1)]),
    }
}
