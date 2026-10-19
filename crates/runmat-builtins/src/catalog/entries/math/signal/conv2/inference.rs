use crate::catalog::inference::{argument_error, literal_text};
use crate::BuiltinCatalogEntry;
use runmat_types::*;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    let form = form(request, &mut diagnostics);
    let output = match form {
        Some(Form::Direct(mode)) => direct(request, mode, &mut diagnostics),
        Some(Form::Separable(mode)) => separable(request, mode, &mut diagnostics),
        None => ValueFact::unknown(DynamicReason::RuntimeValue),
    };
    let output = if diagnostics.is_empty() {
        output
    } else {
        ValueFact::unknown(DynamicReason::UnsupportedRepresentation)
    };
    let mut contract = CallContract::fixed(vec![output]);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    result.dynamic_outputs = request.outputs.requested.known_count().is_none();
    result.diagnostics.extend(diagnostics);
    result
}

#[derive(Clone, Copy)]
enum Form {
    Direct(Option<ConvolutionMode>),
    Separable(Option<ConvolutionMode>),
}

fn form(request: &CallRequest, diagnostics: &mut Vec<InferenceDiagnostic>) -> Option<Form> {
    let arguments = &request.arguments;
    match arguments.len() {
        2 => Some(Form::Direct(Some(ConvolutionMode::Full))),
        3 => match arguments[2].kind {
            ValueKindFact::Character | ValueKindFact::String => {
                Some(Form::Direct(selector(request, 2, diagnostics)))
            }
            ValueKindFact::Numeric(_) | ValueKindFact::Logical => {
                Some(Form::Separable(Some(ConvolutionMode::Full)))
            }
            ValueKindFact::Unknown => None,
            _ => {
                diagnostics.push(argument_error(
                    "RM-CATALOG-CONV2-THIRD",
                    "conv2 expects a shape selector or a separable signal matrix",
                    2,
                ));
                None
            }
        },
        4 => Some(Form::Separable(selector(request, 3, diagnostics))),
        _ => {
            diagnostics.push(argument_error(
                "RM-CATALOG-CONV2-ARITY",
                "conv2 expects two to four inputs",
                arguments.len().saturating_sub(1),
            ));
            None
        }
    }
}

fn selector(
    request: &CallRequest,
    index: usize,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) -> Option<ConvolutionMode> {
    let value = &request.arguments[index];
    if !matches!(
        value.kind,
        ValueKindFact::Unknown | ValueKindFact::Character | ValueKindFact::String
    ) || matches!(value.kind, ValueKindFact::String) && value.shape.is_definitely_nonscalar()
        || matches!(value.kind, ValueKindFact::Character)
            && character_is_known_multirow(&value.shape)
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-CONV2-SHAPE",
            "conv2 shape must be full, same, or valid",
            index,
        ));
        return None;
    }
    let literal = request
        .literals
        .literal_args
        .get(index)
        .and_then(literal_text)?;
    match ConvolutionMode::decode(&literal) {
        Some(mode) => Some(mode),
        None => {
            diagnostics.push(argument_error(
                "RM-CATALOG-CONV2-SHAPE",
                "conv2 shape must be full, same, or valid",
                index,
            ));
            None
        }
    }
}

fn character_is_known_multirow(shape: &ShapeFact) -> bool {
    matches!(shape, ShapeFact::Shaped { dims } if dims.len() > 2 || matches!(dims.first(), Some(DimensionFact::Known(rows)) if *rows != 1))
}

#[derive(Clone, Copy)]
struct MatrixFact {
    class: Option<NumericClass>,
    domain: Option<NumericDomain>,
    dims: Option<(usize, usize)>,
    host: bool,
}

fn matrix(
    value: &ValueFact,
    index: usize,
    vector: bool,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) -> Option<MatrixFact> {
    if matches!(value.storage, StorageFact::Sparse | StorageFact::Opaque) {
        diagnostics.push(argument_error(
            "RM-CATALOG-CONV2-STORAGE",
            "conv2 requires dense inputs",
            index,
        ));
        return None;
    }
    let (class, domain) = match value.kind {
        ValueKindFact::Numeric(fact) => (Some(fact.class), Some(fact.domain)),
        ValueKindFact::Logical => (Some(NumericClass::Double), Some(NumericDomain::Real)),
        ValueKindFact::Unknown => (None, None),
        _ => {
            diagnostics.push(argument_error(
                "RM-CATALOG-CONV2-INPUT",
                "conv2 expects numeric or logical inputs",
                index,
            ));
            return None;
        }
    };
    let dims = match &value.shape {
        ShapeFact::Scalar => Some((1, 1)),
        ShapeFact::Shaped { dims } => {
            if dims.len() > 2
                && dims[2..]
                    .iter()
                    .any(|dim| matches!(dim, DimensionFact::Known(n) if *n != 1))
            {
                diagnostics.push(argument_error(
                    "RM-CATALOG-CONV2-MATRIX",
                    "conv2 inputs must be 2-D",
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
            known.and_then(|known| {
                let (rows, cols) = match known.as_slice() {
                    [] => (1, 1),
                    [rows] => (*rows, 1),
                    [rows, cols, ..] => (*rows, *cols),
                };
                rows.checked_mul(cols)?;
                if vector && rows > 1 && cols > 1 {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-CONV2-VECTOR",
                        "separable kernel components must be vectors",
                        index,
                    ));
                    return None;
                }
                Some((rows, cols))
            })
        }
        ShapeFact::Unknown | ShapeFact::Ranked { .. } => None,
    };
    Some(MatrixFact {
        class,
        domain,
        dims,
        host: value.residency == ResidencyFact::Host,
    })
}

fn direct(
    request: &CallRequest,
    mode: Option<ConvolutionMode>,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) -> ValueFact {
    let left = request
        .arguments
        .first()
        .and_then(|value| matrix(value, 0, false, diagnostics));
    let right = request
        .arguments
        .get(1)
        .and_then(|value| matrix(value, 1, false, diagnostics));
    output(left.into_iter().chain(right).collect(), mode, false)
}

fn separable(
    request: &CallRequest,
    mode: Option<ConvolutionMode>,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) -> ValueFact {
    let column = request
        .arguments
        .first()
        .and_then(|value| matrix(value, 0, true, diagnostics));
    let row = request
        .arguments
        .get(1)
        .and_then(|value| matrix(value, 1, true, diagnostics));
    let signal = request
        .arguments
        .get(2)
        .and_then(|value| matrix(value, 2, false, diagnostics));
    output(
        column.into_iter().chain(row).chain(signal).collect(),
        mode,
        true,
    )
}

fn output(parts: Vec<MatrixFact>, mode: Option<ConvolutionMode>, separable: bool) -> ValueFact {
    if parts.len() != if separable { 3 } else { 2 } {
        return ValueFact::unknown(DynamicReason::RuntimeValue);
    }
    let class = if parts
        .iter()
        .any(|part| part.class == Some(NumericClass::Single))
    {
        NumericClass::Single
    } else if parts.iter().all(|part| part.class.is_some()) {
        NumericClass::Double
    } else {
        return ValueFact::unknown(DynamicReason::RuntimeValue);
    };
    let domain = if parts
        .iter()
        .any(|part| part.domain == Some(NumericDomain::Complex))
    {
        NumericDomain::Complex
    } else if parts.iter().all(|part| part.domain.is_some()) {
        NumericDomain::Real
    } else {
        return ValueFact::unknown(DynamicReason::RuntimeValue);
    };
    let dimensions = match mode {
        Some(mode) if separable => match (parts[0].dims, parts[1].dims, parts[2].dims) {
            (Some(column), Some(row), Some(signal)) => {
                let kernel = (column.0.checked_mul(column.1), row.0.checked_mul(row.1));
                kernel
                    .0
                    .zip(kernel.1)
                    .and_then(|(rows, cols)| output_dims(signal, (rows, cols), mode))
            }
            _ => None,
        },
        Some(mode) => parts[0]
            .dims
            .zip(parts[1].dims)
            .and_then(|(left, right)| output_dims(left, right, mode)),
        None => None,
    };
    let shape = dimensions.map_or(ShapeFact::Unknown, |(rows, cols)| {
        ShapeFact::from(vec![Some(rows), Some(cols)])
    });
    let host = parts.iter().all(|part| part.host);
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

fn output_dims(
    left: (usize, usize),
    right: (usize, usize),
    mode: ConvolutionMode,
) -> Option<(usize, usize)> {
    let empty = left.0 == 0 || left.1 == 0 || right.0 == 0 || right.1 == 0;
    match mode {
        ConvolutionMode::Full if empty => Some((0, 0)),
        ConvolutionMode::Full => Some((
            left.0.checked_add(right.0)?.checked_sub(1)?,
            left.1.checked_add(right.1)?.checked_sub(1)?,
        )),
        ConvolutionMode::Same => Some(left),
        ConvolutionMode::Valid if empty => Some((0, 0)),
        ConvolutionMode::Valid => Some((
            left.0
                .checked_sub(right.0)
                .and_then(|n| n.checked_add(1))
                .unwrap_or(0),
            left.1
                .checked_sub(right.1)
                .and_then(|n| n.checked_add(1))
                .unwrap_or(0),
        )),
    }
}
