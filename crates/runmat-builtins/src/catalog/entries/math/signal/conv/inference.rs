use crate::catalog::inference::{argument_error, literal_text};
use crate::BuiltinCatalogEntry;
use runmat_types::*;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if !(2..=3).contains(&request.arguments.len()) {
        diagnostics.push(argument_error(
            "RM-CATALOG-CONV-ARITY",
            "conv expects two vectors and an optional shape selector",
            request.arguments.len().saturating_sub(1),
        ));
    }
    let mode = selector(request, &mut diagnostics);
    let left = request
        .arguments
        .first()
        .and_then(|value| vector(value, 0, &mut diagnostics));
    let right = request
        .arguments
        .get(1)
        .and_then(|value| vector(value, 1, &mut diagnostics));
    let output = match (left, right) {
        (Some(left), Some(right)) if diagnostics.is_empty() => output(left, right, mode),
        _ => ValueFact::unknown(DynamicReason::UnsupportedRepresentation),
    };
    let mut contract = CallContract::fixed(vec![output]);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    result.dynamic_outputs = request.outputs.requested.known_count().is_none();
    result.diagnostics.extend(diagnostics);
    result
}

fn selector(
    request: &CallRequest,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) -> Option<ConvolutionMode> {
    let Some(value) = request.arguments.get(2) else {
        return Some(ConvolutionMode::Full);
    };
    if !matches!(
        value.kind,
        ValueKindFact::Unknown | ValueKindFact::Character | ValueKindFact::String
    ) || matches!(value.kind, ValueKindFact::String) && value.shape.is_definitely_nonscalar()
        || matches!(value.kind, ValueKindFact::Character)
            && character_is_known_multirow(&value.shape)
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-CONV-SHAPE",
            "conv shape must be full, same, or valid",
            2,
        ));
        return None;
    }
    let literal = request
        .literals
        .literal_args
        .get(2)
        .and_then(literal_text)?;
    match ConvolutionMode::decode(&literal) {
        Some(mode) => Some(mode),
        None => {
            diagnostics.push(argument_error(
                "RM-CATALOG-CONV-SHAPE",
                "conv shape must be full, same, or valid",
                2,
            ));
            None
        }
    }
}

fn character_is_known_multirow(shape: &ShapeFact) -> bool {
    matches!(shape, ShapeFact::Shaped { dims } if dims.len() > 2 || matches!(dims.first(), Some(DimensionFact::Known(rows)) if *rows != 1))
}

#[derive(Clone, Copy)]
struct VectorFact {
    class: Option<NumericClass>,
    domain: Option<NumericDomain>,
    geometry: Option<VectorGeometry>,
    host: bool,
}

#[derive(Clone, Copy)]
struct VectorGeometry {
    len: usize,
    orientation: Orientation,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Orientation {
    Row,
    Column,
    Scalar,
    Empty,
}

fn vector(
    value: &ValueFact,
    index: usize,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) -> Option<VectorFact> {
    if matches!(value.storage, StorageFact::Sparse | StorageFact::Opaque) {
        diagnostics.push(argument_error(
            "RM-CATALOG-CONV-STORAGE",
            "conv requires dense vectors",
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
                "RM-CATALOG-CONV-INPUT",
                "conv expects numeric or logical vectors",
                index,
            ));
            return None;
        }
    };
    let geometry = match &value.shape {
        ShapeFact::Scalar => Some(VectorGeometry {
            len: 1,
            orientation: Orientation::Scalar,
        }),
        ShapeFact::Shaped { dims } => {
            if dims.len() > 2
                && dims[2..]
                    .iter()
                    .any(|dim| matches!(dim, DimensionFact::Known(n) if *n != 1))
            {
                diagnostics.push(argument_error(
                    "RM-CATALOG-CONV-VECTOR",
                    "conv inputs must be vectors",
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
                let len = known
                    .iter()
                    .try_fold(1_usize, |product, dim| product.checked_mul(*dim))?;
                let (rows, cols) = match known.as_slice() {
                    [] => (1, 1),
                    [cols] => (1, *cols),
                    [rows, cols, ..] => (*rows, *cols),
                };
                if rows > 1 && cols > 1 && len != 0 {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-CONV-VECTOR",
                        "conv inputs must be vectors",
                        index,
                    ));
                    return None;
                }
                let orientation = if rows == 1 && cols != 1 {
                    Orientation::Row
                } else if cols == 1 && rows != 1 {
                    Orientation::Column
                } else if rows == 1 && cols == 1 {
                    Orientation::Scalar
                } else {
                    Orientation::Empty
                };
                Some(VectorGeometry { len, orientation })
            })
        }
        ShapeFact::Unknown | ShapeFact::Ranked { .. } => None,
    };
    Some(VectorFact {
        class,
        domain,
        geometry,
        host: value.residency == ResidencyFact::Host,
    })
}

fn output(left: VectorFact, right: VectorFact, mode: Option<ConvolutionMode>) -> ValueFact {
    let class = match (left.class, right.class) {
        (Some(NumericClass::Single), _) | (_, Some(NumericClass::Single)) => NumericClass::Single,
        (Some(_), Some(_)) => NumericClass::Double,
        _ => return ValueFact::unknown(DynamicReason::RuntimeValue),
    };
    let domain = match (left.domain, right.domain) {
        (Some(NumericDomain::Complex), _) | (_, Some(NumericDomain::Complex)) => {
            NumericDomain::Complex
        }
        (Some(_), Some(_)) => NumericDomain::Real,
        _ => return ValueFact::unknown(DynamicReason::RuntimeValue),
    };
    let shape = match (left.geometry, right.geometry, mode) {
        (Some(lhs), Some(rhs), Some(mode)) => {
            output_shape(lhs, rhs, mode).unwrap_or(ShapeFact::Unknown)
        }
        _ => ShapeFact::Unknown,
    };
    let storage = match shape.element_count() {
        _ if !left.host || !right.host => StorageFact::Unknown,
        Some(1) if class == NumericClass::Double => StorageFact::Scalar,
        Some(_) => StorageFact::Dense,
        None => StorageFact::Unknown,
    };
    let mut result = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact { class, domain }),
        shape,
        storage,
    );
    if !left.host || !right.host {
        result.residency = ResidencyFact::Unknown;
        result.layout = LayoutFact::Unknown;
        result.contiguity = ContiguityFact::Unknown;
        result.view = ViewFact::Unknown;
        result.alias = AliasFact::Unknown;
    }
    result
}

fn output_shape(
    lhs: VectorGeometry,
    rhs: VectorGeometry,
    mode: ConvolutionMode,
) -> Option<ShapeFact> {
    let len = match mode {
        ConvolutionMode::Full if lhs.len == 0 || rhs.len == 0 => 0,
        ConvolutionMode::Full => lhs.len.checked_add(rhs.len)?.checked_sub(1)?,
        ConvolutionMode::Same => {
            if lhs.len == 0 || rhs.len == 0 {
                0
            } else {
                lhs.len
            }
        }
        ConvolutionMode::Valid if rhs.len == 0 => lhs.len,
        ConvolutionMode::Valid if lhs.len == 0 => 0,
        ConvolutionMode::Valid => lhs
            .len
            .checked_sub(rhs.len)
            .and_then(|n| n.checked_add(1))
            .unwrap_or(0),
    };
    let orientation = if mode == ConvolutionMode::Full {
        if lhs.orientation == Orientation::Column && rhs.orientation == Orientation::Column {
            Orientation::Column
        } else {
            Orientation::Row
        }
    } else {
        match lhs.orientation {
            Orientation::Row | Orientation::Column => lhs.orientation,
            Orientation::Scalar | Orientation::Empty => {
                if rhs.orientation == Orientation::Column {
                    Orientation::Column
                } else {
                    Orientation::Row
                }
            }
        }
    };
    Some(if orientation == Orientation::Column {
        ShapeFact::from(vec![Some(len), Some(1)])
    } else {
        ShapeFact::from(vec![Some(1), Some(len)])
    })
}
