use crate::BuiltinCatalogEntry;
use runmat_types::{
    infer_call, standard, CallContract, CallInference, CallRequest, CertaintyFact, ContiguityFact,
    DynamicReason, InferenceDiagnostic, LayoutFact, MutationFact, NumericClass, NumericDomain,
    NumericFact, ResidencyFact, ShapeFact, StorageFact, ValueFact, ValueKindFact, ViewFact,
};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let args = &request.arguments;
    let mut diagnostics = Vec::new();
    let (n, m) = {
        let object_form = matches!(
            args.first().map(|arg| &arg.kind),
            Some(ValueKindFact::Object(_))
        );
        let length_valid = if object_form {
            (3..=4).contains(&args.len())
        } else if matches!(
            args.first().map(|arg| &arg.kind),
            Some(ValueKindFact::Unknown)
        ) {
            (3..=5).contains(&args.len())
        } else {
            (4..=5).contains(&args.len())
        };
        if !length_valid {
            diagnostics.push(error(
                "RM-CATALOG-LQR-ARITY",
                "lqr expects (A,B,Q,R[,N]) or (sys,Q,R[,N])",
                args.len().saturating_sub(1),
            ));
        }
        if object_form {
            infer_object_inputs(args, &mut diagnostics)
        } else {
            infer_matrix_inputs(args, &mut diagnostics)
        }
    };
    let k = real_double_matrix(ShapeFact::from(vec![m, n]));
    let s = real_double_matrix(ShapeFact::from(vec![n, n]));
    let pole_shape = ShapeFact::from(vec![n, Some(1)]);
    let poles = if n == Some(1) {
        real_double_matrix(pole_shape)
    } else {
        let mut fact = ValueFact::unknown(DynamicReason::RuntimeValue);
        fact.shape = pole_shape;
        fact.storage = StorageFact::Dense;
        fact.layout = LayoutFact::ColumnMajor;
        fact.contiguity = ContiguityFact::Contiguous;
        fact.view = ViewFact::Materialized;
        fact.residency = ResidencyFact::Host;
        fact.mutation = MutationFact::ValueSemantics;
        fact
    };
    let mut contract = CallContract::fixed(vec![k, s, poles]);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    result.diagnostics.extend(diagnostics);
    result
}

fn infer_object_inputs(
    args: &[ValueFact],
    diagnostics: &mut Vec<InferenceDiagnostic>,
) -> (Option<usize>, Option<usize>) {
    let Some(first) = args.first() else {
        return (None, None);
    };
    let ValueKindFact::Object(object) = &first.kind else {
        return (None, None);
    };
    if match object.runtime_class.as_ref() {
        Some(class) => class != &standard::STATE_SPACE.owned(),
        None => object.class.is_some(),
    } || first.shape.element_count().is_some_and(|count| count != 1)
        || matches!(first.residency, ResidencyFact::Remote { .. })
    {
        diagnostics.push(error(
            "RM-CATALOG-LQR-MODEL",
            "sys must be a scalar state-space ss object",
            0,
        ));
    }
    let a = object.properties.get("A");
    let b = object.properties.get("B");
    if object.properties_complete && (a.is_none() || b.is_none()) {
        diagnostics.push(error(
            "RM-CATALOG-LQR-MODEL",
            "state-space model must contain A and B",
            0,
        ));
    }
    let n = a
        .and_then(|value| {
            matrix_dims(value, 0, diagnostics).map(|(rows, columns)| {
                if rows.is_some() && columns.is_some() && rows != columns {
                    diagnostics.push(error("RM-CATALOG-LQR-DIMENSIONS", "A must be square", 0));
                }
                rows
            })
        })
        .flatten();
    let b_dims = b.and_then(|value| matrix_dims(value, 0, diagnostics));
    if let (Some(n), Some((Some(rows), _))) = (n, b_dims) {
        if rows != n {
            diagnostics.push(error(
                "RM-CATALOG-LQR-DIMENSIONS",
                "B row count must match A",
                0,
            ));
        }
    }
    let m = b_dims.and_then(|(_, columns)| columns);
    check_nonempty_state_and_input(n, m, 0, diagnostics);
    check_weights(args, 1, n, m, diagnostics);
    (n, m)
}

fn infer_matrix_inputs(
    args: &[ValueFact],
    diagnostics: &mut Vec<InferenceDiagnostic>,
) -> (Option<usize>, Option<usize>) {
    let a = args
        .first()
        .and_then(|value| matrix_dims(value, 0, diagnostics));
    let b = args
        .get(1)
        .and_then(|value| matrix_dims(value, 1, diagnostics));
    let n = a.and_then(|(rows, columns)| {
        if rows.is_some() && columns.is_some() && rows != columns {
            diagnostics.push(error("RM-CATALOG-LQR-DIMENSIONS", "A must be square", 0));
        }
        rows
    });
    if let (Some(n), Some((Some(rows), _))) = (n, b) {
        if rows != n {
            diagnostics.push(error(
                "RM-CATALOG-LQR-DIMENSIONS",
                "B row count must match A",
                1,
            ));
        }
    }
    let m = b.and_then(|(_, columns)| columns);
    check_nonempty_state_and_input(n, m, 0, diagnostics);
    check_weights(args, 2, n, m, diagnostics);
    (n, m)
}

fn check_nonempty_state_and_input(
    n: Option<usize>,
    m: Option<usize>,
    index: usize,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) {
    if n == Some(0) || m == Some(0) {
        diagnostics.push(error(
            "RM-CATALOG-LQR-DIMENSIONS",
            "LQR requires at least one state and one input",
            index,
        ));
    }
}

fn check_weights(
    args: &[ValueFact],
    start: usize,
    n: Option<usize>,
    m: Option<usize>,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) {
    for (offset, expected) in [(0, (n, n)), (1, (m, m)), (2, (n, m))] {
        let index = start + offset;
        let Some(value) = args.get(index) else {
            continue;
        };
        if let Some(actual) = matrix_dims(value, index, diagnostics) {
            if mismatched(actual.0, expected.0) || mismatched(actual.1, expected.1) {
                diagnostics.push(error(
                    "RM-CATALOG-LQR-DIMENSIONS",
                    "weight matrix has incompatible dimensions",
                    index,
                ));
            }
        }
    }
}

fn matrix_dims(
    value: &ValueFact,
    index: usize,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) -> Option<(Option<usize>, Option<usize>)> {
    if !matches!(value.kind, ValueKindFact::Unknown)
        && !value
            .numeric()
            .is_some_and(|numeric| numeric.domain == NumericDomain::Real)
    {
        diagnostics.push(error(
            "RM-CATALOG-LQR-INPUT",
            "matrix must have real numeric values",
            index,
        ));
        return None;
    }
    if value.shape.rank().is_some_and(|rank| rank > 2) {
        diagnostics.push(error(
            "RM-CATALOG-LQR-DIMENSIONS",
            "matrix must be two-dimensional",
            index,
        ));
        return None;
    }
    value
        .shape
        .known_dims()
        .and_then(|dims| match dims.as_slice() {
            [rows, columns] => Some((*rows, *columns)),
            _ => None,
        })
}

fn mismatched(actual: Option<usize>, expected: Option<usize>) -> bool {
    actual
        .zip(expected)
        .is_some_and(|(actual, expected)| actual != expected)
}
fn error(code: &str, message: &str, index: usize) -> InferenceDiagnostic {
    let mut diagnostic = InferenceDiagnostic::error(code, message);
    diagnostic.argument = Some(index);
    diagnostic
}
fn real_double_matrix(shape: ShapeFact) -> ValueFact {
    let mut fact = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        shape,
        StorageFact::Dense,
    );
    fact.certainty = CertaintyFact::Proven;
    fact
}
