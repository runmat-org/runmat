use crate::BuiltinCatalogEntry;
use runmat_types::{
    infer_call, CallContract, CallInference, CallRequest, CertaintyFact, ContiguityFact,
    DynamicReason, InferenceDiagnostic, LayoutFact, MutationFact, NumericClass, NumericDomain,
    NumericFact, ResidencyFact, ShapeFact, StorageFact, ValueFact, ValueKindFact, ViewFact,
};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_standard(request, entry, StandardRiccatiApi::Care)
}

#[derive(Clone, Copy)]
pub(crate) enum StandardRiccatiApi {
    Care,
    Dlqr,
}

pub(crate) fn infer_standard(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    api: StandardRiccatiApi,
) -> CallInference {
    let args = &request.arguments;
    let mut diagnostics = Vec::new();
    let arity = match api {
        StandardRiccatiApi::Care => 3..=5,
        StandardRiccatiApi::Dlqr => 4..=5,
    };
    if !arity.contains(&args.len()) {
        diagnostics.push(error(
            "RM-CATALOG-RICCATI-ARITY",
            match api {
                StandardRiccatiApi::Care => "care expects (A,B,Q[,R[,S]])",
                StandardRiccatiApi::Dlqr => "dlqr expects (A,B,Q,R[,N])",
            },
            args.len().saturating_sub(1),
        ));
    }
    let a = args
        .first()
        .and_then(|v| matrix_dims(v, 0, &mut diagnostics));
    let b = args
        .get(1)
        .and_then(|v| matrix_dims(v, 1, &mut diagnostics));
    let n = a.and_then(|(rows, columns)| {
        if mismatch(rows, columns) {
            diagnostics.push(error(
                "RM-CATALOG-RICCATI-DIMENSIONS",
                "A must be square",
                0,
            ));
        }
        rows
    });
    let m = b.and_then(|(rows, columns)| {
        if mismatch(rows, n) {
            diagnostics.push(error(
                "RM-CATALOG-RICCATI-DIMENSIONS",
                "B rows must match A",
                1,
            ));
        }
        columns
    });
    if n == Some(0) || m == Some(0) {
        diagnostics.push(error(
            "RM-CATALOG-RICCATI-DIMENSIONS",
            "at least one state and input are required",
            0,
        ));
    }
    for (index, expected) in [(2, (n, n)), (3, (m, m)), (4, (n, m))] {
        if let Some(actual) = args
            .get(index)
            .and_then(|v| matrix_dims(v, index, &mut diagnostics))
        {
            if mismatch(actual.0, expected.0) || mismatch(actual.1, expected.1) {
                diagnostics.push(error(
                    "RM-CATALOG-RICCATI-DIMENSIONS",
                    "weight matrix has incompatible dimensions",
                    index,
                ));
            }
        }
    }
    let solution = real_double_matrix(ShapeFact::from(vec![n, n]));
    let gain = real_double_matrix(ShapeFact::from(vec![m, n]));
    let poles = if n == Some(1) {
        real_double_matrix(ShapeFact::from(vec![n, Some(1)]))
    } else {
        let mut value = ValueFact::unknown(DynamicReason::RuntimeValue);
        value.shape = ShapeFact::from(vec![n, Some(1)]);
        value.storage = StorageFact::Dense;
        value.layout = LayoutFact::ColumnMajor;
        value.contiguity = ContiguityFact::Contiguous;
        value.view = ViewFact::Materialized;
        value.residency = ResidencyFact::Host;
        value.mutation = MutationFact::ValueSemantics;
        value
    };
    let outputs = match api {
        StandardRiccatiApi::Care => vec![solution, poles, gain],
        StandardRiccatiApi::Dlqr => vec![gain, solution, poles],
    };
    let mut contract = CallContract::fixed(outputs);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut inferred = infer_call(&contract, request);
    inferred.diagnostics.extend(diagnostics);
    inferred
}

fn matrix_dims(
    value: &ValueFact,
    index: usize,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) -> Option<(Option<usize>, Option<usize>)> {
    if matches!(value.residency, ResidencyFact::Remote { .. })
        || !matches!(
            value.storage,
            StorageFact::Scalar | StorageFact::Dense | StorageFact::Unknown
        )
        || (!matches!(value.kind, ValueKindFact::Unknown)
            && !value
                .numeric()
                .is_some_and(|numeric| numeric.domain == NumericDomain::Real))
    {
        diagnostics.push(error(
            "RM-CATALOG-RICCATI-INPUT",
            "matrix must have real numeric storage",
            index,
        ));
        return None;
    }
    if value.shape.rank().is_some_and(|rank| rank > 2) {
        diagnostics.push(error(
            "RM-CATALOG-RICCATI-DIMENSIONS",
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

fn mismatch(actual: Option<usize>, expected: Option<usize>) -> bool {
    actual.zip(expected).is_some_and(|(a, e)| a != e)
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
