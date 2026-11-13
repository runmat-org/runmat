use super::super::program_facts as facts;
use crate::{catalog::inference::argument_error, BuiltinCatalogEntry};
use runmat_types::*;
pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    let mut normalized = request.clone();
    let mut problem = false;
    if request.arguments.len() == 1 {
        problem = true;
        match &request.arguments[0].kind {
            ValueKindFact::Struct(s) => {
                if request.arguments[0]
                    .shape
                    .element_count()
                    .is_some_and(|n| n != 1)
                {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-NNLS-PROBLEM",
                        "problem must be scalar",
                        0,
                    ));
                }
                for field in ["C", "d", "solver", "options"] {
                    if s.fields_complete && !s.fields.contains_key(field) {
                        diagnostics.push(argument_error(
                            "RM-CATALOG-NNLS-PROBLEM",
                            "problem requires C,d,solver and options",
                            0,
                        ));
                    }
                }
                normalized.arguments = ["C", "d", "options"]
                    .into_iter()
                    .map(|name| {
                        s.fields
                            .get(name)
                            .cloned()
                            .unwrap_or_else(|| ValueFact::unknown(DynamicReason::RuntimeValue))
                    })
                    .collect();
            }
            ValueKindFact::Unknown => normalized.arguments.clear(),
            _ => {
                diagnostics.push(argument_error(
                    "RM-CATALOG-NNLS-PROBLEM",
                    "single argument must be a scalar problem struct",
                    0,
                ));
                normalized.arguments.clear();
            }
        }
    } else if !matches!(request.arguments.len(), 2 | 3) {
        diagnostics.push(argument_error(
            "RM-CATALOG-NNLS-ARITY",
            "expected C,d, optional options, or problem struct",
            0,
        ));
    }
    for (i, value) in normalized.arguments.iter().take(2).enumerate() {
        facts::numeric(
            &normalized,
            i,
            facts::OptimNumericInput::NonemptySparseMatrix,
            &mut diagnostics,
        );
        if i == 1
            && value.shape.known_dims().is_some_and(|dims| {
                dims.len() == 2
                    && dims[0].is_some_and(|n| n != 1)
                    && dims[1].is_some_and(|n| n != 1)
            })
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-NNLS-VECTOR",
                "d must be a vector",
                1,
            ));
        }
    }
    let geometry = normalized
        .arguments
        .first()
        .and_then(|c| c.shape.known_dims())
        .map(|dims| {
            (
                dims.first().copied().flatten(),
                dims.get(1)
                    .copied()
                    .flatten()
                    .or(Some(1).filter(|_| dims.len() < 2)),
            )
        });
    let m = geometry.and_then(|(m, _)| m);
    let n = geometry.and_then(|(_, n)| n);
    if let Some(m) = m {
        if normalized
            .arguments
            .get(1)
            .and_then(|v| v.shape.element_count())
            .is_some_and(|d| d != m)
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-NNLS-RHS",
                "d must have one element per C row",
                1,
            ));
        }
    }
    super::super::nonlinear_facts::options(normalized.arguments.get(2), 2, true, &mut diagnostics);
    if let Some(ValueFact {
        kind: ValueKindFact::Struct(options),
        ..
    }) = normalized.arguments.get(2)
    {
        for (name, value) in &options.fields {
            use super::super::{OptimParameterLookup, OptimParameterName};
            let parameter = OptimParameterName::resolve(name);
            let supported = match parameter {
                OptimParameterLookup::Matched(name) => {
                    super::LsqnonnegOption::from_parameter(name).is_some()
                }
                _ => false,
            };
            let definitely_invalid = matches!(
                parameter,
                OptimParameterLookup::Ambiguous | OptimParameterLookup::Unknown
            ) || (!supported
                && value.shape.element_count().is_some_and(|n| n > 0));
            if definitely_invalid {
                diagnostics.push(argument_error(
                    "RM-CATALOG-NNLS-OPTION",
                    "unknown, ambiguous or unsupported nonempty option",
                    2,
                ));
            }
        }
    }
    if problem {
        for diagnostic in &mut diagnostics {
            diagnostic.argument = Some(0);
        }
    }
    let column = |rows| facts::double(ShapeFact::from(vec![rows, Some(1)]), StorageFact::Dense);
    let scalar = || facts::double(ShapeFact::Scalar, StorageFact::Scalar);
    let information = ValueFact::scalar(ValueKindFact::Struct(StructFact::scalar(
        std::collections::BTreeMap::from([
            ("iterations".into(), scalar()),
            ("algorithm".into(), ValueFact::scalar(ValueKindFact::String)),
            ("message".into(), ValueFact::scalar(ValueKindFact::String)),
        ]),
        true,
    )));
    facts::finish(
        entry,
        request,
        vec![
            column(n),
            scalar(),
            column(m),
            scalar(),
            information,
            column(n),
        ],
        diagnostics,
    )
}
