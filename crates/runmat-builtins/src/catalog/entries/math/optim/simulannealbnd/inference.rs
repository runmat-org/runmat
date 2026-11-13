use super::super::{nonlinear_facts as facts, scalar_facts};
use crate::{catalog::inference::argument_error, BuiltinCatalogEntry};
use runmat_types::*;
pub(super) fn solver(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if !matches!(request.arguments.len(), 1 | 2 | 4 | 5) {
        diagnostics.push(argument_error(
            "RM-CATALOG-SA-ARITY",
            "expected fun,x0[,lb,ub[,options]] or one complete problem",
            0,
        ));
    }
    let problem = if request.arguments.len() == 1 {
        match &request.arguments[0].kind {
            ValueKindFact::Struct(s) => Some(s),
            ValueKindFact::Unknown => None,
            _ => {
                diagnostics.push(argument_error(
                    "RM-CATALOG-SA-PROBLEM",
                    "one argument requires a scalar problem structure",
                    0,
                ));
                None
            }
        }
    } else {
        None
    };
    if let Some(p) = problem {
        for name in ["objective", "x0", "lb", "ub", "solver", "options"] {
            if p.fields_complete && !p.fields.contains_key(name) {
                diagnostics.push(argument_error(
                    "RM-CATALOG-SA-PROBLEM-FIELD",
                    "problem is missing a required field",
                    0,
                ));
            }
        }
    }
    let x = problem
        .and_then(|p| p.fields.get("x0"))
        .or_else(|| request.arguments.get(1));
    let options = problem
        .and_then(|p| p.fields.get("options"))
        .or_else(|| request.arguments.get(4));
    facts::options(options, 4, true, &mut diagnostics);
    if let Some(ValueFact {
        kind: ValueKindFact::Struct(fields),
        ..
    }) = options
    {
        for name in fields.fields.keys() {
            if name.eq_ignore_ascii_case("Solver") {
                continue;
            }
            if crate::OptimOptionField::parse_for_solver(name, crate::OptimSolver::Simulannealbnd)
                .is_none()
            {
                diagnostics.push(argument_error(
                    "RM-CATALOG-SA-OPTION",
                    "unsupported annealing option",
                    4,
                ));
            }
        }
    }
    // DataType can be selected at runtime, including through mutable options.
    // The returned representation is therefore not falsely forced to double.
    let default_double=options.is_none_or(|v|v.shape.element_count()==Some(0)||
        matches!(&v.kind,ValueKindFact::Struct(s)if s.fields_complete&&!s.fields.keys().any(|name|crate::OptimOptionField::parse_for_solver(name,crate::OptimSolver::Simulannealbnd)==Some(crate::OptimOptionField::DataType))));
    let unbounded = if let Some(p) = problem {
        ["lb", "ub"].iter().all(|name| {
            p.fields
                .get(*name)
                .is_some_and(|v| v.shape.element_count() == Some(0))
        })
    } else {
        request.arguments.len() == 2
            || [2, 3].iter().all(|&i| {
                request
                    .arguments
                    .get(i)
                    .is_some_and(|v| v.shape.element_count() == Some(0))
            })
    };
    let point = if default_double {
        if let Some(x) = x {
            let valid = matches!(x.kind, ValueKindFact::Unknown)
                || matches!(x.kind,ValueKindFact::Numeric(n)if n.class==NumericClass::Double&&n.domain==NumericDomain::Real);
            let vector = x.shape.known_dims().is_none_or(|d| {
                d.len() <= 2 && d.iter().filter(|v| v.is_some_and(|d| d != 1)).count() <= 1
            });
            if !valid
                || !vector
                || x.storage == StorageFact::Sparse
                || x.shape.element_count() == Some(0)
            {
                diagnostics.push(argument_error(
                    "RM-CATALOG-SA-X0",
                    "double x0 must be a nonempty full real double vector",
                    1,
                ));
            }
        }
        facts::double(
            if unbounded {
                x.map(|v| v.shape.clone()).unwrap_or(ShapeFact::Unknown)
            } else {
                ShapeFact::Unknown
            },
            StorageFact::Dense,
        )
    } else {
        ValueFact::unknown(DynamicReason::RuntimeValue)
    };
    let callback = if request.arguments.len() > 1 {
        scalar_facts::callback(request, &mut diagnostics);
        scalar_facts::callback_invocation_with_arguments(request, vec![point.clone()])
    } else {
        None
    };
    let mut output = std::collections::BTreeMap::new();
    for name in ["iterations", "funccount", "totaltime"] {
        output.insert(name.into(), facts::scalar());
    }
    for name in ["message", "problemtype"] {
        output.insert(name.into(), ValueFact::scalar(ValueKindFact::String));
    }
    output.insert("temperature".into(), facts::matrix(None, Some(1)));
    output.insert(
        "rngstate".into(),
        ValueFact::unknown(DynamicReason::RuntimeValue),
    );
    facts::finish(
        request,
        entry,
        vec![
            point,
            facts::double(ShapeFact::Unknown, StorageFact::Unknown),
            facts::scalar(),
            ValueFact::scalar(ValueKindFact::Struct(StructFact::scalar(output, true))),
        ],
        diagnostics,
        callback,
    )
}
pub(super) fn helper(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    let signature = &entry.descriptor.signatures[0];
    if request.arguments.len() != signature.inputs.len() {
        diagnostics.push(argument_error(
            "RM-CATALOG-SA-HELPER-ARITY",
            "expected the helper's documented input count",
            0,
        ));
    }
    let result = match entry.identity.name {
        "acceptancesa" | "saplotbestf" | "saplotbestx" | "saplotf" | "saplotx"
        | "saplotstopping" | "saplottemperature" => ValueFact::scalar(ValueKindFact::Logical),
        "temperatureexp" | "temperaturefast" | "temperatureboltz" => facts::matrix(None, Some(1)),
        _ => request
            .arguments
            .first()
            .and_then(|v| match &v.kind {
                ValueKindFact::Struct(s) => s.fields.get("x").cloned(),
                _ => None,
            })
            .unwrap_or_else(|| facts::double(ShapeFact::Unknown, StorageFact::Dense)),
    };
    facts::finish(request, entry, vec![result], diagnostics, None)
}
