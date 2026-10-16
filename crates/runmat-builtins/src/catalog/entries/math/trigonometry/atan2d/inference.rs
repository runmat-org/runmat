use crate::catalog::inference::{argument_error, finish_fixed_outputs};
use crate::BuiltinCatalogEntry;
use runmat_types::{
    broadcast_shape, standard, AliasFact, CallInference, CallRequest, DynamicReason, MutationFact,
    NumericClass, NumericDomain, NumericFact, ResidencyFact, StaticClassIdentity, StorageFact,
    ValueFact, ValueKindFact, ViewFact,
};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 2 {
        diagnostics.push(argument_error(
            "RM-CATALOG-ATAN2D-ARITY",
            "atan2d requires exactly two inputs",
            request.arguments.len().min(1),
        ));
    }
    let mut output = ValueFact::unknown(DynamicReason::RuntimeValue);
    if let (Some(y), Some(x)) = (request.arguments.first(), request.arguments.get(1)) {
        if let Some(tabular) = infer_tabular(y, x, &mut diagnostics) {
            return finish_fixed_outputs(entry, request, vec![tabular], diagnostics);
        }
        if matches!(y.storage, StorageFact::Sparse) || matches!(x.storage, StorageFact::Sparse) {
            diagnostics.push(argument_error(
                "RM-CATALOG-ATAN2D-SPARSE",
                "atan2d does not accept sparse input",
                usize::from(!matches!(y.storage, StorageFact::Sparse)),
            ));
        }
        let y_kind = numeric(&y.kind, 0, &mut diagnostics);
        let x_kind = numeric(&x.kind, 1, &mut diagnostics);
        if let (Some(y_kind), Some(x_kind)) = (y_kind, x_kind) {
            if y_kind.domain == NumericDomain::Complex || x_kind.domain == NumericDomain::Complex {
                diagnostics.push(argument_error(
                    "RM-CATALOG-ATAN2D-COMPLEX",
                    "atan2d requires real operands",
                    usize::from(x_kind.domain == NumericDomain::Complex),
                ));
            } else {
                output.kind = ValueKindFact::Numeric(NumericFact {
                    class: if y_kind.class == NumericClass::Single
                        || x_kind.class == NumericClass::Single
                    {
                        NumericClass::Single
                    } else {
                        NumericClass::Double
                    },
                    domain: NumericDomain::Real,
                });
            }
        }
        output.shape = match broadcast_shape(&y.shape, &x.shape) {
            Ok(shape) => shape,
            Err(diagnostic) => {
                diagnostics.push(diagnostic);
                runmat_types::ShapeFact::Unknown
            }
        };
        output.storage = StorageFact::Dense;
        output.residency =
            if y.residency == ResidencyFact::Host && x.residency == ResidencyFact::Host {
                ResidencyFact::Host
            } else {
                ResidencyFact::Unknown
            };
    }
    finish_fixed_outputs(entry, request, vec![output], diagnostics)
}

fn infer_tabular(
    y: &ValueFact,
    x: &ValueFact,
    diagnostics: &mut Vec<runmat_types::InferenceDiagnostic>,
) -> Option<ValueFact> {
    let y_class = tabular_class(y);
    let x_class = tabular_class(x);
    if y_class.is_none() && x_class.is_none() {
        return None;
    }
    let unsupported_object = (matches!(y.kind, ValueKindFact::Object(_)) && y_class.is_none())
        || (matches!(x.kind, ValueKindFact::Object(_)) && x_class.is_none());
    if unsupported_object || (y_class.is_some() && x_class.is_some() && y_class != x_class) {
        diagnostics.push(argument_error(
            "RM-CATALOG-ATAN2D-TABULAR-PAIR",
            "atan2d requires table and timetable operands to have the same container identity",
            usize::from(y_class.is_some()),
        ));
        return Some(ValueFact::unknown(DynamicReason::UnsupportedRepresentation));
    }
    let mut output = if y_class.is_some() {
        y.clone()
    } else {
        x.clone()
    };
    output.residency = ResidencyFact::Host;
    output.alias = AliasFact::Unique;
    output.view = ViewFact::Materialized;
    output.mutation = MutationFact::ValueSemantics;
    if let ValueKindFact::Object(object) = &mut output.kind {
        object.properties.clear();
        object.properties_complete = false;
    }
    Some(output)
}

fn tabular_class(fact: &ValueFact) -> Option<StaticClassIdentity> {
    let ValueKindFact::Object(object) = &fact.kind else {
        return None;
    };
    let identity = object.runtime_class.as_ref()?;
    [standard::TABLE, standard::TIMETABLE]
        .into_iter()
        .find(|candidate| identity.is(*candidate))
}

fn numeric(
    kind: &ValueKindFact,
    argument: usize,
    diagnostics: &mut Vec<runmat_types::InferenceDiagnostic>,
) -> Option<NumericFact> {
    match kind {
        ValueKindFact::Numeric(numeric) => Some(*numeric),
        ValueKindFact::Logical | ValueKindFact::Character => Some(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        ValueKindFact::Unknown => None,
        _ => {
            diagnostics.push(argument_error(
                "RM-CATALOG-ATAN2D-INPUT",
                "atan2d requires real numeric input",
                argument,
            ));
            None
        }
    }
}
