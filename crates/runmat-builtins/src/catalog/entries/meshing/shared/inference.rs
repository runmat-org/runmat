//! Shared static admission for the meshing builtins.

use super::MESHING_MESH_OBJECT_CLASS;
use crate::catalog::{argument_error, finish_fixed};
use crate::fea_contract::constructor_inference::host_text;
use crate::*;
use runmat_types::{
    CallInference, CallRequest, InferenceDiagnostic, ObjectFact, ResidencyFact, ShapeFact,
    StaticClassIdentity, StorageFact, ValueFact, ValueKindFact,
};
use std::collections::BTreeMap;

pub(in crate::catalog::entries::meshing) fn host_object(class: StaticClassIdentity) -> ValueFact {
    let mut fact = ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(class.owned()),
            properties: BTreeMap::new(),
            properties_complete: false,
            handle_semantics: Some(false),
        }),
        ShapeFact::Scalar,
        StorageFact::Opaque,
    );
    fact.residency = ResidencyFact::Host;
    fact
}

pub(in crate::catalog::entries::meshing) fn possible_object(
    fact: &ValueFact,
    class: StaticClassIdentity,
) -> bool {
    !matches!(fact.residency, ResidencyFact::Device { .. })
        && !matches!(fact.shape.element_count(), Some(count) if count != 1)
        && match &fact.kind {
            ValueKindFact::Unknown => true,
            ValueKindFact::Object(object) => object
                .runtime_class
                .as_ref()
                .is_none_or(|runtime_class| runtime_class.is(class)),
            _ => false,
        }
}

pub(in crate::catalog::entries::meshing) fn arity(
    diagnostics: &mut Vec<InferenceDiagnostic>,
    code: &str,
    name: &str,
    count: usize,
    minimum: usize,
    maximum: Option<usize>,
) {
    if count < minimum || maximum.is_some_and(|maximum| count > maximum) {
        let expected = match maximum {
            Some(maximum) if maximum == minimum => format!("exactly {minimum}"),
            Some(maximum) => format!("{minimum} to {maximum}"),
            None => format!("at least {minimum}"),
        };
        diagnostics.push(argument_error(
            format!("RM-CATALOG-MESHING-{code}-ARITY"),
            format!("{name} requires {expected} arguments"),
            count,
        ));
    }
}

pub(in crate::catalog::entries::meshing) fn require_mesh(
    diagnostics: &mut Vec<InferenceDiagnostic>,
    code: &str,
    name: &str,
    request: &CallRequest,
    index: usize,
) {
    if request
        .arguments
        .get(index)
        .is_some_and(|fact| !possible_object(fact, MESHING_MESH_OBJECT_CLASS))
    {
        diagnostics.push(argument_error(
            format!("RM-CATALOG-MESHING-{code}-MESH"),
            format!("{name} requires a host meshing.Mesh"),
            index,
        ));
    }
}

pub(in crate::catalog::entries::meshing) fn require_text(
    diagnostics: &mut Vec<InferenceDiagnostic>,
    code: &str,
    name: &str,
    request: &CallRequest,
    index: usize,
) {
    if request
        .arguments
        .get(index)
        .is_some_and(|fact| !host_text(fact))
    {
        diagnostics.push(argument_error(
            format!("RM-CATALOG-MESHING-{code}-PATH"),
            format!("{name} paths must be scalar text"),
            index,
        ));
    }
}

/// Name, Value pairs starting at `first`. Every literal option name is checked
/// against `known` after removing non-alphanumeric characters and folding case,
/// matching the runtime option decoder.
pub(in crate::catalog::entries::meshing) fn options(
    diagnostics: &mut Vec<InferenceDiagnostic>,
    code: &str,
    name: &str,
    request: &CallRequest,
    first: usize,
    known: &[&str],
) {
    let count = request.arguments.len();
    if count > first && !(count - first).is_multiple_of(2) {
        diagnostics.push(argument_error(
            format!("RM-CATALOG-MESHING-{code}-OPTIONS"),
            format!("{name} options must be Name, Value pairs"),
            count,
        ));
    }
    for index in (first..count).step_by(2) {
        if !host_text(&request.arguments[index]) {
            diagnostics.push(argument_error(
                format!("RM-CATALOG-MESHING-{code}-OPTION"),
                format!("{name} option names must be scalar text"),
                index,
            ));
            continue;
        }
        if let Some(option) = request.literals.literal_string_at(index) {
            let normalized = option
                .chars()
                .filter(char::is_ascii_alphanumeric)
                .collect::<String>();
            if !known.contains(&normalized.as_str()) {
                diagnostics.push(argument_error(
                    format!("RM-CATALOG-MESHING-{code}-OPTION"),
                    format!("{name} does not accept option `{option}`"),
                    index,
                ));
            }
        }
    }
}

pub(in crate::catalog::entries::meshing) fn finish(
    entry: &BuiltinCatalogEntry,
    request: &CallRequest,
    output: ValueFact,
    diagnostics: Vec<InferenceDiagnostic>,
) -> CallInference {
    finish_fixed(entry, request, output, diagnostics)
}

/// Shared inference for the single-argument inspection queries.
pub(in crate::catalog::entries::meshing) fn infer_inspection(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    code: &str,
    output: ValueFact,
) -> CallInference {
    let name = entry.identity.name;
    let mut diagnostics = Vec::new();
    arity(
        &mut diagnostics,
        code,
        name,
        request.arguments.len(),
        1,
        Some(1),
    );
    require_mesh(&mut diagnostics, code, name, request, 0);
    finish(entry, request, output, diagnostics)
}

pub(in crate::catalog::entries::meshing) fn uint64_scalar() -> ValueFact {
    ValueFact::scalar(ValueKindFact::Numeric(runmat_types::NumericFact {
        class: runmat_types::NumericClass::UInt64,
        domain: runmat_types::NumericDomain::Real,
    }))
}

/// `meshing.Mesh` with its public summary properties.
pub(in crate::catalog::entries::meshing) fn mesh_result() -> ValueFact {
    let mut fact = host_object(MESHING_MESH_OBJECT_CLASS);
    if let ValueKindFact::Object(object) = &mut fact.kind {
        object.properties.insert(
            "canonical_digest".into(),
            ValueFact::scalar(ValueKindFact::String),
        );
        for count in ["node_count", "element_count", "boundary_face_count"] {
            object.properties.insert(count.into(), uint64_scalar());
        }
    }
    fact
}

/// Scalar struct result with a complete, known field set.
pub(in crate::catalog::entries::meshing) fn struct_result(
    fields: Vec<(&str, ValueFact)>,
) -> ValueFact {
    let mut fact = ValueFact::proven(
        ValueKindFact::Struct(runmat_types::StructFact {
            fields: fields
                .into_iter()
                .map(|(name, value)| (name.to_string(), value))
                .collect(),
            fields_complete: true,
            elements: Vec::new(),
            elements_complete: false,
        }),
        ShapeFact::Scalar,
        StorageFact::Dense,
    );
    fact.residency = ResidencyFact::Host;
    fact
}

/// Column of `uint64` identifiers or connectivity with a runtime row count.
pub(in crate::catalog::entries::meshing) fn uint64_matrix(columns: Option<usize>) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(runmat_types::NumericFact {
            class: runmat_types::NumericClass::UInt64,
            domain: runmat_types::NumericDomain::Real,
        }),
        ShapeFact::from(vec![None, columns]),
        StorageFact::Dense,
    )
}

pub(in crate::catalog::entries::meshing) fn double_matrix(columns: Option<usize>) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(runmat_types::NumericFact {
            class: runmat_types::NumericClass::Double,
            domain: runmat_types::NumericDomain::Real,
        }),
        ShapeFact::from(vec![None, columns]),
        StorageFact::Dense,
    )
}

pub(in crate::catalog::entries::meshing) fn double_scalar() -> ValueFact {
    ValueFact::scalar(ValueKindFact::Numeric(runmat_types::NumericFact {
        class: runmat_types::NumericClass::Double,
        domain: runmat_types::NumericDomain::Real,
    }))
}

/// Cell array of runtime length whose elements share `element`.
pub(in crate::catalog::entries::meshing) fn cell_of(
    element: ValueFact,
    dims: Vec<Option<usize>>,
) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Cell(runmat_types::CellFact {
            element: Box::new(element),
            elements: Vec::new(),
            elements_complete: false,
        }),
        ShapeFact::from(dims),
        StorageFact::Dense,
    )
}

pub(in crate::catalog::entries::meshing) fn text_scalar() -> ValueFact {
    ValueFact::scalar(ValueKindFact::String)
}
