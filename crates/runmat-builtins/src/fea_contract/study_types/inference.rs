//! Static inference shared by every typed study-contract constructor.
//!
//! Each catalog leaf declares its positional argument kinds and result class
//! once; this module turns that table into arity, kind, and residency
//! diagnostics plus the exact immutable result object. Facts that cannot be
//! known statically (unknown kinds, unknown object classes, unknown cell
//! contents) never produce diagnostics: the runtime remains the authority for
//! value-level validation such as nonempty identifiers or positive counts.

use crate::{
    catalog::{argument_error, finish_fixed},
    BuiltinCatalogEntry,
};
use runmat_types::{
    CallInference, CallRequest, CertaintyFact, DimensionFact, DynamicReason, InferenceDiagnostic,
    NumericDomain, ObjectFact, ResidencyFact, ShapeFact, StaticClassIdentity, StorageFact,
    ValueFact, ValueKindFact,
};
use std::collections::BTreeMap;

/// Statically checkable kind of one positional constructor argument.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StudyArgument {
    /// Host string scalar or character row (identifier or digest).
    Text,
    /// Host text scalar, string array, or cell array of text.
    TextList,
    /// Finite real numeric scalar; all integer classes are admitted.
    RealScalar,
    /// Nonnegative integer-valued numeric scalar (count or interval).
    Count,
    /// Nonempty real numeric scalar or array of coordinates.
    RealValues,
    /// N-by-3 real numeric matrix of positions in meters, N >= 2.
    Positions,
    /// One typed object of the listed classes; an empty list admits any object.
    Object(&'static [StaticClassIdentity]),
    /// One typed object or a cell array of objects of the listed classes.
    ObjectList(&'static [StaticClassIdentity]),
}

/// Static class of the constructed immutable value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StudyResult {
    /// The constructor always returns this class.
    Class(StaticClassIdentity),
    /// The constructor returns the class of one admitted input object.
    SameClassAs {
        argument: usize,
        admitted: &'static [StaticClassIdentity],
    },
}

/// Positional contract for one typed study constructor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StudyConstructorSignature {
    pub arguments: &'static [StudyArgument],
    pub result: StudyResult,
}

impl StudyConstructorSignature {
    pub const fn new(arguments: &'static [StudyArgument], result: StaticClassIdentity) -> Self {
        Self {
            arguments,
            result: StudyResult::Class(result),
        }
    }
}

pub const ARITY_CODE: &str = "RM-CATALOG-FEA-STUDY-ARITY";
pub const TEXT_CODE: &str = "RM-CATALOG-FEA-STUDY-TEXT";
pub const NUMERIC_CODE: &str = "RM-CATALOG-FEA-STUDY-NUMERIC";
pub const POSITIONS_CODE: &str = "RM-CATALOG-FEA-STUDY-POSITIONS";
pub const OBJECT_CODE: &str = "RM-CATALOG-FEA-STUDY-OBJECT";
pub const RESIDENT_CODE: &str = "RM-CATALOG-FEA-STUDY-RESIDENT";

pub fn infer_study_constructor(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    signature: &StudyConstructorSignature,
) -> CallInference {
    let name = entry.identity.name;
    let mut diagnostics = Vec::new();
    let expected = signature.arguments.len();
    if request.arguments.len() != expected {
        diagnostics.push(argument_error(
            ARITY_CODE,
            format!(
                "{name} requires exactly {expected} argument{}",
                if expected == 1 { "" } else { "s" }
            ),
            request.arguments.len().min(expected),
        ));
    }
    for (index, (fact, kind)) in request
        .arguments
        .iter()
        .zip(signature.arguments.iter())
        .enumerate()
    {
        if let Some(diagnostic) = check_argument(name, index, fact, *kind) {
            diagnostics.push(diagnostic);
        }
    }
    finish_fixed(
        entry,
        request,
        result_fact(request, signature.result),
        diagnostics,
    )
}

fn check_argument(
    name: &str,
    index: usize,
    fact: &ValueFact,
    kind: StudyArgument,
) -> Option<InferenceDiagnostic> {
    if matches!(fact.residency, ResidencyFact::Device { .. }) {
        return Some(argument_error(
            RESIDENT_CODE,
            format!("{name} builds a host study value and does not gather device-resident inputs"),
            index,
        ));
    }
    let (valid, code, message) = match kind {
        StudyArgument::Text => (
            host_text(fact),
            TEXT_CODE,
            "identifiers must be host text scalars",
        ),
        StudyArgument::TextList => (
            text_list(fact),
            TEXT_CODE,
            "identifier lists must be text, a string array, or a cell array of text",
        ),
        StudyArgument::RealScalar | StudyArgument::Count => (
            real_numeric(fact) && !known_count_other_than(fact, 1),
            NUMERIC_CODE,
            "this argument must be a real numeric scalar",
        ),
        StudyArgument::RealValues => (
            real_numeric(fact) && !matches!(fact.shape.element_count(), Some(0)),
            NUMERIC_CODE,
            "coordinates must be a nonempty real numeric scalar or array",
        ),
        StudyArgument::Positions => (
            real_numeric(fact) && positions_shape(&fact.shape),
            POSITIONS_CODE,
            "positions must be an N-by-3 real numeric matrix with N at least two",
        ),
        StudyArgument::Object(classes) => (
            single_object(fact, classes),
            OBJECT_CODE,
            "this argument must be one typed FEA object of the documented class",
        ),
        StudyArgument::ObjectList(classes) => (
            object_list(fact, classes),
            OBJECT_CODE,
            "this argument must be a typed FEA object or a cell array of them",
        ),
    };
    (!valid).then(|| argument_error(code, format!("{name}: {message}"), index))
}

/// Host string scalar or character row. Unknown kinds stay admissible.
pub fn host_text(fact: &ValueFact) -> bool {
    if matches!(fact.residency, ResidencyFact::Device { .. }) {
        return false;
    }
    match &fact.kind {
        ValueKindFact::Unknown => true,
        ValueKindFact::String => !known_count_other_than(fact, 1),
        ValueKindFact::Character => match &fact.shape {
            ShapeFact::Shaped { dims } => {
                !matches!(dims.first(), Some(DimensionFact::Known(rows)) if *rows != 1)
            }
            ShapeFact::Scalar | ShapeFact::Ranked { rank: 2.. } | ShapeFact::Unknown => true,
            _ => false,
        },
        _ => false,
    }
}

fn known_count_other_than(fact: &ValueFact, count: usize) -> bool {
    matches!(fact.shape.element_count(), Some(actual) if actual != count)
}

fn real_numeric(fact: &ValueFact) -> bool {
    match &fact.kind {
        ValueKindFact::Unknown => true,
        ValueKindFact::Numeric(numeric) => numeric.domain == NumericDomain::Real,
        _ => false,
    }
}

fn positions_shape(shape: &ShapeFact) -> bool {
    match shape {
        ShapeFact::Scalar => false,
        ShapeFact::Shaped { dims } => {
            if dims.len() != 2 {
                return false;
            }
            let rows_ok = !matches!(dims[0], DimensionFact::Known(rows) if rows < 2);
            let columns_ok = !matches!(dims[1], DimensionFact::Known(columns) if columns != 3);
            rows_ok && columns_ok
        }
        ShapeFact::Ranked { rank } => *rank <= 2,
        ShapeFact::Unknown => true,
    }
}

fn text_list(fact: &ValueFact) -> bool {
    match &fact.kind {
        ValueKindFact::Unknown | ValueKindFact::String => true,
        ValueKindFact::Character => host_text(fact),
        ValueKindFact::Cell(cell) => {
            possible_text(&cell.element) && cell.elements.iter().all(possible_text)
        }
        _ => false,
    }
}

fn possible_text(fact: &ValueFact) -> bool {
    matches!(fact.kind, ValueKindFact::Never) || host_text(fact)
}

fn object_class_matches(object: &ObjectFact, classes: &[StaticClassIdentity]) -> bool {
    classes.is_empty()
        || object
            .runtime_class
            .as_ref()
            .is_none_or(|class| classes.iter().any(|expected| class.is(*expected)))
}

fn single_object(fact: &ValueFact, classes: &[StaticClassIdentity]) -> bool {
    match &fact.kind {
        ValueKindFact::Unknown => true,
        ValueKindFact::Object(object) => {
            !known_count_other_than(fact, 1) && object_class_matches(object, classes)
        }
        _ => false,
    }
}

fn possible_list_element(fact: &ValueFact, classes: &[StaticClassIdentity]) -> bool {
    matches!(fact.kind, ValueKindFact::Never) || single_object(fact, classes)
}

fn object_list(fact: &ValueFact, classes: &[StaticClassIdentity]) -> bool {
    match &fact.kind {
        ValueKindFact::Cell(cell) => {
            possible_list_element(&cell.element, classes)
                && cell
                    .elements
                    .iter()
                    .all(|element| possible_list_element(element, classes))
        }
        _ => single_object(fact, classes),
    }
}

fn result_fact(request: &CallRequest, result: StudyResult) -> ValueFact {
    let runtime_class = match result {
        StudyResult::Class(class) => Some(class),
        StudyResult::SameClassAs { argument, admitted } => request
            .arguments
            .get(argument)
            .and_then(|fact| match &fact.kind {
                ValueKindFact::Object(object) => object.runtime_class.as_ref(),
                _ => None,
            })
            .and_then(|class| {
                admitted
                    .iter()
                    .copied()
                    .find(|expected| class.is(*expected))
            }),
    };
    let mut fact = ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: runtime_class.map(StaticClassIdentity::owned),
            properties: BTreeMap::new(),
            properties_complete: false,
            handle_semantics: Some(false),
        }),
        ShapeFact::Scalar,
        StorageFact::Opaque,
    );
    if runtime_class.is_none() {
        fact.certainty = CertaintyFact::Dynamic(DynamicReason::RuntimeValue);
    }
    fact
}
