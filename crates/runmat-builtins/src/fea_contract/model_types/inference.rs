//! Static argument checks for typed FEA model-object constructors.
//!
//! Every constructor in this family has a fixed positional signature and
//! returns one immutable host object of a known class. A rule rejects an
//! argument only when its kind, shape, or residency is proven incompatible
//! with runtime admission; unknown facts never produce a diagnostic. Geometry
//! binding, identifier emptiness, and physical ranges stay runtime checks.

use crate::{
    catalog::{argument_error, finish_fixed},
    fea_contract::constructor_inference::{
        host_text, object_fact, possible_object, possible_object_or_cell, resident,
    },
    BuiltinCatalogEntry,
};
use runmat_types::{
    CallInference, CallRequest, InferenceDiagnostic, NumericDomain, ResidencyFact,
    StaticClassIdentity, ValueFact, ValueKindFact,
};

pub const ARITY: &str = "RM-CATALOG-FEA-MODEL-ARITY";
pub const TEXT: &str = "RM-CATALOG-FEA-MODEL-TEXT";
pub const TYPED: &str = "RM-CATALOG-FEA-MODEL-TYPED";
pub const NUMERIC: &str = "RM-CATALOG-FEA-MODEL-NUMERIC";
pub const RESIDENT: &str = "RM-CATALOG-FEA-MODEL-RESIDENT";

/// Statically checkable admission rule for one positional argument.
#[derive(Clone, Copy, Debug)]
pub enum ModelArg {
    /// Character row vector or string scalar (identifiers, step ids).
    Text,
    /// Text, a string array, or a cell array of text (lists of step ids).
    TextList,
    /// Real numeric scalar; integer classes are admitted.
    RealScalar,
    /// Real numeric vector with exactly three components.
    RealVector3,
    /// One host object of any listed class.
    Typed(&'static [StaticClassIdentity]),
    /// One host object of a listed class, or a cell array whose known elements
    /// are such objects (the runtime also decodes nested component values).
    TypedOrCell(&'static [StaticClassIdentity]),
    /// One host object of an `object` class, or a cell array whose known
    /// elements are of a `cell` class (for example a signal, or components).
    ObjectOrCellOf {
        object: &'static [StaticClassIdentity],
        cell: &'static [StaticClassIdentity],
    },
}

/// The complete static call shape of one model-object constructor.
#[derive(Clone, Copy, Debug)]
pub struct ModelObjectCall {
    pub result: StaticClassIdentity,
    pub arguments: &'static [ModelArg],
}

pub fn infer_model_object(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    call: &ModelObjectCall,
) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != call.arguments.len() {
        diagnostics.push(argument_error(
            ARITY,
            format!(
                "{} requires exactly {} argument(s)",
                entry.identity.name,
                call.arguments.len()
            ),
            request.arguments.len().min(call.arguments.len()),
        ));
    }
    for (index, (fact, rule)) in request.arguments.iter().zip(call.arguments).enumerate() {
        if let Some(diagnostic) = check_argument(fact, *rule, index) {
            diagnostics.push(diagnostic);
        }
    }
    let mut output = object_fact(call.result);
    output.residency = ResidencyFact::Host;
    finish_fixed(entry, request, output, diagnostics)
}

fn check_argument(fact: &ValueFact, rule: ModelArg, index: usize) -> Option<InferenceDiagnostic> {
    if resident(fact) {
        return Some(argument_error(
            RESIDENT,
            "FEA model-object constructors do not gather device-resident arguments",
            index,
        ));
    }
    let (admitted, code, message) = match rule {
        ModelArg::Text => (
            host_text(fact),
            TEXT,
            "expected a character row vector or string scalar",
        ),
        ModelArg::TextList => (
            matches!(
                fact.kind,
                ValueKindFact::Unknown | ValueKindFact::Cell(_) | ValueKindFact::String
            ) || host_text(fact),
            TEXT,
            "expected text, a string array, or a cell array of text",
        ),
        ModelArg::RealScalar => (
            real_numeric(fact) && !known_count_other_than(fact, 1),
            NUMERIC,
            "expected a real numeric scalar",
        ),
        ModelArg::RealVector3 => (
            real_numeric(fact) && !known_count_other_than(fact, 3),
            NUMERIC,
            "expected a real numeric vector with three components",
        ),
        ModelArg::Typed(classes) => (
            possible_object(fact, classes),
            TYPED,
            "expected a typed FEA object of the documented class",
        ),
        ModelArg::TypedOrCell(classes) => (
            possible_object_or_cell(fact, classes),
            TYPED,
            "expected a typed FEA object of the documented class, or a cell array of them",
        ),
        ModelArg::ObjectOrCellOf { object, cell } => (
            match &fact.kind {
                ValueKindFact::Cell(_) => possible_object_or_cell(fact, cell),
                _ => possible_object(fact, object),
            },
            TYPED,
            "expected a typed FEA object of the documented class, or a cell array of its documented components",
        ),
    };
    (!admitted).then(|| argument_error(code, message, index))
}

fn real_numeric(fact: &ValueFact) -> bool {
    match &fact.kind {
        ValueKindFact::Unknown => true,
        ValueKindFact::Numeric(number) => number.domain == NumericDomain::Real,
        _ => false,
    }
}

fn known_count_other_than(fact: &ValueFact, expected: usize) -> bool {
    matches!(fact.shape.element_count(), Some(count) if count != expected)
}
