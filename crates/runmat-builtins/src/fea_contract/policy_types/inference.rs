//! Static argument checks for typed FEA policy constructors.
//!
//! Every policy constructor has a positional signature (required arguments,
//! then optional trailing ones) and returns one immutable host object of a known class. The checks below only reject
//! arguments whose kind, shape, or residency is proven incompatible with the
//! runtime admission rules; unknown facts never produce a diagnostic.

use crate::{
    catalog::{argument_error, finish_fixed},
    BuiltinCatalogEntry,
};
use runmat_types::{
    CallInference, CallRequest, DimensionFact, InferenceDiagnostic, NumericDomain, ObjectFact,
    ResidencyFact, ShapeFact, StaticClassIdentity, StorageFact, ValueFact, ValueKindFact,
};
use std::collections::BTreeMap;

pub const ARITY: &str = "RM-CATALOG-FEA-POLICY-ARITY";
pub const TEXT: &str = "RM-CATALOG-FEA-POLICY-TEXT";
pub const NUMERIC: &str = "RM-CATALOG-FEA-POLICY-NUMERIC";
pub const SHAPE: &str = "RM-CATALOG-FEA-POLICY-SHAPE";
pub const LOGICAL: &str = "RM-CATALOG-FEA-POLICY-LOGICAL";
pub const TYPED: &str = "RM-CATALOG-FEA-POLICY-TYPED";
pub const RESIDENT: &str = "RM-CATALOG-FEA-POLICY-RESIDENT";

/// Statically checkable admission rule for one positional argument.
#[derive(Clone, Copy, Debug)]
pub enum PolicyArg {
    /// Character row vector or string scalar.
    Text,
    /// Real numeric scalar; integer classes are admitted.
    RealScalar,
    /// Nonnegative integer-valued numeric scalar (sample counts, ceilings).
    Count,
    /// Nonempty real numeric row or column vector.
    RealVector,
    /// Real numeric vector with exactly three components.
    RealVector3,
    /// Real numeric scalar or three-component vector (signal ordinates).
    ScalarOrVector3,
    /// Logical scalar; numeric zero and one are rejected by the runtime.
    Logical,
    /// One host object of any listed class.
    Typed(&'static [StaticClassIdentity]),
    /// One host object of a listed class, or a cell array of such objects.
    TypedOrCell(&'static [StaticClassIdentity]),
    /// A row or column cell array holding at least `minimum` such objects.
    TypedCell {
        classes: &'static [StaticClassIdentity],
        minimum: usize,
    },
    /// A trailing argument the caller may omit; when present it follows the inner rule.
    /// Only trailing positions may be optional.
    Optional(&'static PolicyArg),
}

impl PolicyArg {
    pub const fn is_optional(self) -> bool {
        matches!(self, PolicyArg::Optional(_))
    }

    /// The rule a supplied argument must satisfy.
    pub const fn supplied(self) -> PolicyArg {
        match self {
            PolicyArg::Optional(inner) => *inner,
            rule => rule,
        }
    }
}

/// The complete static call shape of one typed policy constructor.
#[derive(Clone, Copy, Debug)]
pub struct PolicyCall {
    pub result: StaticClassIdentity,
    pub arguments: &'static [PolicyArg],
}

impl PolicyCall {
    /// The number of leading arguments a call must supply.
    pub fn required(&self) -> usize {
        self.arguments
            .iter()
            .take_while(|rule| !rule.is_optional())
            .count()
    }
}

pub fn infer_policy(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    call: &PolicyCall,
) -> CallInference {
    let mut diagnostics = Vec::new();
    let (required, maximum) = (call.required(), call.arguments.len());
    if !(required..=maximum).contains(&request.arguments.len()) {
        let expected = if required == maximum {
            format!("exactly {maximum}")
        } else {
            format!("{required} to {maximum}")
        };
        diagnostics.push(argument_error(
            ARITY,
            format!("{} requires {expected} argument(s)", entry.identity.name),
            request.arguments.len().min(maximum),
        ));
    }
    for (index, (fact, rule)) in request.arguments.iter().zip(call.arguments).enumerate() {
        if let Some(diagnostic) = check_argument(fact, *rule, index) {
            diagnostics.push(diagnostic);
        }
    }
    finish_fixed(entry, request, policy_object(call.result), diagnostics)
}

pub fn policy_object(class: StaticClassIdentity) -> ValueFact {
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

fn check_argument(fact: &ValueFact, rule: PolicyArg, index: usize) -> Option<InferenceDiagnostic> {
    if matches!(fact.residency, ResidencyFact::Device { .. }) {
        return Some(argument_error(
            RESIDENT,
            "typed FEA policy arguments are host values; gather device data explicitly",
            index,
        ));
    }
    let failure = match rule.supplied() {
        PolicyArg::Optional(_) => unreachable!("optional rules wrap a supplied rule"),
        PolicyArg::Text => (!text_scalar(fact)).then_some((TEXT, "expected a text scalar")),
        PolicyArg::RealScalar | PolicyArg::Count => numeric(fact, Some(1..=1)),
        PolicyArg::RealVector => numeric(fact, None),
        PolicyArg::RealVector3 => numeric(fact, Some(3..=3)),
        PolicyArg::ScalarOrVector3 => numeric(fact, Some(1..=3)).or_else(|| {
            (count(fact) == Some(2))
                .then_some((SHAPE, "expected a scalar or three-component vector"))
        }),
        PolicyArg::Logical => {
            (!matches!(fact.kind, ValueKindFact::Unknown | ValueKindFact::Logical)
                || count(fact).is_some_and(|count| count != 1))
            .then_some((LOGICAL, "expected a logical scalar"))
        }
        PolicyArg::Typed(classes) => (!possible_object(fact, classes)).then_some((
            TYPED,
            "expected a typed FEA policy object of the documented class",
        )),
        PolicyArg::TypedOrCell(classes) => {
            (!possible_object(fact, classes) && !possible_cell(fact, classes)).then_some((
                TYPED,
                "expected a typed FEA object of the documented class or a cell array of them",
            ))
        }
        PolicyArg::TypedCell { classes, minimum } => {
            if !possible_cell(fact, classes) {
                Some((
                    TYPED,
                    "expected a cell array of typed FEA objects of the documented class",
                ))
            } else if count(fact).is_some_and(|count| count < minimum) || !vector_shape(fact) {
                Some((
                    SHAPE,
                    "expected a row or column cell array with enough elements",
                ))
            } else {
                None
            }
        }
    };
    failure.map(|(code, message)| argument_error(code, message, index))
}

fn count(fact: &ValueFact) -> Option<usize> {
    fact.shape.element_count()
}

/// `elements` bounds the known element count; `None` means any nonempty vector.
fn numeric(
    fact: &ValueFact,
    elements: Option<std::ops::RangeInclusive<usize>>,
) -> Option<(&'static str, &'static str)> {
    let real = match &fact.kind {
        ValueKindFact::Unknown => true,
        ValueKindFact::Numeric(numeric) => numeric.domain == NumericDomain::Real,
        _ => false,
    };
    if !real {
        return Some((NUMERIC, "expected a real numeric value"));
    }
    let known = count(fact);
    let count_ok = match (&elements, known) {
        (_, Some(0)) => false,
        (Some(range), Some(count)) => range.contains(&count),
        _ => true,
    };
    (!count_ok || !vector_shape(fact)).then_some((
        SHAPE,
        "numeric argument has the wrong number of elements or is not a vector",
    ))
}

/// False only when the shape is proven to have two non-singleton extents.
fn vector_shape(fact: &ValueFact) -> bool {
    let ShapeFact::Shaped { dims } = &fact.shape else {
        return true;
    };
    dims.iter()
        .filter(|dim| matches!(dim, DimensionFact::Known(extent) if *extent != 1))
        .count()
        <= 1
}

fn possible_object(fact: &ValueFact, classes: &[StaticClassIdentity]) -> bool {
    match &fact.kind {
        ValueKindFact::Unknown => true,
        ValueKindFact::Object(object) => {
            count(fact).is_none_or(|count| count == 1)
                && object
                    .runtime_class
                    .as_ref()
                    .is_none_or(|class| classes.iter().any(|expected| class.is(*expected)))
        }
        _ => false,
    }
}

fn possible_cell(fact: &ValueFact, classes: &[StaticClassIdentity]) -> bool {
    match &fact.kind {
        ValueKindFact::Unknown => true,
        ValueKindFact::Cell(cell) => {
            let element_ok = |element: &ValueFact| {
                matches!(element.kind, ValueKindFact::Never) || possible_object(element, classes)
            };
            element_ok(&cell.element) && cell.elements.iter().all(element_ok)
        }
        _ => false,
    }
}

/// Character row vector or string scalar; unknown kinds and shapes are admitted.
fn text_scalar(fact: &ValueFact) -> bool {
    match &fact.kind {
        ValueKindFact::Unknown => true,
        ValueKindFact::String => count(fact).is_none_or(|count| count == 1),
        ValueKindFact::Character => match &fact.shape {
            ShapeFact::Shaped { dims } => {
                dims.len() <= 2
                    && !matches!(dims.first(), Some(DimensionFact::Known(rows)) if *rows != 1)
            }
            ShapeFact::Ranked { rank } => *rank <= 2,
            ShapeFact::Scalar | ShapeFact::Unknown => true,
        },
        _ => false,
    }
}
