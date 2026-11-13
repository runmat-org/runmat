//! Shared static admission for the exact-geometry builder.
//!
//! Each identity supplies its positional argument roles; the rules here mirror
//! the runtime conversion boundary (`generated/primitives.rs::numeric_values` and
//! the immutable-class codec) without claiming facts that only the feature-graph
//! validator can establish.

use crate::{
    catalog::{argument_error, finish_fixed},
    fea_contract::constructor_inference::host_text,
    BuiltinCatalogEntry,
};
use runmat_types::{
    CallInference, CallRequest, InferenceDiagnostic, LiteralValue, ObjectFact, ResidencyFact,
    ShapeFact, StaticClassIdentity, StorageFact, ValueFact, ValueKindFact,
};
use std::collections::BTreeMap;

use super::descriptor::{
    GEOMETRY_FEATURE_CLASS, GEOMETRY_GENERATED_CLASS, GEOMETRY_REGION_OUTPUT_CLASS,
};

/// Argument role at the runtime conversion boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum GeometryArgument {
    /// Strictly positive real scalar (length, radius, depth, distance, or angle).
    PositiveScalar,
    /// Finite three-element real vector with no further constraint (a point).
    Point,
    /// Finite three-element real vector that must not be all zero.
    Direction,
    /// 4-by-4 affine transform with exactly 16 finite real values.
    Transform,
    Feature,
    RegionOutput,
    Generated,
    Text,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct GeometrySignature {
    pub code: &'static str,
    pub name: &'static str,
    pub positional: &'static [GeometryArgument],
    /// Repeated trailing role and the minimum number of repetitions.
    pub variadic: Option<(GeometryArgument, usize)>,
    pub result: StaticClassIdentity,
}

pub(super) const FEATURE_RESULT: StaticClassIdentity = GEOMETRY_FEATURE_CLASS;
pub(super) const GENERATED_RESULT: StaticClassIdentity = GEOMETRY_GENERATED_CLASS;
pub(super) const REGION_OUTPUT_RESULT: StaticClassIdentity = GEOMETRY_REGION_OUTPUT_CLASS;

pub(super) fn infer_geometry(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    signature: GeometrySignature,
) -> CallInference {
    let mut diagnostics = Vec::new();
    let count = request.arguments.len();
    let positional = signature.positional.len();
    let arity_valid = match signature.variadic {
        Some((_, minimum)) => count >= positional + minimum,
        None => count == positional,
    };
    if !arity_valid {
        let expected = match signature.variadic {
            Some((_, minimum)) => format!("at least {} arguments", positional + minimum),
            None => format!("exactly {positional} arguments"),
        };
        diagnostics.push(argument_error(
            format!("RM-CATALOG-GEOMETRY-{}-ARITY", signature.code),
            format!("{} requires {expected}", signature.name),
            count,
        ));
    }
    for (index, argument) in request.arguments.iter().enumerate() {
        let role = match signature.positional.get(index) {
            Some(role) => *role,
            None => match signature.variadic {
                Some((role, _)) => role,
                None => continue,
            },
        };
        if let Some(diagnostic) = check_argument(request, index, argument, role, signature) {
            diagnostics.push(diagnostic);
        }
    }
    finish_fixed(entry, request, object_result(signature.result), diagnostics)
}

pub(super) fn object_result(class: StaticClassIdentity) -> ValueFact {
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

fn check_argument(
    request: &CallRequest,
    index: usize,
    argument: &ValueFact,
    role: GeometryArgument,
    signature: GeometrySignature,
) -> Option<InferenceDiagnostic> {
    let error = |suffix: &str, message: String| {
        Some(argument_error(
            format!("RM-CATALOG-GEOMETRY-{}-{suffix}", signature.code),
            message,
            index,
        ))
    };
    match role {
        GeometryArgument::PositiveScalar => {
            if !real_numeric(argument) || count_is_not(argument, 1) {
                return error(
                    "SCALAR",
                    format!(
                        "{} argument {} must be a real numeric scalar",
                        signature.name,
                        index + 1
                    ),
                );
            }
            match request
                .literals
                .literal_args
                .get(index)
                .and_then(literal_number)
            {
                Some(value) if !(value.is_finite() && value > 0.0) => error(
                    "NONPOSITIVE",
                    format!(
                        "{} argument {} must be finite and positive",
                        signature.name,
                        index + 1
                    ),
                ),
                _ => None,
            }
        }
        GeometryArgument::Point | GeometryArgument::Direction | GeometryArgument::Transform => {
            let (expected, label) = match role {
                GeometryArgument::Transform => (16, "a 4-by-4 real matrix"),
                _ => (3, "a three-element real vector"),
            };
            if !real_numeric(argument)
                || count_is_not(argument, expected)
                || integer_array(argument)
            {
                return error(
                    if role == GeometryArgument::Transform {
                        "MATRIX"
                    } else {
                        "VECTOR"
                    },
                    format!("{} argument {} must be {label}", signature.name, index + 1),
                );
            }
            if role == GeometryArgument::Direction
                && literal_vector(request, index).is_some_and(|values| {
                    values.len() == 3 && values.iter().all(|value| *value == 0.0)
                })
            {
                return error(
                    "DIRECTION",
                    format!(
                        "{} argument {} must be a nonzero direction",
                        signature.name,
                        index + 1
                    ),
                );
            }
            None
        }
        GeometryArgument::Feature
        | GeometryArgument::RegionOutput
        | GeometryArgument::Generated => {
            let class = match role {
                GeometryArgument::Feature => GEOMETRY_FEATURE_CLASS,
                GeometryArgument::RegionOutput => GEOMETRY_REGION_OUTPUT_CLASS,
                _ => GEOMETRY_GENERATED_CLASS,
            };
            if possible_object(argument, class) {
                None
            } else {
                error(
                    "OBJECT",
                    format!(
                        "{} argument {} must be a {} value",
                        signature.name,
                        index + 1,
                        class.display_name()
                    ),
                )
            }
        }
        GeometryArgument::Text => {
            if host_text(argument) {
                None
            } else {
                error(
                    "TEXT",
                    format!(
                        "{} argument {} must be scalar text",
                        signature.name,
                        index + 1
                    ),
                )
            }
        }
    }
}

fn real_numeric(fact: &ValueFact) -> bool {
    !matches!(fact.residency, ResidencyFact::Device { .. })
        && match &fact.kind {
            ValueKindFact::Unknown => true,
            ValueKindFact::Numeric(numeric) => {
                !matches!(numeric.domain, runmat_types::NumericDomain::Complex)
            }
            _ => false,
        }
}

fn count_is_not(fact: &ValueFact, expected: usize) -> bool {
    matches!(fact.shape.element_count(), Some(count) if count != expected)
}

/// Integer-class arrays (as opposed to integer-class scalars) are rejected by
/// the runtime vector boundary.
fn integer_array(fact: &ValueFact) -> bool {
    matches!(&fact.kind, ValueKindFact::Numeric(numeric) if numeric.class.integer_class().is_some())
        && matches!(fact.shape.element_count(), Some(count) if count > 1)
}

fn possible_object(fact: &ValueFact, class: StaticClassIdentity) -> bool {
    !matches!(fact.residency, ResidencyFact::Device { .. })
        && !count_is_not(fact, 1)
        && match &fact.kind {
            ValueKindFact::Unknown => true,
            ValueKindFact::Object(object) => object
                .runtime_class
                .as_ref()
                .is_none_or(|runtime_class| runtime_class.is(class)),
            _ => false,
        }
}

fn literal_number(value: &LiteralValue) -> Option<f64> {
    match value {
        LiteralValue::Number(value) => Some(*value),
        LiteralValue::Real { text, .. } => text.parse().ok(),
        LiteralValue::Integer { .. } => value.integer_value_i128().map(|value| value as f64),
        _ => None,
    }
}

fn literal_vector(request: &CallRequest, index: usize) -> Option<Vec<f64>> {
    let values = match request.literals.literal_args.get(index)? {
        LiteralValue::Vector(values) => values.clone(),
        LiteralValue::Matrix(rows) => rows.iter().flatten().cloned().collect(),
        _ => return None,
    };
    values.iter().map(literal_number).collect()
}
