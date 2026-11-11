//! Checks of PMI entity instances against the AP242 schema
//! (`ap242_managed_model_based_3d_engineering`): each entity's explicit attribute count, and
//! for each reference attribute the entities it may name. Partial entities of complex
//! instances are checked by their own attributes only.

use super::{entity_has_datums, CHARACTERISTICS};
use crate::step::p21::{Data, Param};

/// What an attribute may hold: `Any`, or a reference to one of these entities (a complex
/// instance qualifies when it includes one of them).
#[derive(Clone, Copy)]
enum A {
    Any,
    Ref(&'static [&'static str]),
}

const ASPECTS: &[&str] = &[
    "SHAPE_ASPECT",
    "DATUM_FEATURE",
    "DATUM",
    "DATUM_SYSTEM",
    "TOLERANCE_ZONE",
    "DATUM_REFERENCE_COMPARTMENT",
];
const PDS: &[&str] = &["PRODUCT_DEFINITION_SHAPE"];
const REPS: &[&str] = &[
    "ADVANCED_BREP_SHAPE_REPRESENTATION",
    "MANIFOLD_SURFACE_SHAPE_REPRESENTATION",
    "FACETED_BREP_SHAPE_REPRESENTATION",
    "SHAPE_REPRESENTATION",
];
const FACES: &[&str] = &["ADVANCED_FACE", "FACE_SURFACE"];
const MEASURES: &[&str] = &["LENGTH_MEASURE_WITH_UNIT", "MEASURE_WITH_UNIT"];
const UNITS: &[&str] = &["LENGTH_UNIT", "NAMED_UNIT"];
const SIZED: &[&str] = &["DIMENSIONAL_SIZE", "DIMENSIONAL_LOCATION"];

/// Explicit attributes per entity (inherited first), for a simple instance of it.
fn simple(name: &str) -> Option<Vec<A>> {
    use A::*;
    let aspect = vec![Any, Any, Ref(PDS), Any];
    Some(match name {
        "SHAPE_ASPECT" | "DATUM_FEATURE" => aspect,
        "DATUM" => [aspect, vec![Any]].concat(),
        "GEOMETRIC_ITEM_SPECIFIC_USAGE" => vec![Any, Any, Ref(ASPECTS), Ref(REPS), Ref(FACES)],
        "SHAPE_ASPECT_RELATIONSHIP" => vec![Any, Any, Ref(ASPECTS), Ref(ASPECTS)],
        "DATUM_REFERENCE_COMPARTMENT" => {
            [aspect, vec![Ref(&["DATUM", "COMMON_DATUM"]), Any]].concat()
        }
        "DATUM_SYSTEM" => [aspect, vec![Any]].concat(),
        "TOLERANCE_ZONE" => [aspect, vec![Any, Ref(&["TOLERANCE_ZONE_FORM"])]].concat(),
        "TOLERANCE_ZONE_FORM" => vec![Any],
        "LENGTH_MEASURE_WITH_UNIT" => vec![Any, Ref(UNITS)],
        "DIMENSIONAL_SIZE" => vec![Ref(ASPECTS), Any],
        "DIMENSIONAL_LOCATION" => vec![Any, Any, Ref(ASPECTS), Ref(ASPECTS)],
        "DIMENSIONAL_CHARACTERISTIC_REPRESENTATION" => {
            vec![Ref(SIZED), Ref(&["SHAPE_DIMENSION_REPRESENTATION"])]
        }
        "SHAPE_DIMENSION_REPRESENTATION" => vec![Any, Any, Any],
        "TOLERANCE_VALUE" => vec![Ref(MEASURES), Ref(MEASURES)],
        "PLUS_MINUS_TOLERANCE" => vec![Ref(&["TOLERANCE_VALUE", "LIMITS_AND_FITS"]), Ref(SIZED)],
        "LIMITS_AND_FITS" => vec![Any, Any, Any, Any],
        n => {
            let (c, _) = CHARACTERISTICS.iter().find(|(_, e)| *e == n)?;
            let gt = vec![
                Any,
                Any,
                Ref(MEASURES),
                Ref(&[
                    "SHAPE_ASPECT",
                    "DATUM_FEATURE",
                    "DIMENSIONAL_SIZE",
                    "DIMENSIONAL_LOCATION",
                    "PRODUCT_DEFINITION_SHAPE",
                ]),
            ];
            if entity_has_datums(*c) {
                [gt, vec![Any]].concat()
            } else {
                gt
            }
        }
    })
}

/// Own explicit attributes of a partial entity in a complex instance.
fn partial(name: &str) -> Option<usize> {
    Some(match name {
        "GEOMETRIC_TOLERANCE" => 4,
        "GEOMETRIC_TOLERANCE_WITH_DATUM_REFERENCE" | "GEOMETRIC_TOLERANCE_WITH_MODIFIERS" => 1,
        "MEASURE_WITH_UNIT" => 2,
        "REPRESENTATION_ITEM" => 1,
        "LENGTH_MEASURE_WITH_UNIT" | "MEASURE_REPRESENTATION_ITEM" => 0,
        n if CHARACTERISTICS.iter().any(|(_, e)| *e == n) => 0,
        _ => return None,
    })
}

/// Problems with the file's PMI instances, empty when they conform.
pub fn check(d: &Data) -> Vec<String> {
    let mut out = Vec::new();
    let is = |id: u64, allowed: &[&str]| {
        d.get(id)
            .is_some_and(|i| i.parts.iter().any(|(n, _)| allowed.contains(&n.as_str())))
    };
    for (id, i) in &d.instances {
        if i.is_complex() {
            for (n, p) in &i.parts {
                if let Some(k) = partial(n) {
                    if p.len() != k {
                        out.push(format!(
                            "#{id}: {n} takes {k} attributes in a complex instance, not {}",
                            p.len()
                        ));
                    }
                }
            }
            if i.has("GEOMETRIC_TOLERANCE") && !CHARACTERISTICS.iter().any(|(_, e)| i.has(e)) {
                out.push(format!(
                    "#{id}: a geometric tolerance without its characteristic's entity"
                ));
            }
            let names: Vec<&str> = i.parts.iter().map(|(n, _)| n.as_str()).collect();
            let sorted = {
                let mut s = names.clone();
                s.sort();
                s
            };
            if names != sorted {
                out.push(format!(
                    "#{id}: partial entities are not in alphabetical order"
                ));
            }
            continue;
        }
        let name = &i.parts[0].0;
        let Some(attrs) = simple(name) else { continue };
        let p = &i.parts[0].1;
        if p.len() != attrs.len() {
            out.push(format!(
                "#{id}: {name} takes {} attributes, not {}",
                attrs.len(),
                p.len()
            ));
            continue;
        }
        for (k, (a, v)) in attrs.iter().zip(p).enumerate() {
            if let (A::Ref(allowed), Param::Ref(r)) = (a, v) {
                if !is(*r, allowed) {
                    out.push(format!(
                        "#{id}: {name} attribute {} names #{r}, which is not one of {allowed:?}",
                        k + 1
                    ));
                }
            }
            if let (A::Ref(_), Param::Unset | Param::Str(_) | Param::Real(_) | Param::Int(_)) =
                (a, v)
            {
                out.push(format!(
                    "#{id}: {name} attribute {} must be a reference",
                    k + 1
                ));
            }
        }
    }
    out
}
