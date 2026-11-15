//! An annotation's definition as the field values of the Annotate command that makes it, so
//! Properties shows the command's own controls filled from the file, and a change re-applies
//! the command (lengths in millimetres, the catalog unit).

use crate::drawing::values::length_mm;
use crate::drawing_edits::Values;
use runmat_cad::drawing::{AnnotationKind, WeldSide};
use runmat_cad::model::PartDocument;
use runmat_cad::pmi::{MaterialCondition, ModelAnnotation, Removal};
use runmat_cad::refs::reference::Reference;
use runmat_cad::syntax::refs::print_reference;
use runmat_cad::value::ValueExpr;
use serde_json::{json, Value};

fn picks(refs: &[Reference]) -> Value {
    Value::Array(
        refs.iter()
            .filter_map(print_reference)
            .map(Value::String)
            .collect(),
    )
}

fn mm(v: &ValueExpr) -> Value {
    length_mm(v).map_or(Value::Null, |x| json!((x * 1.0e6).round() / 1.0e6))
}

fn condition(m: MaterialCondition) -> &'static str {
    match m {
        MaterialCondition::Rfs => "none",
        MaterialCondition::Mmc => "mmc",
        MaterialCondition::Lmc => "lmc",
    }
}

fn condition_suffix(m: MaterialCondition) -> &'static str {
    match m {
        MaterialCondition::Rfs => "",
        MaterialCondition::Mmc => "(M)",
        MaterialCondition::Lmc => "(L)",
    }
}

/// The characteristic as the form's choice writes it.
pub fn characteristic_word(c: runmat_cad::pmi::Characteristic) -> String {
    let debug = format!("{c:?}");
    let mut out = String::new();
    for (i, ch) in debug.chars().enumerate() {
        if ch.is_ascii_uppercase() && i > 0 {
            out.push('_');
        }
        out.push(ch.to_ascii_lowercase());
    }
    out
}

/// The command and its values that make `a` as it is written.
pub fn values(doc: &PartDocument, a: &ModelAnnotation) -> Option<(&'static str, Values)> {
    let pmi = &doc.pmi;
    let mut v = Values::new();
    let command = match &a.kind {
        AnnotationKind::DatumSymbol { label } => {
            let d = pmi.datums.iter().find(|d| &d.label == label)?;
            v.insert("label".into(), json!(label));
            v.insert("target".into(), picks(std::slice::from_ref(&d.on)));
            "annotate.datum"
        }
        AnnotationKind::Gdt { tolerance } => {
            let t = pmi.tolerances.iter().find(|t| &t.name == tolerance)?;
            v.insert(
                "characteristic".into(),
                json!(characteristic_word(t.characteristic)),
            );
            v.insert("tolerance".into(), mm(&t.tolerance));
            v.insert("diametral".into(), json!(t.diametral));
            v.insert("modifier".into(), json!(condition(t.modifier)));
            let datums: Vec<String> = t
                .datums
                .iter()
                .map(|d| format!("{}{}", d.label, condition_suffix(d.modifier)))
                .collect();
            v.insert("datums".into(), json!(datums.join("|")));
            v.insert("projected".into(), json!(t.projected.is_some()));
            if let Some(p) = &t.projected {
                v.insert("projected_height".into(), mm(p));
            }
            v.insert("targets".into(), picks(&t.on));
            "annotate.gdt"
        }
        AnnotationKind::Dimension {
            references,
            pmi: name,
            ..
        } => {
            v.insert("targets".into(), picks(references));
            let name = name.as_deref()?;
            let (value, fit) = match pmi.sizes.iter().find(|s| s.name == name) {
                Some(s) => (&s.size, s.fit.clone()),
                None => (
                    &pmi.locations.iter().find(|l| l.name == name)?.distance,
                    None,
                ),
            };
            match (value, fit) {
                (_, Some(f)) => {
                    v.insert("tolerance".into(), json!("fit"));
                    v.insert("fit".into(), json!(f));
                }
                (ValueExpr::Toleranced { plus, minus, .. }, None) => {
                    let (p, m) = (length_mm(plus)?, length_mm(minus)?);
                    let kind = if (p - m).abs() < 1e-9 {
                        "symmetric"
                    } else {
                        "bilateral"
                    };
                    v.insert("tolerance".into(), json!(kind));
                    v.insert("plus".into(), mm(plus));
                    v.insert("minus".into(), mm(minus));
                }
                _ => {
                    v.insert("tolerance".into(), json!("none"));
                }
            }
            "annotate.dimension"
        }
        AnnotationKind::SurfaceFinish { on } => {
            let f = pmi.finishes.iter().find(|f| f.on.contains(on))?;
            v.insert("faces".into(), picks(&f.on));
            // Ra in micrometres, the field's unit.
            v.insert(
                "ra".into(),
                length_mm(&f.ra).map_or(Value::Null, |x| json!((x * 1.0e6).round() / 1.0e3)),
            );
            let removal = match f.removal {
                Removal::Any => "any",
                Removal::Required => "removal_required",
                Removal::Prohibited => "removal_prohibited",
            };
            v.insert("removal".into(), json!(removal));
            v.insert(
                "method".into(),
                json!(f.process.clone().unwrap_or_default()),
            );
            v.insert(
                "lay".into(),
                json!(f.lay.clone().unwrap_or_else(|| "none".into())),
            );
            "annotate.surface-finish"
        }
        AnnotationKind::Coating { treatment } => {
            let t = pmi.treatments.iter().find(|t| &t.name == treatment)?;
            v.insert("faces".into(), picks(&t.on));
            v.insert("spec".into(), json!(t.specification));
            "annotate.coating"
        }
        AnnotationKind::Weld {
            on,
            symbol,
            size,
            all_around,
            field,
            side,
            length,
            pitch,
            tail,
        } => {
            v.insert("edges".into(), picks(std::slice::from_ref(on)));
            v.insert("type".into(), json!(symbol));
            v.insert("size".into(), size.as_ref().map_or(json!(0), mm));
            v.insert("all_around".into(), json!(all_around));
            v.insert("field".into(), json!(field));
            let side = match side {
                WeldSide::Arrow => "arrow",
                WeldSide::Other => "other",
                WeldSide::Both => "both",
            };
            v.insert("side".into(), json!(side));
            v.insert("intermittent".into(), json!(length.is_some()));
            if let (Some(l), Some(p)) = (length, pitch) {
                v.insert("length".into(), mm(l));
                v.insert("pitch".into(), mm(p));
            }
            v.insert("tail".into(), json!(tail.clone().unwrap_or_default()));
            "annotate.weld-symbol"
        }
        AnnotationKind::Note { text, leader, .. } => {
            v.insert("text".into(), json!(text));
            v.insert(
                "anchor".into(),
                picks(leader.as_ref().map(std::slice::from_ref).unwrap_or(&[])),
            );
            "annotate.note"
        }
        _ => return None,
    };
    Some((command, v))
}
