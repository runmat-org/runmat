//! Each Annotate command's form as the PMI it declares and the annotations that show it.
//!
//! GD&T semantics follow the public summaries of ASME Y14.5 and ISO 1101: form tolerances
//! (straightness, flatness, circularity, cylindricity) take no datums; orientation, location and
//! runout tolerances need a datum reference frame; a material condition applies to features of
//! size under straightness, the orientation tolerances and position. Weld symbols follow AWS
//! A2.4 and ISO 2553: the weld's symbol, size and length on the reference line, the arrow side
//! below it and the other side above, an all-around circle and a field-weld flag at the bend,
//! an intermittent weld's length and pitch, and the tail for the process.

use crate::drawing::values;
use crate::drawing_edits::pmi::{self as shared, Frame, Tolerance};
use crate::drawing_edits::{flag, number, text, Values};
use crate::edit::model;
use crate::features::picks_of;
use crate::picks::Picks;
use runmat_cad::drawing::{AnnotationKind, ArcCondition, DimensionStyle, WeldSide};
use runmat_cad::pmi::{DatumReference, MaterialCondition, PmiSet, Removal, Treatment};
use runmat_cad::refs::reference::Reference;
use runmat_cad::syntax::edit::EditFailure;
use runmat_cad::syntax::refs::print_reference;
use runmat_cad::value::ValueExpr;
use serde_json::{json, Value};

/// A pick field's picks as model references.
pub fn references(
    picks: &Picks,
    values: &Values,
    field: &str,
) -> Result<Vec<Reference>, EditFailure> {
    picks_of(values, field)
        .iter()
        .map(|p| {
            let v = picks.reference(p)?;
            serde_json::from_value(v).map_err(|e| model(format!("`{p}`: {e}")))
        })
        .collect()
}

/// The datums a frame names, `A|B(M)|C` (or with commas or spaces): each label with its
/// material condition.
pub fn datum_frame(words: &str) -> Vec<DatumReference> {
    words
        .split(['|', ',', ' '])
        .map(str::trim)
        .filter(|w| !w.is_empty())
        .map(|w| {
            let (label, modifier) = shared::condition(w);
            DatumReference {
                label: label.to_uppercase(),
                modifier,
            }
        })
        .collect()
}

fn condition(word: &str) -> MaterialCondition {
    match word {
        "mmc" => MaterialCondition::Mmc,
        "lmc" => MaterialCondition::Lmc,
        _ => MaterialCondition::Rfs,
    }
}

/// An optional text field, trimmed; `None` when empty.
fn optional(values: &Values, id: &str) -> Option<String> {
    let t = text(values, id, "").trim();
    (!t.is_empty()).then(|| t.to_owned())
}

/// The part's display as the drawing's measuring helpers read it (`{ data: { bodies } }`).
pub fn display(picks: &Picks) -> Value {
    json!({ "kind": "part", "data": { "bodies": picks.bodies() } })
}

/// Declares the PMI `command`'s values ask for in `pmi` (under `name` when an edit keeps it),
/// answering the annotations that show it.
pub fn declare(
    command: &str,
    values: &Values,
    picks: &Picks,
    pmi: &mut PmiSet,
    name: Option<&str>,
) -> Result<Vec<AnnotationKind>, EditFailure> {
    Ok(match command {
        "annotate.datum" => {
            let label = text(values, "label", "A").trim().to_uppercase();
            let on = references(picks, values, "target")?
                .into_iter()
                .next()
                .ok_or_else(|| model(format!("pick the face or axis to label datum {label}")))?;
            shared::declare_datum(pmi, &label, &on)?;
            vec![AnnotationKind::DatumSymbol { label }]
        }
        "annotate.gdt" => {
            let on = references(picks, values, "targets")?;
            if on.is_empty() {
                return Err(model("pick the faces the frame applies to"));
            }
            let characteristic =
                shared::characteristic(text(values, "characteristic", "position"))?;
            let projected =
                flag(values, "projected", false).then(|| number(values, "projected_height", 10.0));
            let frame = Frame {
                characteristic,
                zone: number(values, "tolerance", 0.1),
                diametral: flag(values, "diametral", false),
                modifier: condition(text(values, "modifier", "none")),
                datums: datum_frame(text(values, "datums", "")),
                projected,
            };
            let tolerance = shared::declare_tolerance(pmi, frame, on, name)?;
            vec![AnnotationKind::Gdt { tolerance }]
        }
        "annotate.dimension" => {
            let references = references(picks, values, "targets")?;
            let (style, nominal) = measure(&display(picks), &references)?;
            let tolerance = match text(values, "tolerance", "none") {
                "symmetric" => {
                    let plus = number(values, "plus", 0.1);
                    Some(Tolerance::Deviation { plus, minus: plus })
                }
                "bilateral" | "limit" => Some(Tolerance::Deviation {
                    plus: number(values, "plus", 0.1),
                    minus: number(values, "minus", 0.1),
                }),
                "fit" => Some(Tolerance::Fit(text(values, "fit", "H7").trim().to_owned())),
                _ => None,
            };
            if let Some(Tolerance::Deviation { plus, minus }) = &tolerance {
                if *plus < 0.0 || *minus < 0.0 {
                    return Err(model(
                        "write the plus and minus deviations as sizes (0.1), without signs",
                    ));
                }
            }
            let size =
                shared::declare_dimension(pmi, &references, nominal, tolerance.as_ref(), name)?;
            vec![AnnotationKind::Dimension {
                style,
                references,
                pmi: Some(size),
                group: None,
                reference: false,
                condition: ArcCondition::Center,
                occurrences: Vec::new(),
            }]
        }
        "annotate.surface-finish" => {
            let on = references(picks, values, "faces")?;
            let first = on
                .first()
                .cloned()
                .ok_or_else(|| model("pick the faces the finish applies to"))?;
            let removal = match text(values, "removal", "any") {
                "removal_required" | "required" => Removal::Required,
                "removal_prohibited" | "prohibited" => Removal::Prohibited,
                _ => Removal::Any,
            };
            let lay = optional(values, "lay").filter(|l| l != "none");
            shared::declare_finish(
                pmi,
                on,
                number(values, "ra", 1.6) / 1000.0,
                removal,
                optional(values, "method"),
                lay,
                name,
            )?;
            vec![AnnotationKind::SurfaceFinish { on: first }]
        }
        "annotate.coating" => {
            let specification = optional(values, "spec")
                .ok_or_else(|| model("write the coating's specification"))?;
            let on = references(picks, values, "faces")?;
            let treatment = match name {
                Some(n) => n.to_owned(),
                None => {
                    let names: Vec<&str> = pmi.treatments.iter().map(|t| t.name.as_str()).collect();
                    shared::fresh(&names, "coating")
                }
            };
            pmi.treatments.push(Treatment {
                name: treatment.clone(),
                on,
                specification,
            });
            vec![AnnotationKind::Coating { treatment }]
        }
        "annotate.weld-symbol" => {
            let edges = references(picks, values, "edges")?;
            if edges.is_empty() {
                return Err(model("pick the edges to weld"));
            }
            let symbol = text(values, "type", "fillet").to_owned();
            if !["fillet", "groove", "plug", "spot"].contains(&symbol.as_str()) {
                return Err(model(format!("there is no `{symbol}` weld")));
            }
            let side = match text(values, "side", "arrow") {
                "other" => WeldSide::Other,
                "both" => WeldSide::Both,
                _ => WeldSide::Arrow,
            };
            if side == WeldSide::Both && (symbol == "plug" || symbol == "spot") {
                return Err(model(format!(
                    "a {symbol} weld is on one side of the joint"
                )));
            }
            let size = number(values, "size", 4.0);
            let intermittent = flag(values, "intermittent", false);
            let (length, pitch) = if intermittent {
                let (l, p) = (
                    number(values, "length", 50.0),
                    number(values, "pitch", 100.0),
                );
                if l <= 0.0 || p < l {
                    return Err(model("an intermittent weld's pitch is at least its length"));
                }
                (Some(ValueExpr::mm(l)), Some(ValueExpr::mm(p)))
            } else {
                (None, None)
            };
            let all_around = flag(values, "all_around", false);
            let field = flag(values, "field", false);
            let tail = optional(values, "tail");
            edges
                .into_iter()
                .map(|on| AnnotationKind::Weld {
                    on,
                    symbol: symbol.clone(),
                    size: (size > 0.0).then(|| ValueExpr::mm(size)),
                    all_around,
                    field,
                    side,
                    length: length.clone(),
                    pitch: pitch.clone(),
                    tail: tail.clone(),
                })
                .collect()
        }
        "annotate.note" => {
            let body = text(values, "text", "").trim().to_owned();
            if body.is_empty() {
                return Err(model("write the note's text"));
            }
            let leader = references(picks, values, "anchor")?.into_iter().next();
            vec![AnnotationKind::Note {
                text: body,
                leader,
                width: None,
            }]
        }
        other => return Err(model(format!("`{other}` isn't an Annotate command"))),
    })
}

/// What one or two picked faces measure: a cylinder's diameter, or the distance between two
/// parallel planes, a plane and a hole's axis, or two holes' axes (mm).
pub fn measure(
    display: &Value,
    references: &[Reference],
) -> Result<(DimensionStyle, f64), EditFailure> {
    let faces: Vec<_> = references
        .iter()
        .map(|r| {
            print_reference(r)
                .and_then(|t| values::face(display, &t))
                .ok_or_else(|| model("pick faces: a hole for its diameter, or two faces for the distance between them"))
        })
        .collect::<Result<_, _>>()?;
    match faces.as_slice() {
        [] => Err(model(
            "pick a hole for its diameter, or two faces for the distance between them",
        )),
        [(_, s)] => values::cylinder(s)
            .map(|(_, _, r)| (DimensionStyle::Diameter, 2.0 * r))
            .ok_or_else(|| {
                model("one face measures a diameter: pick a hole or a boss, or a second face")
            }),
        [a, b] => {
            if values::angle(a, b).is_some_and(|deg| deg > 0.5 && deg < 179.5) {
                return Err(model("the faces aren't parallel: pick two parallel faces, a face and a hole, or two holes"));
            }
            values::linear(a, b)
                .map(|d| (DimensionStyle::Linear, d))
                .ok_or_else(|| {
                    model("the picks don't measure a distance; pick parallel faces or holes")
                })
        }
        _ => Err(model("a dimension measures one face or two")),
    }
}
