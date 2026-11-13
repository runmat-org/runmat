//! Annotations placed by hand on a view: dimensions (linear, angular, radius, diameter,
//! ordinate, baseline and chain, with a tolerance), hole callouts, GD&T frames, datum symbols
//! and targets, surface finishes, weld symbols, notes, balloons, center marks and centerlines.
//!
//! Each names the model faces it measures or points at by reference, so it follows the model
//! through edits. Tolerances, datums and finishes are declared in the part's PMI ([`super::pmi`]).

use super::pmi::{self, Tolerance};
use super::*;
use crate::drawing::values;
use runmat_cad::drawing::{Annotation, AnnotationKind, ArcCondition, DimensionStyle, TableKind};
use runmat_cad::value::ValueExpr;

/// The view the annotation goes on: the picks' view, else the `view` field, else the
/// drawing's only view.
fn view_of(d: &Drawing, values: &Values, fields: &[&str]) -> Result<String, EditFailure> {
    for field in fields.iter().chain(["view"].iter()) {
        if let Some(v) = view_pick(values, field)? {
            d.view(&v)?;
            return Ok(v);
        }
    }
    match d.document.views.as_slice() {
        [only] => Ok(only.name.clone()),
        [] => Err(model("place a view first")),
        _ => Err(model("pick in the view the annotation goes on")),
    }
}

fn refs(values: &Values, field: &str) -> Result<Vec<Reference>, EditFailure> {
    Ok(edges(values, field)?
        .into_iter()
        .map(|(_, r, _)| r)
        .collect())
}

fn one(values: &Values, field: &str, what: &str) -> Result<Reference, EditFailure> {
    refs(values, field)?
        .into_iter()
        .next()
        .ok_or_else(|| model(format!("pick {what}")))
}

/// The feature an `attach` pick names: a model edge's face, or a dimension's first reference.
fn attached(
    d: &Drawing,
    values: &Values,
    what: &str,
) -> Result<(Reference, Option<String>), EditFailure> {
    for p in picks(values, "attach")? {
        match p {
            SheetPick::Edge {
                reference, view, ..
            } => return Ok((reference, view)),
            SheetPick::Annotation(id) => {
                let a = d
                    .document
                    .annotations
                    .iter()
                    .find(|a| a.id == id)
                    .ok_or_else(|| model(format!("the drawing has no annotation `{id}`")))?;
                let on = match &a.kind {
                    AnnotationKind::Dimension { references, .. } => references.first().cloned(),
                    AnnotationKind::HoleCallout { hole } => Some(hole.clone()),
                    _ => None,
                };
                return on
                    .map(|r| (r, Some(a.view.clone())))
                    .ok_or_else(|| model(format!("attach {what} to a dimension or a model edge")));
            }
            _ => {}
        }
    }
    Err(model(format!("pick what {what} goes on")))
}

fn at(values: &Values) -> Result<[f64; 2], EditFailure> {
    Ok(points(values, "at")?
        .into_iter()
        .next()
        .unwrap_or([0.0, 0.0]))
}

fn style(word: &str) -> Option<DimensionStyle> {
    Some(match word {
        "linear" => DimensionStyle::Linear,
        "angular" => DimensionStyle::Angular,
        "radius" => DimensionStyle::Radius,
        "diameter" => DimensionStyle::Diameter,
        _ => return None,
    })
}

type Face<'v> = (Vec<[f64; 3]>, &'v Value);

fn face<'v>(display: &'v Value, r: &Reference) -> Option<Face<'v>> {
    values::face(display, &runmat_cad::syntax::refs::print_reference(r)?)
}

/// What one or two picked faces measure when the form leaves it to the picks: a cylinder its
/// diameter, two planes at an angle that angle, anything else the distance.
fn inferred(display: Option<&Value>, references: &[Reference]) -> DimensionStyle {
    let faces: Vec<Face> = display
        .map(|d| references.iter().filter_map(|r| face(d, r)).collect())
        .unwrap_or_default();
    match faces.as_slice() {
        [(_, s)] if values::cylinder(s).is_some() => DimensionStyle::Diameter,
        [a, b] if values::angle(a, b).is_some_and(|deg| deg > 0.5 && deg < 179.5) => {
            DimensionStyle::Angular
        }
        _ => DimensionStyle::Linear,
    }
}

/// The value a dimension measures (mm), for its tolerance's nominal.
fn measured(
    display: Option<&Value>,
    style: DimensionStyle,
    references: &[Reference],
) -> Result<f64, EditFailure> {
    let faces: Vec<Face> = display
        .map(|d| references.iter().filter_map(|r| face(d, r)).collect())
        .unwrap_or_default();
    let none =
        || model("the picks don't measure on this model; pick faces seen edge-on, or a hole");
    match (style, faces.as_slice()) {
        (DimensionStyle::Diameter, [(_, s)]) => values::cylinder(s)
            .map(|(_, _, r)| 2.0 * r)
            .ok_or_else(none),
        (DimensionStyle::Radius, [(_, s)]) => {
            values::cylinder(s).map(|(_, _, r)| r).ok_or_else(none)
        }
        (DimensionStyle::Linear | DimensionStyle::Aligned, [a, b]) => {
            values::linear(a, b).ok_or_else(none)
        }
        _ => Err(model(
            "a tolerance goes on a linear, radius or diameter dimension",
        )),
    }
}

fn dimension(
    d: &mut Drawing,
    view: &str,
    id: String,
    style: DimensionStyle,
    references: Vec<Reference>,
    pmi: Option<String>,
    at: [f64; 2],
) {
    d.document.annotations.push(Annotation {
        id,
        view: view.to_owned(),
        kind: AnnotationKind::Dimension {
            style,
            references,
            pmi,
            group: None,
            reference: false,
            condition: ArcCondition::Center,
        },
        at,
    });
}

fn add(d: &mut Drawing, view: &str, stem: &str, kind: AnnotationKind, at: [f64; 2]) -> String {
    let id = fresh(&d.document, stem);
    d.document.annotations.push(Annotation {
        id: id.clone(),
        view: view.to_owned(),
        kind,
        at,
    });
    id
}

const COMMANDS: &[&str] = &[
    "drawing.smart-dimension",
    "drawing.ordinate",
    "drawing.baseline",
    "drawing.chain",
    "drawing.hole-callout",
    "drawing.gdt",
    "drawing.datum",
    "drawing.datum-target",
    "drawing.surface-finish",
    "drawing.weld-symbol",
    "drawing.note",
    "drawing.balloon",
    "drawing.center-mark",
    "drawing.centerline",
];

pub fn edit(
    path: &str,
    text_: &str,
    files: &Files,
    command: &str,
    values: &Values,
) -> Option<Result<(Vec<FileEdit>, String), EditFailure>> {
    if !COMMANDS.contains(&command) {
        return None;
    }
    Some((|| {
        let mut d = Drawing::open(path, text_, files)?;
        // Changes to the drawn part (its PMI), written with the drawing's.
        let mut others: Vec<FileEdit> = Vec::new();
        let summary = match command {
            "drawing.smart-dimension" => {
                let view = view_of(&d, values, &["items"])?;
                let references = refs(values, "items")?;
                if references.is_empty() || references.len() > 2 {
                    return Err(model(
                        "pick one face (a hole's size) or two (the distance or angle between them)",
                    ));
                }
                let display = d.display(&view);
                let style = match style(text(values, "measure", "auto")) {
                    Some(s) => s,
                    None => inferred(display.as_ref(), &references),
                };
                let (lo, hi) = style.arity();
                if references.len() < lo || references.len() > hi {
                    return Err(model(
                        format!(
                            "a {style:?} dimension takes {lo} pick{}",
                            if lo == 1 { "" } else { "s" }
                        )
                        .to_lowercase(),
                    ));
                }
                let tolerance = match text(values, "tolerance", "none") {
                    "none" => None,
                    "symmetric" => {
                        let t = number(values, "plus", 0.1).abs();
                        Some(Tolerance::Deviation { plus: t, minus: t })
                    }
                    "bilateral" => Some(Tolerance::Deviation {
                        plus: number(values, "plus", 0.1).abs(),
                        minus: number(values, "minus", 0.1).abs(),
                    }),
                    "fit" => Some(Tolerance::Fit(text(values, "fit", "H7").trim().to_owned())),
                    other => return Err(model(format!("there is no `{other}` tolerance"))),
                };
                let pmi = match tolerance {
                    None => None,
                    Some(t) => {
                        let nominal = measured(display.as_ref(), style, &references)?;
                        let (name, edit) =
                            pmi::add_dimension_tolerance(&d, &view, &references, nominal, t)?;
                        others.push(edit);
                        Some(name)
                    }
                };
                let stem = match style {
                    DimensionStyle::Diameter => "dia",
                    DimensionStyle::Radius => "rad",
                    DimensionStyle::Angular => "angle",
                    _ => "dim",
                };
                let id = fresh(&d.document, stem);
                dimension(
                    &mut d,
                    &view,
                    id.clone(),
                    style,
                    references,
                    pmi,
                    at(values)?,
                );
                format!("Added dimension {id}")
            }
            "drawing.ordinate" | "drawing.baseline" => {
                let (zero_field, stem) = if command == "drawing.ordinate" {
                    ("origin", "ordinate")
                } else {
                    ("base", "baseline")
                };
                let view = view_of(&d, values, &[zero_field, "items"])?;
                let zero = one(
                    values,
                    zero_field,
                    if command == "drawing.ordinate" {
                        "the zero edge"
                    } else {
                        "the baseline"
                    },
                )?;
                let items = refs(values, "items")?;
                if items.is_empty() {
                    return Err(model("pick the edges to dimension"));
                }
                let n = items.len();
                let place = at(values)?;
                if command == "drawing.ordinate" {
                    // One per feature, each from the zero.
                    let mut ids = Vec::new();
                    for item in items {
                        let id = fresh(&d.document, stem);
                        dimension(
                            &mut d,
                            &view,
                            id.clone(),
                            DimensionStyle::Ordinate,
                            vec![zero.clone(), item],
                            None,
                            place,
                        );
                        ids.push(id);
                    }
                    format!("Added ordinate dimensions {}", ids.join(", "))
                } else {
                    let mut references = vec![zero];
                    references.extend(items);
                    let id = fresh(&d.document, stem);
                    dimension(
                        &mut d,
                        &view,
                        id.clone(),
                        DimensionStyle::Baseline,
                        references,
                        None,
                        place,
                    );
                    format!(
                        "Added baseline {id} ({n} dimension{})",
                        if n == 1 { "" } else { "s" }
                    )
                }
            }
            "drawing.chain" => {
                let view = view_of(&d, values, &["items"])?;
                let references = refs(values, "items")?;
                if references.len() < 2 {
                    return Err(model("pick at least two edges to chain"));
                }
                let id = fresh(&d.document, "chain");
                dimension(
                    &mut d,
                    &view,
                    id.clone(),
                    DimensionStyle::Chain,
                    references,
                    None,
                    at(values)?,
                );
                format!("Added chain {id}")
            }
            "drawing.hole-callout" => {
                let view = view_of(&d, values, &["hole"])?;
                let hole = one(values, "hole", "a hole")?;
                if let Some(display) = d.display(&view) {
                    if face(&display, &hole).is_none_or(|(_, s)| values::cylinder(s).is_none()) {
                        return Err(model("pick a hole (its rim in the view)"));
                    }
                }
                let id = add(
                    &mut d,
                    &view,
                    "hole",
                    AnnotationKind::HoleCallout { hole },
                    at(values)?,
                );
                format!("Added hole callout {id}")
            }
            "drawing.gdt" => {
                let (on, v) = attached(&d, values, "the frame")?;
                let view = match v {
                    Some(v) => v,
                    None => view_of(&d, values, &[])?,
                };
                let (name, edit) = pmi::add_tolerance(&d, &view, on, text(values, "frame", ""))?;
                others.push(edit);
                let id = add(
                    &mut d,
                    &view,
                    &name,
                    AnnotationKind::Gdt {
                        tolerance: name.clone(),
                    },
                    at(values)?,
                );
                format!("Added GD&T frame {id}")
            }
            "drawing.datum" => {
                let (on, v) = attached(&d, values, "the datum")?;
                let view = match v {
                    Some(v) => v,
                    None => view_of(&d, values, &[])?,
                };
                let label = text(values, "label", "A").trim().to_uppercase();
                others.extend(pmi::add_datum(&d, &view, on, &label)?);
                let id = add(
                    &mut d,
                    &view,
                    &format!("datum_{label}"),
                    AnnotationKind::DatumSymbol {
                        label: label.clone(),
                    },
                    at(values)?,
                );
                format!("Added datum {label} ({id})")
            }
            "drawing.datum-target" => {
                let (on, v) = attached(&d, values, "the target")?;
                let view = match v {
                    Some(v) => v,
                    None => view_of(&d, values, &[])?,
                };
                let label = text(values, "label", "A1").trim().to_uppercase();
                others.push(pmi::add_target(
                    &d,
                    &view,
                    on,
                    &label,
                    text(values, "shape", "point"),
                    number(values, "size", 5.0),
                )?);
                let id = add(
                    &mut d,
                    &view,
                    &format!("target_{label}"),
                    AnnotationKind::DatumTarget {
                        label: label.clone(),
                    },
                    at(values)?,
                );
                format!("Added datum target {label} ({id})")
            }
            "drawing.surface-finish" => {
                let (on, v) = attached(&d, values, "the finish")?;
                let view = match v {
                    Some(v) => v,
                    None => view_of(&d, values, &[])?,
                };
                others.push(pmi::add_finish(
                    &d,
                    &view,
                    on.clone(),
                    number(values, "ra", 1.6),
                )?);
                let id = add(
                    &mut d,
                    &view,
                    "finish",
                    AnnotationKind::SurfaceFinish { on },
                    at(values)?,
                );
                format!("Added surface finish {id}")
            }
            "drawing.weld-symbol" => {
                let (on, v) = attached(&d, values, "the weld symbol")?;
                let view = match v {
                    Some(v) => v,
                    None => view_of(&d, values, &[])?,
                };
                let symbol = text(values, "type", "fillet").to_owned();
                if !["fillet", "groove", "plug", "spot"].contains(&symbol.as_str()) {
                    return Err(model(format!("there is no `{symbol}` weld")));
                }
                let size = number(values, "size", 4.0);
                let kind = AnnotationKind::Weld {
                    on,
                    symbol,
                    size: (size > 0.0).then(|| ValueExpr::mm(size)),
                    all_around: flag(values, "all_around", false),
                    field: flag(values, "field", false),
                };
                let id = add(&mut d, &view, "weld", kind, at(values)?);
                format!("Added weld symbol {id}")
            }
            "drawing.note" => {
                let body = text(values, "text", "").trim().to_owned();
                if body.is_empty() {
                    return Err(model("write the note's text"));
                }
                let leader = refs(values, "attach")
                    .ok()
                    .and_then(|r| r.into_iter().next());
                let mut place = at(values)?;
                // A note goes with the view it points at, else the view nearest it.
                let view = match view_of(&d, values, &["attach"]) {
                    Ok(v) => v,
                    Err(_) => d
                        .document
                        .views
                        .iter()
                        .min_by(|a, b| {
                            let dist = |v: &runmat_cad::drawing::View| {
                                (v.at[0] - place[0]).powi(2) + (v.at[1] - place[1]).powi(2)
                            };
                            dist(a).total_cmp(&dist(b))
                        })
                        .map(|v| v.name.clone())
                        .ok_or_else(|| model("place a view first"))?,
                };
                if place == [0.0, 0.0] && leader.is_none() {
                    // Above the title block on the right, where general notes go (ISO 7200).
                    place = [300.0, 70.0];
                }
                let id = add(
                    &mut d,
                    &view,
                    "note",
                    AnnotationKind::Note { text: body, leader },
                    place,
                );
                format!("Added note {id}")
            }
            "drawing.balloon" => {
                let view = view_of(&d, values, &["attach"])?;
                let occurrence = picks(values, "attach")?
                    .into_iter()
                    .find_map(|p| match p {
                        SheetPick::Edge { text, .. } => {
                            text.strip_prefix("occurrence:").map(str::to_owned)
                        }
                        _ => None,
                    })
                    .ok_or_else(|| model("pick a component of an assembly view"))?;
                let occurrence: runmat_cad::assembly::model::RelPath =
                    serde_json::from_value(Value::String(occurrence.clone()))
                        .map_err(|_| model(format!("`{occurrence}` isn't a component path")))?;
                let table = d
                    .document
                    .tables
                    .iter()
                    .find(|t| matches!(t.kind, TableKind::Bom { .. }))
                    .map(|t| t.name.clone())
                    .ok_or_else(|| {
                        model("add a BOM table first: balloons take its item numbers")
                    })?;
                let id = add(
                    &mut d,
                    &view,
                    "balloon",
                    AnnotationKind::Balloon { occurrence, table },
                    at(values)?,
                );
                format!("Added balloon {id}")
            }
            "drawing.center-mark" => {
                let view = view_of(&d, values, &["edges"])?;
                let mut circles = refs(values, "edges")?;
                if flag(values, "auto", false) {
                    let display = d
                        .display(&view)
                        .ok_or_else(|| model("marks every hole of part views"))?;
                    for h in crate::drawing::auto::holes(&display) {
                        circles.push(reference(&h)?);
                    }
                }
                circles.retain(|c| {
                    !d.document.annotations.iter().any(|a| {
                        a.view == view
                            && matches!(&a.kind, AnnotationKind::CenterMark { on } if on == c)
                    })
                });
                circles.dedup();
                if circles.is_empty() {
                    return Err(model(
                        "pick the circles to mark (or mark all holes in the view)",
                    ));
                }
                let n = circles.len();
                for on in circles {
                    add(
                        &mut d,
                        &view,
                        "center",
                        AnnotationKind::CenterMark { on },
                        [0.0, 0.0],
                    );
                }
                format!("Added {n} center mark{}", if n == 1 { "" } else { "s" })
            }
            "drawing.centerline" => {
                let view = view_of(&d, values, &["edges"])?;
                let between = refs(values, "edges")?;
                if between.is_empty() || between.len() > 2 {
                    return Err(model(
                        "pick a cylinder, or two edges to draw the centerline between",
                    ));
                }
                let id = add(
                    &mut d,
                    &view,
                    "centerline",
                    AnnotationKind::Centerline { between },
                    [0.0, 0.0],
                );
                format!("Added centerline {id}")
            }
            _ => unreachable!(),
        };
        let mut out = vec![(path.to_owned(), text_.to_owned(), d.change()?)];
        out.extend(others);
        Ok((out, summary))
    })())
}
