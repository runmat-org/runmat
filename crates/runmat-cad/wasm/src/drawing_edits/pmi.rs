//! Tolerances, datums, datum targets and finishes placed on the sheet are declared in the drawn
//! part's PMI, by the face they apply to, and the drawing shows them by name. The part stays
//! the one place a tolerance is written: its other drawings, its model view and the exchange
//! files read the same declaration.

use super::*;
use runmat_cad::pmi::{
    Characteristic, DatumFeature, DatumReference, DatumTarget, GeometricTolerance,
    LocationTolerance, MaterialCondition, PmiSet, Removal, SizeTolerance, SurfaceFinish,
    TargetShape,
};
use runmat_cad::syntax::edit::edit_part;
use runmat_cad::syntax::parse_part;
use runmat_cad::value::ValueExpr;

/// The part a view draws, as a file to edit: its path and text.
pub fn part_of<'f>(d: &'f Drawing, view: &str) -> Result<(String, &'f String), EditFailure> {
    let file = d.view(view)?.model.file.as_str().to_owned();
    if !file.ends_with(".part") {
        return Err(model(
            "tolerances, datums and finishes go on part views; this view draws an assembly",
        ));
    }
    let text = d
        .files
        .get(&file)
        .ok_or_else(|| model(format!("{file} isn't in the workspace")))?;
    Ok((file, text))
}

pub fn fresh(names: &[&str], stem: &str) -> String {
    (1..)
        .map(|n| format!("{stem}{n}"))
        .find(|n| !names.contains(&n.as_str()))
        .expect("a free name")
}

/// Edits the part's PMI with `add`, which answers the name it declared.
fn edit_pmi(
    file: &str,
    text: &str,
    add: impl Fn(&mut runmat_cad::pmi::PmiSet) -> Result<String, EditFailure>,
) -> Result<(String, FileEdit), EditFailure> {
    let mut name = String::new();
    let change = edit_part(text, file, |document| {
        name = add(&mut document.pmi)?;
        Ok(())
    })?;
    Ok((name, (file.to_owned(), text.to_owned(), change)))
}

pub fn characteristic(word: &str) -> Result<Characteristic, EditFailure> {
    let w = word.trim().to_lowercase().replace([' ', '-'], "_");
    Ok(match w.as_str() {
        "straightness" => Characteristic::Straightness,
        "flatness" => Characteristic::Flatness,
        "circularity" | "roundness" => Characteristic::Circularity,
        "cylindricity" => Characteristic::Cylindricity,
        "profile_of_line" | "line_profile" => Characteristic::ProfileOfLine,
        "profile_of_surface" | "profile" | "surface_profile" => Characteristic::ProfileOfSurface,
        "parallelism" | "parallel" => Characteristic::Parallelism,
        "perpendicularity" | "perpendicular" => Characteristic::Perpendicularity,
        "angularity" => Characteristic::Angularity,
        "position" | "true_position" => Characteristic::Position,
        "concentricity" => Characteristic::Concentricity,
        "symmetry" => Characteristic::Symmetry,
        "circular_runout" | "runout" => Characteristic::CircularRunout,
        "total_runout" => Characteristic::TotalRunout,
        other => {
            return Err(model(format!(
                "there is no `{other}` tolerance characteristic"
            )))
        }
    })
}

pub fn condition(word: &str) -> (String, MaterialCondition) {
    let w = word.trim();
    for (suffix, m) in [
        ("(M)", MaterialCondition::Mmc),
        ("Ⓜ", MaterialCondition::Mmc),
        ("(L)", MaterialCondition::Lmc),
        ("Ⓛ", MaterialCondition::Lmc),
    ] {
        if let Some(rest) = w.strip_suffix(suffix) {
            return (rest.trim().to_owned(), m);
        }
    }
    (w.to_owned(), MaterialCondition::Rfs)
}

/// A frame as the form writes it, `position|⌀0.1(M)|A|B|C`: characteristic, zone (`⌀` for a
/// diametral zone, `(M)`/`(L)` for a material condition), then the datums in order.
pub fn frame(
    text: &str,
) -> Result<
    (
        Characteristic,
        f64,
        bool,
        MaterialCondition,
        Vec<DatumReference>,
    ),
    EditFailure,
> {
    let cells: Vec<&str> = text
        .split('|')
        .map(str::trim)
        .filter(|c| !c.is_empty())
        .collect();
    let [kind, zone, datums @ ..] = cells.as_slice() else {
        return Err(model(
            "write the frame as characteristic|zone|datums (position|0.1|A|B)",
        ));
    };
    let c = characteristic(kind)?;
    let (zone, modifier) = condition(zone);
    let diametral = zone.starts_with('⌀') || zone.starts_with("dia");
    let value: f64 = zone
        .trim_start_matches('⌀')
        .trim_start_matches("dia")
        .trim()
        .parse()
        .map_err(|_| model(format!("`{zone}` isn't a tolerance zone (a number in mm)")))?;
    if value <= 0.0 {
        return Err(model("the tolerance zone must be more than zero"));
    }
    let datums: Vec<DatumReference> = datums
        .iter()
        .map(|d| {
            let (label, modifier) = condition(d);
            DatumReference {
                label: label.to_uppercase(),
                modifier,
            }
        })
        .collect();
    if c.needs_datum() && datums.is_empty() {
        return Err(model(format!("a {kind} tolerance needs a datum")));
    }
    Ok((c, value, diametral, modifier, datums))
}

/// Declares a geometric tolerance on `on`.
pub fn add_tolerance(
    d: &Drawing,
    view: &str,
    on: Reference,
    frame_text: &str,
) -> Result<(String, FileEdit), EditFailure> {
    let (c, zone, diametral, modifier, datums) = frame(frame_text)?;
    let (file, text) = part_of(d, view)?;
    edit_pmi(&file, text, |pmi| {
        declare_tolerance(
            pmi,
            Frame {
                characteristic: c,
                zone,
                diametral,
                modifier,
                datums: datums.clone(),
                projected: None,
            },
            vec![on.clone()],
            None,
        )
    })
}

/// A feature control frame's cells.
#[derive(Clone, Debug)]
pub struct Frame {
    pub characteristic: Characteristic,
    /// The zone (mm).
    pub zone: f64,
    pub diametral: bool,
    pub modifier: MaterialCondition,
    pub datums: Vec<DatumReference>,
    /// A projected zone's height (mm).
    pub projected: Option<f64>,
}

/// Declares a geometric tolerance in `pmi` (named `name`, else a fresh name from the
/// characteristic), answering its name. Its datums must be declared already.
pub fn declare_tolerance(
    pmi: &mut PmiSet,
    frame: Frame,
    on: Vec<Reference>,
    name: Option<&str>,
) -> Result<String, EditFailure> {
    let c = frame.characteristic;
    if c.needs_datum() && frame.datums.is_empty() {
        return Err(model(format!(
            "{} needs a datum reference frame (A, A|B or A|B|C)",
            characteristic_words(c)
        )));
    }
    if c.is_form() && !frame.datums.is_empty() {
        return Err(model(format!(
            "{} is a form tolerance and takes no datums",
            characteristic_words(c)
        )));
    }
    if frame.modifier != MaterialCondition::Rfs && !c.takes_material_condition() {
        return Err(model(format!(
            "{} takes no material condition",
            characteristic_words(c)
        )));
    }
    if let Some(missing) = frame
        .datums
        .iter()
        .find(|r| !pmi.datums.iter().any(|d| d.label == r.label))
    {
        return Err(model(format!(
            "the part declares no datum {}; add it with Datum first",
            missing.label
        )));
    }
    if frame.zone <= 0.0 {
        return Err(model("the tolerance zone must be more than zero"));
    }
    let name = match name {
        Some(n) => n.to_owned(),
        None => {
            let names: Vec<&str> = pmi.tolerances.iter().map(|t| t.name.as_str()).collect();
            fresh(&names, &format!("{c:?}").to_lowercase())
        }
    };
    pmi.tolerances.push(GeometricTolerance {
        name: name.clone(),
        characteristic: c,
        on,
        tolerance: ValueExpr::mm(frame.zone),
        diametral: frame.diametral,
        modifier: frame.modifier,
        projected: frame.projected.map(ValueExpr::mm),
        datums: frame.datums,
    });
    Ok(name)
}

/// A characteristic as a sentence names it (`Position`, `Profile of a surface`).
pub fn characteristic_words(c: Characteristic) -> &'static str {
    match c {
        Characteristic::Straightness => "Straightness",
        Characteristic::Flatness => "Flatness",
        Characteristic::Circularity => "Circularity",
        Characteristic::Cylindricity => "Cylindricity",
        Characteristic::ProfileOfLine => "Profile of a line",
        Characteristic::ProfileOfSurface => "Profile of a surface",
        Characteristic::Parallelism => "Parallelism",
        Characteristic::Perpendicularity => "Perpendicularity",
        Characteristic::Angularity => "Angularity",
        Characteristic::Position => "Position",
        Characteristic::Concentricity => "Concentricity",
        Characteristic::Symmetry => "Symmetry",
        Characteristic::CircularRunout => "Circular runout",
        Characteristic::TotalRunout => "Total runout",
    }
}

/// Declares datum `label` on `on`, or answers the label when the part has it on that face.
pub fn add_datum(
    d: &Drawing,
    view: &str,
    on: Reference,
    label: &str,
) -> Result<Option<FileEdit>, EditFailure> {
    let (file, text) = part_of(d, view)?;
    if let Some(p) = parse_part(&file, text).document {
        if !declare_datum(&mut p.pmi.clone(), label, &on)? {
            return Ok(None);
        }
    }
    let (_, edit) = edit_pmi(&file, text, |pmi| {
        declare_datum(pmi, label, &on)?;
        Ok(label.to_owned())
    })?;
    Ok(Some(edit))
}

/// Declares datum `label` on `on` in `pmi`: false when it is there on that face already.
pub fn declare_datum(pmi: &mut PmiSet, label: &str, on: &Reference) -> Result<bool, EditFailure> {
    let ok = !label.is_empty()
        && label.chars().all(|c| c.is_ascii_uppercase() || c.is_ascii_digit())
        && !label.contains(['I', 'O', 'Q']);
    if !ok {
        return Err(model(format!(
            "`{label}` isn't a datum label: use capital letters other than I, O and Q (A, B, AA)"
        )));
    }
    if let Some(existing) = pmi.datums.iter().find(|x| x.label == label) {
        return if existing.on == *on {
            Ok(false)
        } else {
            Err(model(format!(
                "the part has datum {label} on another face already"
            )))
        };
    }
    pmi.datums.push(DatumFeature {
        label: label.to_owned(),
        on: on.clone(),
    });
    Ok(true)
}

/// Declares datum target `label` (`A1`: datum A, target 1) of `shape` on `on`.
pub fn add_target(
    d: &Drawing,
    view: &str,
    on: Reference,
    label: &str,
    shape: &str,
    size: f64,
) -> Result<FileEdit, EditFailure> {
    let split = label.find(|c: char| c.is_ascii_digit()).ok_or_else(|| {
        model(format!(
            "write the target as its datum and number (A1), not `{label}`"
        ))
    })?;
    let (datum, index) = label.split_at(split);
    let index: u32 = index
        .parse()
        .map_err(|_| model(format!("`{label}` isn't a datum target (A1)")))?;
    let shape = match shape {
        "" | "point" => TargetShape::Point,
        "line" => TargetShape::Line {
            length: ValueExpr::mm(size),
        },
        "circle" => TargetShape::Circle {
            diameter: ValueExpr::mm(size),
        },
        "rectangle" => TargetShape::Rectangle {
            width: ValueExpr::mm(size),
            height: ValueExpr::mm(size),
        },
        other => return Err(model(format!("there is no `{other}` target shape"))),
    };
    let (file, text) = part_of(d, view)?;
    let datum = datum.to_uppercase();
    Ok(edit_pmi(&file, text, |pmi| {
        if pmi
            .targets
            .iter()
            .any(|t| t.datum == datum && t.index == index)
        {
            return Err(model(format!("the part has target {datum}{index} already")));
        }
        pmi.targets.push(DatumTarget {
            datum: datum.clone(),
            index,
            shape: shape.clone(),
            on: on.clone(),
        });
        Ok(format!("{datum}{index}"))
    })?
    .1)
}

/// Declares a surface finish (Ra in µm) on `on`.
pub fn add_finish(
    d: &Drawing,
    view: &str,
    on: Reference,
    ra_um: f64,
) -> Result<FileEdit, EditFailure> {
    let (file, text) = part_of(d, view)?;
    Ok(edit_pmi(&file, text, |pmi| {
        declare_finish(
            pmi,
            vec![on.clone()],
            ra_um / 1000.0,
            Removal::Any,
            None,
            None,
            None,
        )
    })?
    .1)
}

/// Declares a surface finish of `ra_mm` on `on` in `pmi`, answering its name.
pub fn declare_finish(
    pmi: &mut PmiSet,
    on: Vec<Reference>,
    ra_mm: f64,
    removal: Removal,
    process: Option<String>,
    lay: Option<String>,
    name: Option<&str>,
) -> Result<String, EditFailure> {
    if ra_mm <= 0.0 {
        return Err(model("Ra must be more than zero"));
    }
    if pmi
        .finishes
        .iter()
        .any(|f| f.on.iter().any(|r| on.contains(r)))
    {
        return Err(model("the part has a finish on that face already"));
    }
    let name = match name {
        Some(n) => n.to_owned(),
        None => {
            let names: Vec<&str> = pmi.finishes.iter().map(|f| f.name.as_str()).collect();
            fresh(&names, "finish")
        }
    };
    pmi.finishes.push(SurfaceFinish {
        name: name.clone(),
        on,
        ra: ValueExpr::mm(ra_mm),
        process,
        lay,
        removal,
    });
    Ok(name)
}

/// A dimension's tolerance as the form gives it.
pub enum Tolerance {
    /// `+plus/-minus` mm about the nominal (symmetric, bilateral and limit forms all write it).
    Deviation { plus: f64, minus: f64 },
    /// An ISO 286 fit (`H7`).
    Fit(String),
}

/// Declares a size tolerance (one reference: a diameter) or a location tolerance (two: the
/// distance between them) at the measured `nominal` (mm).
pub fn add_dimension_tolerance(
    d: &Drawing,
    view: &str,
    references: &[Reference],
    nominal: f64,
    tolerance: Tolerance,
) -> Result<(String, FileEdit), EditFailure> {
    let (file, text) = part_of(d, view)?;
    edit_pmi(&file, text, |pmi| {
        declare_dimension(pmi, references, nominal, Some(&tolerance), None)
    })
}

/// Declares a size (one reference) or location (two) at `nominal` mm in `pmi`, toleranced by
/// `tolerance` (a plain value without one), answering its name.
pub fn declare_dimension(
    pmi: &mut PmiSet,
    references: &[Reference],
    nominal: f64,
    tolerance: Option<&Tolerance>,
    name: Option<&str>,
) -> Result<String, EditFailure> {
    let nominal = ValueExpr::mm((nominal * 1.0e4).round() / 1.0e4);
    let toleranced = |plus: f64, minus: f64| ValueExpr::Toleranced {
        nominal: Box::new(nominal.clone()),
        minus: Box::new(ValueExpr::mm(minus)),
        plus: Box::new(ValueExpr::mm(plus)),
    };
    let value = |t: Option<&Tolerance>| match t {
        Some(Tolerance::Deviation { plus, minus }) => toleranced(*plus, *minus),
        Some(Tolerance::Fit(_)) => toleranced(0.0, 0.0),
        None => nominal.clone(),
    };
    match (references, tolerance) {
        ([on], t) => {
            let name = match name {
                Some(n) => n.to_owned(),
                None => {
                    let names: Vec<&str> = pmi.sizes.iter().map(|s| s.name.as_str()).collect();
                    fresh(&names, "size")
                }
            };
            pmi.sizes.push(SizeTolerance {
                name: name.clone(),
                on: vec![on.clone()],
                size: value(t),
                fit: match t {
                    Some(Tolerance::Fit(f)) => Some(f.clone()),
                    _ => None,
                },
            });
            Ok(name)
        }
        ([_, _], Some(Tolerance::Fit(_))) => Err(model(
            "a fit tolerances a size (a diameter); a distance takes a plus/minus tolerance",
        )),
        ([from, to], t) => {
            let name = match name {
                Some(n) => n.to_owned(),
                None => {
                    let names: Vec<&str> =
                        pmi.locations.iter().map(|s| s.name.as_str()).collect();
                    fresh(&names, "distance")
                }
            };
            pmi.locations.push(LocationTolerance {
                name: name.clone(),
                from: from.clone(),
                to: to.clone(),
                distance: value(t),
            });
            Ok(name)
        }
        _ => Err(model(
            "a dimension measures one reference (a diameter) or two (the distance between them)",
        )),
    }
}
