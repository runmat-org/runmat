//! Views placed by hand: a standard three-view layout, a model view in any orientation, an
//! isometric view, and the views drawn from another one (projected, section through a picked
//! edge, detail around a picked point, auxiliary normal to a picked edge, broken-out inside a
//! boundary picked on the sheet).
//!
//! A view lands where the `at` point pick puts it, else beside its parent (or, for a first
//! view, in the sheet's open area); the sheet keeps views clear of each other.

use super::*;
use runmat_cad::drawing::{Display, Orientation, ProjectDirection, View, ViewKind};
use runmat_cad::value::ValueExpr;

/// The gap between a view and the one beside it (sheet mm).
const GAP: f64 = 30.0;

fn display(word: &str) -> Result<Display, EditFailure> {
    Ok(match word {
        "" | "hidden_removed" | "visible_lines" => Display::VisibleLines,
        "hidden_visible" | "hidden_lines" => Display::HiddenLines,
        "shaded" => Display::Shaded,
        other => return Err(model(format!("there is no `{other}` display style"))),
    })
}

fn orientation(word: &str) -> Result<Orientation, EditFailure> {
    Ok(match word {
        "" | "front" => Orientation::Front,
        "back" => Orientation::Back,
        "top" => Orientation::Top,
        "bottom" => Orientation::Bottom,
        "left" => Orientation::Left,
        "right" => Orientation::Right,
        "isometric" => Orientation::Isometric,
        other => return Err(model(format!("there is no `{other}` orientation"))),
    })
}

fn direction(word: &str) -> Result<ProjectDirection, EditFailure> {
    Ok(match word {
        "" | "right" => ProjectDirection::Right,
        "left" => ProjectDirection::Left,
        "up" => ProjectDirection::Up,
        "down" => ProjectDirection::Down,
        other => return Err(model(format!("there is no `{other}` projection direction"))),
    })
}

/// The model's size along x, y and z (mm), from its display's bounds.
fn size_of(display: Option<&Value>) -> [f64; 3] {
    let b = display.map(|d| &d["data"]["bounds"]);
    let f = |k: &str| -> Vec<f64> {
        b.and_then(|b| b[k].as_array())
            .map(|a| a.iter().filter_map(Value::as_f64).collect())
            .unwrap_or_default()
    };
    let (lo, hi) = (f("min"), f("max"));
    if lo.len() == 3 && hi.len() == 3 {
        [0, 1, 2].map(|i| (hi[i] - lo[i]) * 1000.0)
    } else {
        [60.0; 3]
    }
}

/// The scale a view draws at: its own, else its sheet's.
fn drawn_scale(d: &Drawing, view: &View) -> f64 {
    let (n, m) = view
        .scale
        .or_else(|| {
            d.document
                .sheets
                .iter()
                .find(|s| s.name == view.sheet)
                .map(|s| s.scale)
        })
        .unwrap_or((1, 1));
    n as f64 / m.max(1) as f64
}

/// The drawn width and height (sheet mm) of a model seen in `orientation`.
fn extent(size: [f64; 3], orientation: Orientation, scale: f64) -> [f64; 2] {
    let [x, y, z] = size;
    let [w, h] = match orientation {
        Orientation::Front | Orientation::Back => [x, z],
        Orientation::Top | Orientation::Bottom => [x, y],
        Orientation::Left | Orientation::Right => [y, z],
        _ => [(x + y) * 0.71, z + (x + y) * 0.41],
    };
    [w * scale, h * scale]
}

/// What a view looks along, following projected views back to their standard one.
fn seen(d: &Drawing, name: &str) -> Orientation {
    let mut at = name;
    for _ in 0..8 {
        let Some(view) = d.document.view(at) else {
            break;
        };
        match &view.kind {
            ViewKind::Standard { orientation, .. } => return *orientation,
            ViewKind::Projected { direction, .. } => {
                return match direction {
                    ProjectDirection::Up => Orientation::Top,
                    ProjectDirection::Down => Orientation::Bottom,
                    ProjectDirection::Left => Orientation::Left,
                    ProjectDirection::Right => Orientation::Right,
                }
            }
            kind => match kind.parent() {
                Some(p) => at = p,
                None => break,
            },
        }
    }
    Orientation::Front
}

/// Beside `parent` toward `step` (x, y), clear of it by the gap.
fn beside(d: &Drawing, parent: &View, own: [f64; 2], step: [f64; 2]) -> [f64; 2] {
    let size = size_of(d.display(&parent.name).as_ref());
    let p = extent(size, seen(d, &parent.name), drawn_scale(d, parent));
    [
        parent.at[0] + step[0] * (p[0] / 2.0 + own[0] / 2.0 + GAP),
        parent.at[1] + step[1] * (p[1] / 2.0 + own[1] / 2.0 + GAP),
    ]
}

fn round(p: [f64; 2]) -> [f64; 2] {
    p.map(|v| (v * 10.0).round() / 10.0)
}

/// The `at` point pick, if any.
fn placed(values: &Values) -> Result<Option<[f64; 2]>, EditFailure> {
    Ok(points(values, "at")?.into_iter().next())
}

fn parent_of<'d>(d: &'d Drawing, values: &Values) -> Result<&'d View, EditFailure> {
    let name =
        view_pick(values, "parent")?.ok_or_else(|| model("pick the view to draw it from"))?;
    d.view(&name)
}

fn child(parent: &View, name: String, kind: ViewKind, at: [f64; 2]) -> View {
    View {
        name,
        sheet: parent.sheet.clone(),
        model: parent.model.clone(),
        kind,
        at: round(at),
        scale: None,
        display: parent.display,
        exclude: Vec::new(),
        exploded: None,
    }
}

/// The one edge pick of `field`, which must lie in `view` when it names one.
fn edge_in(values: &Values, field: &str, view: &str, what: &str) -> Result<Reference, EditFailure> {
    let (on, reference, _) = edges(values, field)?
        .into_iter()
        .next()
        .ok_or_else(|| model(format!("pick {what}")))?;
    if on.as_deref().is_some_and(|v| v != view) {
        return Err(model(format!("pick {what} in view `{view}`")));
    }
    Ok(reference)
}

/// A label letter, upper case.
fn label(values: &Values, default: &str) -> String {
    text(values, "label", default).trim().to_uppercase()
}

pub fn edit(
    path: &str,
    text_: &str,
    files: &Files,
    command: &str,
    values: &Values,
) -> Option<Result<(SourceChange, String), EditFailure>> {
    let known = [
        "drawing.standard-3",
        "drawing.model-view",
        "drawing.isometric",
        "drawing.projected",
        "drawing.section",
        "drawing.detail",
        "drawing.auxiliary",
        "drawing.broken-out",
    ];
    if !known.contains(&command) {
        return None;
    }
    Some((|| {
        let mut d = Drawing::open(path, text_, files)?;
        let summary = match command {
            "drawing.standard-3" | "drawing.model-view" | "drawing.isometric" => {
                let model_ref = model_ref(&d, values)?;
                let sheet = d.sheet()?;
                let scale = scale(text(values, "scale", ""))?;
                let s = scale.map(|(n, m)| n as f64 / m as f64).unwrap_or_else(|| {
                    d.document
                        .sheets
                        .iter()
                        .find(|x| x.name == sheet)
                        .map(|x| x.scale.0 as f64 / x.scale.1.max(1) as f64)
                        .unwrap_or(1.0)
                });
                let size = size_of(display_of(model_ref.file.as_str(), files).as_ref());
                let new =
                    |_: &Drawing, name: String, kind: ViewKind, at: [f64; 2], display: Display| {
                        View {
                            name,
                            sheet: sheet.clone(),
                            model: model_ref.clone(),
                            kind,
                            at: round(at),
                            scale,
                            display,
                            exclude: Vec::new(),
                            exploded: None,
                        }
                    };
                // A first view sits low left (room for dimensions under it and the title block
                // along the bottom); a later one right of the views already there.
                let start = |d: &Drawing, own: [f64; 2]| -> [f64; 2] {
                    let right = d
                        .document
                        .views
                        .iter()
                        .filter(|v| v.sheet == sheet)
                        .map(|v| {
                            let e = extent(
                                size_of(d.display(&v.name).as_ref()),
                                seen(d, &v.name),
                                drawn_scale(d, v),
                            );
                            v.at[0] + e[0] / 2.0
                        })
                        .fold(f64::NAN, f64::max);
                    if right.is_nan() {
                        [60.0 + own[0] / 2.0, 90.0 + own[1] / 2.0]
                    } else {
                        [right + GAP + own[0] / 2.0, 90.0 + own[1] / 2.0]
                    }
                };
                match command {
                    "drawing.standard-3" => {
                        let first = text(values, "projection", "third") == "first";
                        let (fo, to, ro) = (
                            extent(size, Orientation::Front, s),
                            extent(size, Orientation::Top, s),
                            extent(size, Orientation::Right, s),
                        );
                        let front_at = placed(values)?.unwrap_or_else(|| start(&d, fo));
                        // Third angle: the top view above the front, the right view right of it;
                        // first angle (ISO 5456-2): below and left.
                        let sign = if first { -1.0 } else { 1.0 };
                        let top_at = [
                            front_at[0],
                            front_at[1] + sign * (fo[1] / 2.0 + to[1] / 2.0 + GAP),
                        ];
                        let right_at = [
                            front_at[0] + sign * (fo[0] / 2.0 + ro[0] / 2.0 + GAP),
                            front_at[1],
                        ];
                        let f = fresh(&d.document, "front");
                        let front = new(
                            &d,
                            f.clone(),
                            ViewKind::Standard {
                                orientation: Orientation::Front,
                                frame: None,
                            },
                            front_at,
                            Display::HiddenLines,
                        );
                        d.document.views.push(front);
                        let t = fresh(&d.document, "top");
                        let mut top = new(
                            &d,
                            t.clone(),
                            ViewKind::Projected {
                                from: f.clone(),
                                direction: ProjectDirection::Up,
                            },
                            top_at,
                            Display::HiddenLines,
                        );
                        top.scale = None;
                        d.document.views.push(top);
                        let r = fresh(&d.document, "right");
                        let mut right = new(
                            &d,
                            r.clone(),
                            ViewKind::Projected {
                                from: f.clone(),
                                direction: ProjectDirection::Right,
                            },
                            right_at,
                            Display::HiddenLines,
                        );
                        right.scale = None;
                        d.document.views.push(right);
                        if let Some(sheet) = d.document.sheets.iter_mut().find(|x| x.name == sheet)
                        {
                            sheet.projection = if first {
                                runmat_cad::drawing::Projection::FirstAngle
                            } else {
                                runmat_cad::drawing::Projection::ThirdAngle
                            };
                        }
                        format!("Placed views {f}, {t} and {r}")
                    }
                    _ => {
                        let iso = command == "drawing.isometric";
                        let o = if iso {
                            Orientation::Isometric
                        } else {
                            orientation(text(values, "orientation", "front"))?
                        };
                        let own = extent(size, o, s);
                        let at = placed(values)?.unwrap_or_else(|| start(&d, own));
                        let stem = if iso {
                            "isometric".to_owned()
                        } else {
                            format!("{o:?}").to_lowercase()
                        };
                        let name = fresh(&d.document, &stem);
                        let disp = display(text(
                            values,
                            "display",
                            if iso {
                                "hidden_removed"
                            } else {
                                "hidden_visible"
                            },
                        ))?;
                        let view = new(
                            &d,
                            name.clone(),
                            ViewKind::Standard {
                                orientation: o,
                                frame: None,
                            },
                            at,
                            disp,
                        );
                        d.document.views.push(view);
                        format!("Placed view {name}")
                    }
                }
            }
            "drawing.projected" => {
                let parent = parent_of(&d, values)?.clone();
                let dir = direction(text(values, "direction", "right"))?;
                let step = match dir {
                    ProjectDirection::Up => [0.0, 1.0],
                    ProjectDirection::Down => [0.0, -1.0],
                    ProjectDirection::Left => [-1.0, 0.0],
                    ProjectDirection::Right => [1.0, 0.0],
                };
                let name = fresh(
                    &d.document,
                    &format!("{dir:?}")
                        .to_lowercase()
                        .replace("up", "top")
                        .replace("down", "bottom"),
                );
                let kind = ViewKind::Projected {
                    from: parent.name.clone(),
                    direction: dir,
                };
                let mut view = child(&parent, name.clone(), kind, parent.at);
                view.name = name.clone();
                let own = extent(
                    size_of(d.display(&parent.name).as_ref()),
                    {
                        d.document.views.push(view.clone());
                        let o = seen(&d, &name);
                        d.document.views.pop();
                        o
                    },
                    drawn_scale(&d, &parent),
                );
                view.at = round(placed(values)?.unwrap_or_else(|| beside(&d, &parent, own, step)));
                d.document.views.push(view);
                format!("Projected view {name} from {}", parent.name)
            }
            "drawing.section" => {
                let parent = parent_of(&d, values)?.clone();
                let plane = edge_in(
                    values,
                    "through",
                    &parent.name,
                    "where the cutting line passes (an edge or a hole)",
                )?;
                let letter = label(values, "A");
                let name = fresh(&d.document, &format!("{}_{}", letter, letter));
                let own = extent(
                    size_of(d.display(&parent.name).as_ref()),
                    seen(&d, &parent.name),
                    drawn_scale(&d, &parent),
                );
                let at = placed(values)?.unwrap_or_else(|| beside(&d, &parent, own, [1.0, 0.0]));
                let kind = ViewKind::Section {
                    from: parent.name.clone(),
                    plane: runmat_cad::model::PlaneRef::Face { face: plane },
                    label: letter.clone(),
                    aligned: flag(values, "aligned", false),
                };
                let mut view = child(&parent, name.clone(), kind, at);
                // Sections draw what the cut shows: no hidden lines (ISO 128-40).
                view.display = Display::VisibleLines;
                d.document.views.push(view);
                format!("Cut section {letter}-{letter} from {}", parent.name)
            }
            "drawing.detail" => {
                let parent = parent_of(&d, values)?.clone();
                let center = edge_in(
                    values,
                    "center",
                    &parent.name,
                    "the detail's center (an edge or a hole)",
                )?;
                let letter = label(values, "B");
                let radius = number(values, "radius", 5.0);
                if radius <= 0.0 {
                    return Err(model("the detail's radius must be more than zero"));
                }
                let scale = scale(text(values, "scale", "2:1"))?;
                let name = fresh(&d.document, &format!("detail_{letter}"));
                let s = scale.map(|(n, m)| n as f64 / m as f64).unwrap_or(1.0);
                let own = [2.0 * radius * s, 2.0 * radius * s];
                let at = placed(values)?.unwrap_or_else(|| beside(&d, &parent, own, [1.0, 0.0]));
                let kind = ViewKind::Detail {
                    from: parent.name.clone(),
                    center,
                    radius: ValueExpr::mm(radius),
                    label: letter.clone(),
                };
                let mut view = child(&parent, name.clone(), kind, at);
                view.scale = scale;
                d.document.views.push(view);
                format!("Placed detail {letter} of {}", parent.name)
            }
            "drawing.auxiliary" => {
                let parent = parent_of(&d, values)?.clone();
                let edge = edge_in(values, "edge", &parent.name, "the edge to look square to")?;
                let letter = label(values, "C");
                let name = fresh(&d.document, &format!("view_{letter}"));
                let own = extent(
                    size_of(d.display(&parent.name).as_ref()),
                    seen(&d, &parent.name),
                    drawn_scale(&d, &parent),
                );
                let at = placed(values)?.unwrap_or_else(|| beside(&d, &parent, own, [1.0, 1.0]));
                let kind = ViewKind::Auxiliary {
                    from: parent.name.clone(),
                    edge,
                    label: letter.clone(),
                };
                d.document
                    .views
                    .push(child(&parent, name.clone(), kind, at));
                format!("Placed auxiliary view {letter} from {}", parent.name)
            }
            "drawing.broken-out" => {
                let parent = parent_of(&d, values)?.clone();
                let boundary = points(values, "boundary")?;
                if boundary.len() < 3 {
                    return Err(model(
                        "pick at least three points around the area to break out",
                    ));
                }
                // The boundary in the parent's view millimetres, about the view's centre (the
                // point the sheet puts at its `at`).
                let s = drawn_scale(&d, &parent);
                let boundary: Vec<[f64; 2]> = boundary
                    .iter()
                    .map(|p| round([(p[0] - parent.at[0]) / s, (p[1] - parent.at[1]) / s]))
                    .collect();
                let depth = number(values, "depth", 10.0);
                if depth <= 0.0 {
                    return Err(model("the break-out depth must be more than zero"));
                }
                let name = fresh(&d.document, &format!("{}_broken", parent.name));
                let own = extent(
                    size_of(d.display(&parent.name).as_ref()),
                    seen(&d, &parent.name),
                    s,
                );
                let at = placed(values)?.unwrap_or_else(|| beside(&d, &parent, own, [1.0, 0.0]));
                let kind = ViewKind::BrokenOut {
                    from: parent.name.clone(),
                    boundary,
                    depth: ValueExpr::mm(depth),
                };
                let mut view = child(&parent, name.clone(), kind, at);
                view.scale = parent.scale;
                d.document.views.push(view);
                format!("Broke out {name} from {}", parent.name)
            }
            _ => unreachable!(),
        };
        Ok((d.change()?, summary))
    })())
}
