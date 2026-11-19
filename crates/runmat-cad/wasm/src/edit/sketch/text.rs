//! `sketch.text`: lettering set in the single-line font, whose stroke outline extrudes and cuts.
//!
//! Values, in sketch-plane millimetres and degrees:
//!
//! - `text`: the string (`\n` starts a new line);
//! - `at`: the left end of the first line's baseline, a point value;
//! - `height`: the capital height; `angle`: the baseline's direction (0 when left out);
//! - `font`: `single_line`, the one font sketches set text in;
//! - `stroke`: the pen width (0.15 of the height when left out).
//!
//! Text along a curve isn't in the sketch model; `along` says so.

use super::{model, num, point, EditFailure};
use runmat_cad::ids::SketchEntityId;
use runmat_cad::model::sketch::{SketchConstraint, SketchEntity, SketchGeometry as G, TextFont};
use runmat_cad::value::ValueExpr;
use serde_json::{Map, Value};

pub(super) fn draw(
    values: &Map<String, Value>,
    name: &mut dyn FnMut(&str) -> String,
) -> Result<(Vec<SketchEntity>, Vec<SketchConstraint>), EditFailure> {
    if values
        .get("along")
        .is_some_and(|v| !v.is_null() && v != "" && v != &Value::Array(vec![]))
    {
        return Err(model(
            "text along a curve isn't in the sketch model yet; place it on a straight baseline",
        ));
    }
    let text = values
        .get("text")
        .and_then(Value::as_str)
        .filter(|t| t.chars().any(|c| !c.is_whitespace()))
        .ok_or_else(|| model("type the text to place"))?
        .to_owned();
    let font = match values.get("font").and_then(Value::as_str) {
        None | Some("") => TextFont::default(),
        Some(word) => TextFont::from_word(word).ok_or_else(|| {
            model(format!(
                "sketch text is set in the single-line font (`single_line`), not `{word}`"
            ))
        })?,
    };
    let at = point(values, "at")?;
    let height = values
        .get("height")
        .and_then(num)
        .filter(|h| *h > 0.0)
        .ok_or_else(|| model("give the text a height"))?;
    let stroke = match values.get("stroke").and_then(num) {
        None => None,
        Some(w) if w > 0.0 && w < height => Some(super::mm(w)),
        Some(_) => {
            return Err(model(
                "the stroke is wider than nothing and narrower than the text",
            ))
        }
    };
    let angle = values.get("angle").and_then(num).unwrap_or(0.0);
    Ok((
        vec![SketchEntity {
            id: SketchEntityId::new(name("t")),
            geometry: G::Text {
                text,
                at,
                height: super::mm(height),
                angle: (angle != 0.0).then(|| ValueExpr::deg((angle * 1e4).round() / 1e4)),
                font,
                stroke,
            },
            construction: false,
            projected: None,
        }],
        Vec::new(),
    ))
}
