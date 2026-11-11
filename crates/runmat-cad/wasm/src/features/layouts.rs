//! Canvas appliers for pattern seeds, table and variable layouts, skipped instances, and the
//! body operations (move, rotate, copy, indent). Each returns model JSON, like the other arms
//! of [`super::feature_kind`].
//!
//! - **Seeds**: features, faces or bodies; one kind per pattern.
//! - **Table and variable layouts**: the `table` field holds one instance per line (or per
//!   `;`): an optional `key:`, then the offset `dx, dy, dz` (plain numbers are millimetres),
//!   then, in a variable pattern, `input = value` pairs the seed is rebuilt with.
//! - **Skip**: instance numbers in layout order, 1 being the first copy after the seed. A grid
//!   counts along its second direction first.
//! - **Move/copy body**: a translation, a rotation about an axis, or copies (each copy moved
//!   and turned once more than the last).

use super::*;
use crate::picks::{lower_text, topo};

/// Features, faces or bodies to repeat.
pub(super) fn seeds(values: &Values, picks: &Picks) -> Result<Value, EditFailure> {
    let all = some(values, "seeds", "the features, faces or bodies to repeat")?;
    let features: Vec<&str> = all
        .iter()
        .filter_map(|p| p.strip_prefix("feature:"))
        .collect();
    if features.len() == all.len() {
        return Ok(json!({ "seeds": "features", "features": features }));
    }
    if !features.is_empty() {
        return Err(model(
            "a pattern repeats features, faces or bodies; pick one kind",
        ));
    }
    let refs = picks.references(&all)?;
    if refs.iter().all(|r| topo(r) == "face") {
        return Ok(json!({ "seeds": "faces", "faces": refs }));
    }
    if refs.iter().all(|r| topo(r) == "body") {
        return Ok(json!({ "seeds": "bodies", "bodies": refs }));
    }
    Err(model(
        "a pattern repeats features, faces or bodies; pick one kind (edges and vertices aren't repeated)",
    ))
}

/// One offset component or input value: a plain number is millimetres.
fn quantity(token: &str) -> Result<Value, EditFailure> {
    let t = token.trim();
    match t.parse::<f64>() {
        Ok(n) => Ok(json!({ "form": "quantity", "value": n, "unit": "mm" })),
        Err(_) => lower_text(t).map_err(|_| model(format!("`{t}` isn't a value"))),
    }
}

/// The `table` field's instances: `(key, [dx, dy, dz], values)` per line.
fn rows(values: &Values, variable: bool) -> Result<Vec<Value>, EditFailure> {
    let table = text(values, "table", "");
    let mut out = Vec::new();
    for (n, line) in table
        .split(['\n', ';'])
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .enumerate()
    {
        let (key, rest) = match line.split_once(':') {
            Some((k, r)) => (k.trim().to_owned(), r),
            None => (format!("r{}", n + 1), line),
        };
        if key.is_empty() || !key.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
            return Err(model(format!("row {}: `{key}` isn't a name", n + 1)));
        }
        let parts: Vec<&str> = rest.split(',').map(str::trim).collect();
        if parts.len() < 3 {
            return Err(model(format!(
                "row {} needs an offset `dx, dy, dz` (millimetres)",
                n + 1
            )));
        }
        let mut row = json!({
            "key": key,
            "dx": quantity(parts[0])?,
            "dy": quantity(parts[1])?,
            "dz": quantity(parts[2])?,
        });
        let extra = &parts[3..];
        if !variable && !extra.is_empty() {
            return Err(model(format!(
                "row {} has values; a table pattern takes offsets only (a variable pattern takes values)",
                n + 1
            )));
        }
        if variable {
            let mut inputs = Map::new();
            for pair in extra {
                let (name, v) = pair.split_once('=').ok_or_else(|| {
                    model(format!("row {}: `{pair}` isn't `input = value`", n + 1))
                })?;
                inputs.insert(name.trim().to_owned(), quantity(v)?);
            }
            row["values"] = Value::Object(inputs);
        }
        out.push(row);
    }
    if out.is_empty() {
        return Err(model(
            "list the instances, one per line: `key: dx, dy, dz` in millimetres",
        ));
    }
    Ok(out)
}

/// A table or variable layout from the `table` field.
pub(super) fn table_layout(values: &Values, variable: bool) -> Result<Value, EditFailure> {
    let rows = rows(values, variable)?;
    Ok(if variable {
        json!({ "layout": "variable", "instances": rows })
    } else {
        json!({ "layout": "table", "rows": rows })
    })
}

fn whole(v: &Value) -> Option<u32> {
    let n = match v {
        Value::Number(n) => n.as_f64()?,
        Value::String(s) => s.trim().parse().ok()?,
        Value::Object(o) => o.get("value").and_then(|x| x.get("value").or(Some(x)))?.as_f64()?,
        _ => return None,
    };
    (n >= 1.0 && n.fract() == 0.0).then_some(n as u32)
}

/// The skipped instances' keys, by instance number in layout order.
pub(super) fn skip_keys(values: &Values, layout: &Value) -> Result<Vec<Value>, EditFailure> {
    let numbers: Vec<&Value> = match values.get("skip") {
        Some(Value::Array(items)) => items.iter().collect(),
        Some(Value::Null) | None => Vec::new(),
        Some(other) => vec![other],
    };
    let mut out = Vec::new();
    for v in numbers {
        let n = whole(v).ok_or_else(|| model(format!("`{v}` isn't an instance number")))?;
        let key = match layout["layout"].as_str().unwrap_or("") {
            "linear" => json!({ "instance": "linear", "step": n }),
            "circular" => json!({ "instance": "circular", "step": n }),
            "curve" => json!({ "instance": "curve", "step": n }),
            "grid" => {
                let second = values.get("instances2").and_then(whole).unwrap_or(1);
                json!({ "instance": "grid", "first": n / second, "second": n % second })
            }
            "table" | "variable" => {
                let list = if layout["layout"] == "table" {
                    &layout["rows"]
                } else {
                    &layout["instances"]
                };
                let row = list.get(n as usize - 1).ok_or_else(|| {
                    model(format!("there is no instance {n} to skip"))
                })?;
                json!({ "instance": "row", "key": row["key"] })
            }
            other => {
                return Err(model(format!(
                    "a {other} pattern's instances are skipped in the source editor"
                )))
            }
        };
        out.push(key);
    }
    Ok(out)
}

fn moves(values: &Values) -> bool {
    ["dx", "dy", "dz"].iter().any(|f| {
        values
            .get(*f)
            .and_then(Value::as_f64)
            .is_some_and(|v| v != 0.0)
    })
}

/// Move, rotate or copy bodies; the feature's stem and model JSON.
pub(super) fn body_move(
    values: &Values,
    picks: &Picks,
) -> Result<(&'static str, Value), EditFailure> {
    let bodies = picks.references(&some(values, "bodies", "the bodies to move")?)?;
    let turns = values
        .get("angle")
        .and_then(Value::as_f64)
        .is_some_and(|a| a != 0.0);
    let axis = || -> Result<Value, EditFailure> {
        picks.axis(&one(values, "axis", "the axis to rotate about")?)
    };
    if flag(values, "copy") {
        let mut kind = json!({
            "feature": "body_op",
            "op": "copy",
            "bodies": bodies,
            "dx": value(values, "dx", "mm")?,
            "dy": value(values, "dy", "mm")?,
            "dz": value(values, "dz", "mm")?,
            "count": number(values, "copies").unwrap_or(json!({ "form": "number", "value": 1 })),
        });
        if turns {
            kind["axis"] = axis()?;
            kind["angle"] = value(values, "angle", "deg")?;
        }
        return Ok(("copy", kind));
    }
    if turns {
        if moves(values) {
            return Err(model(
                "move and rotate in two steps: a body moves by an offset or turns about an axis",
            ));
        }
        return Ok((
            "rotate",
            json!({
                "feature": "body_op",
                "op": "rotate",
                "bodies": bodies,
                "axis": axis()?,
                "angle": value(values, "angle", "deg")?,
            }),
        ));
    }
    Ok((
        "move",
        json!({
            "feature": "body_op",
            "op": "translate",
            "bodies": bodies,
            "dx": value(values, "dx", "mm")?,
            "dy": value(values, "dy", "mm")?,
            "dz": value(values, "dz", "mm")?,
        }),
    ))
}

/// Press a tool body into a target body.
pub(super) fn body_indent(values: &Values, picks: &Picks) -> Result<Value, EditFailure> {
    let mut kind = json!({
        "feature": "body_op",
        "op": "indent",
        "target": picks.reference(&one(values, "target", "the target body")?)?,
        "tool_body": picks.reference(&one(values, "tool", "the tool body")?)?,
    });
    if values.get("clearance").is_some() {
        kind["clearance"] = value(values, "clearance", "mm")?;
    }
    Ok(kind)
}
