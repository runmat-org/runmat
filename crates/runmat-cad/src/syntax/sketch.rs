//! Sketches as blocks: entities, constraints and the `solved { }` cache.
//!
//! ```text
//! sketch s1 on origin.xy {
//!     l1 = line(start = [0 mm, 0 mm], end = [80 mm, 0 mm])
//!     l2 = line(start = l1.end, end = [80 mm, 50 mm])
//!     c1 = circle(center = [40 mm, 25 mm], radius = 5 mm, construction)
//!     constrain horizontal(l1)
//!     constrain distance(l1.start, l2.end, id = d1) = 60 mm
//!     constrain limit(length(l1, value = 80 mm), min = 60 mm, max = 90 mm)
//!     solved {
//!         l1.start = [0, 0]
//!         c1.radius = 0.005
//!     }
//! }
//! ```
//!
//! - A point is `[x, y]` (or `[x, y] <unit>`), or another entity's point (`l1.end`), which
//!   makes the two the same point.
//! - A constraint names entities (`l1`) and their points (`l1.start`); a dimension takes its
//!   value after `=`. `id = <name>` names the constraint.
//! - `solved { }` holds the solver's last output in SI units, so the file reloads without
//!   re-solving: `<entity>.<point> = [x, y]`, `<entity>.radius = r`, and
//!   `<constraint>.hidden_<n> = v` for variables the solver adds.
//! - `suppressed = true` and `frozen = "<key>"` are the feature's own settings.

use super::refs::{self, print_plane_or_face, role_of};
use super::value::{lower_value, print_value};
use crate::ids::{ConstraintId, SketchEntityId};
use crate::model::datum::PlaneRef;
use crate::model::sketch::{
    PointExpr, PointRole, Sketch, SketchConstraint, SketchConstraintKind as K, SketchEntity,
    SketchGeometry, SketchTarget, SolvedCache, SolvedKey,
};
use crate::value::ValueExpr;
use runmat_design_syntax::ast::{self, Argument, Expr, Headed, Statement};
use runmat_design_syntax::document::BlockEntry;
use runmat_design_syntax::json::{
    format_f64, is_identifier, lower_expr, print_json, Json, LowerError, ValueSyntax, INDENT,
};
use runmat_types::Span;

fn error(span: Span, message: impl Into<String>) -> LowerError {
    LowerError::new(span, message)
}

fn encode<T: serde::Serialize>(value: &T, span: Span) -> Result<Json, LowerError> {
    Json::from_serialize(value).map_err(|e| error(span, e.to_string()))
}

// ----------------------------------------------------------------------------- printing

fn role_word(role: PointRole) -> String {
    role.word()
}

fn name(text: &str) -> Option<&str> {
    is_identifier(text).then_some(text)
}

fn print_point(point: &PointExpr) -> Option<String> {
    Some(match point {
        PointExpr::Coords { x, y } => format!("[{}, {}]", print_value(x)?, print_value(y)?),
        PointExpr::Of { entity, role } => format!("{}.{}", name(&entity.0)?, role_word(*role)),
    })
}

fn print_target(target: &SketchTarget) -> Option<String> {
    Some(match target {
        SketchTarget::Entity { entity } => name(&entity.0)?.to_string(),
        SketchTarget::Point { entity, role } => {
            format!("{}.{}", name(&entity.0)?, role_word(*role))
        }
    })
}

fn print_entity(entity: &SketchEntity) -> Option<String> {
    let mut arguments = match &entity.geometry {
        SketchGeometry::Point { at } => vec![format!("at = {}", print_point(at)?)],
        SketchGeometry::Line { start, end } => vec![
            format!("start = {}", print_point(start)?),
            format!("end = {}", print_point(end)?),
        ],
        SketchGeometry::Arc { center, start, end } => vec![
            format!("center = {}", print_point(center)?),
            format!("start = {}", print_point(start)?),
            format!("end = {}", print_point(end)?),
        ],
        SketchGeometry::Circle { center, radius } => vec![
            format!("center = {}", print_point(center)?),
            format!("radius = {}", print_value(radius)?),
        ],
        SketchGeometry::Spline { points, closed } => {
            let points: Option<Vec<String>> = points.iter().map(print_point).collect();
            let mut arguments = vec![format!("points = [{}]", points?.join(", "))];
            if *closed {
                arguments.push("closed".into());
            }
            arguments
        }
    };
    if entity.construction {
        arguments.push("construction".into());
    }
    let kind = match entity.geometry {
        SketchGeometry::Point { .. } => "point",
        SketchGeometry::Line { .. } => "line",
        SketchGeometry::Arc { .. } => "arc",
        SketchGeometry::Circle { .. } => "circle",
        SketchGeometry::Spline { .. } => "spline",
    };
    Some(format!(
        "{} = {kind}({})",
        name(&entity.id.0)?,
        arguments.join(", ")
    ))
}

/// The constraint's call and its value (`distance(l1.start, l2.end)`, `60 mm`).
fn print_kind(kind: &K, extra: Option<String>) -> Option<(String, Option<String>)> {
    let target = |t: &SketchTarget| print_target(t);
    let (word, mut arguments, value) = match kind {
        K::Coincident { a, b } => ("coincident", vec![target(a)?, target(b)?], None),
        K::Parallel { a, b } => ("parallel", vec![target(a)?, target(b)?], None),
        K::Perpendicular { a, b } => ("perpendicular", vec![target(a)?, target(b)?], None),
        K::Tangent { a, b } => ("tangent", vec![target(a)?, target(b)?], None),
        K::Equal { a, b } => ("equal", vec![target(a)?, target(b)?], None),
        K::Horizontal { line } => ("horizontal", vec![target(line)?], None),
        K::Vertical { line } => ("vertical", vec![target(line)?], None),
        K::Fix { target: t } => ("fix", vec![target(t)?], None),
        K::Midpoint { point, line } => ("midpoint", vec![target(point)?, target(line)?], None),
        K::Concentric { a, b } => ("concentric", vec![target(a)?, target(b)?], None),
        K::Coradial { a, b } => ("coradial", vec![target(a)?, target(b)?], None),
        K::Collinear { a, b } => ("collinear", vec![target(a)?, target(b)?], None),
        K::Symmetric { a, b, axis } => (
            "symmetric",
            vec![target(a)?, target(b)?, target(axis)?],
            None,
        ),
        K::Distance { a, b, value } => ("distance", vec![target(a)?, target(b)?], Some(value)),
        K::Angle { a, b, value } => ("angle", vec![target(a)?, target(b)?], Some(value)),
        K::Length { line, value } => ("length", vec![target(line)?], Some(value)),
        K::Radius { curve, value } => ("radius", vec![target(curve)?], Some(value)),
        K::Diameter { curve, value } => ("diameter", vec![target(curve)?], Some(value)),
        K::Limit {
            dimension,
            min,
            max,
        } => {
            let (inner, inner_value) = print_kind(dimension, None)?;
            let inner = match inner_value {
                Some(value) => format!("{}, value = {value})", inner.strip_suffix(')')?),
                None => inner,
            };
            (
                "limit",
                vec![
                    inner,
                    format!("min = {}", print_value(min)?),
                    format!("max = {}", print_value(max)?),
                ],
                None,
            )
        }
    };
    if let Some(extra) = extra {
        arguments.push(extra);
    }
    let value = match value {
        Some(value) => Some(print_value(value)?),
        None => None,
    };
    Some((format!("{word}({})", arguments.join(", ")), value))
}

fn print_constraint(constraint: &SketchConstraint) -> Option<String> {
    let id = match &constraint.id {
        Some(id) => Some(format!("id = {}", name(&id.0)?)),
        None => None,
    };
    let (call, value) = print_kind(&constraint.kind, id)?;
    Some(match value {
        Some(value) => format!("constrain {call} = {value}"),
        None => format!("constrain {call}"),
    })
}

fn solved_target(key: &SolvedKey) -> Option<String> {
    Some(match key {
        SolvedKey::Point { entity, role } => {
            format!("{}.{}", name(&entity.0)?, role_word(*role))
        }
        SolvedKey::Radius { entity } => format!("{}.radius", name(&entity.0)?),
        SolvedKey::Hidden { constraint, index } => {
            format!("{}.hidden_{index}", name(&constraint.0)?)
        }
    })
}

fn number(value: f64) -> Option<String> {
    value.is_finite().then(|| format_f64(value))
}

fn print_solved(solved: &SolvedCache) -> Option<Vec<String>> {
    let mut lines = Vec::new();
    for (key, [x, y]) in &solved.points {
        lines.push(format!(
            "{} = [{}, {}]",
            solved_target(key)?,
            number(*x)?,
            number(*y)?
        ));
    }
    for (key, value) in &solved.scalars {
        lines.push(format!("{} = {}", solved_target(key)?, number(*value)?));
    }
    Some(lines)
}

/// The sketch feature `name` as a block, or `None` when part of it has no block form (the
/// record form is used then).
pub fn print(name_text: &str, fields: &[(String, Json)], hook: &dyn ValueSyntax) -> Option<String> {
    let mut kind = fields.iter().find(|(k, _)| k == "kind")?.1.clone();
    if let Json::Object(entries) = &mut kind {
        entries.retain(|(k, _)| k != "feature");
    }
    let sketch: Sketch = kind.decode().ok()?;
    let mut lines = Vec::new();
    for entity in &sketch.entities {
        lines.push(print_entity(entity)?);
    }
    for constraint in &sketch.constraints {
        lines.push(print_constraint(constraint)?);
    }
    let solved = print_solved(&sketch.solved)?;
    if !solved.is_empty() {
        lines.push("solved {".into());
        for line in solved {
            lines.push(format!("{INDENT}{line}"));
        }
        lines.push("}".into());
    }
    for (key, value) in fields {
        match key.as_str() {
            "kind" => {}
            "suppressed" | "frozen" => {
                lines.push(format!("{key} = {}", print_json(value, hook, 1)))
            }
            _ => return None,
        }
    }
    let plane = match print_plane_or_face(&sketch.plane) {
        Some(text) => text,
        None => print_json(&Json::from_serialize(&sketch.plane).ok()?, hook, 0),
    };
    if plane.contains('\n') {
        return None;
    }
    let mut out = format!("sketch {} on {plane} {{\n", name(name_text)?);
    for line in lines {
        out.push_str(INDENT);
        out.push_str(&line);
        out.push('\n');
    }
    out.push('}');
    Some(out)
}

// ----------------------------------------------------------------------------- reading

fn literal_number(expr: &Expr) -> Result<f64, LowerError> {
    let (text, negative) = match expr {
        Expr::Number(number) => (number.text(), false),
        Expr::Unary(unary) if unary.is_negative() => match unary.operand() {
            Some(Expr::Number(number)) => (number.text(), true),
            _ => return Err(error(expr.span(), "expected a number")),
        },
        _ => return Err(error(expr.span(), "expected a number")),
    };
    let value: f64 = text
        .parse()
        .map_err(|_| error(expr.span(), format!("`{text}` is not a number")))?;
    Ok(if negative { -value } else { value })
}

fn entity_point(expr: &Expr) -> Option<Result<(SketchEntityId, PointRole), LowerError>> {
    let Expr::Reference(reference) = expr else {
        return None;
    };
    match reference.segments().as_slice() {
        [entity, role] => Some(
            role_of(role)
                .map(|role| (SketchEntityId::new(entity), role))
                .ok_or_else(|| error(expr.span(), format!("`{role}` is not a sketch point"))),
        ),
        _ => None,
    }
}

fn lower_point(expr: &Expr) -> Result<PointExpr, LowerError> {
    if let Some(of) = entity_point(expr) {
        let (entity, role) = of?;
        return Ok(PointExpr::Of { entity, role });
    }
    match expr {
        Expr::List(list) => match list.items().as_slice() {
            [x, y] => Ok(PointExpr::Coords {
                x: lower_value(x)?,
                y: lower_value(y)?,
            }),
            _ => Err(error(expr.span(), "a sketch point is `[x, y]`")),
        },
        Expr::Quantity(quantity) => match quantity.value() {
            Some(Expr::List(list)) => match list.items().as_slice() {
                [x, y] => {
                    let unit = quantity.unit();
                    let component = |e: &Expr| -> Result<ValueExpr, LowerError> {
                        Ok(ValueExpr::Quantity {
                            value: literal_number(e)?,
                            unit: unit.clone(),
                        })
                    };
                    Ok(PointExpr::Coords {
                        x: component(x)?,
                        y: component(y)?,
                    })
                }
                _ => Err(error(expr.span(), "a sketch point is `[x, y] <unit>`")),
            },
            _ => Err(error(expr.span(), "expected a sketch point")),
        },
        _ => Err(error(
            expr.span(),
            "a sketch point is `[x, y]` or another entity's point (`l1.end`)",
        )),
    }
}

fn lower_target(expr: &Expr) -> Result<SketchTarget, LowerError> {
    if let Some(of) = entity_point(expr) {
        let (entity, role) = of?;
        return Ok(SketchTarget::Point { entity, role });
    }
    match expr {
        Expr::Reference(reference) => match reference.segments().as_slice() {
            [entity] => Ok(SketchTarget::Entity {
                entity: SketchEntityId::new(entity),
            }),
            _ => Err(error(expr.span(), "expected an entity or its point")),
        },
        _ => Err(error(
            expr.span(),
            "expected an entity (`l1`) or its point (`l1.end`)",
        )),
    }
}

struct Call {
    word: String,
    positional: Vec<Expr>,
    named: Vec<(String, Expr)>,
    flags: Vec<String>,
}

fn call_parts(expr: &Expr) -> Result<Call, LowerError> {
    let Expr::Call(call) = expr else {
        return Err(error(expr.span(), "expected `<kind>(…)`"));
    };
    let word = match call.callee() {
        Some(Expr::Reference(reference)) if reference.segments().len() == 1 => {
            reference.segments().remove(0)
        }
        _ => return Err(error(expr.span(), "expected `<kind>(…)`")),
    };
    let mut parts = Call {
        word,
        positional: Vec::new(),
        named: Vec::new(),
        flags: Vec::new(),
    };
    for argument in call.arguments() {
        match argument {
            Argument::Positional(value) => parts.positional.push(value),
            Argument::Named(named) => {
                let key = named.name().map(|n| n.text()).unwrap_or_default();
                let value = named
                    .value()
                    .ok_or_else(|| error(named.span(), format!("`{key}` needs a value")))?;
                if parts.named.iter().any(|(k, _)| *k == key) {
                    return Err(error(named.span(), format!("`{key}` is given twice")));
                }
                parts.named.push((key, value));
            }
        }
    }
    Ok(parts)
}

impl Call {
    fn take(&mut self, key: &str) -> Option<Expr> {
        let position = self.named.iter().position(|(k, _)| k == key)?;
        Some(self.named.remove(position).1)
    }

    fn need(&mut self, key: &str, span: Span) -> Result<Expr, LowerError> {
        self.take(key)
            .ok_or_else(|| error(span, format!("`{}` needs `{key} = …`", self.word)))
    }

    fn done(&self, span: Span) -> Result<(), LowerError> {
        if let Some((key, _)) = self.named.first() {
            return Err(error(span, format!("`{}` takes no `{key}`", self.word)));
        }
        if let Some(flag) = self.flags.first() {
            return Err(error(span, format!("`{}` takes no `{flag}`", self.word)));
        }
        Ok(())
    }
}

fn lower_entity(id: &str, value: &Expr) -> Result<SketchEntity, LowerError> {
    let span = value.span();
    let mut call = call_parts(value)?;
    let mut construction = false;
    let mut closed = false;
    let positional = std::mem::take(&mut call.positional);
    for flag in positional {
        match &flag {
            Expr::Reference(r) if r.segments() == ["construction"] => construction = true,
            Expr::Reference(r) if r.segments() == ["closed"] && call.word == "spline" => {
                closed = true
            }
            other => {
                return Err(error(
                    other.span(),
                    "an entity takes `name = value` arguments",
                ))
            }
        }
    }
    let geometry = match call.word.as_str() {
        "point" => SketchGeometry::Point {
            at: lower_point(&call.need("at", span)?)?,
        },
        "line" => SketchGeometry::Line {
            start: lower_point(&call.need("start", span)?)?,
            end: lower_point(&call.need("end", span)?)?,
        },
        "arc" => SketchGeometry::Arc {
            center: lower_point(&call.need("center", span)?)?,
            start: lower_point(&call.need("start", span)?)?,
            end: lower_point(&call.need("end", span)?)?,
        },
        "circle" => SketchGeometry::Circle {
            center: lower_point(&call.need("center", span)?)?,
            radius: lower_value(&call.need("radius", span)?)?,
        },
        "spline" => {
            let points = call.need("points", span)?;
            let Expr::List(list) = &points else {
                return Err(error(
                    points.span(),
                    "a spline's `points` is a list of sketch points",
                ));
            };
            let points = list
                .items()
                .iter()
                .map(lower_point)
                .collect::<Result<Vec<_>, _>>()?;
            if points.len() < 2 {
                return Err(error(span, "a spline passes through at least two points"));
            }
            SketchGeometry::Spline { points, closed }
        }
        other => {
            return Err(error(
                span,
                format!("`{other}` is not a sketch entity (point, line, arc, circle, spline)"),
            ))
        }
    };
    if let Some(flag) = call.take("construction") {
        construction = match &flag {
            Expr::Reference(r) if r.segments() == ["true"] => true,
            Expr::Reference(r) if r.segments() == ["false"] => false,
            other => return Err(error(other.span(), "`construction` is true or false")),
        };
    }
    call.done(span)?;
    Ok(SketchEntity {
        id: SketchEntityId::new(id),
        geometry,
        construction,
    })
}

fn lower_kind(expr: &Expr, value: Option<&Expr>) -> Result<(K, Option<ConstraintId>), LowerError> {
    let span = expr.span();
    let mut call = call_parts(expr)?;
    let id = match call.take("id") {
        Some(Expr::Reference(r)) if r.segments().len() == 1 => {
            Some(ConstraintId::new(r.segments().remove(0)))
        }
        Some(other) => return Err(error(other.span(), "a constraint id is a name")),
        None => None,
    };
    let value = match (value, call.take("value")) {
        (Some(_), Some(other)) => {
            return Err(error(other.span(), "give the value once, after `=`"))
        }
        (Some(value), None) => Some(lower_value(value)?),
        (None, Some(value)) => Some(lower_value(&value)?),
        (None, None) => None,
    };
    let positional = std::mem::take(&mut call.positional);
    let targets = |count: usize| -> Result<Vec<SketchTarget>, LowerError> {
        if positional.len() != count {
            return Err(error(
                span,
                format!("`{}` takes {count} entities or points", call.word),
            ));
        }
        positional.iter().map(lower_target).collect()
    };
    let need_value = |value: Option<ValueExpr>| {
        value.ok_or_else(|| error(span, format!("`{}` needs `= <value>`", call.word)))
    };
    let no_value = |value: &Option<ValueExpr>| match value {
        Some(_) => Err(error(span, format!("`{}` takes no value", call.word))),
        None => Ok(()),
    };
    let pair = |t: Vec<SketchTarget>| {
        let mut t = t.into_iter();
        (t.next().unwrap(), t.next().unwrap())
    };
    let kind = match call.word.as_str() {
        "coincident" | "parallel" | "perpendicular" | "tangent" | "equal" | "concentric"
        | "coradial" | "collinear" => {
            no_value(&value)?;
            let (a, b) = pair(targets(2)?);
            match call.word.as_str() {
                "coincident" => K::Coincident { a, b },
                "parallel" => K::Parallel { a, b },
                "perpendicular" => K::Perpendicular { a, b },
                "tangent" => K::Tangent { a, b },
                "concentric" => K::Concentric { a, b },
                "coradial" => K::Coradial { a, b },
                "collinear" => K::Collinear { a, b },
                _ => K::Equal { a, b },
            }
        }
        "symmetric" => {
            no_value(&value)?;
            let mut t = targets(3)?.into_iter();
            let (a, b, axis) = (t.next().unwrap(), t.next().unwrap(), t.next().unwrap());
            if !matches!(a, SketchTarget::Point { .. })
                || !matches!(b, SketchTarget::Point { .. })
                || !matches!(axis, SketchTarget::Entity { .. })
            {
                return Err(error(
                    span,
                    "`symmetric` takes two points, then the line between them",
                ));
            }
            K::Symmetric { a, b, axis }
        }
        "midpoint" => {
            no_value(&value)?;
            let (point, line) = pair(targets(2)?);
            if !matches!(point, SketchTarget::Point { .. })
                || !matches!(line, SketchTarget::Entity { .. })
            {
                return Err(error(span, "`midpoint` takes a point, then a line"));
            }
            K::Midpoint { point, line }
        }
        "horizontal" | "vertical" | "fix" => {
            no_value(&value)?;
            let t = targets(1)?.remove(0);
            match call.word.as_str() {
                "horizontal" => K::Horizontal { line: t },
                "vertical" => K::Vertical { line: t },
                _ => K::Fix { target: t },
            }
        }
        "distance" | "angle" => {
            let (a, b) = pair(targets(2)?);
            let value = need_value(value)?;
            if call.word == "distance" {
                K::Distance { a, b, value }
            } else {
                K::Angle { a, b, value }
            }
        }
        "length" => K::Length {
            line: targets(1)?.remove(0),
            value: need_value(value)?,
        },
        "radius" => K::Radius {
            curve: targets(1)?.remove(0),
            value: need_value(value)?,
        },
        "diameter" => K::Diameter {
            curve: targets(1)?.remove(0),
            value: need_value(value)?,
        },
        "limit" => {
            no_value(&value)?;
            let [dimension] = positional.as_slice() else {
                return Err(error(span, "`limit` takes one dimension"));
            };
            let (dimension, _) = lower_kind(dimension, None)?;
            let min = lower_value(&call.need("min", span)?)?;
            let max = lower_value(&call.need("max", span)?)?;
            K::Limit {
                dimension: Box::new(dimension),
                min,
                max,
            }
        }
        other => return Err(error(span, format!("`{other}` is not a sketch constraint"))),
    };
    call.done(span)?;
    Ok((kind, id))
}

fn solved_key(target: &[String], span: Span) -> Result<SolvedKey, LowerError> {
    match target {
        [entity, field] if field == "radius" => Ok(SolvedKey::Radius {
            entity: SketchEntityId::new(entity),
        }),
        [owner, field] if field.starts_with("hidden_") => {
            let index = field["hidden_".len()..]
                .parse::<u32>()
                .map_err(|_| error(span, format!("`{field}` is `hidden_<n>`")))?;
            Ok(SolvedKey::Hidden {
                constraint: ConstraintId::new(owner),
                index,
            })
        }
        [entity, role] => Ok(SolvedKey::Point {
            entity: SketchEntityId::new(entity),
            role: role_of(role)
                .ok_or_else(|| error(span, format!("`{role}` is not a sketch point")))?,
        }),
        _ => Err(error(span, "a solved value is `<entity>.<point> = [x, y]`")),
    }
}

fn lower_solved(block: &ast::Block) -> Result<SolvedCache, LowerError> {
    let mut solved = SolvedCache::default();
    for statement in block.statements() {
        let span = statement.span();
        let Statement::Assignment(assignment) = &statement else {
            return Err(error(span, "a solved value is `<entity>.<point> = [x, y]`"));
        };
        let key = solved_key(&assignment.target(), assignment.target_span())?;
        let value = assignment
            .value()
            .ok_or_else(|| error(span, "a solved value needs `= <value>`"))?;
        let duplicate = match &value {
            Expr::List(list) => match list.items().as_slice() {
                [x, y] => solved
                    .points
                    .insert(key, [literal_number(x)?, literal_number(y)?])
                    .is_some(),
                _ => return Err(error(value.span(), "a solved point is `[x, y]`")),
            },
            other => solved.scalars.insert(key, literal_number(other)?).is_some(),
        };
        if duplicate {
            return Err(error(span, "this solved value is given twice"));
        }
    }
    Ok(solved)
}

fn lower_plane(expr: &Expr, hook: &dyn ValueSyntax) -> Result<PlaneRef, LowerError> {
    if let Some(face) = refs::lower_reference(expr) {
        return Ok(PlaneRef::Face { face: face? });
    }
    let json = lower_expr(expr, hook, "", &mut Vec::new())?;
    json.decode::<PlaneRef>().map_err(|_| {
        error(
            expr.span(),
            format!(
                "expected a plane: {}; for example `sketch base on origin.xy {{ … }}`",
                crate::model::PLANE_REFERENCE_FORMS
            ),
        )
    })
}

/// A constraint's key among the sketch's written lines, for structured edits: its id, or
/// what it constrains (its dimension values left out, so a value edit keeps the key). A
/// repeated key takes its ordinal.
fn constraint_key(constraint: &SketchConstraint, written: &[(String, Json, Span)]) -> String {
    fn strip(value: &mut serde_json::Value) {
        if let serde_json::Value::Object(map) = value {
            for key in ["value", "min", "max"] {
                map.remove(key);
            }
            map.values_mut().for_each(strip);
        }
    }
    let base = match &constraint.id {
        Some(id) => format!("constrain.{}", id.0),
        None => {
            let mut value = serde_json::to_value(&constraint.kind).unwrap_or_default();
            strip(&mut value);
            format!("constrain {value}")
        }
    };
    let mut key = base.clone();
    let mut ordinal = 1;
    while written.iter().any(|(k, _, _)| *k == key) {
        ordinal += 1;
        key = format!("{base}~{ordinal}");
    }
    key
}

/// Reads `sketch <name> on <plane> { … }` as a feature.
pub fn lower(
    block: &ast::BlockStatement,
    hook: &dyn ValueSyntax,
) -> Result<BlockEntry, LowerError> {
    let span = block.span();
    let name = block
        .name()
        .ok_or_else(|| error(block.keyword_span(), "a sketch needs a name"))?
        .text();
    let clauses = block.clauses();
    let plane_expr = match clauses.as_slice() {
        [clause] if clause.word() == "on" => clause
            .value()
            .ok_or_else(|| error(clause.span(), "`on` needs a plane"))?,
        _ => return Err(error(span, "a sketch is `sketch <name> on <plane> { … }`")),
    };
    let body = block
        .block()
        .ok_or_else(|| error(span, "a sketch needs `{ … }`"))?;
    let plane = lower_plane(&plane_expr, hook)?;
    let mut spans: Vec<(String, Span)> = vec![("kind.plane".into(), plane_expr.span())];
    let mut written: Vec<(String, Json, Span)> =
        vec![("plane".into(), encode(&plane, span)?, plane_expr.span())];
    let mut entities: Vec<SketchEntity> = Vec::new();
    let mut constraints: Vec<SketchConstraint> = Vec::new();
    let mut solved = None;
    let mut extras: Vec<(String, Json)> = Vec::new();
    let (mut entity_span, mut constraint_span): (Option<Span>, Option<Span>) = (None, None);
    let widen = |slot: &mut Option<Span>, at: Span| {
        *slot = Some(match slot {
            Some(s) => Span {
                start: s.start.min(at.start),
                end: s.end.max(at.end),
            },
            None => at,
        })
    };
    for statement in body.statements() {
        let at = statement.span();
        match &statement {
            Statement::Assignment(assignment) => {
                let target = assignment.target();
                let value = assignment
                    .value()
                    .ok_or_else(|| error(at, "expected `<name> = <value>`"))?;
                let [key] = target.as_slice() else {
                    return Err(error(assignment.target_span(), "expected a name"));
                };
                if key == "suppressed" || key == "frozen" {
                    let mut value_spans = Vec::new();
                    let item = lower_expr(&value, hook, key, &mut value_spans)?;
                    spans.extend(value_spans);
                    written.push((key.clone(), item.clone(), at));
                    extras.push((key.clone(), item));
                    continue;
                }
                if entities.iter().any(|e| e.id.0 == *key) {
                    return Err(error(at, format!("`{key}` is declared twice")));
                }
                let entity = lower_entity(key, &value)?;
                spans.push((format!("kind.entities[{}]", entities.len()), at));
                written.push((key.clone(), encode(&entity, at)?, at));
                widen(&mut entity_span, at);
                entities.push(entity);
            }
            Statement::Constrain(constrain) => {
                let expr = constrain
                    .expression()
                    .ok_or_else(|| error(at, "`constrain` needs a constraint"))?;
                let (kind, id) = lower_kind(&expr, constrain.equals().as_ref())?;
                let constraint = SketchConstraint { id, kind };
                spans.push((format!("kind.constraints[{}]", constraints.len()), at));
                let key = constraint_key(&constraint, &written);
                written.push((key, encode(&constraint, at)?, at));
                widen(&mut constraint_span, at);
                constraints.push(constraint);
            }
            Statement::Solved(block) => {
                if solved.is_some() {
                    return Err(error(at, "a sketch has one `solved { }` block"));
                }
                let inner = block
                    .block()
                    .ok_or_else(|| error(at, "`solved` needs `{ … }`"))?;
                let cache = lower_solved(&inner)?;
                spans.push(("kind.solved".into(), at));
                written.push(("solved".into(), encode(&cache, at)?, at));
                solved = Some(cache);
            }
            _ => return Err(error(
                at,
                "a sketch holds entities (`l1 = line(…)`), `constrain …` lines and `solved { }`",
            )),
        }
    }
    if let Some(at) = entity_span {
        spans.push(("kind.entities".into(), at));
    }
    if let Some(at) = constraint_span {
        spans.push(("kind.constraints".into(), at));
    }
    let sketch = Sketch {
        plane,
        entities,
        constraints,
        solved: solved.unwrap_or_default(),
    };
    let Json::Object(mut kind) = encode(&sketch, span)? else {
        return Err(error(span, "a sketch serialises to an object"));
    };
    kind.insert(0, ("feature".into(), Json::String("sketch".into())));
    let mut fields = vec![("kind".to_string(), Json::Object(kind))];
    fields.extend(extras);
    Ok(BlockEntry {
        word: "feature".into(),
        name: Some(name),
        fields,
        spans,
        written,
        head_end: body.span().start,
        value_span: Some(body.span()),
    })
}
