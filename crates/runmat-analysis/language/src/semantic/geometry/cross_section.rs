//! `cross_section { ... }` geometry: a board stackup and the traces on one copper layer.
//!
//! ```text
//! cross_section {
//!     layer "F.Mask" : solder_mask { thickness = 20 um; permittivity = 3.8 }
//!     layer "F.Cu" : signal { thickness = 35 um }
//!     layer "dielectric 1" : dielectric { thickness = 0.2 mm; permittivity = 4.1 }
//!     layer "In1.Cu" : plane { thickness = 35 um }
//!     trace line : signal { layer = "F.Cu"; width = 0.36 mm; etch_angle = 60 deg }
//! }
//! ```
//!
//! `stackup = "<file>.circuit.layout"` (with `stackup_name = <name>` for a layout with several)
//! takes the layers from the board layout instead; the module graph loaded that file.
mod stackup;

use super::error;
use crate::{
    BoundFeaProgram, CheckedExpression, CheckedExpressionKind, CheckedFeaDeclaration,
    CheckedFeaStatement, CheckedFeaStatementKind, FeaDiagnostic,
};
use runmat_geometry_core::{
    CrossSectionDefinition, CrossSectionEnclosure, CrossSectionLayer, CrossSectionLayerRole,
    CrossSectionStackupSource, CrossSectionTrace, CrossSectionTraceBase, CrossSectionTraceRole,
    GeometryDefinitionSource, GeometryDigest,
};
use runmat_types::Span;

pub(super) fn source(
    bound: &BoundFeaProgram,
    declaration: &CheckedFeaDeclaration,
    statements: &[CheckedFeaStatement],
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<GeometryDefinitionSource> {
    let errors = diagnostics.len();
    let mut reader = Reader {
        declaration,
        diagnostics,
    };
    let mut layers = Vec::new();
    let mut traces = Vec::new();
    let mut enclosure = None;
    let mut stackup_path = None;
    let mut stackup_name = None;
    for statement in statements {
        match &statement.kind {
            CheckedFeaStatementKind::Block {
                header,
                statements: body,
                ..
            } => {
                let header = header.trim();
                if let Some(rest) = header.strip_prefix("layer") {
                    if let Some(layer) = reader.layer(rest, body, statement.span) {
                        layers.push(layer);
                    }
                } else if let Some(rest) = header.strip_prefix("trace") {
                    if let Some(trace) = reader.trace(rest, body, statement.span) {
                        traces.push(trace);
                    }
                } else if header == "enclosure" {
                    enclosure = reader.enclosure(body, statement.span);
                } else {
                    reader.error(
                        statement.span,
                        format!("`{header}` does not belong in a cross_section; expected `layer`, `trace`, `enclosure`, `stackup` or `stackup_name`"),
                    );
                }
            }
            CheckedFeaStatementKind::Assignment { target, value, .. }
                if target.as_slice() == ["stackup"] =>
            {
                match &value.kind {
                    CheckedExpressionKind::String(path) => stackup_path = Some((path, value.span)),
                    _ => reader.error(value.span, "`stackup` is the path of a board layout file"),
                }
            }
            CheckedFeaStatementKind::Assignment { target, value, .. }
                if target.as_slice() == ["stackup_name"] =>
            {
                stackup_name = reader.word(value, "stackup_name");
            }
            _ => reader.error(
                statement.span,
                "a cross_section holds `layer`, `trace` and `enclosure` blocks and the `stackup` and `stackup_name` settings",
            ),
        }
    }
    let mut stackup = None;
    match stackup_path {
        Some((path, span)) => {
            if !layers.is_empty() {
                reader.error(
                    span,
                    "a cross_section takes its layers from `stackup` or from `layer` blocks, not both",
                );
            }
            let (resolved, source) =
                reader.stackup(bound, path, stackup_name.as_deref(), &traces, span)?;
            layers = resolved;
            stackup = Some(source);
        }
        None => {
            if let Some(name) = &stackup_name {
                reader.error(
                    declaration.span,
                    format!("`stackup_name = {name}` needs a `stackup` layout file"),
                );
            }
        }
    }
    if traces.is_empty() {
        reader.error(
            declaration.span,
            "a cross_section needs at least one `trace`",
        );
    }
    if reader.diagnostics.len() != errors {
        return None;
    }
    Some(GeometryDefinitionSource::CrossSection(
        CrossSectionDefinition {
            layers,
            stackup,
            traces,
            enclosure,
        },
    ))
}

struct Reader<'a> {
    declaration: &'a CheckedFeaDeclaration,
    diagnostics: &'a mut Vec<FeaDiagnostic>,
}

impl Reader<'_> {
    fn error(&mut self, span: Span, message: impl Into<String>) {
        error(self.declaration, span, message, self.diagnostics);
    }

    /// `layer "<name>" : <role> { ... }`
    fn layer(
        &mut self,
        rest: &str,
        body: &[CheckedFeaStatement],
        span: Span,
    ) -> Option<CrossSectionLayer> {
        let Some((name, role)) = quoted_name_and_role(rest) else {
            self.error(span, "write a layer as `layer \"<name>\" : <role> { ... }`");
            return None;
        };
        let (role, fields): (_, &[&str]) = match role {
            "dielectric" => (0, &["thickness", "permittivity"]),
            "solder_mask" => (1, &["thickness", "permittivity"]),
            "signal" => (2, &["thickness", "fill_permittivity"]),
            "plane" => (3, &["thickness"]),
            other => {
                self.error(
                    span,
                    format!("`{other}` is not a layer role; expected dielectric, solder_mask, signal or plane"),
                );
                return None;
            }
        };
        self.only(body, fields, &format!("layer \"{name}\""));
        let thickness_m = self.length(body, "thickness", span)?;
        let role = match role {
            0 => CrossSectionLayerRole::Dielectric {
                permittivity: self.number(body, "permittivity", span)?,
            },
            1 => CrossSectionLayerRole::SolderMask {
                permittivity: self.number(body, "permittivity", span)?,
            },
            2 => CrossSectionLayerRole::Signal {
                fill_permittivity: match field(body, "fill_permittivity") {
                    Some(_) => Some(self.number(body, "fill_permittivity", span)?),
                    None => None,
                },
            },
            _ => CrossSectionLayerRole::Plane,
        };
        Some(CrossSectionLayer {
            name,
            thickness_m,
            role,
        })
    }

    /// `trace <name> : signal|ground { layer, width, center, etch_angle, base }`
    fn trace(
        &mut self,
        rest: &str,
        body: &[CheckedFeaStatement],
        span: Span,
    ) -> Option<CrossSectionTrace> {
        let parsed = rest.split_once(':').and_then(|(name, role)| {
            let name = name.trim();
            let role = role.trim();
            (is_identifier(name) && is_identifier(role)).then_some((name, role))
        });
        let Some((name, role)) = parsed else {
            self.error(span, "write a trace as `trace <name> : signal { ... }` or `trace <name> : ground { ... }`");
            return None;
        };
        let role = match role {
            "signal" => CrossSectionTraceRole::Signal,
            "ground" => CrossSectionTraceRole::Ground,
            other => {
                self.error(
                    span,
                    format!("`{other}` is not a trace role; expected signal or ground"),
                );
                return None;
            }
        };
        self.only(
            body,
            &["layer", "width", "center", "etch_angle", "base"],
            &format!("trace {name}"),
        );
        let layer = match field(body, "layer") {
            Some(CheckedExpression {
                kind: CheckedExpressionKind::String(layer),
                ..
            }) => layer.clone(),
            Some(value) => {
                self.error(
                    value.span,
                    "a trace `layer` is the copper layer name in quotes",
                );
                return None;
            }
            None => {
                self.error(span, format!("trace {name} needs `layer`"));
                return None;
            }
        };
        let width_m = self.length(body, "width", span)?;
        let center_m = match field(body, "center") {
            Some(_) => self.length(body, "center", span)?,
            None => 0.0,
        };
        let etch_angle_rad = match field(body, "etch_angle") {
            Some(value) => self.angle(value)?,
            None => std::f64::consts::FRAC_PI_2,
        };
        let base = match field(body, "base") {
            Some(value) => match self.word(value, "base")?.as_str() {
                "top" => Some(CrossSectionTraceBase::Top),
                "bottom" => Some(CrossSectionTraceBase::Bottom),
                other => {
                    self.error(
                        value.span,
                        format!("`base = {other}`: expected top or bottom"),
                    );
                    return None;
                }
            },
            None => None,
        };
        Some(CrossSectionTrace {
            name: name.to_string(),
            role,
            layer,
            center_m,
            width_m,
            etch_angle_rad,
            base,
        })
    }

    fn enclosure(
        &mut self,
        body: &[CheckedFeaStatement],
        span: Span,
    ) -> Option<CrossSectionEnclosure> {
        self.only(body, &["half_width", "air"], "enclosure");
        Some(CrossSectionEnclosure {
            half_width_m: self.length(body, "half_width", span)?,
            air_m: self.length(body, "air", span)?,
        })
    }

    fn stackup(
        &mut self,
        bound: &BoundFeaProgram,
        relative: &str,
        name: Option<&str>,
        traces: &[CrossSectionTrace],
        span: Span,
    ) -> Option<(Vec<CrossSectionLayer>, CrossSectionStackupSource)> {
        let path = crate::resolve_logical_path(
            &bound.graph.source_root,
            &self.declaration.module,
            relative,
        )
        .ok()?;
        // A missing file was reported where the module graph tried to load it.
        let resource = bound.graph.resources.get(&path)?;
        let text = match std::str::from_utf8(&resource.source) {
            Ok(text) => text,
            Err(_) => {
                self.error(span, format!("stackup `{relative}` is not UTF-8 text"));
                return None;
            }
        };
        let stackup = match stackup::read(text, name) {
            Ok(stackup) => stackup,
            Err(message) => {
                self.error(span, format!("stackup `{relative}`: {message}"));
                return None;
            }
        };
        let signal_layers = traces
            .iter()
            .map(|trace| trace.layer.as_str())
            .collect::<std::collections::BTreeSet<_>>();
        let layers = stackup
            .layers
            .into_iter()
            .filter_map(|layer| {
                let role = match layer.kind {
                    stackup::Kind::Mask => CrossSectionLayerRole::SolderMask {
                        permittivity: layer.dk,
                    },
                    stackup::Kind::Dielectric => CrossSectionLayerRole::Dielectric {
                        permittivity: layer.dk,
                    },
                    // The copper the traces are etched on is the signal layer; every other
                    // copper layer of the board is a plane.
                    stackup::Kind::Copper if signal_layers.contains(layer.name.as_str()) => {
                        CrossSectionLayerRole::Signal {
                            fill_permittivity: None,
                        }
                    }
                    stackup::Kind::Copper => CrossSectionLayerRole::Plane,
                    // Legend ink and paste do not change the field.
                    stackup::Kind::Other => return None,
                };
                Some(CrossSectionLayer {
                    name: layer.name,
                    thickness_m: layer.thickness_m,
                    role,
                })
            })
            .collect();
        Some((
            layers,
            CrossSectionStackupSource {
                logical_path: path.as_str().to_string(),
                stackup_name: name.map(str::to_string),
                content_digest: GeometryDigest::from_bytes(resource.source_digest.0),
            },
        ))
    }

    fn only(&mut self, body: &[CheckedFeaStatement], allowed: &[&str], owner: &str) {
        for statement in body {
            match &statement.kind {
                CheckedFeaStatementKind::Assignment { target, .. }
                    if target.len() == 1 && allowed.contains(&target[0].as_str()) => {}
                _ => self.error(
                    statement.span,
                    format!("{owner} accepts {}", allowed.join(", ")),
                ),
            }
        }
    }

    fn length(&mut self, body: &[CheckedFeaStatement], name: &str, span: Span) -> Option<f64> {
        let Some(value) = field(body, name) else {
            self.error(span, format!("`{name}` is required"));
            return None;
        };
        let is_length = value
            .value_type
            .is_quantity(runmat_types::PhysicalDimension::LENGTH.into());
        match scalar(value).filter(|_| is_length) {
            Some(value) => Some(value),
            None => {
                self.error(value.span, format!("`{name}` is a length such as `0.2 mm`"));
                None
            }
        }
    }

    fn number(&mut self, body: &[CheckedFeaStatement], name: &str, span: Span) -> Option<f64> {
        let Some(value) = field(body, name) else {
            self.error(span, format!("`{name}` is required"));
            return None;
        };
        match &value.kind {
            CheckedExpressionKind::Number(text) => text.parse().ok(),
            _ => None,
        }
        .or_else(|| {
            self.error(value.span, format!("`{name}` is a number"));
            None
        })
    }

    fn angle(&mut self, value: &CheckedExpression) -> Option<f64> {
        match &value.kind {
            CheckedExpressionKind::Quantity { unit, .. } if unit.angle_power() == 1 => {
                scalar(value)
            }
            _ => None,
        }
        .or_else(|| {
            self.error(value.span, "`etch_angle` is an angle such as `60 deg`");
            None
        })
    }

    fn word(&mut self, value: &CheckedExpression, name: &str) -> Option<String> {
        match &value.kind {
            CheckedExpressionKind::Reference(path) if path.len() == 1 => Some(path[0].clone()),
            CheckedExpressionKind::Enum(word) | CheckedExpressionKind::String(word) => {
                Some(word.clone())
            }
            _ => {
                self.error(value.span, format!("`{name}` is a name"));
                None
            }
        }
    }
}

fn field<'a>(body: &'a [CheckedFeaStatement], name: &str) -> Option<&'a CheckedExpression> {
    body.iter().find_map(|statement| match &statement.kind {
        CheckedFeaStatementKind::Assignment { target, value, .. }
            if target.len() == 1 && target[0] == name =>
        {
            Some(value)
        }
        _ => None,
    })
}

fn scalar(value: &CheckedExpression) -> Option<f64> {
    match &value.kind {
        CheckedExpressionKind::Number(source) => source.parse().ok(),
        CheckedExpressionKind::Quantity { value, unit, .. } => {
            unit.to_canonical(scalar(value)?).ok()
        }
        CheckedExpressionKind::Unary { operator, operand } => {
            let scalar = scalar(operand)?;
            Some(match operator {
                runmat_design_syntax::FeaUnaryOperator::Positive => scalar,
                runmat_design_syntax::FeaUnaryOperator::Negative => -scalar,
            })
        }
        _ => None,
    }
}

/// `"<name>" : <role>`
fn quoted_name_and_role(rest: &str) -> Option<(String, &str)> {
    let rest = rest.trim_start().strip_prefix('"')?;
    let close = rest.find('"')?;
    let name = rest[..close].to_string();
    let role = rest[close + 1..].trim_start().strip_prefix(':')?.trim();
    (!name.trim().is_empty() && is_identifier(role)).then_some((name, role))
}

fn is_identifier(text: &str) -> bool {
    !text.is_empty() && text.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
}
