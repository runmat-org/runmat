//! The `.circuit` reader's value forms: records, words, numbers, quantities, tolerances,
//! parameters, parts, fittings, variants, ports, components, nets, instances and endpoints.

use super::*;

pub(super) fn record_of(expr: Option<Expr>, span: Span) -> Read<Vec<(String, Expr)>> {
    match expr {
        Some(Expr::Record(record)) => record
            .entries()
            .into_iter()
            .map(|entry| {
                let name = entry.name().map(|n| n.text()).unwrap_or_default();
                let value = entry
                    .value()
                    .ok_or_else(|| diagnostic(entry.span(), format!("`{name}` needs a value")))?;
                Ok((name, value))
            })
            .collect(),
        Some(other) => Err(diagnostic(other.span(), "expected a record `{ … }`")),
        None => Err(diagnostic(span, "expected `= { … }`")),
    }
}

pub(super) fn word_of(expr: Option<Expr>, span: Span) -> Read<String> {
    match expr {
        Some(Expr::Reference(reference)) if reference.segments().len() == 1 => {
            Ok(reference.segments().remove(0))
        }
        Some(Expr::String(string)) => Ok(string.value()),
        Some(other) => Err(diagnostic(other.span(), "expected a name")),
        None => Err(diagnostic(span, "expected `= <name>`")),
    }
}

pub(super) fn words_of(expr: Option<Expr>, span: Span) -> Read<Vec<String>> {
    match expr {
        Some(Expr::List(list)) => list
            .items()
            .into_iter()
            .map(|item| word_of(Some(item), span))
            .collect(),
        Some(other) => Err(diagnostic(other.span(), "expected a list of names")),
        None => Err(diagnostic(span, "expected `= [ … ]`")),
    }
}

pub(super) fn bool_of(expr: Expr) -> Read<bool> {
    match &expr {
        Expr::Reference(reference) if reference.segments() == ["true"] => Ok(true),
        Expr::Reference(reference) if reference.segments() == ["false"] => Ok(false),
        _ => Err(diagnostic(expr.span(), "expected `true` or `false`")),
    }
}

pub(super) fn u32_of(expr: Expr) -> Read<u32> {
    match &expr {
        Expr::Number(number) => number
            .text()
            .parse::<u32>()
            .map_err(|_| diagnostic(expr.span(), "expected a whole number")),
        _ => Err(diagnostic(expr.span(), "expected a whole number")),
    }
}

pub(super) fn number_of(expr: &Expr) -> Read<f64> {
    match expr {
        Expr::Number(number) => number
            .text()
            .parse::<f64>()
            .map_err(|_| diagnostic(expr.span(), "expected a number")),
        Expr::Unary(unary) if unary.is_negative() => {
            Ok(-number_of(&unary.operand().ok_or_else(|| {
                diagnostic(expr.span(), "expected a number")
            })?)?)
        }
        _ => Err(diagnostic(expr.span(), "expected a number")),
    }
}

/// A number with an optional unit; `%` is returned as its own marker.
pub(super) fn quantity_of(expr: &Expr) -> Read<(f64, Option<String>)> {
    match expr {
        Expr::Quantity(quantity) => {
            let value = quantity
                .value()
                .ok_or_else(|| diagnostic(expr.span(), "expected a number"))?;
            Ok((number_of(&value)?, Some(quantity.unit())))
        }
        Expr::Unary(unary) if unary.is_negative() => {
            let operand = unary
                .operand()
                .ok_or_else(|| diagnostic(expr.span(), "expected a number"))?;
            let (value, unit) = quantity_of(&operand)?;
            Ok((-value, unit))
        }
        _ => Ok((number_of(expr)?, None)),
    }
}

pub(super) fn si_of(expr: &Expr) -> Read<Quantity> {
    let (value, unit) = quantity_of(expr)?;
    match unit {
        None => Ok(Quantity::scalar(value)),
        Some(unit) => parse_quantity(&format!("{} {unit}", format_f64(value)), &BuiltinUnits)
            .map_err(|message| diagnostic(expr.span(), message)),
    }
}

/// `± 1 %`, `± [1 %, 2 %]` (plus, minus), `± 0.1 V`, `± [0.2 V, 0.1 V]`.
pub(super) fn tolerance_of(expr: &Expr) -> Read<Tolerance> {
    let pair = match expr {
        Expr::List(list) => match list.items().as_slice() {
            [plus, minus] => Some((plus.clone(), minus.clone())),
            _ => {
                return Err(diagnostic(
                    expr.span(),
                    "an asymmetric tolerance is `[plus, minus]`",
                ))
            }
        },
        _ => None,
    };
    let (plus, minus) = pair.unwrap_or_else(|| (expr.clone(), expr.clone()));
    let (plus_value, plus_unit) = quantity_of(&plus)?;
    let (minus_value, minus_unit) = quantity_of(&minus)?;
    if plus_unit.as_deref() == Some("%") && minus_unit.as_deref() == Some("%") {
        return Ok(Tolerance::Relative {
            minus: minus_value / 100.0,
            plus: plus_value / 100.0,
        });
    }
    Ok(Tolerance::Absolute {
        minus: si_of(&minus)?.value,
        plus: si_of(&plus)?.value,
    })
}

pub(super) fn param_value(expr: &Expr) -> Read<ParamValue> {
    match expr {
        Expr::String(string) => Ok(ParamValue::Text(string.value())),
        Expr::Reference(reference) if matches!(reference.segments().as_slice(), [w] if w == "true" || w == "false") => {
            Ok(ParamValue::Bool(reference.segments()[0] == "true"))
        }
        Expr::Toleranced(toleranced) => {
            let nominal = toleranced
                .nominal()
                .ok_or_else(|| diagnostic(expr.span(), "missing nominal"))?;
            let tolerance = toleranced
                .tolerance()
                .ok_or_else(|| diagnostic(expr.span(), "missing tolerance"))?;
            Ok(ParamValue::Quantity {
                nominal: si_of(&nominal)?,
                tol: tolerance_of(&tolerance)?,
            })
        }
        Expr::Record(_) => {
            let mut nominal = None;
            let mut range = None;
            for (field, value) in record_of(Some(expr.clone()), expr.span())? {
                match field.as_str() {
                    "nominal" => nominal = Some(si_of(&value)?),
                    "range" => match &value {
                        Expr::List(list) if list.items().len() == 2 => {
                            let items = list.items();
                            range = Some((si_of(&items[0])?.value, si_of(&items[1])?.value));
                        }
                        _ => return Err(diagnostic(value.span(), "a range is `[min, max]`")),
                    },
                    other => {
                        return Err(diagnostic(
                            value.span(),
                            format!("a ranged value has no `{other}`"),
                        ))
                    }
                }
            }
            match (nominal, range) {
                (Some(nominal), Some((min, max))) => Ok(ParamValue::Quantity {
                    nominal,
                    tol: Tolerance::Range { min, max },
                }),
                _ => Err(diagnostic(
                    expr.span(),
                    "a ranged value is `{ nominal = …, range = [min, max] }`",
                )),
            }
        }
        other => Ok(ParamValue::Quantity {
            nominal: si_of(other)?,
            tol: Tolerance::Exact,
        }),
    }
}

pub(super) fn read_param(expr: Option<Expr>, span: Span, source: SourceKey) -> Read<Param> {
    let expr = expr.ok_or_else(|| diagnostic(span, "a param needs `= <value>`"))?;
    param_of(&expr, source)
}

/// A param from its value: a literal, or a call of a `.m` function whose result the runtime
/// fills in (`feedbackResistor(vout = 3.3 V, r1 = 10 kohm) as ohm`).
pub(super) fn param_of(expr: &Expr, source: SourceKey) -> Read<Param> {
    if let Some((call, unit)) = value_call(expr) {
        let function = match call.callee() {
            Some(Expr::Reference(r)) => r.segments().join("."),
            _ => {
                return Err(diagnostic(
                    expr.span(),
                    "a computed value calls a function by name",
                ))
            }
        };
        let mut arguments = Vec::new();
        for argument in call.arguments() {
            match argument {
                ast::Argument::Named(named) => {
                    let name = named.name().map(|n| n.text()).unwrap_or_default();
                    let value = named.value().ok_or_else(|| {
                        diagnostic(expr.span(), format!("`{name}` needs a value"))
                    })?;
                    arguments.push((name, si_of(&value)?));
                }
                ast::Argument::Positional(value) => arguments.push((String::new(), si_of(&value)?)),
            }
        }
        // The dimension the result takes; the number waits for the runtime.
        let dim = match &unit {
            Some(u) => {
                parse_quantity(&format!("1 {u}"), &BuiltinUnits)
                    .map_err(|m| diagnostic(expr.span(), m))?
                    .dim
            }
            None => Quantity::scalar(0.0).dim,
        };
        return Ok(Param {
            value: ParamValue::Quantity {
                nominal: Quantity::new(f64::NAN, dim),
                tol: Tolerance::Exact,
            },
            source,
            call: Some(ValueCall {
                text: expr.syntax().text().trim().to_string(),
                function,
                arguments,
                unit,
            }),
            written: None,
        });
    }
    let value = param_value(expr)?;
    // A quantity keeps its written text, so the printer writes it back in the file's units.
    let written = matches!(value, ParamValue::Quantity { .. })
        .then(|| expr.syntax().text().trim().to_string());
    Ok(Param {
        value,
        source,
        call: None,
        written,
    })
}

/// `f(…) as unit` or `f(…)`: the call and its unit.
fn value_call(expr: &Expr) -> Option<(ast::Call, Option<String>)> {
    match expr {
        Expr::Coerce(c) => match c.expr()? {
            Expr::Call(call) => Some((call, Some(c.unit()))),
            _ => None,
        },
        Expr::Call(call) => Some((call.clone(), None)),
        _ => None,
    }
}

pub(super) fn read_params(
    expr: Option<Expr>,
    span: Span,
    source: &SourceKey,
) -> Read<BTreeMap<String, Param>> {
    let mut params = BTreeMap::new();
    for (name, value) in record_of(expr, span)? {
        params.insert(name, param_of(&value, source.clone())?);
    }
    Ok(params)
}

pub(super) fn read_interface(name: String, expr: Option<Expr>, span: Span) -> Read<InterfaceDef> {
    let mut interface = InterfaceDef {
        name,
        members: Vec::new(),
        diff_pairs: Vec::new(),
    };
    for (field, value) in record_of(expr, span)? {
        match field.as_str() {
            "members" => interface.members = words_of(Some(value), span)?,
            "diff_pairs" => {
                let Expr::List(list) = &value else {
                    return Err(diagnostic(value.span(), "diff pairs are `[[p, n], …]`"));
                };
                for pair in list.items() {
                    match words_of(Some(pair.clone()), span)?.as_slice() {
                        [p, n] => interface.diff_pairs.push((p.clone(), n.clone())),
                        _ => return Err(diagnostic(pair.span(), "a diff pair is `[p, n]`")),
                    }
                }
            }
            other => {
                return Err(diagnostic(
                    value.span(),
                    format!("an interface has no `{other}`"),
                ))
            }
        }
    }
    Ok(interface)
}

pub(super) fn part_field(
    part: &mut Option<PartRef>,
    field: &str,
    value: &Expr,
    span: Span,
) -> Read<bool> {
    match field {
        "part" => *part = Some(PartRef::Part(word_of(Some(value.clone()), span)?)),
        "entity" => {
            let package = match part.take() {
                Some(PartRef::Entity { package, .. }) => package,
                _ => None,
            };
            *part = Some(PartRef::Entity {
                entity: word_of(Some(value.clone()), span)?,
                package,
            })
        }
        "package" => match part {
            Some(PartRef::Entity { package, .. }) => {
                *package = Some(word_of(Some(value.clone()), span)?)
            }
            _ => return Err(diagnostic(value.span(), "`package` follows `entity`")),
        },
        _ => return Ok(false),
    }
    Ok(true)
}

pub(super) fn fitting_of(expr: &Expr, span: Span, source: &SourceKey) -> Read<Fitting> {
    match expr {
        Expr::Reference(reference) if reference.segments() == ["fitted"] => Ok(Fitting::Fitted),
        Expr::Reference(reference) if reference.segments() == ["not_fitted"] => {
            Ok(Fitting::NotFitted)
        }
        Expr::Record(_) => {
            let mut part = None;
            let mut params = BTreeMap::new();
            for (field, value) in record_of(Some(expr.clone()), span)? {
                if !part_field(&mut part, &field, &value, span)? {
                    params.insert(field, param_of(&value, source.clone())?);
                }
            }
            Ok(Fitting::Alternate { part, params })
        }
        _ => Err(diagnostic(
            expr.span(),
            "a fitting is `fitted`, `not_fitted` or `{ part = …, … }`",
        )),
    }
}

pub(super) fn fittings_of(
    expr: Expr,
    span: Span,
    source: &SourceKey,
) -> Read<BTreeMap<String, Fitting>> {
    let mut out = BTreeMap::new();
    for (name, value) in record_of(Some(expr), span)? {
        out.insert(name, fitting_of(&value, span, source)?);
    }
    Ok(out)
}

pub(super) fn read_variant(name: String, expr: Option<Expr>, span: Span) -> Read<VariantDef> {
    let mut variant = VariantDef {
        name,
        ..Default::default()
    };
    for (field, value) in record_of(expr, span)? {
        match field.as_str() {
            "base" => variant.base = Some(word_of(Some(value), span)?),
            "overrides" => variant.overrides = fittings_of(value, span, &SourceKey::default())?,
            other => {
                return Err(diagnostic(
                    value.span(),
                    format!("a variant has no `{other}`"),
                ))
            }
        }
    }
    Ok(variant)
}

pub(super) fn read_port(
    name: String,
    value: &ast::ValueStatement,
    span: Span,
    source: SourceKey,
) -> Read<PortDecl> {
    let direction = value.type_text().unwrap_or_default();
    let dir = DIRECTIONS
        .iter()
        .find(|(word, _)| *word == direction)
        .map(|(_, dir)| *dir)
        .ok_or_else(|| {
            diagnostic(
                span,
                "a port is `port <name> : in | out | inout | passive | power`",
            )
        })?;
    let kind = match value.value() {
        None => PortKind::Signal,
        Some(Expr::PrefixApplication(application)) => {
            let word = match application.callee() {
                Some(Expr::Reference(r)) if r.segments().len() == 1 => r.segments().remove(0),
                _ => String::new(),
            };
            match word.as_str() {
                "bus" => PortKind::Bus(words_of(application.argument(), span)?),
                "interface" => PortKind::Interface(word_of(application.argument(), span)?),
                _ => {
                    return Err(diagnostic(
                        span,
                        "a port kind is `bus [ … ]` or `interface <name>`",
                    ))
                }
            }
        }
        Some(other) => {
            return Err(diagnostic(
                other.span(),
                "a port kind is `bus [ … ]` or `interface <name>`",
            ))
        }
    };
    Ok(PortDecl {
        name,
        dir,
        kind,
        source,
    })
}

pub(super) fn read_component(
    name: String,
    expr: Option<Expr>,
    span: Span,
    source: SourceKey,
) -> Read<ComponentDecl> {
    let mut part = None;
    let mut component = ComponentDecl::new(&name, PartRef::Part(String::new()));
    for (field, value) in record_of(expr, span)? {
        if part_field(&mut part, &field, &value, span)? {
            continue;
        }
        match field.as_str() {
            "classes" => component.classes = words_of(Some(value), span)?,
            "power" => component.power = Some(si_of(&value)?),
            "renamed_from" => component.renamed_from = Some(word_of(Some(value), span)?),
            "variants" => component.variants = fittings_of(value, span, &source)?,
            "properties" => {
                for (key, text) in record_of(Some(value), span)? {
                    component.properties.insert(key, word_of(Some(text), span)?);
                }
            }
            _ => {
                component
                    .params
                    .insert(field, param_of(&value, source.clone())?);
            }
        }
    }
    component.part =
        part.ok_or_else(|| diagnostic(span, "a component names its `part` or `entity`"))?;
    component.source = source;
    Ok(component)
}

pub(super) fn read_net(
    name: String,
    expr: Option<Expr>,
    span: Span,
    source: SourceKey,
) -> Read<NetDecl> {
    let mut net = NetDecl {
        name,
        class: None,
        global: false,
        params: BTreeMap::new(),
        renamed_from: None,
        source: source.clone(),
    };
    for (field, value) in record_of(expr, span)? {
        match field.as_str() {
            "class" => net.class = Some(word_of(Some(value), span)?),
            "global" => net.global = bool_of(value)?,
            "renamed_from" => net.renamed_from = Some(word_of(Some(value), span)?),
            _ => {
                net.params.insert(field, param_of(&value, source.clone())?);
            }
        }
    }
    Ok(net)
}

/// `blanket <name> = { nets = [A, B], class = power, <params> }`: the nets are required.
pub(super) fn read_blanket(
    name: String,
    expr: Option<Expr>,
    span: Span,
    source: SourceKey,
) -> Read<BlanketDecl> {
    let mut blanket = BlanketDecl {
        name,
        nets: Vec::new(),
        class: None,
        params: BTreeMap::new(),
        source: source.clone(),
    };
    for (field, value) in record_of(expr, span)? {
        match field.as_str() {
            "nets" => blanket.nets = words_of(Some(value), span)?,
            "class" => blanket.class = Some(word_of(Some(value), span)?),
            "global" | "renamed_from" => {
                return Err(diagnostic(
                    span,
                    format!("a blanket can't set `{field}`; write it on the net"),
                ))
            }
            _ => {
                blanket
                    .params
                    .insert(field, param_of(&value, source.clone())?);
            }
        }
    }
    if blanket.nets.is_empty() {
        return Err(diagnostic(
            span,
            "a blanket names its nets: `nets = [A, B]`",
        ));
    }
    Ok(blanket)
}

pub(super) fn read_instance(
    name: String,
    expr: Option<Expr>,
    span: Span,
    source: SourceKey,
) -> Read<SheetInstance> {
    let mut instance = SheetInstance {
        name,
        sheet: String::new(),
        repeat: None,
        bindings: Vec::new(),
        renamed_from: None,
        designators: None,
        source,
    };
    for (field, value) in record_of(expr, span)? {
        match field.as_str() {
            "sheet" => instance.sheet = word_of(Some(value), span)?,
            "repeat" => instance.repeat = Some(u32_of(value)?),
            "designators" => {
                let template = word_of(Some(value), span)?;
                crate::elab::check_channel_designators(&template)
                    .map_err(|m| diagnostic(span, m))?;
                instance.designators = Some(template);
            }
            "renamed_from" => instance.renamed_from = Some(word_of(Some(value), span)?),
            "bind" => {
                for (port, binding) in record_of(Some(value), span)? {
                    let binding = match &binding {
                        Expr::Record(_) => {
                            let fields = record_of(Some(binding.clone()), span)?;
                            let get = |key: &str| {
                                fields
                                    .iter()
                                    .find(|(k, _)| k == key)
                                    .map(|(_, v)| v.clone())
                            };
                            let offset = get("offset").map(u32_of).transpose()?.unwrap_or(0);
                            match (get("bus"), get("net_prefix")) {
                                (Some(bus), None) => Binding::BusIndexed {
                                    bus: word_of(Some(bus), span)?,
                                    offset,
                                },
                                (None, Some(prefix)) => Binding::NetIndexed {
                                    prefix: word_of(Some(prefix), span)?,
                                    offset,
                                },
                                _ => return Err(diagnostic(
                                    binding.span(),
                                    "a channel binding is `{ bus = … }` or `{ net_prefix = … }`",
                                )),
                            }
                        }
                        other => Binding::Shared(endpoint_of(other)?),
                    };
                    instance.bindings.push((port, binding));
                }
            }
            other => {
                return Err(diagnostic(
                    value.span(),
                    format!("an instance has no `{other}`"),
                ))
            }
        }
    }
    if instance.sheet.is_empty() {
        return Err(diagnostic(span, "an instance names its `sheet`"));
    }
    Ok(instance)
}

pub(super) fn endpoints(expr: Option<Expr>, span: Span) -> Read<Vec<Endpoint>> {
    match expr {
        Some(Expr::List(list)) => list.items().iter().map(endpoint_of).collect(),
        Some(other) => Err(diagnostic(other.span(), "expected a list of endpoints")),
        None => Err(diagnostic(span, "expected `= [ … ]`")),
    }
}

pub(super) fn endpoint_of(expr: &Expr) -> Read<Endpoint> {
    let text = match expr {
        Expr::Reference(reference) => reference.segments().join("."),
        Expr::String(string) => string.value(),
        _ => {
            return Err(diagnostic(
                expr.span(),
                "an endpoint is `<component>.<pin>`, `net.<name>`, `port.<name>`, …",
            ))
        }
    };
    endpoint_from_text(&text)
        .ok_or_else(|| diagnostic(expr.span(), format!("`{text}` is not an endpoint")))
}

/// Reads the endpoint text form (see the module docs).
pub fn endpoint_from_text(text: &str) -> Option<Endpoint> {
    let (head, rest) = text.split_once('.')?;
    let two = || {
        rest.split_once('.')
            .map(|(a, b)| (a.to_string(), b.to_string()))
    };
    Some(match head {
        "net" => Endpoint::Net(rest.to_string()),
        "port" => Endpoint::Port(rest.to_string()),
        "bus" => {
            let (bus, member) = two()?;
            Endpoint::BusMember { bus, member }
        }
        "harness" => {
            let (harness, member) = two()?;
            Endpoint::HarnessMember { harness, member }
        }
        "instance" => {
            let (instance, port) = two()?;
            let (port, channel) = match port.rsplit_once('#') {
                Some((port, channel)) => (port.to_string(), Some(channel.parse().ok()?)),
                None => (port, None),
            };
            Endpoint::InstancePort {
                instance,
                port,
                channel,
            }
        }
        component => Endpoint::Pin {
            component: component.to_string(),
            pin: rest.to_string(),
        },
    })
}

/// The endpoint text form.
pub fn endpoint_text(endpoint: &Endpoint) -> String {
    match endpoint {
        Endpoint::Pin { component, pin } => format!("{component}.{pin}"),
        Endpoint::Net(net) => format!("net.{net}"),
        Endpoint::Port(port) => format!("port.{port}"),
        Endpoint::BusMember { bus, member } => format!("bus.{bus}.{member}"),
        Endpoint::HarnessMember { harness, member } => format!("harness.{harness}.{member}"),
        Endpoint::InstancePort {
            instance,
            port,
            channel,
        } => match channel {
            Some(channel) => format!("instance.{instance}.{port}#{channel}"),
            None => format!("instance.{instance}.{port}"),
        },
    }
}
