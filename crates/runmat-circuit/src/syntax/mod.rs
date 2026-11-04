//! The `.circuit` front end on the shared design grammar.
//!
//! It reads a `.circuit` source's lossless tree into a [`Design`] through plain construction,
//! with a [`SourceKey`] on every declaration, and prints a design back. The statement words are
//! this front end's registry ([`words`]); the grammar knows none of them.
//!
//! ```text
//! runmat 0.7
//! std 1
//!
//! circuit sensor {
//!     root = main
//!     param supply = 3.3 V ± 5 %
//!     net_class power = { trace_width = 0.0005 m }
//!     interface usb2 = { members = [DP, DN], diff_pairs = [[DP, DN]] }
//!     variant lite = { overrides = { U3 = not_fitted } }
//!
//!     sheet main {
//!         port VIN : power
//!         port D : in = bus [D0, D1]
//!         component R1 = { part = "R0402-10K", value = 10000 ohm ± 1 %, classes = [power] }
//!         net GND = { class = power, global = true }
//!         net SW
//!         bus DATA = [D0, D1]
//!         harness USB = usb2
//!         instance pwr = { sheet = power, repeat = 2, bind = { VIN = net.VBUS } }
//!         connect = [U1.VIN, net.VIN, port.VIN]
//!         no_erc = [U1.PG]
//!         diff_pair usb = { p = DP, n = DN }
//!     }
//! }
//! ```
//!
//! Quantities print in SI units (`10000 ohm`), so a design reads back bit for bit; any unit
//! the rules' unit table knows (`10 kohm`, `0.5 mm`) reads in. Endpoints are references
//! (`R1.VIN`, `net.GND`, `port.VIN`, `bus.D.D0`, `harness.USB.DP`, `instance.pwr.VIN`), or
//! the same text quoted when a name is not a word (`"C1.1"`, `"net.+3V3"`,
//! `"instance.ch.VIN#2"` for one channel).

pub mod board;
pub mod board_edit;
pub mod layout;
pub(crate) mod values;

use crate::ir::{
    Binding, BusDecl, ComponentDecl, Connection, Design, DiffPairDecl, Endpoint, Fitting,
    HarnessDecl, InterfaceDef, NetClass, NetDecl, Param, ParamValue, PartRef, PortDecl, PortDir,
    PortKind, SheetDef, SheetInstance, Tolerance, VariantDef,
};
use runmat_design_syntax::ast::{self, Expr, Headed, Item, Statement};
use runmat_design_syntax::json::{format_f64, is_bare_key, is_bare_word, quote, INDENT};
use runmat_design_syntax::{parse_source, Dialect, FeaSyntaxDiagnostic, FeaSyntaxDiagnosticKind};
use runmat_rules::{parse_quantity, BuiltinUnits, Dim, Quantity, SourceKey};
use runmat_types::Span;
use std::collections::BTreeMap;

/// The `.circuit` statement words: the front end's registry.
pub mod words {
    pub const CIRCUIT: &str = "circuit";
    pub const SHEET: &str = "sheet";
    pub const ROOT: &str = "root";
    pub const PARAM: &str = "param";
    pub const NET_CLASS: &str = "net_class";
    pub const INTERFACE: &str = "interface";
    pub const VARIANT: &str = "variant";
    pub const PORT: &str = "port";
    pub const COMPONENT: &str = "component";
    pub const NET: &str = "net";
    pub const BUS: &str = "bus";
    pub const HARNESS: &str = "harness";
    pub const INSTANCE: &str = "instance";
    pub const CONNECT: &str = "connect";
    pub const NO_ERC: &str = "no_erc";
    pub const DIFF_PAIR: &str = "diff_pair";
    /// Every statement word, for completion and the highlighter.
    pub const ALL: &[&str] = &[
        CIRCUIT, SHEET, ROOT, PARAM, NET_CLASS, INTERFACE, VARIANT, PORT, COMPONENT, NET, BUS,
        HARNESS, INSTANCE, CONNECT, NO_ERC, DIFF_PAIR,
    ];
}

/// Port directions by their written word.
const DIRECTIONS: &[(&str, PortDir)] = &[
    ("in", PortDir::In),
    ("out", PortDir::Out),
    ("inout", PortDir::InOut),
    ("passive", PortDir::Passive),
    ("power", PortDir::Power),
];

/// The SI symbol each printed dimension takes (scale 1, so values print exactly).
const SI_SYMBOLS: &[(&str, Dim)] = &[
    ("m", Dim::LENGTH),
    ("kg", Dim::MASS),
    ("s", Dim::TIME),
    ("A", Dim::CURRENT),
    ("K", Dim::TEMPERATURE),
    ("V", Dim::VOLTAGE),
    ("ohm", Dim::RESISTANCE),
    ("S", Dim::CONDUCTANCE),
    ("F", Dim::CAPACITANCE),
    ("H", Dim::INDUCTANCE),
    ("Hz", Dim::FREQUENCY),
    ("W", Dim::POWER),
    ("J", Dim::ENERGY),
    ("C", Dim::CHARGE),
    ("N", Dim::FORCE),
    ("Pa", Dim::PRESSURE),
    ("K/W", Dim::THERMAL_RESISTANCE),
    ("W/mK", Dim::THERMAL_CONDUCTIVITY),
];

/// A parsed `.circuit` file.
#[derive(Debug, Clone, PartialEq)]
pub struct ParsedCircuit {
    /// The design, when the source is free of errors.
    pub design: Option<Design>,
    pub diagnostics: Vec<FeaSyntaxDiagnostic>,
}

fn diagnostic(span: Span, message: impl Into<String>) -> FeaSyntaxDiagnostic {
    FeaSyntaxDiagnostic {
        kind: FeaSyntaxDiagnosticKind::InvalidExpression,
        span,
        message: message.into(),
    }
}

type Read<T> = Result<T, FeaSyntaxDiagnostic>;

struct Reader<'a> {
    path: &'a str,
    diagnostics: Vec<FeaSyntaxDiagnostic>,
}

impl Reader<'_> {
    fn key(&self, span: Span) -> SourceKey {
        SourceKey::new(self.path, span.start as u32, span.end as u32)
    }

    fn note<T>(&mut self, result: Read<T>) -> Option<T> {
        match result {
            Ok(value) => Some(value),
            Err(error) => {
                self.diagnostics.push(error);
                None
            }
        }
    }
}

/// Parses a `.circuit` source.
pub fn parse_circuit(path: &str, source: &str) -> ParsedCircuit {
    let syntax = parse_source(Dialect::Circuit, source);
    let mut reader = Reader {
        path,
        diagnostics: syntax.diagnostics.clone(),
    };
    let mut design: Option<Design> = None;
    let mut loose_sheets = Vec::new();
    for item in syntax.file().items() {
        match item {
            Item::Declaration(declaration) if declaration.keyword() == words::CIRCUIT => {
                if design.is_some() {
                    reader
                        .diagnostics
                        .push(diagnostic(declaration.span(), "a file holds one `circuit`"));
                    continue;
                }
                design = Some(read_circuit(&mut reader, &declaration));
            }
            Item::Declaration(declaration) if declaration.keyword() == words::SHEET => {
                loose_sheets.push(declaration);
            }
            Item::Declaration(_) | Item::Error(_) => {}
            Item::Use(declaration) => reader.diagnostics.push(diagnostic(
                declaration.span(),
                "`use` in a circuit is not read yet; library parts come from the pool",
            )),
            Item::Import(import) => reader.diagnostics.push(diagnostic(
                import.span(),
                "a circuit reads other files with `use`",
            )),
        }
    }
    let mut design = design.unwrap_or_else(|| {
        reader.diagnostics.push(diagnostic(
            Span::default(),
            "expected a `circuit <name> { … }` declaration",
        ));
        Design::new(&Dialect::Circuit.stem(path))
    });
    for sheet in loose_sheets {
        read_sheet(&mut reader, &mut design, &sheet);
    }
    ParsedCircuit {
        design: reader.diagnostics.is_empty().then_some(design),
        diagnostics: reader.diagnostics,
    }
}

fn read_circuit(reader: &mut Reader<'_>, declaration: &ast::Declaration) -> Design {
    let name = declaration.name().map(|n| n.text()).unwrap_or_default();
    let mut design = Design::new(&name);
    let Some(block) = declaration.block() else {
        return design;
    };
    for statement in block.statements() {
        let span = statement.span();
        match &statement {
            Statement::Block(sheet) if sheet.keyword() == words::SHEET => {
                read_sheet(reader, &mut design, sheet)
            }
            Statement::Assignment(assignment) if assignment.target() == [words::ROOT] => {
                if let Some(root) = reader.note(word_of(assignment.value(), span)) {
                    design.root = root;
                }
            }
            Statement::Value(value) => {
                let word = value.keyword();
                let name = value.name().map(|n| n.text()).unwrap_or_default();
                let key = reader.key(span);
                match word.as_str() {
                    words::PARAM => {
                        if let Some(param) = reader.note(read_param(value.value(), span, key)) {
                            design.params.insert(name, param);
                        }
                    }
                    words::NET_CLASS => {
                        if let Some(params) = reader.note(read_params(value.value(), span, &key)) {
                            design.net_classes.insert(
                                name.clone(),
                                NetClass {
                                    name,
                                    params,
                                    source: key,
                                },
                            );
                        }
                    }
                    words::INTERFACE => {
                        if let Some(interface) =
                            reader.note(read_interface(name.clone(), value.value(), span))
                        {
                            design.interfaces.insert(name, interface);
                        }
                    }
                    words::VARIANT => {
                        if let Some(variant) =
                            reader.note(read_variant(name.clone(), value.value(), span))
                        {
                            design.variants.insert(name, variant);
                        }
                    }
                    other => reader.diagnostics.push(diagnostic(
                        span,
                        format!("`{other}` does not start a circuit statement"),
                    )),
                }
            }
            _ => reader.diagnostics.push(diagnostic(
                span,
                "expected a sheet, `root`, `param`, `net_class`, `interface` or `variant`",
            )),
        }
    }
    design
}

fn read_sheet<H: Headed>(reader: &mut Reader<'_>, design: &mut Design, header: &H) {
    let name = header.name().map(|n| n.text()).unwrap_or_default();
    if design.sheets.contains_key(&name) {
        reader.diagnostics.push(diagnostic(
            header.keyword_span(),
            format!("sheet `{name}` is declared twice"),
        ));
        return;
    }
    let mut sheet = SheetDef {
        name: name.clone(),
        ..Default::default()
    };
    if let Some(block) = header.block() {
        for statement in block.statements() {
            read_sheet_statement(reader, &mut sheet, &statement);
        }
    }
    design.sheets.insert(name, sheet);
}

fn read_sheet_statement(reader: &mut Reader<'_>, sheet: &mut SheetDef, statement: &Statement) {
    let span = statement.span();
    let key = reader.key(span);
    match statement {
        Statement::Value(value) => {
            let word = value.keyword();
            let name = value.name().map(|n| n.text()).unwrap_or_default();
            let result: Read<()> = (|| {
                match word.as_str() {
                    words::PARAM => {
                        sheet
                            .params
                            .insert(name, read_param(value.value(), span, key)?);
                    }
                    words::PORT => sheet.ports.push(read_port(name, value, span, key)?),
                    words::COMPONENT => {
                        sheet
                            .components
                            .push(read_component(name, value.value(), span, key)?)
                    }
                    words::NET => sheet.nets.push(read_net(name, value.value(), span, key)?),
                    words::BUS => sheet.buses.push(BusDecl {
                        name,
                        members: words_of(value.value(), span)?,
                        source: key,
                    }),
                    words::HARNESS => sheet.harnesses.push(HarnessDecl {
                        name,
                        interface: word_of(value.value(), span)?,
                        source: key,
                    }),
                    words::INSTANCE => {
                        sheet
                            .instances
                            .push(read_instance(name, value.value(), span, key)?)
                    }
                    words::DIFF_PAIR => {
                        let record = record_of(value.value(), span)?;
                        let mut pair = DiffPairDecl {
                            name,
                            p: String::new(),
                            n: String::new(),
                            class: None,
                            source: key,
                        };
                        for (field, expr) in record {
                            match field.as_str() {
                                "p" => pair.p = word_of(Some(expr), span)?,
                                "n" => pair.n = word_of(Some(expr), span)?,
                                "class" => pair.class = Some(word_of(Some(expr), span)?),
                                other => {
                                    return Err(diagnostic(
                                        span,
                                        format!("a diff pair has no `{other}`"),
                                    ))
                                }
                            }
                        }
                        sheet.diff_pairs.push(pair);
                    }
                    other => {
                        return Err(diagnostic(
                            span,
                            format!("`{other}` does not start a sheet statement"),
                        ))
                    }
                }
                Ok(())
            })();
            reader.note(result);
        }
        Statement::Command(command) if command.word() == words::NET => {
            let name = match command.argument() {
                Some(Expr::Reference(r)) if r.segments().len() == 1 => Some(r.segments().remove(0)),
                Some(Expr::String(s)) => Some(s.value()),
                _ => None,
            };
            match name {
                Some(name) => sheet.nets.push(NetDecl {
                    name,
                    class: None,
                    global: false,
                    params: BTreeMap::new(),
                    renamed_from: None,
                    source: key,
                }),
                None => reader
                    .diagnostics
                    .push(diagnostic(span, "a net is `net <name>`")),
            }
        }
        Statement::Assignment(assignment) => {
            let target = assignment.target();
            let result: Read<()> = (|| {
                match target.as_slice() {
                    [word] if word == words::CONNECT => sheet.connections.push(Connection {
                        ends: endpoints(assignment.value(), span)?,
                        source: key,
                    }),
                    [word] if word == words::NO_ERC => {
                        sheet.no_erc.extend(endpoints(assignment.value(), span)?)
                    }
                    _ => {
                        return Err(diagnostic(
                            span,
                            "expected `connect = [ … ]` or `no_erc = [ … ]`",
                        ))
                    }
                }
                Ok(())
            })();
            reader.note(result);
        }
        _ => reader
            .diagnostics
            .push(diagnostic(span, "this is not a sheet statement")),
    }
}

fn record_of(expr: Option<Expr>, span: Span) -> Read<Vec<(String, Expr)>> {
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

fn word_of(expr: Option<Expr>, span: Span) -> Read<String> {
    match expr {
        Some(Expr::Reference(reference)) if reference.segments().len() == 1 => {
            Ok(reference.segments().remove(0))
        }
        Some(Expr::String(string)) => Ok(string.value()),
        Some(other) => Err(diagnostic(other.span(), "expected a name")),
        None => Err(diagnostic(span, "expected `= <name>`")),
    }
}

fn words_of(expr: Option<Expr>, span: Span) -> Read<Vec<String>> {
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

fn bool_of(expr: Expr) -> Read<bool> {
    match &expr {
        Expr::Reference(reference) if reference.segments() == ["true"] => Ok(true),
        Expr::Reference(reference) if reference.segments() == ["false"] => Ok(false),
        _ => Err(diagnostic(expr.span(), "expected `true` or `false`")),
    }
}

fn u32_of(expr: Expr) -> Read<u32> {
    match &expr {
        Expr::Number(number) => number
            .text()
            .parse::<u32>()
            .map_err(|_| diagnostic(expr.span(), "expected a whole number")),
        _ => Err(diagnostic(expr.span(), "expected a whole number")),
    }
}

fn number_of(expr: &Expr) -> Read<f64> {
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
fn quantity_of(expr: &Expr) -> Read<(f64, Option<String>)> {
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

fn si_of(expr: &Expr) -> Read<Quantity> {
    let (value, unit) = quantity_of(expr)?;
    match unit {
        None => Ok(Quantity::scalar(value)),
        Some(unit) => parse_quantity(&format!("{} {unit}", format_f64(value)), &BuiltinUnits)
            .map_err(|message| diagnostic(expr.span(), message)),
    }
}

/// `± 1 %`, `± [1 %, 2 %]` (plus, minus), `± 0.1 V`, `± [0.2 V, 0.1 V]`.
fn tolerance_of(expr: &Expr) -> Read<Tolerance> {
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

fn param_value(expr: &Expr) -> Read<ParamValue> {
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

fn read_param(expr: Option<Expr>, span: Span, source: SourceKey) -> Read<Param> {
    let expr = expr.ok_or_else(|| diagnostic(span, "a param needs `= <value>`"))?;
    Ok(Param {
        value: param_value(&expr)?,
        source,
    })
}

fn read_params(
    expr: Option<Expr>,
    span: Span,
    source: &SourceKey,
) -> Read<BTreeMap<String, Param>> {
    let mut params = BTreeMap::new();
    for (name, value) in record_of(expr, span)? {
        params.insert(
            name,
            Param {
                value: param_value(&value)?,
                source: source.clone(),
            },
        );
    }
    Ok(params)
}

fn read_interface(name: String, expr: Option<Expr>, span: Span) -> Read<InterfaceDef> {
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

fn part_field(part: &mut Option<PartRef>, field: &str, value: &Expr, span: Span) -> Read<bool> {
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

fn fitting_of(expr: &Expr, span: Span, source: &SourceKey) -> Read<Fitting> {
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
                    params.insert(
                        field,
                        Param {
                            value: param_value(&value)?,
                            source: source.clone(),
                        },
                    );
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

fn fittings_of(expr: Expr, span: Span, source: &SourceKey) -> Read<BTreeMap<String, Fitting>> {
    let mut out = BTreeMap::new();
    for (name, value) in record_of(Some(expr), span)? {
        out.insert(name, fitting_of(&value, span, source)?);
    }
    Ok(out)
}

fn read_variant(name: String, expr: Option<Expr>, span: Span) -> Read<VariantDef> {
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

fn read_port(
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

fn read_component(
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
                component.params.insert(
                    field,
                    Param {
                        value: param_value(&value)?,
                        source: source.clone(),
                    },
                );
            }
        }
    }
    component.part =
        part.ok_or_else(|| diagnostic(span, "a component names its `part` or `entity`"))?;
    component.source = source;
    Ok(component)
}

fn read_net(name: String, expr: Option<Expr>, span: Span, source: SourceKey) -> Read<NetDecl> {
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
                net.params.insert(
                    field,
                    Param {
                        value: param_value(&value)?,
                        source: source.clone(),
                    },
                );
            }
        }
    }
    Ok(net)
}

fn read_instance(
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
        source,
    };
    for (field, value) in record_of(expr, span)? {
        match field.as_str() {
            "sheet" => instance.sheet = word_of(Some(value), span)?,
            "repeat" => instance.repeat = Some(u32_of(value)?),
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

fn endpoints(expr: Option<Expr>, span: Span) -> Read<Vec<Endpoint>> {
    match expr {
        Some(Expr::List(list)) => list.items().iter().map(endpoint_of).collect(),
        Some(other) => Err(diagnostic(other.span(), "expected a list of endpoints")),
        None => Err(diagnostic(span, "expected `= [ … ]`")),
    }
}

fn endpoint_of(expr: &Expr) -> Read<Endpoint> {
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

// ------------------------------------------------------------------------------------ printing

/// Why a design could not be printed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrintError(pub String);

impl std::fmt::Display for PrintError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for PrintError {}

type Print = Result<String, PrintError>;

fn name(text: &str) -> String {
    if is_bare_word(text) {
        text.to_string()
    } else {
        quote(text)
    }
}

fn endpoint(endpoint: &Endpoint) -> Print {
    let text = endpoint_text(endpoint);
    if endpoint_from_text(&text).as_ref() != Some(endpoint) {
        return Err(PrintError(format!(
            "the endpoint `{text}` has no text form"
        )));
    }
    let segments: Vec<&str> = text.split('.').collect();
    let reference = segments.iter().all(|segment| is_bare_key(segment))
        && endpoint_from_text(&segments.join(".")).as_ref() == Some(endpoint)
        && !text.contains('#');
    Ok(if reference { text } else { quote(&text) })
}

fn si(quantity: &Quantity) -> Print {
    if !quantity.value.is_finite() {
        return Err(PrintError("a non-finite quantity".into()));
    }
    let number = format_f64(quantity.value);
    if quantity.dim == Dim::NONE {
        return Ok(number);
    }
    let symbol = SI_SYMBOLS
        .iter()
        .find(|(_, dim)| *dim == quantity.dim)
        .map(|(symbol, _)| *symbol)
        .ok_or_else(|| PrintError(format!("no unit to print the dimension {:?}", quantity.dim)))?;
    Ok(format!("{number} {symbol}"))
}

/// A percentage that reads back (divided by 100) as exactly `fraction`.
fn percent(fraction: f64) -> Print {
    let first = format_f64(fraction * 100.0);
    if first.parse::<f64>().ok().map(|p| p / 100.0) == Some(fraction) {
        return Ok(format!("{first} %"));
    }
    for digits in 1..=20 {
        let text = format!("{:.*}", digits, fraction * 100.0);
        if text.parse::<f64>().ok().map(|p| p / 100.0) == Some(fraction) {
            return Ok(format!("{text} %"));
        }
    }
    Err(PrintError(format!(
        "the relative tolerance {fraction} has no exact percentage"
    )))
}

fn param_text(value: &ParamValue) -> Print {
    Ok(match value {
        ParamValue::Text(text) => quote(text),
        ParamValue::Bool(flag) => flag.to_string(),
        ParamValue::Quantity { nominal, tol } => {
            let nominal_text = si(nominal)?;
            let with_unit = |v: f64| si(&Quantity::new(v, nominal.dim));
            match tol {
                Tolerance::Exact => nominal_text,
                Tolerance::Relative { minus, plus } if minus == plus => {
                    format!("{nominal_text} ± {}", percent(*plus)?)
                }
                Tolerance::Relative { minus, plus } => {
                    format!(
                        "{nominal_text} ± [{}, {}]",
                        percent(*plus)?,
                        percent(*minus)?
                    )
                }
                Tolerance::Absolute { minus, plus } if minus == plus => {
                    format!("{nominal_text} ± {}", with_unit(*plus)?)
                }
                Tolerance::Absolute { minus, plus } => {
                    format!(
                        "{nominal_text} ± [{}, {}]",
                        with_unit(*plus)?,
                        with_unit(*minus)?
                    )
                }
                Tolerance::Range { min, max } => format!(
                    "{{ nominal = {nominal_text}, range = [{}, {}] }}",
                    with_unit(*min)?,
                    with_unit(*max)?
                ),
            }
        }
    })
}

fn key(text: &str) -> String {
    if is_bare_key(text) {
        text.to_string()
    } else {
        quote(text)
    }
}

fn record(entries: &[(String, String)]) -> String {
    if entries.is_empty() {
        return "{}".into();
    }
    let inline = format!(
        "{{ {} }}",
        entries
            .iter()
            .map(|(k, v)| format!("{} = {v}", key(k)))
            .collect::<Vec<_>>()
            .join(", ")
    );
    if inline.len() <= 84 && !inline.contains('\n') {
        return inline;
    }
    let mut out = String::from("{\n");
    for (index, (k, v)) in entries.iter().enumerate() {
        out.push_str(INDENT);
        out.push_str(&format!(
            "{} = {}",
            key(k),
            v.replace('\n', &format!("\n{INDENT}"))
        ));
        if index + 1 < entries.len() {
            out.push(',');
        }
        out.push('\n');
    }
    out.push('}');
    out
}

fn list(items: &[String]) -> String {
    format!("[{}]", items.join(", "))
}

fn names(items: &[String]) -> String {
    list(&items.iter().map(|item| name(item)).collect::<Vec<_>>())
}

fn params_entries(
    params: &BTreeMap<String, Param>,
    reserved: &[&str],
) -> Result<Vec<(String, String)>, PrintError> {
    params
        .iter()
        .map(|(k, p)| {
            if reserved.contains(&k.as_str()) {
                return Err(PrintError(format!(
                    "a param named `{k}` collides with a field"
                )));
            }
            Ok((k.clone(), param_text(&p.value)?))
        })
        .collect()
}

fn part_entries(part: &PartRef) -> Vec<(String, String)> {
    match part {
        PartRef::Part(part) => vec![("part".into(), name(part))],
        PartRef::Entity { entity, package } => {
            let mut out = vec![("entity".into(), name(entity))];
            if let Some(package) = package {
                out.push(("package".into(), name(package)));
            }
            out
        }
    }
}

const COMPONENT_FIELDS: &[&str] = &[
    "part",
    "entity",
    "package",
    "classes",
    "power",
    "renamed_from",
    "variants",
    "properties",
];

fn fitting_text(fitting: &Fitting) -> Print {
    Ok(match fitting {
        Fitting::Fitted => "fitted".into(),
        Fitting::NotFitted => "not_fitted".into(),
        Fitting::Alternate { part, params } => {
            let mut entries = part.as_ref().map(part_entries).unwrap_or_default();
            entries.extend(params_entries(params, &["part", "entity", "package"])?);
            if entries.is_empty() {
                return Err(PrintError(
                    "an empty alternate fitting has no text form".into(),
                ));
            }
            record(&entries)
        }
    })
}

fn fittings(map: &BTreeMap<String, Fitting>) -> Print {
    let entries: Result<Vec<_>, _> = map
        .iter()
        .map(|(k, f)| Ok((k.clone(), fitting_text(f)?)))
        .collect();
    Ok(record(&entries?))
}

/// Prints a design as `.circuit` source (sheets in name order, declarations in model order).
pub fn print_circuit(design: &Design) -> Print {
    let mut body: Vec<String> = Vec::new();
    if design.root != "main" {
        body.push(format!("root = {}", name(&design.root)));
    }
    for (k, param) in &design.params {
        body.push(format!("param {} = {}", name(k), param_text(&param.value)?));
    }
    for (k, class) in &design.net_classes {
        if &class.name != k {
            return Err(PrintError(format!(
                "net class `{k}` is keyed by another name"
            )));
        }
        body.push(format!(
            "net_class {} = {}",
            name(k),
            record(&params_entries(&class.params, &[])?)
        ));
    }
    for (k, interface) in &design.interfaces {
        let mut entries = vec![("members".to_string(), names(&interface.members))];
        if !interface.diff_pairs.is_empty() {
            let pairs: Vec<String> = interface
                .diff_pairs
                .iter()
                .map(|(p, n)| names(&[p.clone(), n.clone()]))
                .collect();
            entries.push(("diff_pairs".into(), list(&pairs)));
        }
        body.push(format!("interface {} = {}", name(k), record(&entries)));
    }
    for (k, variant) in &design.variants {
        let mut entries = Vec::new();
        if let Some(base) = &variant.base {
            entries.push(("base".to_string(), name(base)));
        }
        if !variant.overrides.is_empty() {
            entries.push(("overrides".into(), fittings(&variant.overrides)?));
        }
        body.push(format!("variant {} = {}", name(k), record(&entries)));
    }
    for (k, sheet) in &design.sheets {
        if &sheet.name != k {
            return Err(PrintError(format!("sheet `{k}` is keyed by another name")));
        }
        if !body.is_empty() {
            body.push(String::new());
        }
        body.push(print_sheet(sheet)?);
    }
    let mut out = String::from("runmat 0.7\nstd 1\n\n");
    out.push_str(&format!("circuit {} {{\n", name(&design.name)));
    for statement in body {
        if statement.is_empty() {
            out.push('\n');
            continue;
        }
        out.push_str(INDENT);
        out.push_str(&statement.replace('\n', &format!("\n{INDENT}")));
        out.push('\n');
    }
    out.push_str("}\n");
    Ok(out)
}

fn print_sheet(sheet: &SheetDef) -> Print {
    let mut lines = Vec::new();
    for (k, param) in &sheet.params {
        lines.push(format!("param {} = {}", name(k), param_text(&param.value)?));
    }
    for port in &sheet.ports {
        let dir = DIRECTIONS
            .iter()
            .find(|(_, d)| *d == port.dir)
            .map(|(w, _)| *w)
            .unwrap_or("passive");
        let kind = match &port.kind {
            PortKind::Signal => String::new(),
            PortKind::Bus(members) => format!(" = bus {}", names(members)),
            PortKind::Interface(interface) => format!(" = interface {}", name(interface)),
        };
        lines.push(format!("port {} : {dir}{kind}", name(&port.name)));
    }
    for component in &sheet.components {
        let mut entries = part_entries(&component.part);
        if !component.classes.is_empty() {
            entries.push(("classes".into(), names(&component.classes)));
        }
        if let Some(power) = &component.power {
            entries.push(("power".into(), si(power)?));
        }
        if let Some(renamed) = &component.renamed_from {
            entries.push(("renamed_from".into(), name(renamed)));
        }
        if !component.variants.is_empty() {
            entries.push(("variants".into(), fittings(&component.variants)?));
        }
        if !component.properties.is_empty() {
            let properties: Vec<(String, String)> = component
                .properties
                .iter()
                .map(|(k, v)| (k.clone(), quote(v)))
                .collect();
            entries.push(("properties".into(), record(&properties)));
        }
        entries.extend(params_entries(&component.params, COMPONENT_FIELDS)?);
        lines.push(format!(
            "component {} = {}",
            name(&component.name),
            record(&entries)
        ));
    }
    for net in &sheet.nets {
        let mut entries = Vec::new();
        if let Some(class) = &net.class {
            entries.push(("class".to_string(), name(class)));
        }
        if net.global {
            entries.push(("global".into(), "true".into()));
        }
        if let Some(renamed) = &net.renamed_from {
            entries.push(("renamed_from".into(), name(renamed)));
        }
        entries.extend(params_entries(
            &net.params,
            &["class", "global", "renamed_from"],
        )?);
        lines.push(if entries.is_empty() {
            format!("net {}", name(&net.name))
        } else {
            format!("net {} = {}", name(&net.name), record(&entries))
        });
    }
    for bus in &sheet.buses {
        lines.push(format!("bus {} = {}", name(&bus.name), names(&bus.members)));
    }
    for harness in &sheet.harnesses {
        lines.push(format!(
            "harness {} = {}",
            name(&harness.name),
            name(&harness.interface)
        ));
    }
    for instance in &sheet.instances {
        let mut entries = vec![("sheet".to_string(), name(&instance.sheet))];
        if let Some(repeat) = instance.repeat {
            entries.push(("repeat".into(), repeat.to_string()));
        }
        if let Some(renamed) = &instance.renamed_from {
            entries.push(("renamed_from".into(), name(renamed)));
        }
        if !instance.bindings.is_empty() {
            let mut bind = Vec::new();
            for (port, binding) in &instance.bindings {
                if bind.iter().any(|(k, _): &(String, String)| k == port) {
                    return Err(PrintError(format!(
                        "instance `{}` binds `{port}` twice",
                        instance.name
                    )));
                }
                let text = match binding {
                    Binding::Shared(end) => endpoint(end)?,
                    Binding::BusIndexed { bus, offset } => {
                        format!("{{ bus = {}, offset = {offset} }}", name(bus))
                    }
                    Binding::NetIndexed { prefix, offset } => {
                        format!("{{ net_prefix = {}, offset = {offset} }}", name(prefix))
                    }
                };
                bind.push((port.clone(), text));
            }
            entries.push(("bind".into(), record(&bind)));
        }
        lines.push(format!(
            "instance {} = {}",
            name(&instance.name),
            record(&entries)
        ));
    }
    for connection in &sheet.connections {
        let ends: Result<Vec<String>, _> = connection.ends.iter().map(endpoint).collect();
        lines.push(format!("connect = {}", list(&ends?)));
    }
    if !sheet.no_erc.is_empty() {
        let ends: Result<Vec<String>, _> = sheet.no_erc.iter().map(endpoint).collect();
        lines.push(format!("no_erc = {}", list(&ends?)));
    }
    for pair in &sheet.diff_pairs {
        let mut entries = vec![
            ("p".to_string(), name(&pair.p)),
            ("n".to_string(), name(&pair.n)),
        ];
        if let Some(class) = &pair.class {
            entries.push(("class".into(), name(class)));
        }
        lines.push(format!(
            "diff_pair {} = {}",
            name(&pair.name),
            record(&entries)
        ));
    }
    let mut out = format!("sheet {} {{\n", name(&sheet.name));
    for line in lines {
        out.push_str(INDENT);
        out.push_str(&line.replace('\n', &format!("\n{INDENT}")));
        out.push('\n');
    }
    out.push('}');
    Ok(out)
}
