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
pub mod circuit_edit;
pub mod layout;
pub mod footprint_decl;
pub mod part_decl;
pub mod symbol_decl;
pub mod simulation;
pub mod simulation_print;
pub mod simulation_spans;
mod forms;
mod printer;
pub use forms::*;
pub use printer::*;
pub(crate) mod values;

use crate::ir::{
    Binding, BlanketDecl, BusDecl, ComponentDecl, Connection, Design, DiffPairDecl, Endpoint, Fitting,
    HarnessDecl, InterfaceDef, NetClass, NetDecl, Param, ParamValue, PartRef, PortDecl, PortDir,
    PortKind, SheetDef, SheetInstance, Tolerance, ValueCall, VariantDef,
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
    /// One directive over several nets (`blanket rails = { nets = [VIN, GND], class = power }`).
    pub const BLANKET: &str = "blanket";
    /// A note on a sheet (`text note1 = "5 V in from J1"`); its place is in the lock.
    pub const TEXT: &str = "text";
    pub const SIMULATION: &str = "simulation";
    /// Statements inside a `simulation` section.
    pub const SOURCE: &str = "source";
    pub const MODEL: &str = "model";
    pub const TOLERANCE: &str = "tolerance";
    pub const ANALYSIS: &str = "analysis";
    pub const PROBE: &str = "probe";
    pub const MEASURE: &str = "measure";
    pub const OPTION: &str = "option";
    /// A catalog part embedded in the circuit (`part "<mpn>" = { … }`).
    pub const PART: &str = "part";
    /// A footprint declared in the file (`footprint <name> = { … }`).
    pub const FOOTPRINT: &str = "footprint";
    /// A schematic symbol declared in the file (`symbol <name> = { … }`).
    pub const SYMBOL: &str = "symbol";
    /// Every statement word, for completion and the highlighter.
    pub const ALL: &[&str] = &[
        CIRCUIT, SHEET, ROOT, PARAM, NET_CLASS, INTERFACE, VARIANT, PORT, COMPONENT, NET, BUS,
        HARNESS, INSTANCE, CONNECT, NO_ERC, DIFF_PAIR, BLANKET, TEXT, SIMULATION, SOURCE, MODEL, TOLERANCE,
        ANALYSIS, PROBE, MEASURE, OPTION, PART, FOOTPRINT, SYMBOL,
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
    let mut uses: Vec<crate::ir::CircuitUse> = Vec::new();
    let mut declared = Design::default();
    for item in syntax.file().items() {
        match item {
            Item::Declaration(declaration)
                if [words::FOOTPRINT, words::SYMBOL, words::PART].contains(&declaration.keyword().as_str()) =>
            {
                let name = declaration.name().map(|n| n.text()).unwrap_or_default();
                read_declared(&mut reader, &mut declared, &declaration.keyword(), name, declaration.value(), declaration.span());
            }
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
            Item::Use(declaration) => {
                if let Some(u) = reader.note(read_use(&declaration, &uses)) {
                    uses.push(u);
                }
            }
            Item::Import(import) => reader.diagnostics.push(diagnostic(
                import.span(),
                "a circuit reads other files with `use`",
            )),
        }
    }
    let library = design.is_none() && loose_sheets.is_empty();
    let mut design = design.unwrap_or_else(|| {
        if !library {
            reader.diagnostics.push(diagnostic(
                Span::default(),
                "expected a `circuit <name> { … }` declaration",
            ));
        }
        Design::new(&Dialect::Circuit.stem(path))
    });
    design.library = library;
    for sheet in loose_sheets {
        read_sheet(&mut reader, &mut design, &sheet);
    }
    for (name, f) in declared.footprints {
        if design.footprints.insert(name.clone(), f).is_some() {
            reader.diagnostics.push(diagnostic(Span::default(), format!("footprint `{name}` is declared twice")));
        }
    }
    for (name, s) in declared.symbols {
        if design.symbols.insert(name.clone(), s).is_some() {
            reader.diagnostics.push(diagnostic(Span::default(), format!("symbol `{name}` is declared twice")));
        }
    }
    for (name, p) in declared.parts {
        if design.parts.insert(name.clone(), p).is_some() {
            reader.diagnostics.push(diagnostic(Span::default(), format!("part `{name}` is declared twice")));
        }
    }
    design.uses = uses;
    ParsedCircuit {
        design: reader.diagnostics.is_empty().then_some(design),
        diagnostics: reader.diagnostics,
    }
}

/// A `use` line, checked against the kinds a circuit takes and the names already used.
fn read_use(
    declaration: &ast::UseDeclaration,
    earlier: &[crate::ir::CircuitUse],
) -> Read<crate::ir::CircuitUse> {
    let item = board::use_item(declaration)?;
    let u = crate::ir::CircuitUse {
        symbol: item.symbol,
        from: item.from,
        alias: item.alias,
        revision: item.revision,
    };
    if let Some((kind, _)) = &u.symbol {
        if !crate::ir::CircuitUse::KINDS.contains(&kind.as_str()) {
            return Err(diagnostic(
                declaration.span(),
                format!(
                    "a circuit uses {}; `{kind}` is not one of them",
                    crate::ir::CircuitUse::KINDS.join(", ")
                ),
            ));
        }
    } else if u.is_library_file() {
        if u.alias.is_some() {
            return Err(diagnostic(
                declaration.span(),
                format!("a library brings its declarations in by their own names: write `use \"{}\"`", u.from),
            ));
        }
        if earlier.iter().any(|e| e.from == u.from) {
            return Err(diagnostic(declaration.span(), format!("\"{}\" is already used", u.from)));
        }
    } else if u.alias.is_none() {
        return Err(diagnostic(
            declaration.span(),
            "a whole folder is used under a name: `use \"<folder>\" as <name>`; a library file is used as `use \"<path>.circuit\"`",
        ));
    }
    if let Some(name) = u.local_name() {
        if earlier.iter().any(|e| e.local_name() == Some(name)) {
            return Err(diagnostic(
                declaration.span(),
                format!("`{name}` is already used; give one of them `as <name>`"),
            ));
        }
    }
    Ok(u)
}

fn use_text(u: &crate::ir::CircuitUse) -> String {
    let mut out = String::from("use ");
    if let Some((kind, name)) = &u.symbol {
        out.push_str(&format!("{kind} {name}"));
        if !u.from.is_empty() {
            out.push_str(" from ");
        }
    }
    if !u.from.is_empty() {
        out.push_str(&quote(&u.from));
    }
    if let Some(r) = &u.revision {
        out.push_str(&format!(" @ {}", quote(r)));
    }
    if let Some(a) = &u.alias {
        out.push_str(&format!(" as {a}"));
    }
    out
}

/// Brings in what a design's `use` lines name: the sheets, interfaces and net classes of used
/// circuits (`load` resolves a `use` line to the design that declares it). A `sheet` or `interface`
/// use takes that one item under its local name, with the sheets it instances. The design's
/// own declarations win over used ones of the same name; a name two uses bring in is an
/// error. `library` uses name part libraries, which the pool resolves.
pub fn resolve_uses(
    design: &mut Design,
    load: &dyn Fn(&crate::ir::CircuitUse) -> Result<Design, String>,
) -> Result<(), String> {
    let mut brought: BTreeMap<String, String> = BTreeMap::new();
    let mut claim = |kind: &str, name: &str, from: &str| -> Result<(), String> {
        match brought.insert(format!("{kind} {name}"), from.to_owned()) {
            Some(other) if other != from => Err(format!(
                "{kind} `{name}` comes from both \"{other}\" and \"{from}\""
            )),
            _ => Ok(()),
        }
    };
    for u in design.uses.clone() {
        let Some((kind, name)) = &u.symbol else {
            continue;
        };
        if kind == "library" {
            continue;
        }
        let used = load(&u)?;
        let local = u.local_name().unwrap_or(name).to_owned();
        let mut add_sheet = |design: &mut Design, sheet: &str, as_name: &str| -> Result<(), String> {
            // The sheet and every sheet it instances, transitively.
            let mut todo = vec![(sheet.to_owned(), as_name.to_owned())];
            while let Some((s, local)) = todo.pop() {
                let mut def = used
                    .sheets
                    .get(&s)
                    .cloned()
                    .ok_or_else(|| format!("\"{}\" has no sheet `{s}`", u.from))?;
                for i in &def.instances {
                    todo.push((i.sheet.clone(), i.sheet.clone()));
                }
                if design.sheets.contains_key(&local) {
                    continue;
                }
                claim("sheet", &local, &u.from)?;
                def.name = local.clone();
                design.sheets.insert(local, def);
            }
            Ok(())
        };
        match kind.as_str() {
            "circuit" => {
                let sheets: Vec<String> = used.sheets.keys().cloned().collect();
                for s in sheets {
                    add_sheet(design, &s, &s)?;
                }
                for (k, i) in &used.interfaces {
                    claim("interface", k, &u.from)?;
                    design.interfaces.entry(k.clone()).or_insert_with(|| i.clone());
                }
                for (k, c) in &used.net_classes {
                    claim("net_class", k, &u.from)?;
                    design.net_classes.entry(k.clone()).or_insert_with(|| c.clone());
                }
            }
            "sheet" => add_sheet(design, name, &local)?,
            "interface" => {
                let i = used
                    .interfaces
                    .get(name)
                    .ok_or_else(|| format!("\"{}\" has no interface `{name}`", u.from))?;
                claim("interface", &local, &u.from)?;
                design.interfaces.entry(local).or_insert_with(|| i.clone());
            }
            "net_class" => {
                let c = used
                    .net_classes
                    .get(name)
                    .ok_or_else(|| format!("\"{}\" has no net class `{name}`", u.from))?;
                claim("net_class", &local, &u.from)?;
                let mut c = c.clone();
                c.name = local.clone();
                design.net_classes.entry(local).or_insert(c);
            }
            _ => {}
        }
    }
    Ok(())
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
            Statement::Block(sim) if sim.keyword() == words::SIMULATION => {
                if let Some(s) = simulation::read(reader, sim) {
                    if design.simulations.contains_key(&s.name) {
                        reader.diagnostics.push(diagnostic(span, format!("simulation `{}` is declared twice", s.name)));
                    } else {
                        design.simulations.insert(s.name.clone(), s);
                    }
                }
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
                    words::PART | words::FOOTPRINT | words::SYMBOL => {
                        read_declared(reader, &mut design, &word, name, value.value(), span)
                    }
                    other => reader.diagnostics.push(diagnostic(
                        span,
                        format!("`{other}` does not start a circuit statement"),
                    )),
                }
            }
            _ => reader.diagnostics.push(diagnostic(
                span,
                "expected a sheet, a simulation, `root`, `param`, `net_class`, `interface`, `variant`, `part`, `footprint` or `symbol`",
            )),
        }
    }
    design
}

/// A `part`, `footprint` or `symbol` declaration, in a circuit block or at file level.
fn read_declared(reader: &mut Reader<'_>, design: &mut Design, word: &str, name: String, value: Option<Expr>, span: Span) {
    let twice = |kind: &str| diagnostic(span, format!("{kind} `{name}` is declared twice"));
    match word {
        words::PART => {
            if design.parts.contains_key(&name) {
                reader.diagnostics.push(twice("part"));
            } else if let Some(part) = reader.note(part_decl::read(name.clone(), value, span)) {
                design.parts.insert(name, part);
            }
        }
        words::FOOTPRINT => {
            if design.footprints.contains_key(&name) {
                reader.diagnostics.push(twice("footprint"));
            } else if let Some(f) = reader.note(footprint_decl::read(name.clone(), value, span)) {
                design.footprints.insert(name, f);
            }
        }
        _ => {
            if design.symbols.contains_key(&name) {
                reader.diagnostics.push(twice("symbol"));
            } else if let Some(s) = reader.note(symbol_decl::read(name.clone(), value, span)) {
                design.symbols.insert(name, s);
            }
        }
    }
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
                    words::TEXT => match value.value() {
                        Some(Expr::String(s)) => sheet.texts.push(crate::ir::TextDecl { name, text: s.value(), source: key }),
                        _ => return Err(diagnostic(span, "a text is `text <name> = \"…\"`")),
                    },
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
                    words::BLANKET => sheet.blankets.push(read_blanket(name, value.value(), span, key)?),
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
