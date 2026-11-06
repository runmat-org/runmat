//! Schematic canvas commands as text edits on the `.circuit` file.
//!
//! A `.circuit` file holds connectivity, and the schematic is drawn from it, so each canvas
//! tool writes the statement that makes the connection it draws:
//!
//! - **Wire** (`schematic.place-wire`): two picks (pins, or a wire to continue from) join one
//!   net. The pick already on a net takes the other end in its `connect` statement; two free
//!   pins get a new `connect = [A, B]`. The drawing routes the wire with its corners and
//!   junction dots, and re-routes it when a symbol moves.
//! - **Disconnect** (`schematic.disconnect`): takes a pin out of its `connect` statements.
//! - **Net label, power port, port** ([`wire`]): a name on a pin or wire names the net; a power
//!   port joins a global rail; a port joins the sheet to its parent.
//! - **Sheet symbol, sheet entry, repeat channel, no-ERC, directive, bus, harness** ([`sheet`]).
//!
//! Each edit replaces or inserts whole statements and keeps every other byte, comments
//! included. The edited source is compiled before it's returned, so a wire that would join two
//! differently named nets is refused with the compiler's message.

mod sheet;
mod wire;

use super::edit::{EditError, EditOutput, EditRequest};
use super::{file_change, load_with, Files, Project};
use crate::elab::find_pins;
use crate::ids::NamePath;
use crate::ir::{Connection, Endpoint, SheetDef};
use crate::syntax::board_edit::SourceChange;
use crate::syntax::{connection_statement, endpoint, parse_circuit, words};
use runmat_design_syntax::ast::{Headed, Item, Statement};
use runmat_design_syntax::edit::{apply_edits, TextEdit};
use runmat_design_syntax::json::INDENT;
use runmat_design_syntax::{parse_source, Dialect};
use runmat_types::Span;
use serde_json::Value;

/// The commands this module applies.
pub(super) const COMMANDS: &[&str] = &[
    "schematic.place-wire",
    "schematic.disconnect",
    "schematic.net-label",
    "schematic.power-port",
    "schematic.port",
    "schematic.sheet-symbol",
    "schematic.sheet-entry",
    "schematic.repeat-channel",
    "schematic.no-erc",
    "schematic.directive",
    "schematic.place-bus",
    "schematic.place-harness",
];

pub(super) fn run(
    project: &Project,
    files: &Files,
    request: &EditRequest,
) -> Result<EditOutput, EditError> {
    let mut e = Editor::new(project, request);
    let summary = match request.command.as_str() {
        "schematic.place-wire" => wire::place_wire(&mut e)?,
        "schematic.disconnect" => wire::disconnect(&mut e)?,
        "schematic.net-label" => wire::net_label(&mut e)?,
        "schematic.power-port" => wire::power_port(&mut e)?,
        "schematic.port" => wire::port(&mut e)?,
        "schematic.sheet-symbol" => sheet::sheet_symbol(&mut e)?,
        "schematic.sheet-entry" => sheet::sheet_entry(&mut e)?,
        "schematic.repeat-channel" => sheet::repeat_channel(&mut e)?,
        "schematic.no-erc" => sheet::no_erc(&mut e)?,
        "schematic.directive" => sheet::directive(&mut e)?,
        "schematic.place-bus" => sheet::bus(&mut e)?,
        "schematic.place-harness" => sheet::harness(&mut e)?,
        other => return Err(EditError::new(format!("`{other}` isn't a schematic edit"))),
    };
    e.finish(files, summary)
}

/* ------------------------------------------------------------------------------------------------
 * Where statements are
 * --------------------------------------------------------------------------------------------- */

/// A statement of a sheet block: its first word (`connect`, `net`, `port` …) and its span.
#[derive(Clone, Debug)]
pub(super) struct At {
    pub word: String,
    pub start: usize,
    pub end: usize,
}

#[derive(Clone, Debug)]
pub(super) struct SheetBlock {
    pub name: String,
    /// Just after the block's opening brace.
    pub open: usize,
    /// The block's closing brace.
    pub close: usize,
    pub statements: Vec<At>,
}

/// The `circuit` block's sheets, in source order, and where a new sheet goes.
pub(super) struct CircuitBlocks {
    pub sheets: Vec<SheetBlock>,
    /// Just after the last sheet block (or the circuit's opening brace).
    pub after_sheets: usize,
}

fn first_word(text: &str) -> String {
    text.split(|c: char| c.is_whitespace() || c == '=')
        .next()
        .unwrap_or_default()
        .to_owned()
}

pub(super) fn blocks(text: &str) -> Option<CircuitBlocks> {
    let syntax = parse_source(Dialect::Circuit, text);
    for item in syntax.file().items() {
        let Item::Declaration(d) = item else { continue };
        if d.keyword() != words::CIRCUIT {
            continue;
        }
        let block = d.block()?;
        let open = text[block.span().start..].find('{').map(|i| block.span().start + i + 1)?;
        let mut out = CircuitBlocks { sheets: Vec::new(), after_sheets: open };
        for statement in block.statements() {
            let Statement::Block(sheet) = &statement else { continue };
            if sheet.keyword() != words::SHEET {
                continue;
            }
            out.after_sheets = statement.span().end;
            let Some(body) = sheet.block() else { continue };
            let span = body.span();
            let open = text[span.start..].find('{').map_or(span.start, |i| span.start + i + 1);
            let statements = body
                .statements()
                .iter()
                .map(|s| At {
                    word: first_word(&text[s.span().start..s.span().end]),
                    start: s.span().start,
                    end: s.span().end,
                })
                .collect();
            out.sheets.push(SheetBlock {
                name: sheet.name().map(|n| n.text()).unwrap_or_default(),
                open,
                close: span.end.saturating_sub(1),
                statements,
            });
        }
        return Some(out);
    }
    None
}

pub(super) fn line_start(text: &str, at: usize) -> usize {
    text[..at].rfind('\n').map_or(0, |i| i + 1)
}

pub(super) fn indent_at(text: &str, at: usize) -> String {
    text[line_start(text, at)..]
        .chars()
        .take_while(|c| *c == ' ' || *c == '\t')
        .collect()
}

/* ------------------------------------------------------------------------------------------------
 * Picks
 * --------------------------------------------------------------------------------------------- */

/// A canvas pick, resolved against the compiled netlist.
#[derive(Clone, Debug, PartialEq)]
pub(super) enum End {
    /// `pin:U1.VIN`: a component pin (indices into the netlist).
    Pin { comp: usize, pin: usize },
    /// `net:GND`: a wire, a label or a sheet entry of a net.
    Net(usize),
}

/// The edit under construction: the project, the request and the text edits so far.
pub(super) struct Editor<'a> {
    pub project: &'a Project,
    pub request: &'a EditRequest,
    pub edits: Vec<TextEdit>,
}

impl<'a> Editor<'a> {
    fn new(project: &'a Project, request: &'a EditRequest) -> Self {
        Editor { project, request, edits: Vec::new() }
    }

    pub fn text(&self) -> &'a str {
        &self.project.circuit_text
    }

    pub fn value_str(&self, id: &str) -> Option<String> {
        match self.request.values.get(id) {
            Some(Value::String(s)) if !s.trim().is_empty() => Some(s.trim().to_owned()),
            Some(Value::Array(a)) => a.iter().find_map(|v| v.as_str()).map(str::to_owned),
            _ => None,
        }
    }

    pub fn value_strs(&self, id: &str) -> Vec<String> {
        match self.request.values.get(id) {
            Some(Value::String(s)) if !s.trim().is_empty() => {
                s.split(',').map(|p| p.trim().to_owned()).filter(|p| !p.is_empty()).collect()
            }
            Some(Value::Array(a)) => a.iter().filter_map(|v| v.as_str().map(str::to_owned)).collect(),
            _ => Vec::new(),
        }
    }

    pub fn value_u32(&self, id: &str) -> Option<u32> {
        self.request.values.get(id).and_then(|v| match v {
            Value::Number(n) => n.as_f64().map(|f| f.round() as u32),
            Value::String(s) => s.trim().parse().ok(),
            _ => None,
        })
    }

    /// The pick in field `id`, or the selection's `n`th pick of the kinds a field takes.
    pub fn pick(&self, id: &str, n: usize) -> Option<String> {
        self.value_str(id).or_else(|| {
            self.request
                .selection
                .iter()
                .filter(|s| s.starts_with("pin:") || s.starts_with("net:"))
                .nth(n)
                .cloned()
        })
    }

    pub fn resolve(&self, id: &str) -> Result<End, EditError> {
        let nl = &self.project.netlist;
        if let Some(rest) = id.strip_prefix("pin:") {
            let (designator, pin) = rest
                .split_once('.')
                .ok_or_else(|| EditError::new(format!("`{id}` names no pin")))?;
            let (comp, c) = nl
                .component(designator)
                .ok_or_else(|| EditError::new(format!("{designator} isn't in the design")))?;
            let pin = c
                .pins
                .iter()
                .position(|p| p.pin == pin || format!("{}.{}", p.gate, p.pin) == pin)
                .or_else(|| c.pins.iter().position(|p| p.pads.iter().any(|pad| pad == pin)))
                .ok_or_else(|| EditError::new(format!("{designator} has no pin `{pin}`")))?;
            return Ok(End::Pin { comp, pin });
        }
        if let Some(name) = id.strip_prefix("net:") {
            let net = nl
                .nets
                .iter()
                .position(|n| n.name == name)
                .ok_or_else(|| EditError::new(format!("the design has no net `{name}`")))?;
            return Ok(End::Net(net));
        }
        Err(EditError::new(format!("pick a pin or a wire (not `{id}`)")))
    }

    /// The net an end is on, if any.
    pub fn net_of(&self, end: &End) -> Option<usize> {
        match end {
            End::Pin { comp, pin } => self.project.netlist.components[*comp].pins[*pin].net,
            End::Net(n) => Some(*n),
        }
    }

    /// The sheet instance an end is drawn on (`/`, `/ch[0]`).
    pub fn instance_of(&self, end: &End) -> NamePath {
        let nl = &self.project.netlist;
        match end {
            End::Pin { comp, .. } => nl.components[*comp].path.parent().unwrap_or_else(NamePath::root),
            End::Net(n) => {
                let net = &nl.nets[*n];
                // A net's own path names it in its instance; a global rail lives at the root.
                net.path.parent().unwrap_or_else(NamePath::root)
            }
        }
    }

    /// The sheet definition an instance draws.
    pub fn sheet_def_of(&self, instance: &NamePath) -> Result<String, EditError> {
        if *instance == NamePath::root() {
            return Ok(self.project.design.root.clone());
        }
        self.project
            .netlist
            .sheets
            .iter()
            .find(|(p, _)| p == instance)
            .map(|(_, s)| s.clone())
            .ok_or_else(|| EditError::new(format!("no sheet instance `{instance}`")))
    }

    /// The sheet `name` as this file declares it (sheets brought in by `use` are read-only here).
    pub fn own_sheet(&self, name: &str) -> Result<&'a SheetDef, EditError> {
        self.project.own.sheets.get(name).ok_or_else(|| {
            EditError::new(format!(
                "sheet `{name}` comes from another file; open that file to edit it"
            ))
        })
    }

    pub fn block(&self, name: &str) -> Result<SheetBlock, EditError> {
        blocks(self.text())
            .and_then(|b| b.sheets.into_iter().find(|s| s.name == name))
            .ok_or_else(|| EditError::new(format!("{} has no sheet `{name}`", self.project.circuit_path)))
    }

    /// The endpoint text a pin end takes in its sheet: the component's local name and the pin.
    pub fn pin_endpoint(&self, comp: usize, pin: usize) -> Endpoint {
        let c = &self.project.netlist.components[comp];
        let p = &c.pins[pin];
        // The plain name when it names this pin alone (or every pin of a power group).
        let plain = find_pins(&c.pins, &p.pin).map(|ix| ix.contains(&pin)).unwrap_or(false);
        let name = if plain || p.gate.is_empty() { p.pin.clone() } else { format!("{}.{}", p.gate, p.pin) };
        Endpoint::pin(c.path.leaf(), &name)
    }

    /// Whether `ep`, written in a sheet drawn at `instance`, reaches net `net`.
    pub fn reaches(&self, instance: &NamePath, ep: &Endpoint, net: usize) -> bool {
        let nl = &self.project.netlist;
        match ep {
            Endpoint::Pin { component, pin } => {
                let path = instance.child(component.as_str());
                nl.components.iter().filter(|c| c.path == path).any(|c| {
                    find_pins(&c.pins, pin)
                        .map(|ix| ix.iter().any(|i| c.pins[*i].net == Some(net)))
                        .unwrap_or(false)
                })
            }
            Endpoint::Net(name) => {
                let n = &nl.nets[net];
                let local = instance.child(name.as_str()).to_string();
                n.name == *name || n.path.to_string() == local || n.aliases.iter().any(|a| *a == local || a == name)
            }
            Endpoint::Port(name) => {
                let n = &nl.nets[net];
                let local = instance.child(name.as_str()).to_string();
                n.path.to_string() == local || n.aliases.iter().any(|a| *a == local)
            }
            _ => false,
        }
    }

    /// The `connect` statements of `sheet` (drawn at `instance`) that reach `net`.
    pub fn statements_of(&self, sheet: &SheetDef, instance: &NamePath, net: usize) -> Vec<usize> {
        sheet
            .connections
            .iter()
            .enumerate()
            .filter(|(_, c)| c.ends.iter().any(|ep| self.reaches(instance, ep, net)))
            .map(|(i, _)| i)
            .collect()
    }

    /// Replaces statement `connection` with one holding `ends` (removes it when `ends` has
    /// fewer than two).
    pub fn rewrite_connection(&mut self, connection: &Connection, ends: Vec<Endpoint>) -> Result<(), EditError> {
        let (start, end) = (connection.source.start as usize, connection.source.end as usize);
        if ends.len() < 2 {
            self.remove_statement(start, end);
            return Ok(());
        }
        let text = connection_statement(&Connection { ends, source: connection.source.clone() })
            .map_err(|e| EditError::new(e.0))?;
        self.edits.push(TextEdit::replace(Span { start, end }, text));
        Ok(())
    }

    /// Replaces the statement at `start..end` with `text`.
    pub fn replace(&mut self, start: usize, end: usize, text: String) {
        self.edits.push(TextEdit::replace(Span { start, end }, text));
    }

    /// Removes the statement at `start..end` with its line.
    pub fn remove_statement(&mut self, start: usize, end: usize) {
        let text = self.text();
        let from = line_start(text, start);
        let only = text[from..start].trim().is_empty();
        let to = if only && text[end..].starts_with('\n') { end + 1 } else { end };
        self.edits.push(TextEdit::replace(Span { start: if only { from } else { start }, end: to }, String::new()));
    }

    /// Inserts `statement` into sheet `block`: after the last statement whose word is one of
    /// `after` (in that order of preference), else before the first statement, else into the
    /// empty block.
    pub fn insert(&mut self, block: &SheetBlock, after: &[&str], statement: &str) {
        let text = self.text();
        let anchor = after
            .iter()
            .find_map(|w| block.statements.iter().rev().find(|s| s.word == *w))
            .map(|s| s.end);
        let edit = match (anchor, block.statements.first()) {
            (Some(end), _) => {
                let indent = indent_at(text, end);
                TextEdit::replace(Span { start: end, end }, format!("\n{indent}{statement}"))
            }
            (None, Some(first)) => {
                let indent = indent_at(text, first.start);
                let at = line_start(text, first.start);
                TextEdit::replace(Span { start: at, end: at }, format!("{indent}{statement}\n"))
            }
            (None, None) => {
                let outer = indent_at(text, block.close);
                let inner = format!("{outer}{INDENT}");
                TextEdit::replace(
                    Span { start: block.open, end: block.close },
                    format!("\n{inner}{statement}\n{outer}"),
                )
            }
        };
        self.edits.push(edit);
    }

    pub fn endpoint_text(ep: &Endpoint) -> Result<String, EditError> {
        endpoint(ep).map_err(|e| EditError::new(e.0))
    }

    /// Applies the edits, checks the file compiles, and returns the change.
    fn finish(self, files: &Files, summary: String) -> Result<EditOutput, EditError> {
        let project = self.project;
        let path = &project.circuit_path;
        if self.edits.is_empty() {
            return Ok(EditOutput { files: Vec::new(), summary });
        }
        let source = apply_edits(&project.circuit_text, &self.edits).map_err(|e| EditError::new(format!("{e:?}")))?;
        let read = parse_circuit(path, &source);
        if let Some(d) = read.diagnostics.first() {
            return Err(EditError::new(format!("the edit would not read back: {}", d.message)));
        }
        if let Err(diagnostics) = load_with(path, &source, files, &self.request.options, false) {
            let first = diagnostics.first().map(|d| d.message.clone()).unwrap_or_default();
            return Err(EditError { message: format!("the edit wasn't made: {first}"), diagnostics });
        }
        Ok(EditOutput {
            summary,
            files: vec![file_change(path, &project.circuit_text, SourceChange { edits: self.edits, source, reprinted: false })],
        })
    }
}
