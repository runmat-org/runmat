//! Where a `.circuit` file's `simulation` sections and their statements are, so a structured edit
//! can replace or insert one statement and leave the rest of the file as written.

use super::values::{val, Val, V};
use super::words;
use runmat_design_syntax::ast::{Headed, Item, Statement};
use runmat_design_syntax::{parse_source, Dialect};
use runmat_types::Span;

/// One statement of a section: `probe`, `sheet`, or the keyword of `source <name> = …`,
/// `analysis <name> = …` and the rest, with its name.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StatementAt {
    pub keyword: String,
    pub name: String,
    pub span: Span,
    /// A `model` statement's SPICE library (`library = "models/opa171.lib"`), as written.
    pub library: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SectionAt {
    pub name: String,
    /// The whole `simulation <name> { … }` statement.
    pub span: Span,
    /// The offset of its closing brace.
    pub close: usize,
    pub statements: Vec<StatementAt>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SimulationLayout {
    /// The offset of the `circuit` block's closing brace.
    pub circuit_close: Option<usize>,
    pub sections: Vec<SectionAt>,
}

/// The sections of `source`, in file order.
pub fn simulation_layout(source: &str) -> SimulationLayout {
    let syntax = parse_source(Dialect::Circuit, source);
    let mut out = SimulationLayout::default();
    for item in syntax.file().items() {
        let Item::Declaration(d) = item else { continue };
        if d.keyword() != words::CIRCUIT {
            continue;
        }
        let Some(block) = d.block() else { continue };
        out.circuit_close = Some(block.span().end.saturating_sub(1));
        for statement in block.statements() {
            let Statement::Block(sim) = &statement else { continue };
            if sim.keyword() != words::SIMULATION {
                continue;
            }
            let Some(body) = sim.block() else { continue };
            let statements = body
                .statements()
                .iter()
                .filter_map(|s| match s {
                    Statement::Assignment(a) => Some(StatementAt { keyword: a.target().join("."), name: String::new(), span: s.span(), library: None }),
                    Statement::Value(v) => {
                        let keyword = v.keyword();
                        let library = if keyword == "model" { v.value().and_then(|e| val(&e).ok()).and_then(|x| library_of(&x)) } else { None };
                        Some(StatementAt { keyword, name: v.name().map(|n| n.text()).unwrap_or_default(), span: s.span(), library })
                    }
                    _ => None,
                })
                .collect();
            out.sections.push(SectionAt { name: sim.name().map(|n| n.text()).unwrap_or_default(), span: statement.span(), close: body.span().end.saturating_sub(1), statements });
        }
    }
    out
}

fn library_of(v: &Val) -> Option<String> {
    let V::Rec(fields) = &v.v else { return None };
    fields.iter().find(|(k, _)| k == "library").and_then(|(_, x)| match &x.v {
        V::Str(s) | V::Word(s) => Some(s.clone()),
        _ => None,
    })
}

/// The SPICE libraries the file's simulation sections name, as written (relative to the file).
pub fn model_libraries(source: &str) -> Vec<String> {
    let mut out: Vec<String> = simulation_layout(source).sections.iter().flat_map(|s| s.statements.iter().filter_map(|t| t.library.clone())).collect();
    out.sort();
    out.dedup();
    out
}

impl SectionAt {
    pub fn find(&self, keyword: &str, name: &str) -> Option<&StatementAt> {
        self.statements.iter().find(|s| s.keyword == keyword && s.name == name)
    }

    /// The last statement with one of `keywords`.
    pub fn last_of(&self, keywords: &[&str]) -> Option<&StatementAt> {
        self.statements.iter().rev().find(|s| keywords.contains(&s.keyword.as_str()))
    }
}
