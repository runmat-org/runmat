use super::*;
use std::collections::{btree_map::Entry, BTreeMap};

/// Reads a named part, footprint or symbol without replacing an earlier declaration.
pub(super) fn read_declared(
    reader: &mut Reader<'_>,
    design: &mut Design,
    word: &str,
    name: String,
    value: Option<Expr>,
    span: Span,
) {
    match word {
        words::PART => insert(
            reader,
            &mut design.parts,
            "part",
            name,
            value,
            span,
            part_decl::read,
        ),
        words::FOOTPRINT => insert(
            reader,
            &mut design.footprints,
            "footprint",
            name,
            value,
            span,
            footprint_decl::read,
        ),
        _ => insert(
            reader,
            &mut design.symbols,
            "symbol",
            name,
            value,
            span,
            symbol_decl::read,
        ),
    }
}

fn insert<T>(
    reader: &mut Reader<'_>,
    declarations: &mut BTreeMap<String, T>,
    kind: &str,
    name: String,
    value: Option<Expr>,
    span: Span,
    parse: fn(String, Option<Expr>, Span) -> Read<T>,
) {
    match declarations.entry(name) {
        Entry::Occupied(entry) => reader.diagnostics.push(diagnostic(
            span,
            format!("{kind} `{}` is declared twice", entry.key()),
        )),
        Entry::Vacant(entry) => {
            if let Some(declaration) = reader.note(parse(entry.key().clone(), value, span)) {
                entry.insert(declaration);
            }
        }
    }
}
