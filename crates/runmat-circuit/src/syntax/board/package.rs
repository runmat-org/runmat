//! A layout's packages: `package <key> = { … }`, the footprint record of
//! [`crate::syntax::footprint_decl`] with one layout field, `name`, written when two packages
//! share a name and the key tells them apart (`R0402#2`).

use super::*;
use crate::library::Package;
use crate::syntax::footprint_decl;
use crate::syntax::forms::record_of;
use crate::syntax::printer::record;
use runmat_design_syntax::ast::Expr;
use runmat_design_syntax::json::INDENT;
use runmat_types::Span;

/// Reads the package a layout keys `key`.
pub(super) fn read_package(key: &str, value: Option<Expr>, span: Span) -> Read<Package> {
    let mut fields = record_of(value, span)?;
    let name = match fields.iter().position(|(k, _)| k == words::NAME) {
        Some(i) => {
            let (_, e) = fields.remove(i);
            text(&crate::syntax::values::val(&e)?)?
        }
        None => key.to_owned(),
    };
    footprint_decl::read_fields(name, fields)
}

/// `package <key> = { … }`, its later lines indented one step for the layout block.
pub(super) fn package_text(key: &str, p: &Package) -> String {
    let mut entries = Vec::new();
    if p.name != key {
        entries.push((words::NAME.to_owned(), name_text(&p.name)));
    }
    entries.extend(footprint_decl::entries(p));
    format!(
        "{} {} = {}",
        words::PACKAGE,
        name_text(key),
        record(&entries).replace('\n', &format!("\n{INDENT}"))
    )
}
