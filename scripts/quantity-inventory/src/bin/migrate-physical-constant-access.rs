//! Canonical numerical projections of the two reviewed vacuum constants.
use std::{
    env, fs,
    path::{Path, PathBuf},
};
use syn::{
    parse::Parser,
    spanned::Spanned,
    visit::{self, Visit},
};
struct R {
    lines: Vec<usize>,
    edits: Vec<(usize, usize, String)>,
}
impl<'ast> Visit<'ast> for R {
    fn visit_expr_path(&mut self, x: &'ast syn::ExprPath) {
        if x.path.segments.last().is_some_and(|s| {
            matches!(
                s.ident.to_string().as_str(),
                "VACUUM_PERMITTIVITY_F_PER_M" | "VACUUM_PERMEABILITY_H_PER_M"
            )
        }) {
            let b = x.span().end();
            let b = self.lines[b.line - 1] + b.column;
            self.edits.push((b, b, ".canonical_value()".into()));
        }
        visit::visit_expr_path(self, x);
    }
    fn visit_macro(&mut self, m: &'ast syn::Macro) {
        if let Ok(args) = syn::punctuated::Punctuated::<syn::Expr, syn::Token![,]>::parse_terminated
            .parse2(m.tokens.clone())
        {
            for a in &args {
                self.visit_expr(a);
            }
        }
    }
}
fn paths(p: &Path, out: &mut Vec<PathBuf>) {
    if p.is_file() {
        if p.extension().is_some_and(|x| x == "rs") {
            out.push(p.into());
        }
        return;
    }
    for e in fs::read_dir(p).unwrap() {
        let e = e.unwrap();
        if !e.file_type().unwrap().is_symlink() {
            paths(&e.path(), out);
        }
    }
}
fn main() {
    let mut ps = vec![];
    for a in env::args().skip(1) {
        paths(Path::new(&a), &mut ps);
    }
    for p in ps {
        let s = fs::read_to_string(&p).unwrap();
        let Ok(ast) = syn::parse_file(&s) else {
            continue;
        };
        let mut r = R {
            lines: std::iter::once(0)
                .chain(s.match_indices('\n').map(|(i, _)| i + 1))
                .collect(),
            edits: vec![],
        };
        r.visit_file(&ast);
        r.edits.sort_by_key(|e| e.0);
        r.edits.dedup();
        let mut s = s;
        for (a, b, v) in r.edits.into_iter().rev() {
            s.replace_range(a..b, &v);
        }
        fs::write(p, s).unwrap();
    }
}
