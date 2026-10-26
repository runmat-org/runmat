//! Reviewed acoustic constructor accepts nominal angular frequency, not raw Hz.
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
impl R {
    fn range(&self, s: proc_macro2::Span) -> (usize, usize) {
        let (a, b) = (s.start(), s.end());
        (
            self.lines[a.line - 1] + a.column,
            self.lines[b.line - 1] + b.column,
        )
    }
}
impl<'ast> Visit<'ast> for R {
    fn visit_expr_call(&mut self, x: &'ast syn::ExprCall) {
        if let syn::Expr::Path(p) = x.func.as_ref() {
            if p.path
                .segments
                .last()
                .is_some_and(|s| s.ident == "assemble")
                && p.path
                    .segments
                    .iter()
                    .any(|s| s.ident == "AcousticAssembly")
            {
                for e in x.args.iter().skip(2).take(1) {
                    let (a, b) = self.range(e.span());
                    self.edits.push((
                        a,
                        a,
                        "runmat_types::physical::AngularFrequency::new(".into(),
                    ));
                    self.edits.push((b, b, ")".into()));
                }
            }
        }
        visit::visit_expr_call(self, x);
    }
    fn visit_macro(&mut self, m: &'ast syn::Macro) {
        if let Ok(args) = syn::punctuated::Punctuated::<syn::Expr, syn::Token![,]>::parse_terminated
            .parse2(m.tokens.clone())
        {
            for arg in &args {
                self.visit_expr(arg);
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
    for arg in env::args().skip(1) {
        paths(Path::new(&arg), &mut ps)
    }
    for p in ps {
        if p.ends_with("acoustic/canonical.rs") {
            continue;
        }
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
        if r.edits.is_empty() {
            continue;
        }
        r.edits.sort_by_key(|e| (e.0, e.1));
        r.edits.dedup();
        let mut output = s;
        for (a, b, v) in r.edits.into_iter().rev() {
            output.replace_range(a..b, &v);
        }
        fs::write(p, output).unwrap();
    }
}
