//! Scoped numerical projection of typed portable definition positions.
use std::{
    env, fs,
    path::{Path, PathBuf},
};
use syn::{
    spanned::Spanned,
    visit::{self, Visit},
};
fn bindings(p: &syn::Pat) -> Vec<String> {
    let syn::Pat::Struct(p) = p else {
        return vec![];
    };
    if !p.path.segments.iter().any(|s| {
        matches!(
            s.ident.to_string().as_str(),
            "ExactCurveDefinition" | "ExactSurfaceDefinition"
        )
    }) {
        return vec![];
    }
    p.fields
        .iter()
        .filter_map(|f| {
            let syn::Member::Named(n) = &f.member else {
                return None;
            };
            if !matches!(n.to_string().as_str(), "point_m" | "origin_m" | "center_m") {
                return None;
            }
            let syn::Pat::Ident(n) = f.pat.as_ref() else {
                return None;
            };
            Some(n.ident.to_string())
        })
        .collect()
}
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
    fn prefix(names: Vec<String>) -> String {
        names
            .into_iter()
            .map(|n| format!("let {n}={n}.canonical_values();"))
            .collect::<String>()
    }
}
impl<'ast> Visit<'ast> for R {
    fn visit_arm(&mut self, a: &'ast syn::Arm) {
        let prefix = Self::prefix(bindings(&a.pat));
        if !prefix.is_empty() {
            let (start, end) = self.range(a.body.span());
            self.edits.push((start, start, format!("{{ {prefix} ")));
            self.edits.push((end, end, " }".into()));
        }
        visit::visit_arm(self, a);
    }
    fn visit_expr_if(&mut self, x: &'ast syn::ExprIf) {
        if let syn::Expr::Let(c) = x.cond.as_ref() {
            let prefix = Self::prefix(bindings(&c.pat));
            if !prefix.is_empty() {
                let (_, pos) = self.range(x.then_branch.brace_token.span.open());
                self.edits.push((pos, pos, prefix));
            }
        }
        visit::visit_expr_if(self, x);
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
        let source = fs::read_to_string(&p).unwrap();
        let Ok(ast) = syn::parse_file(&source) else {
            continue;
        };
        let mut r = R {
            lines: std::iter::once(0)
                .chain(source.match_indices('\n').map(|(i, _)| i + 1))
                .collect(),
            edits: vec![],
        };
        r.visit_file(&ast);
        if r.edits.is_empty() {
            continue;
        }
        r.edits.sort_by_key(|e| (e.0, e.1));
        r.edits.dedup();
        let mut output = source;
        for (a, b, v) in r.edits.into_iter().rev() {
            output.replace_range(a..b, &v);
        }
        fs::write(p, output).unwrap();
    }
}
