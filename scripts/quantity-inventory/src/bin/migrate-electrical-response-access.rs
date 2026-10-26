//! Reviewed access projection for equation-owned electrical response fields.
use std::{
    env, fs,
    path::{Path, PathBuf},
};
use syn::{
    parse::Parser,
    spanned::Spanned,
    visit::{self, Visit},
};
struct R<'a> {
    s: &'a str,
    lines: Vec<usize>,
    edits: Vec<(usize, usize, String)>,
    lhs: bool,
}
impl R<'_> {
    fn range(&self, s: proc_macro2::Span) -> (usize, usize) {
        let (a, b) = (s.start(), s.end());
        (
            self.lines[a.line - 1] + a.column,
            self.lines[b.line - 1] + b.column,
        )
    }
    fn getter(n: &syn::Member) -> Option<&'static str> {
        let syn::Member::Named(n) = n else {
            return None;
        };
        match n.to_string().as_str() {
            "potential_v"
            | "potential_increment_v"
            | "element_electric_field_v_per_m"
            | "nodal_outward_flux" => Some("canonical_values"),
            "volume_source_work" | "terminal_work" | "outward_flux" => Some("canonical_value"),
            _ => None,
        }
    }
}
impl<'ast> Visit<'ast> for R<'_> {
    fn visit_expr_field(&mut self, x: &'ast syn::ExprField) {
        if !self.lhs {
            if let Some(g) = Self::getter(&x.member) {
                let (_, b) = self.range(x.span());
                self.edits.push((b, b, format!(".{g}()")));
            }
        }
        visit::visit_expr_field(self, x);
    }
    fn visit_expr_reference(&mut self, x: &'ast syn::ExprReference) {
        if x.mutability.is_none() {
            if let syn::Expr::Field(f) = x.expr.as_ref() {
                if Self::getter(&f.member) == Some("canonical_values") {
                    let (a, b) = self.range(x.span());
                    let (c, d) = self.range(f.span());
                    self.edits
                        .push((a, b, format!("{}.canonical_values()", &self.s[c..d])));
                    return;
                }
            }
        }
        visit::visit_expr_reference(self, x);
    }
    fn visit_expr_assign(&mut self, x: &'ast syn::ExprAssign) {
        let old = self.lhs;
        self.lhs = true;
        self.visit_expr(&x.left);
        self.lhs = old;
        self.visit_expr(&x.right);
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
        paths(Path::new(&arg), &mut ps);
    }
    for p in ps {
        let s = fs::read_to_string(&p).unwrap();
        let Ok(ast) = syn::parse_file(&s) else {
            continue;
        };
        let mut r = R {
            s: &s,
            lines: std::iter::once(0)
                .chain(s.match_indices('\n').map(|(i, _)| i + 1))
                .collect(),
            edits: vec![],
            lhs: false,
        };
        r.visit_file(&ast);
        if r.edits.is_empty() {
            continue;
        }
        r.edits.sort_by_key(|e| (e.0, e.1));
        r.edits.dedup();
        let mut result = s.clone();
        for (a, b, v) in r.edits.into_iter().rev() {
            result.replace_range(a..b, &v);
        }
        fs::write(p, result).unwrap();
    }
}
