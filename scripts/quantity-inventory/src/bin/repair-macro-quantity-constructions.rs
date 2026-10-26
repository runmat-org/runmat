//! Repair reviewed constructor fields inside expression macros (not expanded by the initial AST walk).
use std::{env, fs};
use syn::{
    parse::Parser,
    spanned::Spanned,
    visit::{self, Visit},
};
struct R<'a> {
    source: &'a str,
    lines: Vec<usize>,
    edits: Vec<(usize, usize, String)>,
}
impl R<'_> {
    fn range(&self, s: proc_macro2::Span) -> (usize, usize) {
        let (a, b) = (s.start(), s.end());
        (
            self.lines[a.line - 1] + a.column,
            self.lines[b.line - 1] + b.column,
        )
    }
}
impl<'ast> Visit<'ast> for R<'_> {
    fn visit_expr_struct(&mut self, x: &'ast syn::ExprStruct) {
        for f in &x.fields {
            let syn::Member::Named(n) = &f.member else {
                continue;
            };
            let kind = match n.to_string().as_str() {
                "coordinates_m" => "Position",
                "area_m2" => "Area",
                "iy_m4" | "iz_m4" | "torsion_j_m4" => "SecondMomentOfArea",
                "outer_fiber_y_m"
                | "outer_fiber_z_m"
                | "torsion_outer_radius_m"
                | "thickness_m" => "Length",
                _ => continue,
            };
            let (a, b) = self.range(f.expr.span());
            if self.source[a..b].contains("physical::") {
                continue;
            }
            let prefix = if f.colon_token.is_none() {
                format!("{n}: ")
            } else {
                String::new()
            };
            self.edits.push((a,a,format!("{prefix}runmat_types::physical::Quantity::<runmat_types::physical::kind::{kind},_>::new(")));
            self.edits.push((b, b, ")".into()));
        }
        visit::visit_expr_struct(self, x);
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
fn main() {
    for name in env::args().skip(1) {
        let s = fs::read_to_string(&name).unwrap();
        let ast = syn::parse_file(&s).unwrap();
        let mut r = R {
            source: &s,
            lines: std::iter::once(0)
                .chain(s.match_indices('\n').map(|(i, _)| i + 1))
                .collect(),
            edits: vec![],
        };
        r.visit_file(&ast);
        r.edits.sort_by_key(|e| (e.0, e.1));
        r.edits.dedup();
        let n = r.edits.len();
        let mut out = s.clone();
        for (a, b, v) in r.edits.into_iter().rev() {
            out.replace_range(a..b, &v);
        }
        fs::write(name, out).unwrap();
        println!("{n} constructor edits");
    }
}
