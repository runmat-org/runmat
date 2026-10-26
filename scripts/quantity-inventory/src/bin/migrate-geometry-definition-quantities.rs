//! Reviewed public definition/topology owners; private numerical coefficients excluded.
use quote::ToTokens;
use std::{
    env, fs,
    path::{Path, PathBuf},
};
use syn::{
    spanned::Spanned,
    visit::{self, Visit},
};
fn role(owner: &str, name: &str) -> Option<&'static str> {
    match (owner, name) {
        (
            "ExactCurveDefinition" | "ExactSurfaceDefinition",
            "point_m" | "origin_m" | "center_m",
        ) => Some("Position"),
        ("NurbsCurve3" | "NurbsSurface3", "control_points_m") => Some("Position"),
        ("ExactVertex", "point_m") => Some("Position"),
        ("ExactVertex" | "ExactEdge", "tolerance_m") => Some("Length"),
        ("CartesianOrientation" | "AxisOrientation", "origin_m") => Some("Position"),
        ("LaminatePlyModel" | "AuthoredPly", "angle_rad") => Some("Angle"),
        ("ConductivityFrequencyPoint", "frequency_hz") => Some("Frequency"),
        _ => None,
    }
}
struct R<'a> {
    source: &'a str,
    lines: Vec<usize>,
    edits: Vec<(usize, usize, String)>,
    owner: String,
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
    fn visit_item_struct(&mut self, x: &'ast syn::ItemStruct) {
        let old = std::mem::replace(&mut self.owner, x.ident.to_string());
        visit::visit_item_struct(self, x);
        self.owner = old;
    }
    fn visit_item_enum(&mut self, x: &'ast syn::ItemEnum) {
        let old = std::mem::replace(&mut self.owner, x.ident.to_string());
        visit::visit_item_enum(self, x);
        self.owner = old;
    }
    fn visit_field(&mut self, x: &'ast syn::Field) {
        if let Some(name) = &x.ident {
            if let Some(kind) = role(&self.owner, &name.to_string()) {
                let ty = x.ty.to_token_stream().to_string();
                if ty == "f64" || ty == "[f64 ; 3]" || ty == "Vec < [f64 ; 3] >" {
                    let (a, b) = self.range(x.ty.span());
                    self.edits.push((a,b,format!("runmat_types::physical::Quantity<runmat_types::physical::kind::{kind},{ty}>")));
                    let (a, _) = self.range(x.span());
                    self.edits.push((
                        a,
                        a,
                        "#[serde(with=\"runmat_types::physical::canonical_storage\")]\n".into(),
                    ));
                }
            }
        }
        visit::visit_field(self, x);
    }
    fn visit_expr_struct(&mut self, x: &'ast syn::ExprStruct) {
        for f in &x.fields {
            let syn::Member::Named(name) = &f.member else {
                continue;
            };
            let kind = x
                .path
                .segments
                .iter()
                .find_map(|s| role(&s.ident.to_string(), &name.to_string()));
            if let Some(kind) = kind {
                let (a, b) = self.range(f.expr.span());
                if self.source[a..b].contains("physical::") {
                    continue;
                }
                let prefix = if f.colon_token.is_none() {
                    format!("{name}: ")
                } else {
                    String::new()
                };
                self.edits.push((a,a,format!("{prefix}runmat_types::physical::Quantity::<runmat_types::physical::kind::{kind},_>::new(")));
                self.edits.push((b, b, ")".into()));
            }
        }
        visit::visit_expr_struct(self, x);
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
    let mut count = 0;
    for p in ps {
        let source = fs::read_to_string(&p).unwrap();
        let Ok(ast) = syn::parse_file(&source) else {
            continue;
        };
        let mut r = R {
            source: &source,
            lines: std::iter::once(0)
                .chain(source.match_indices('\n').map(|(i, _)| i + 1))
                .collect(),
            edits: vec![],
            owner: String::new(),
        };
        r.visit_file(&ast);
        if r.edits.is_empty() {
            continue;
        }
        r.edits.sort_by_key(|e| (e.0, e.1));
        r.edits.dedup();
        let edits = std::mem::take(&mut r.edits);
        drop(r);
        let mut result = source;
        for (a, b, v) in edits.into_iter().rev() {
            result.replace_range(a..b, &v);
        }
        fs::write(p, result).unwrap();
        count += 1;
    }
    println!("migrated {count} geometry/authored definition owner files");
}
