//! Reviewed modal coordinates use the retained one-kilogram / length basis convention.
use std::{
    env, fs,
    path::{Path, PathBuf},
};
use syn::{
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
    fn kind(n: &str) -> Option<&'static str> {
        match n {
            "displacement" => Some("Displacement"),
            "velocity" => Some("Velocity"),
            "acceleration" => Some("Acceleration"),
            _ => None,
        }
    }
}
impl<'ast> Visit<'ast> for R {
    fn visit_item_struct(&mut self, x: &'ast syn::ItemStruct) {
        if matches!(
            x.ident.to_string().as_str(),
            "ModalTransientCoordinateState" | "ModalTransientInitialState"
        ) {
            for f in &x.fields {
                if let Some(k) = f.ident.as_ref().and_then(|n| Self::kind(&n.to_string())) {
                    let (a, b) = self.range(f.ty.span());
                    self.edits.push((a,b,format!("runmat_types::physical::Quantity<runmat_types::physical::kind::{k}, Vec<f64>>")));
                    if x.ident == "ModalTransientCoordinateState" {
                        let (a, _) = self.range(f.span());
                        self.edits.push((
                            a,
                            a,
                            "#[serde(with = \"runmat_types::physical::canonical_storage\")]\n"
                                .into(),
                        ));
                    }
                }
            }
        }
        visit::visit_item_struct(self, x);
    }
    fn visit_expr_struct(&mut self, x: &'ast syn::ExprStruct) {
        if x.path.segments.last().is_some_and(|s| {
            matches!(
                s.ident.to_string().as_str(),
                "ModalTransientCoordinateState" | "ModalTransientInitialState"
            )
        }) {
            for f in &x.fields {
                if let syn::Member::Named(n) = &f.member {
                    if let Some(k) = Self::kind(&n.to_string()) {
                        let (a, b) = self.range(f.expr.span());
                        let prefix = if f.colon_token.is_none() {
                            format!("{n}: ")
                        } else {
                            String::new()
                        };
                        self.edits.push((a,a,format!("{prefix}runmat_types::physical::Quantity::<runmat_types::physical::kind::{k},_>::new(")));
                        self.edits.push((b, b, ")".into()));
                    }
                }
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
        paths(Path::new(&arg), &mut ps);
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
        if r.edits.is_empty() {
            continue;
        }
        r.edits.sort_by_key(|e| (e.0, e.1));
        let mut result = s;
        for (a, b, v) in r.edits.into_iter().rev() {
            result.replace_range(a..b, &v);
        }
        fs::write(p, result).unwrap();
    }
}
