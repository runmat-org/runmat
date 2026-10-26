//! Assessment values share the checked dynamic quantity representation.
use std::{
    env, fs,
    path::{Path, PathBuf},
};
use syn::{
    spanned::Spanned,
    visit::{self, Visit},
};
struct R<'a> {
    s: &'a str,
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
        if x.path
            .segments
            .last()
            .is_some_and(|s| s.ident == "AssessmentQuantity")
        {
            let get = |name: &str| {
                x.fields
                    .iter()
                    .find(|f| matches!(&f.member,syn::Member::Named(n) if n==name))
                    .map(|f| {
                        let (a, b) = self.range(f.expr.span());
                        self.s[a..b].to_owned()
                    })
            };
            if let (Some(value), Some(dimension)) = (get("value_si"), get("dimension")) {
                let (a, b) = self.range(x.path.span());
                let path = &self.s[a..b];
                let (a, b) = self.range(x.span());
                self.edits.push((a,b,format!("{path}::try_from_descriptor(({dimension}).into(), {value}).expect(\"checked finite assessment quantity\")")));
                return;
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
            s: &s,
            lines: std::iter::once(0)
                .chain(s.match_indices('\n').map(|(i, _)| i + 1))
                .collect(),
            edits: vec![],
        };
        r.visit_file(&ast);
        if r.edits.is_empty() {
            continue;
        }
        r.edits.sort_by_key(|e| std::cmp::Reverse(e.0));
        let mut result = s.clone();
        for (a, b, v) in r.edits {
            result.replace_range(a..b, &v);
        }
        fs::write(p, result).unwrap();
    }
}
