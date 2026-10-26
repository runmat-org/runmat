//! Full structural loads enter modal projection with their retained physical layout.
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
    fn visit_expr_method_call(&mut self, x: &'ast syn::ExprMethodCall) {
        if x.method == "expand_vector" && x.args.len() == 1 {
            let (a, b) = self.range(x.receiver.span());
            let receiver = &self.s[a..b];
            let (a, b) = self.range(x.args[0].span());
            let argument = &self.s[a..b];
            let kind = if argument.contains(".velocity()") {
                "Velocity"
            } else if argument.contains(".acceleration()") {
                "Acceleration"
            } else {
                "Displacement"
            };
            let quantity = if argument.contains(".displacement_real")
                || argument.contains(".displacement_imaginary")
            {
                argument.to_owned()
            } else {
                format!("&runmat_types::physical::Quantity::<runmat_types::physical::kind::{kind},_>::new({argument})")
            };
            let (a, b) = self.range(x.span());
            self.edits.push((a,b,format!("({receiver}).expand_state({quantity}).map(|field| field.into_canonical_storage())")));
            return;
        }
        visit::visit_expr_method_call(self, x);
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
