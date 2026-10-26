//! Reviewed ThermalEnergyBalance constructors now accept physical Power values.
use std::{
    env, fs,
    path::{Path, PathBuf},
};
use syn::visit::{self, Visit};
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
    fn visit_impl_item_fn(&mut self, x: &'ast syn::ImplItemFn) {
        if matches!(
            x.sig.ident.to_string().as_str(),
            "arc_length_m" | "inverse_project" | "closest_point"
        ) {
            let mut prefix = String::new();
            for arg in &x.sig.inputs {
                let syn::FnArg::Typed(arg) = arg else {
                    continue;
                };
                let syn::Pat::Ident(name) = arg.pat.as_ref() else {
                    continue;
                };
                let syn::Type::Path(t) = arg.ty.as_ref() else {
                    continue;
                };
                if t.path
                    .segments
                    .last()
                    .is_some_and(|s| matches!(s.ident.to_string().as_str(), "Length" | "Position"))
                {
                    prefix.push_str(&format!("\nlet {0} = {0}.canonical_value();", name.ident));
                }
            }
            let (_, pos) = self.range(x.block.brace_token.span.open());
            if !prefix.is_empty() {
                self.edits.push((pos, pos, prefix));
            }
        }
        visit::visit_impl_item_fn(self, x);
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
