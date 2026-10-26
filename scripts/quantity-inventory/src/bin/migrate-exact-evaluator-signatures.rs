//! Typed public physical IO; mathematical evaluator parameters remain bound to IDs.
use quote::ToTokens;
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
    prefix: &'static str,
}
impl R {
    fn range(&self, s: proc_macro2::Span) -> (usize, usize) {
        let (a, b) = (s.start(), s.end());
        (
            self.lines[a.line - 1] + a.column,
            self.lines[b.line - 1] + b.column,
        )
    }
    fn signature(&mut self, s: &syn::Signature) {
        let name = s.ident.to_string();
        let ids = s.inputs.to_token_stream().to_string();
        if !ids.contains("EvaluatorId") {
            return;
        }
        let pcurve = ids.contains("PcurveEvaluatorId");
        let ret = match name.as_str() {
            "point" if pcurve => Some("[GeometryParameterCoordinate;2]"),
            "point" => Some("physical::Position"),
            "curvature_1_per_m" => Some("physical::Curvature"),
            "arc_length_m" => Some("physical::Length"),
            _ => None,
        };
        if let Some(ret) = ret {
            if let syn::ReturnType::Type(_, ty) = &s.output {
                if let syn::Type::Path(t) = ty.as_ref() {
                    if let Some(last) = t.path.segments.last() {
                        if let syn::PathArguments::AngleBracketed(args) = &last.arguments {
                            if let Some(syn::GenericArgument::Type(old)) = args.args.first() {
                                let (a, b) = self.range(old.span());
                                let target = if ret.starts_with('[') {
                                    format!("[{}::GeometryParameterCoordinate;2]", self.prefix)
                                } else {
                                    format!("{}::{ret}", self.prefix)
                                };
                                self.edits.push((a, b, target));
                            }
                        }
                    }
                }
            }
        }
        if matches!(
            name.as_str(),
            "arc_length_m" | "inverse_project" | "closest_point"
        ) {
            for a in &s.inputs {
                let syn::FnArg::Typed(a) = a else { continue };
                let ty = a.ty.to_token_stream().to_string();
                let role = if ty == "f64" && a.pat.to_token_stream().to_string().contains("error") {
                    Some("Length")
                } else if ty == "[f64 ; 3]" {
                    Some("Position")
                } else {
                    None
                };
                if let Some(role) = role {
                    let (a, b) = self.range(a.ty.span());
                    self.edits
                        .push((a, b, format!("{}::physical::{role}", self.prefix)));
                }
            }
        }
    }
}
impl<'ast> Visit<'ast> for R {
    fn visit_signature(&mut self, s: &'ast syn::Signature) {
        self.signature(s);
        visit::visit_signature(self, s)
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
        let s = fs::read_to_string(&p).unwrap();
        let Ok(ast) = syn::parse_file(&s) else {
            continue;
        };
        let prefix = if p.to_string_lossy().contains("runmat-geometry/core/") {
            "crate"
        } else {
            "runmat_geometry_core"
        };
        let mut r = R {
            lines: std::iter::once(0)
                .chain(s.match_indices('\n').map(|(i, _)| i + 1))
                .collect(),
            edits: vec![],
            prefix,
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
        count += 1;
    }
    println!("updated {count} exact evaluator signature owners");
}
