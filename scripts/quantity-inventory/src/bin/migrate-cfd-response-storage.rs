//! Reviewed public physics response owners: no suffix inference or unit-string wrappers.
//! Array storage receives one shared Quantity header, including complex phasors.
use quote::ToTokens;
use std::{
    collections::{BTreeMap, BTreeSet},
    env, fs,
    path::{Path, PathBuf},
};
use syn::{
    parse::Parser,
    spanned::Spanned,
    visit::{self, Visit},
};
fn kind(n: &str) -> Option<&'static str> {
    Some(match n {
        "velocity_m_per_s"
        | "inlet_velocity_m_per_s"
        | "nominal_velocity_m_per_s"
        | "nominal_inlet_velocity_m_per_s" => "Velocity",
        "pressure_fluctuation_pa"
        | "pressure_reference_pa"
        | "pressure_drop_pa"
        | "outlet_pressure_pa" => "Pressure",
        "nominal_time_step_s" => "Time",
        "domain_length_m" | "hydraulic_diameter_m" | "dx_m" => "Length",
        "face_area_m2" => "Area",
        "face_volume_flux_m3_per_s"
        | "cell_continuity_m3_per_s"
        | "global_boundary_flux_m3_per_s"
        | "volume_flow_rate_m3_per_s" => "VolumeFlowRate",
        "mass_flow_rate_kg_per_s" => "MassFlowRate",
        "cell_momentum_n"
        | "cell_algebraic_momentum_n"
        | "global_momentum_n"
        | "advective_momentum_n"
        | "viscous_traction_n"
        | "pressure_traction_n"
        | "pressure_fluctuation_traction_n" => "Force",
        _ => return None,
    })
}

struct R<'a> {
    s: &'a str,
    lines: Vec<usize>,
    edits: Vec<(usize, usize, String)>,
    serializable: bool,
    lhs: bool,
    getters: &'a BTreeMap<String, &'static str>,
}
impl R<'_> {
    fn range(&self, s: proc_macro2::Span) -> (usize, usize) {
        let (a, b) = (s.start(), s.end());
        (
            self.lines[a.line - 1] + a.column,
            self.lines[b.line - 1] + b.column,
        )
    }
    fn field(&mut self, f: &syn::Field) {
        let Some(n) = &f.ident else { return };
        let Some(k) = kind(&n.to_string()) else {
            return;
        };
        let ty = f.ty.to_token_stream().to_string();
        if ty.contains("Quantity") || ty.starts_with("Option") {
            return;
        }
        let (a, b) = self.range(f.ty.span());
        self.edits.push((
            a,
            b,
            format!("runmat_types::physical::Quantity<runmat_types::physical::kind::{k}, {ty}>"),
        ));
        if self.serializable {
            let (a, _) = self.range(f.span());
            self.edits.push((
                a,
                a,
                "#[serde(with = \"runmat_types::physical::canonical_storage\")]\n".into(),
            ));
        }
    }
}
impl<'ast> Visit<'ast> for R<'_> {
    fn visit_item_struct(&mut self, x: &'ast syn::ItemStruct) {
        let old = self.serializable;
        self.serializable = x
            .attrs
            .iter()
            .any(|a| a.to_token_stream().to_string().contains("Serialize"));
        visit::visit_item_struct(self, x);
        self.serializable = old;
    }
    fn visit_item_enum(&mut self, x: &'ast syn::ItemEnum) {
        let old = self.serializable;
        self.serializable = x
            .attrs
            .iter()
            .any(|a| a.to_token_stream().to_string().contains("Serialize"));
        visit::visit_item_enum(self, x);
        self.serializable = old;
    }
    fn visit_field(&mut self, f: &'ast syn::Field) {
        self.field(f);
        visit::visit_field(self, f);
    }
    fn visit_expr_struct(&mut self, x: &'ast syn::ExprStruct) {
        for f in &x.fields {
            let syn::Member::Named(n) = &f.member else {
                continue;
            };
            let Some(k) = kind(&n.to_string()) else {
                continue;
            };
            let (a, b) = self.range(f.expr.span());
            let text = &self.s[a..b];
            if text.contains("physical::") {
                continue;
            }
            let prefix = if f.colon_token.is_none() {
                format!("{n}: ")
            } else {
                String::new()
            };
            self.edits.push((a,a,format!("{prefix}runmat_types::physical::Quantity::<runmat_types::physical::kind::{k},_>::new(")));
            self.edits.push((b, b, ")".into()));
        }
        visit::visit_expr_struct(self, x);
    }
    fn visit_expr_field(&mut self, x: &'ast syn::ExprField) {
        if !self.lhs {
            if let syn::Member::Named(n) = &x.member {
                if let Some(g) = self.getters.get(&n.to_string()) {
                    let (_, b) = self.range(x.span());
                    if !self.s[b..].starts_with(".canonical_") {
                        self.edits.push((b, b, format!(".{g}()")));
                    }
                }
            }
        }
        visit::visit_expr_field(self, x);
    }
    fn visit_expr_reference(&mut self, x: &'ast syn::ExprReference) {
        if x.mutability.is_none() {
            if let syn::Expr::Field(f) = x.expr.as_ref() {
                if let syn::Member::Named(n) = &f.member {
                    if self.getters.get(&n.to_string()) == Some(&"canonical_values") {
                        let (a, b) = self.range(x.span());
                        let (c, d) = self.range(f.span());
                        self.edits
                            .push((a, b, format!("{}.canonical_values()", &self.s[c..d])));
                        return;
                    }
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
struct Discovery(BTreeMap<String, BTreeSet<&'static str>>);
impl<'ast> Visit<'ast> for Discovery {
    fn visit_field(&mut self, f: &'ast syn::Field) {
        if let Some(n) = &f.ident {
            if kind(&n.to_string()).is_some() {
                let t = f.ty.to_token_stream().to_string();
                if !t.starts_with("Option") && !t.contains("Quantity") {
                    self.0
                        .entry(n.to_string())
                        .or_default()
                        .insert(if t.starts_with("Vec") {
                            "canonical_values"
                        } else {
                            "canonical_value"
                        });
                }
            }
        }
        visit::visit_field(self, f);
    }
}
fn main() {
    let mut ps = vec![];
    for arg in env::args().skip(1) {
        paths(Path::new(&arg), &mut ps)
    }
    let mut d = Discovery(BTreeMap::new());
    for p in &ps {
        let s = fs::read_to_string(p).unwrap();
        if let Ok(ast) = syn::parse_file(&s) {
            d.visit_file(&ast);
        }
    }
    let getters =
        d.0.iter()
            .filter(|(_, v)| v.len() == 1)
            .map(|(n, v)| (n.clone(), *v.first().unwrap()))
            .collect();
    let mut count = 0;
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
            serializable: false,
            lhs: false,
            getters: &getters,
        };
        r.visit_file(&ast);
        if r.edits.is_empty() {
            continue;
        }
        r.edits.sort_by_key(|e| (e.0, e.1, usize::from(e.2 == ")")));
        r.edits.dedup();
        let mut result = s.clone();
        for (a, b, v) in r.edits.into_iter().rev() {
            result.replace_range(a..b, &v);
        }
        fs::write(p, result).unwrap();
        count += 1;
    }
    println!(
        "reviewed response storage changed {count} files; ambiguous getters: {:?}",
        d.0.iter().filter(|(_, v)| v.len() > 1).collect::<Vec<_>>()
    );
}
