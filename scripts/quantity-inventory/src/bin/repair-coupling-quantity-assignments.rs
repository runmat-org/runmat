//! Reviewed ThermalEnergyBalance constructors now accept physical Power values.
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
}
impl<'ast> Visit<'ast> for R {
    fn visit_expr_assign(&mut self, x: &'ast syn::ExprAssign) {
        if let syn::Expr::Field(f) = x.left.as_ref() {
            if let syn::Member::Named(n) = &f.member {
                if matches!(
                    n.to_string().as_str(),
                    "max_temperature_jump_k"
                        | "mean_interface_heat_flux_w_per_m2"
                        | "max_advection_temperature_shift_k"
                        | "interface_conductance_w_per_m2k"
                        | "max_interface_work_j_per_m2"
                        | "max_structural_strain_energy_j_per_m2"
                        | "interface_stiffness_pa_per_m"
                        | "mean_interface_pressure_pa"
                        | "max_traction_magnitude_pa"
                        | "max_displacement_transfer_residual_m"
                        | "max_interface_displacement_m"
                ) {
                    let (a, b) = self.range(x.right.span());
                    self.edits
                        .push((a, a, "runmat_types::physical::Quantity::new(".into()));
                    self.edits.push((b, b, ")".into()));
                }
            }
        }
        visit::visit_expr_assign(self, x);
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
        if p.ends_with("thermal_state/energy.rs") {
            continue;
        }
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
