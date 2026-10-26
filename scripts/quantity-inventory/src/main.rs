//! Conservative inventory of numeric and physical-unit declaration surfaces.
//! This is discovery, not unit inference: an unclassified f64 is never silently
//! treated as dimensionless. Function bodies are traversed to find local types.
use quote::ToTokens;
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    env, fs,
    path::{Path, PathBuf},
};
use syn::{
    spanned::Spanned,
    visit::{self, Visit},
};

struct Inventory {
    source: String,
    owners: Vec<String>,
    records: Vec<Value>,
    numeric_aliases: BTreeSet<String>,
    test_depth: usize,
}
fn test_attributes(attrs: &[syn::Attribute]) -> bool {
    attrs.iter().any(|attr| {
        attr.path().is_ident("test")
            || (attr.path().is_ident("cfg")
                && attr.meta.to_token_stream().to_string() == "cfg (test)")
    })
}
fn unit_metadata_name(name: &str) -> bool {
    let name = name.to_ascii_lowercase();
    name.contains("unit") || name.contains("dimension") || name.contains("quantity")
}
impl Inventory {
    fn candidate(&self, ty: &syn::Type) -> bool {
        let text = ty.to_token_stream().to_string();
        text.split(|c: char| !c.is_alphanumeric() && c != '_')
            .any(|name| {
                matches!(
                    name,
                    "f64"
                        | "f32"
                        | "PhysicalDimension"
                        | "QuantityDimension"
                        | "UnitSystem"
                        | "ParsedUnit"
                        | "CanonicalQuantity"
                        | "FiniteQuantity"
                        | "PhysicalQuantityKind"
                        | "ObservationQuantity"
                        | "Ap242PhysicalDimension"
                        | "Ap242Measure"
                        | "ComplexScalar"
                        | "ComplexVector3"
                        | "Q"
                        | "Quantity"
                        | "AnyQuantity"
                        | "SourceQuantity"
                        | "SourceMeasure"
                        | "LengthUnit"
                        | "LengthUnitEvidence"
                        | "CoordinateUnits"
                        | "UnitDefinition"
                        | "Unit"
                ) || self.numeric_aliases.contains(name)
            })
    }
    fn record(&mut self, kind: &str, name: String, ty: &syn::Type) {
        if !self.candidate(ty) && !unit_metadata_name(&name) {
            return;
        }
        self.records
            .push(json!({"path":self.source,"line":ty.span().start().line,
            "owner":self.owners.join("::"),"kind":kind,"name":name,
            "type":ty.to_token_stream().to_string(),"test_context":self.test_depth > 0 || self.source.contains("/tests/") || self.source.ends_with("/tests.rs"),"disposition":"unclassified"}));
    }
}
impl<'ast> Visit<'ast> for Inventory {
    fn visit_item_struct(&mut self, item: &'ast syn::ItemStruct) {
        self.owners.push(item.ident.to_string());
        visit::visit_item_struct(self, item);
        self.owners.pop();
    }
    fn visit_item_enum(&mut self, item: &'ast syn::ItemEnum) {
        if unit_metadata_name(&item.ident.to_string()) {
            self.records.push(json!({"path":self.source,"line":item.span().start().line,
                "owner":self.owners.join("::"),"kind":"unit_enum","name":item.ident.to_string(),
                "type":item.variants.iter().map(|v|v.ident.to_string()).collect::<Vec<_>>().join(" | "),"disposition":"unclassified"}));
        }
        self.owners.push(item.ident.to_string());
        visit::visit_item_enum(self, item);
        self.owners.pop();
    }
    fn visit_variant(&mut self, item: &'ast syn::Variant) {
        self.owners.push(item.ident.to_string());
        visit::visit_variant(self, item);
        self.owners.pop();
    }
    fn visit_item_impl(&mut self, item: &'ast syn::ItemImpl) {
        self.owners.push(item.self_ty.to_token_stream().to_string());
        visit::visit_item_impl(self, item);
        self.owners.pop();
    }
    fn visit_item_mod(&mut self, item: &'ast syn::ItemMod) {
        let test = test_attributes(&item.attrs);
        self.test_depth += usize::from(test);
        self.owners.push(item.ident.to_string());
        visit::visit_item_mod(self, item);
        self.owners.pop();
        self.test_depth -= usize::from(test);
    }
    fn visit_item_fn(&mut self, item: &'ast syn::ItemFn) {
        let test = test_attributes(&item.attrs);
        self.test_depth += usize::from(test);
        self.visit_signature(&item.sig);
        self.owners.push(item.sig.ident.to_string());
        self.visit_block(&item.block);
        self.owners.pop();
        self.test_depth -= usize::from(test);
    }
    fn visit_impl_item_fn(&mut self, item: &'ast syn::ImplItemFn) {
        let test = test_attributes(&item.attrs);
        self.test_depth += usize::from(test);
        self.visit_signature(&item.sig);
        self.owners.push(item.sig.ident.to_string());
        self.visit_block(&item.block);
        self.owners.pop();
        self.test_depth -= usize::from(test);
    }
    fn visit_field(&mut self, item: &'ast syn::Field) {
        self.record(
            "field",
            item.ident
                .as_ref()
                .map(ToString::to_string)
                .unwrap_or_else(|| "<tuple>".into()),
            &item.ty,
        );
        visit::visit_field(self, item);
    }
    fn visit_signature(&mut self, item: &'ast syn::Signature) {
        self.owners.push(item.ident.to_string());
        for arg in &item.inputs {
            if let syn::FnArg::Typed(arg) = arg {
                self.record("parameter", arg.pat.to_token_stream().to_string(), &arg.ty);
            }
        }
        if let syn::ReturnType::Type(_, ty) = &item.output {
            self.record("return", "return".into(), ty);
        }
        self.owners.pop();
    }
    fn visit_item_type(&mut self, item: &'ast syn::ItemType) {
        self.record("alias", item.ident.to_string(), &item.ty);
        visit::visit_item_type(self, item);
    }
    fn visit_pat_type(&mut self, item: &'ast syn::PatType) {
        self.record(
            "annotation",
            item.pat.to_token_stream().to_string(),
            &item.ty,
        );
        visit::visit_pat_type(self, item);
    }
    fn visit_item_const(&mut self, item: &'ast syn::ItemConst) {
        self.record("constant", item.ident.to_string(), &item.ty);
        visit::visit_item_const(self, item);
    }
    fn visit_macro(&mut self, item: &'ast syn::Macro) {
        let name = item.path.to_token_stream().to_string();
        if name == "scalar_quantity" {
            self.records.push(json!({"path":self.source,"line":item.span().start().line,"owner":self.owners.join("::"),"kind":"quantity_macro","name":name,"type":item.tokens.to_string(),"disposition":"unclassified"}));
        }
        visit::visit_macro(self, item);
    }
}
fn collect(root: &Path, paths: &mut Vec<PathBuf>) -> std::io::Result<()> {
    if root.is_file() {
        if root.extension().is_some_and(|e| e == "rs") {
            paths.push(root.to_owned());
        }
        return Ok(());
    }
    let mut entries = fs::read_dir(root)?.collect::<Result<Vec<_>, _>>()?;
    entries.sort_by_key(|e| e.file_name());
    for entry in entries {
        let path = entry.path();
        if entry.file_type()?.is_symlink() {
            continue;
        }
        if entry.file_type()?.is_dir() {
            if !matches!(
                entry.file_name().to_str(),
                Some("target" | "node_modules" | ".git")
            ) {
                collect(&path, paths)?;
            }
        } else if path.extension().is_some_and(|e| e == "rs") {
            paths.push(path);
        }
    }
    Ok(())
}
/// Resolve external test-module ownership through the AST, not file-name guesses.
/// A source used by both production and tests stays a production candidate.
fn external_test_files(files: &[(&PathBuf, syn::File)]) -> BTreeSet<PathBuf> {
    fn normalized(path: &Path) -> PathBuf {
        path.canonicalize().unwrap_or_else(|_| path.to_owned())
    }
    fn module_edges(
        items: &[syn::Item],
        base: &Path,
        path_directory: &Path,
        inherited_test: bool,
        known: &BTreeSet<PathBuf>,
        output: &mut Vec<(PathBuf, bool)>,
    ) {
        for item in items {
            let syn::Item::Mod(module) = item else {
                continue;
            };
            let test = inherited_test || test_attributes(&module.attrs);
            if let Some((_, items)) = &module.content {
                module_edges(
                    items,
                    &base.join(module.ident.to_string()),
                    &base.join(module.ident.to_string()),
                    test,
                    known,
                    output,
                );
            } else {
                // A literal #[path] has an exact source-relative owner. It is
                // common for qualification/evaluator fixtures and must inherit
                // cfg(test) without treating a separately production-owned file
                // as test-only. Nonliteral/macro paths remain conservative.
                if let Some(attribute) = module
                    .attrs
                    .iter()
                    .find(|attr| attr.path().is_ident("path"))
                {
                    if let syn::Meta::NameValue(value) = &attribute.meta {
                        if let syn::Expr::Lit(value) = &value.value {
                            if let syn::Lit::Str(value) = &value.lit {
                                let path = normalized(&path_directory.join(value.value()));
                                if known.contains(&path) {
                                    output.push((path, test));
                                }
                            }
                        }
                    }
                    continue;
                }
                let stem = base.join(module.ident.to_string());
                for path in [stem.with_extension("rs"), stem.join("mod.rs")] {
                    let path = normalized(&path);
                    if known.contains(&path) {
                        output.push((path, test));
                        break;
                    }
                }
            }
        }
    }
    let known = files
        .iter()
        .map(|(p, _)| normalized(p))
        .collect::<BTreeSet<_>>();
    let mut edges = BTreeMap::new();
    let mut incoming = BTreeSet::new();
    for (path, file) in files {
        let base = if matches!(
            path.file_name().and_then(|n| n.to_str()),
            Some("lib.rs" | "main.rs" | "mod.rs")
        ) {
            path.parent().unwrap().to_owned()
        } else {
            path.with_extension("")
        };
        let mut out = Vec::new();
        module_edges(
            &file.items,
            &base,
            path.parent().unwrap(),
            false,
            &known,
            &mut out,
        );
        incoming.extend(out.iter().map(|(p, _)| p.clone()));
        edges.insert(normalized(path), out);
    }
    let mut work = VecDeque::new();
    for root in known.difference(&incoming) {
        work.push_back((root.clone(), false));
    }
    // A cfg(test) edge can be present below an out-of-scope parent; record it too.
    for out in edges.values() {
        for (path, test) in out {
            if *test {
                work.push_back((path.clone(), true));
            }
        }
    }
    let mut seen = BTreeSet::new();
    while let Some((path, test)) = work.pop_front() {
        if !seen.insert((path.clone(), test)) {
            continue;
        }
        if let Some(out) = edges.get(&path) {
            for (child, child_test) in out {
                work.push_back((child.clone(), test || *child_test));
            }
        }
    }
    known
        .into_iter()
        .filter(|path| {
            seen.contains(&(path.clone(), true)) && !seen.contains(&(path.clone(), false))
        })
        .collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let roots: Vec<String> = env::args().skip(1).collect();
    if roots.iter().any(|s| matches!(s.as_str(), "--help" | "-h")) {
        println!("usage: runmat-quantity-inventory <Rust source root>... (JSON to stdout)");
        return Ok(());
    }
    if roots.is_empty() {
        return Err(
            "usage: runmat-quantity-inventory <Rust source root>... (JSON to stdout)".into(),
        );
    }
    let cwd = env::current_dir()?;
    let mut paths = Vec::new();
    for root in &roots {
        collect(Path::new(root), &mut paths)?;
    }
    paths.sort();
    paths.dedup();
    let mut files = Vec::new();
    let mut aliases = BTreeSet::new();
    // The existing macro-generated quantities are declarations too; macros are
    // not expanded by syn, so record and resolve this explicitly known family.
    for path in &paths {
        let source = fs::read_to_string(path)?;
        let parsed = syn::parse_file(&source).map_err(|e| format!("{}: {e}", path.display()))?;
        for line in source
            .lines()
            .filter(|s| s.trim_start().starts_with("scalar_quantity!("))
        {
            if let Some(name) = line
                .split("scalar_quantity!(")
                .nth(1)
                .and_then(|s| s.split(',').next())
            {
                aliases.insert(name.trim().to_string());
            }
        }
        files.push((path, parsed));
    }
    // Resolve direct numeric aliases to a fixed point (e.g. ComplexScalar).
    loop {
        let before = aliases.len();
        for (_, file) in &files {
            for item in &file.items {
                if let syn::Item::Type(item) = item {
                    let text = item.ty.to_token_stream().to_string();
                    if text
                        .split(|c: char| !c.is_alphanumeric() && c != '_')
                        .any(|n| {
                            matches!(
                                n,
                                "f64"
                                    | "f32"
                                    | "Quantity"
                                    | "AnyQuantity"
                                    | "SourceMeasure"
                                    | "SourceQuantity"
                                    | "Unit"
                            ) || aliases.contains(n)
                        })
                    {
                        aliases.insert(item.ident.to_string());
                    }
                }
            }
        }
        if aliases.len() == before {
            break;
        }
    }
    let external_tests = external_test_files(&files);
    let mut inventory = Inventory {
        source: String::new(),
        owners: vec![],
        records: vec![],
        numeric_aliases: aliases,
        test_depth: 0,
    };
    for (path, file) in &files {
        inventory.source = path
            .strip_prefix(&cwd)
            .unwrap_or(path)
            .display()
            .to_string();
        inventory.test_depth = usize::from(
            external_tests.contains(&path.canonicalize().unwrap_or_else(|_| (*path).clone())),
        );
        inventory.visit_file(file);
    }
    let mut counts = serde_json::Map::new();
    for record in &inventory.records {
        let kind = record["kind"].as_str().unwrap();
        let n = counts.get(kind).and_then(Value::as_u64).unwrap_or(0);
        counts.insert(kind.into(), json!(n + 1));
    }
    println!(
        "{}",
        serde_json::to_string_pretty(
            &json!({"schema_version":1,"scope":roots,"files_parsed":files.len(),"counts":counts,"limits":["Records are candidates, not inferred physical dimensions.","Includes tests and cfg-disabled Rust syntax; does not expand arbitrary macros or infer local expression types.","Dependency aliases, FFI-generated declarations, TypeScript and artifact schemas require their own inventory passes.","No unclassified candidate counts as migrated or approved raw storage."],"records":inventory.records})
        )?
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn scan(source: &str) -> Vec<Value> {
        let mut inventory = Inventory {
            source: "fixture.rs".into(),
            owners: vec![],
            records: vec![],
            numeric_aliases: BTreeSet::from(["ElectricCurrent".into()]),
            test_depth: 0,
        };
        inventory.visit_file(&syn::parse_file(source).unwrap());
        inventory.records
    }
    #[test]
    fn records_unit_contracts_and_nested_numeric_storage_without_guessing_units() {
        let records=scan("struct Material { current: ElectricCurrent, coefficient: f64, field: Option<Vec<f64>>, name: String } enum Load { Power(f64), Current(ElectricCurrent) }");
        assert_eq!(records.len(), 5);
        assert!(records.iter().all(|v| v["disposition"] == "unclassified"));
        assert!(records.iter().any(|v| v["owner"] == "Load::Current"));
        assert!(!records.iter().any(|v| v["name"] == "name"));
    }
    #[test]
    fn records_each_signature_parameter_once() {
        let records = scan("fn flux(area: f64, current: ElectricCurrent) -> Vec<f64> { vec![] }");
        assert_eq!(
            records.iter().filter(|v| v["kind"] == "parameter").count(),
            2
        );
        assert_eq!(
            records.iter().filter(|v| v["kind"] == "annotation").count(),
            0
        );
        assert_eq!(records.iter().filter(|v| v["kind"] == "return").count(), 1);
    }
    #[test]
    fn includes_tests_cfg_disabled_code_and_known_quantity_macros() {
        let records=scan("scalar_quantity!(HeatFlux, watts_per_square_meter); #[cfg(test)] mod tests { struct Example { watts: f64 } }");
        assert!(records.iter().any(|v| v["kind"] == "quantity_macro"));
        assert!(records
            .iter()
            .any(|v| v["owner"] == "tests::Example" && v["test_context"] == true));
    }
    #[test]
    fn includes_string_metadata_and_closed_unit_catalogs() {
        let records = scan("enum UnitSystem { Meter, Millimeter } struct Evidence { source_unit: String, dimension: [i8; 8] }");
        assert_eq!(records.len(), 3);
        assert!(records.iter().any(|v| v["kind"] == "unit_enum"));
        assert!(records.iter().any(|v| v["name"] == "source_unit"));
    }
    #[test]
    fn propagates_external_cfg_test_ownership_without_hiding_shared_production() {
        let paths = [
            "/quantity-fixture/lib.rs",
            "/quantity-fixture/qualification.rs",
            "/quantity-fixture/qualification/nested.rs",
            "/quantity-fixture/shared.rs",
        ]
        .map(PathBuf::from);
        let sources = ["#[cfg(test)] mod qualification; #[cfg(test)] mod shared; #[cfg(not(test))] mod shared;", "mod nested;", "struct Case { value: f64 }", "struct Shared { value: f64 }"];
        let files = paths
            .iter()
            .zip(sources)
            .map(|(path, source)| (path, syn::parse_file(source).unwrap()))
            .collect::<Vec<_>>();
        let tests = external_test_files(&files);
        assert!(tests.contains(&paths[1]));
        assert!(tests.contains(&paths[2]));
        assert!(!tests.contains(&paths[0]));
        assert!(!tests.contains(&paths[3]));
    }
    #[test]
    fn literal_path_test_modules_inherit_their_actual_source_owner() {
        let paths = [
            "/quantity-fixture/lib.rs",
            "/quantity-fixture/ops.rs",
            "/quantity-fixture/fixtures/evaluator.rs",
            "/quantity-fixture/fixtures/shared.rs",
        ]
        .map(PathBuf::from);
        let sources = ["mod ops;", "#[cfg(test)] #[path=\"fixtures/evaluator.rs\"] mod evaluator; #[cfg(test)] #[path=\"fixtures/shared.rs\"] mod tests_only; #[cfg(not(test))] #[path=\"fixtures/shared.rs\"] mod live;", "struct Fixture { value:f64 }", "struct Shared { value:f64 }"];
        let files = paths
            .iter()
            .zip(sources)
            .map(|(path, source)| (path, syn::parse_file(source).unwrap()))
            .collect::<Vec<_>>();
        let tests = external_test_files(&files);
        assert!(tests.contains(&paths[2]));
        assert!(!tests.contains(&paths[0]));
        assert!(!tests.contains(&paths[1]));
        assert!(!tests.contains(&paths[3]));
    }
}
