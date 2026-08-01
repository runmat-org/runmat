use anyhow::{Context, Result};
use runmat_config::runtime::RunMatRuntimeConfig;
use runmat_hir::LoweringContext;
use runmat_vm::Instr;
use std::collections::HashMap;
use std::fmt::Write as FmtWrite;
use std::fs;
use std::io::Write;
use std::path::PathBuf;

use crate::diagnostics::parser_compat;

pub fn emit_bytecode(
    source: &str,
    config: &RunMatRuntimeConfig,
    source_name: Option<&str>,
) -> Result<String> {
    let source_catalog = discover_source_catalog(source_name);
    let known_project_symbols = source_catalog
        .as_ref()
        .map(|catalog| catalog.symbols.clone())
        .unwrap_or_default();
    let lowering_context =
        LoweringContext::empty().with_known_project_symbols(&known_project_symbols);
    let analysis = runmat_static_analysis::frontend::analyze_source_with_catalog(
        source,
        parser_compat(config.language.compat),
        &lowering_context,
        source_catalog.as_ref(),
    );
    let bytecode = analysis.bytecode.ok_or_else(|| {
        let detail = analysis
            .diagnostics
            .first()
            .map(|diagnostic| diagnostic.message.as_str())
            .unwrap_or("canonical frontend did not produce bytecode");
        anyhow::anyhow!("Compile error: {detail}")
    })?;
    Ok(disassemble_bytecode(&bytecode))
}

fn discover_source_catalog(
    source_name: Option<&str>,
) -> Option<runmat_package::DiscoveredSourceSymbols> {
    let source_name = source_name?;
    let Ok(cwd) = std::env::current_dir() else {
        return None;
    };
    runmat_package::discover_source_symbols_from_source_name(source_name, &cwd)
        .ok()
        .flatten()
}

pub fn write_bytecode_output(path: &PathBuf, output: &str) -> Result<()> {
    if path.as_os_str() == "-" {
        println!("{output}");
        return Ok(());
    }
    let mut file = fs::File::create(path)
        .with_context(|| format!("Failed to create bytecode output file {}", path.display()))?;
    file.write_all(output.as_bytes())
        .with_context(|| format!("Failed to write bytecode output file {}", path.display()))?;
    Ok(())
}

fn disassemble_bytecode(bytecode: &runmat_vm::Bytecode) -> String {
    let mut out = String::new();
    if !bytecode.var_names.is_empty() {
        let mut entries: Vec<_> = bytecode.var_names.iter().collect();
        entries.sort_by_key(|(idx, _)| *idx);
        let _ = writeln!(&mut out, "# Variables");
        for (idx, name) in entries {
            let _ = writeln!(&mut out, "v{} = {}", idx, name);
        }
        let _ = writeln!(&mut out);
    }
    let _ = writeln!(&mut out, "# Bytecode");
    for (idx, instr) in bytecode.instructions.iter().enumerate() {
        let mut line = format!("{:04}: {}", idx, format_instr(instr, &bytecode.var_names));
        if let Some(span) = bytecode.instr_spans.get(idx) {
            if span.start != 0 || span.end != 0 {
                let _ = write!(line, "  ; span {}..{}", span.start, span.end);
            }
        }
        let _ = writeln!(&mut out, "{line}");
    }
    out
}

fn format_instr(instr: &Instr, var_names: &HashMap<usize, String>) -> String {
    let label = |idx: usize| var_names.get(&idx).map(|n| n.as_str()).unwrap_or("?");
    match instr {
        Instr::LoadVar(idx) => format!("LoadVar {} ({})", idx, label(*idx)),
        Instr::StoreVar(idx) => format!("StoreVar {} ({})", idx, label(*idx)),
        Instr::LoadLocal(idx) => format!("LoadLocal {}", idx),
        Instr::StoreLocal(idx) => format!("StoreLocal {}", idx),
        Instr::EmitVar {
            var_index,
            label: emit,
        } => {
            format!("EmitVar {} ({}) {:?}", var_index, label(*var_index), emit)
        }
        Instr::EmitStackTop { label: emit } => format!("EmitStackTop {:?}", emit),
        Instr::CallFunctionMulti {
            identity,
            fallback_policy,
            arg_count,
            out_count,
        } => format!(
            "CallFunctionMulti {} args={} outputs={} fallback={fallback_policy:?}",
            identity
                .display_name()
                .unwrap_or_else(|| format!("{identity:?}")),
            arg_count,
            out_count
        ),
        Instr::CallFunctionMultiUsingOutputSlot {
            identity,
            fallback_policy,
            arg_count,
            out_count_slot,
        } => format!(
            "CallFunctionMultiUsingOutputSlot {} args={} output_slot={} fallback={fallback_policy:?}",
            identity
                .display_name()
                .unwrap_or_else(|| format!("{identity:?}")),
            arg_count,
            out_count_slot
        ),
        other => format!("{other:?}"),
    }
}

#[cfg(test)]
mod tests {
    use super::{discover_source_catalog, emit_bytecode};
    use crate::test_support::ScopedCurrentDir;
    use std::fs;

    #[test]
    fn discover_known_project_symbols_reads_manifest_source_context() {
        let tmp = tempfile::TempDir::new().expect("tempdir");
        fs::create_dir_all(tmp.path().join("+stats")).expect("create package dir");
        fs::write(
            tmp.path().join("runmat.toml"),
            r#"
[package]
name = "demo"

[sources]
roots = ["."]
"#,
        )
        .expect("write manifest");
        fs::write(
            tmp.path().join("+stats/summarize.m"),
            "function y = summarize(x); y = x; end",
        )
        .expect("write package function");
        fs::write(tmp.path().join("main.m"), "x = 1;").expect("write source file");

        let _cwd = ScopedCurrentDir::enter(tmp.path());
        let source_name = tmp.path().join("main.m");
        let catalog = discover_source_catalog(Some(source_name.to_string_lossy().as_ref()))
            .expect("discover project source catalog");

        assert!(
            catalog.symbols.contains("stats.summarize"),
            "expected project symbol discovery to include package-qualified names"
        );
    }

    #[test]
    fn emit_bytecode_uses_source_context_project_symbols() {
        let tmp = tempfile::TempDir::new().expect("tempdir");
        fs::create_dir_all(tmp.path().join("+stats")).expect("create package dir");
        fs::write(
            tmp.path().join("runmat.toml"),
            r#"
[package]
name = "demo"

[sources]
roots = ["."]
"#,
        )
        .expect("write manifest");
        fs::write(
            tmp.path().join("+stats/summarize.m"),
            "function y = summarize(x); y = x; end",
        )
        .expect("write package function");
        fs::write(tmp.path().join("main.m"), "x = 1;").expect("write source file");

        let _cwd = ScopedCurrentDir::enter(tmp.path());
        let source_name = tmp.path().join("main.m");
        let output = emit_bytecode(
            "import stats.*; y = summarize(1);",
            &runmat_config::runtime::RunMatRuntimeConfig::default(),
            Some(source_name.to_string_lossy().as_ref()),
        )
        .expect("emit bytecode through canonical frontend");

        assert!(
            output.contains("stats.summarize"),
            "expected call instruction identity to resolve to exact package-qualified symbol; output:\n{output}"
        );
    }

    #[test]
    fn discover_source_catalog_requires_existing_local_source_path() {
        let tmp = tempfile::TempDir::new().expect("tempdir");
        fs::create_dir_all(tmp.path().join("+stats")).expect("create package dir");
        fs::write(
            tmp.path().join("runmat.toml"),
            r#"
[package]
name = "demo"

[sources]
roots = ["."]
"#,
        )
        .expect("write manifest");
        fs::write(
            tmp.path().join("+stats/summarize.m"),
            "function y = summarize(x); y = x; end",
        )
        .expect("write package function");

        let _cwd = ScopedCurrentDir::enter(tmp.path());
        let catalog = discover_source_catalog(Some("virtual/nonexistent_remote.m"));

        assert!(
            catalog.is_none(),
            "nonexistent source names should not pull project symbols from local cwd"
        );
    }

    #[test]
    fn discover_source_catalog_rejects_colon_remote_name() {
        let tmp = tempfile::TempDir::new().expect("tempdir");
        fs::create_dir_all(tmp.path().join("+stats")).expect("create package dir");
        fs::write(
            tmp.path().join("runmat.toml"),
            r#"
[package]
name = "demo"

[sources]
roots = ["."]
"#,
        )
        .expect("write manifest");
        fs::write(
            tmp.path().join("+stats/summarize.m"),
            "function y = summarize(x); y = x; end",
        )
        .expect("write package function");
        fs::write(tmp.path().join("main.m"), "x = 1;").expect("write source file");

        let _cwd = ScopedCurrentDir::enter(tmp.path());
        let catalog = discover_source_catalog(Some("remote:main.m"));

        assert!(
            catalog.is_none(),
            "colon-style remote source names should not pull project symbols from local cwd"
        );
    }
}
