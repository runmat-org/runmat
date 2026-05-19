use anyhow::{Context, Result};
use runmat_config::RunMatConfig;
use runmat_hir::LoweringContext;
use runmat_parser::ParserOptions;
use runmat_vm::Instr;
use std::collections::{HashMap, HashSet};
use std::fmt::Write as FmtWrite;
use std::fs;
use std::io::Write;
use std::path::PathBuf;

use crate::diagnostics::parser_compat;

pub fn emit_bytecode(
    source: &str,
    config: &RunMatConfig,
    source_name: Option<&str>,
) -> Result<String> {
    let options = ParserOptions::new(parser_compat(config.language.compat));
    let ast = runmat_parser::parse_with_options(source, options)
        .map_err(|err| anyhow::anyhow!(format!("Parse error: {err:?}")))?;
    let known_project_symbols = discover_known_project_symbols(source_name);
    let lowering = runmat_hir::lower(
        &ast,
        &LoweringContext::empty().with_known_project_symbols(&known_project_symbols),
    )
    .map_err(|err| anyhow::anyhow!(format!("Lowering error: {err:?}")))?;
    let bytecode = compile_bytecode(&lowering)?;
    Ok(disassemble_bytecode(&bytecode))
}

fn discover_known_project_symbols(source_name: Option<&str>) -> HashSet<String> {
    use runmat_config::discover_project_symbols_from_source_name;

    let Some(source_name) = source_name else {
        return HashSet::new();
    };
    let Ok(cwd) = std::env::current_dir() else {
        return HashSet::new();
    };
    let source_path = PathBuf::from(source_name);
    let local_candidate = if source_path.is_absolute() {
        source_path
    } else {
        cwd.join(&source_path)
    };
    if !local_candidate.exists() {
        return HashSet::new();
    }
    let Ok(discovered) = discover_project_symbols_from_source_name(source_name, &cwd) else {
        return HashSet::new();
    };
    discovered
        .map(|discovered| discovered.symbols)
        .unwrap_or_default()
}

fn compile_bytecode(lowering: &runmat_hir::LoweringResult) -> Result<runmat_vm::Bytecode> {
    let entrypoint =
        lowering.assembly.entrypoints.first().ok_or_else(|| {
            anyhow::anyhow!("Compile error: semantic HIR assembly has no entrypoint")
        })?;
    let mir = runmat_mir::lowering::lower_assembly(&lowering.assembly)
        .map_err(|err| anyhow::anyhow!(format!("MIR lowering error: {err:?}")))?;
    let bytecode = runmat_vm::compile(&lowering.assembly, &mir, entrypoint.id)
        .map_err(|err| anyhow::anyhow!(format!("Compile error: {err:?}")))?;
    Ok(bytecode)
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
        other => format!("{other:?}"),
    }
}

#[cfg(test)]
mod tests {
    use super::{discover_known_project_symbols, emit_bytecode};
    use once_cell::sync::Lazy;
    use runmat_config::RunMatConfig;
    use std::fs;
    use std::sync::Mutex;

    static CWD_LOCK: Lazy<Mutex<()>> = Lazy::new(|| Mutex::new(()));

    #[test]
    fn discover_known_project_symbols_reads_manifest_source_context() {
        let _guard = CWD_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
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

        let prev = std::env::current_dir().expect("cwd");
        std::env::set_current_dir(tmp.path()).expect("set cwd");
        let source_name = tmp.path().join("main.m");
        let symbols = discover_known_project_symbols(Some(source_name.to_string_lossy().as_ref()));
        std::env::set_current_dir(prev).expect("restore cwd");

        assert!(
            symbols.contains("stats.summarize"),
            "expected project symbol discovery to include package-qualified names"
        );
    }

    #[test]
    fn emit_bytecode_uses_source_context_project_symbols() {
        let _guard = CWD_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
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

        let prev = std::env::current_dir().expect("cwd");
        std::env::set_current_dir(tmp.path()).expect("set cwd");
        let source_name = tmp.path().join("main.m");
        let output = emit_bytecode(
            "import stats.*; y = summarize(1);",
            &RunMatConfig::default(),
            Some(source_name.to_string_lossy().as_ref()),
        )
        .expect("emit bytecode with project symbol context");
        std::env::set_current_dir(prev).expect("restore cwd");

        assert!(
            output.contains("summarize"),
            "expected disassembly to include imported callable identity"
        );
        assert!(
            output.contains("CallFunctionMulti")
                || output.contains("CallSemanticFunctionMulti")
                || output.contains("CallBuiltinMulti"),
            "expected disassembly to include a call instruction for imported symbol"
        );
    }

    #[test]
    fn discover_known_project_symbols_requires_existing_local_source_path() {
        let _guard = CWD_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
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

        let prev = std::env::current_dir().expect("cwd");
        std::env::set_current_dir(tmp.path()).expect("set cwd");
        let symbols = discover_known_project_symbols(Some("virtual/nonexistent_remote.m"));
        std::env::set_current_dir(prev).expect("restore cwd");

        assert!(
            symbols.is_empty(),
            "nonexistent source names should not pull project symbols from local cwd"
        );
    }
}
