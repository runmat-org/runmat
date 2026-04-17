use anyhow::{Context, Result};
use runmat_config::RunMatConfig;
use runmat_hir::LoweringContext;
use runmat_parser::ParserOptions;
use runmat_vm::instr::Instr;
use std::collections::HashMap;
use std::fmt::Write as FmtWrite;
use std::fs;
use std::io::Write;
use std::path::PathBuf;

use crate::diagnostics::parser_compat;

pub fn emit_bytecode(source: &str, config: &RunMatConfig) -> Result<String> {
    let options = ParserOptions::new(parser_compat(config.language.compat));
    let ast = runmat_parser::parse_with_options(source, options)
        .map_err(|err| anyhow::anyhow!(format!("Parse error: {err:?}")))?;
    let lowering = runmat_hir::lower(&ast, &LoweringContext::empty())
        .map_err(|err| anyhow::anyhow!(format!("Lowering error: {err:?}")))?;
    let mut bytecode = runmat_vm::compile(&lowering.hir, &HashMap::new())
        .map_err(|err| anyhow::anyhow!(format!("Compile error: {err:?}")))?;
    bytecode.var_names = lowering
        .var_names
        .iter()
        .map(|(id, name)| (id.0, name.clone()))
        .collect();
    Ok(disassemble_bytecode(&bytecode))
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
