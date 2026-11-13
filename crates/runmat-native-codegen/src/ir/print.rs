use std::fmt::Write;

use super::*;

pub fn print_native_ir(assembly: &NativeAssembly) -> String {
    let mut output = String::new();
    let _ = writeln!(
        output,
        "native-ir v{} source={} executable={} native={} target={}",
        assembly.schema_version,
        assembly.program.source_digest(),
        assembly.executable_cache_key,
        assembly.native_cache_key,
        match &assembly.target {
            crate::NativeTarget::Native { target, probe } => format!(
                "{:?}-{:?}-{} abi={}:{}",
                target.target.architecture,
                target.target.operating_system,
                probe.pointer_width,
                probe.abi_schema_version,
                probe.abi_encoded_version
            ),
            crate::NativeTarget::PortableWasm32 { schema_version } =>
                format!("portable-wasm32-v{schema_version}"),
            crate::NativeTarget::UnavailableHost => "native-host-unavailable".to_string(),
        }
    );
    for function in &assembly.functions {
        let _ = writeln!(
            output,
            "function f{} {} source={} capabilities={:?} locals={} entry=b{}",
            function.id.0,
            function.name,
            function.source.0,
            function.capabilities,
            function.local_count(),
            function.entry.0
        );
        for local in &function.locals {
            let _ = writeln!(
                output,
                "  local l{} {:?} binding={:?} name={:?}",
                local.id.0, local.kind, local.binding, local.name
            );
        }
        for validation in &function.argument_validations {
            let _ = writeln!(
                output,
                "  argument-validation l{} size={:?} class={:?} validators={:?} default={:?}",
                validation.input.0,
                validation.size,
                validation.class_name,
                validation.validators,
                validation.default_value
            );
        }
        for block in &function.blocks {
            let _ = write!(output, "  b{}(", block.id.0);
            for (index, parameter) in block.parameters.iter().enumerate() {
                if index != 0 {
                    output.push_str(", ");
                }
                let _ = write!(
                    output,
                    "l{}=v{}:{:?}",
                    parameter.local.0, parameter.value.0, parameter.value_type
                );
            }
            let _ = writeln!(output, ") effect=v{}", block.side_effect_epoch.0);
            for boundary in &block.region_boundaries {
                let _ = writeln!(
                    output,
                    "    region {:?} r{} live={:?} guards={} epoch=v{} @ {}:{}:{}",
                    boundary.kind,
                    boundary.region.ordinal,
                    boundary.live_values,
                    boundary.guards.len(),
                    boundary.frame_state.side_effect_epoch.0,
                    boundary.point.function.0,
                    boundary.point.block,
                    boundary.point.position
                );
            }
            for instruction in &block.instructions {
                let _ = writeln!(
                    output,
                    "    i{} {:?} {:?} embedded={:?} effects={:?} caps={:?} in={:?} out={:?} next_effect={:?} safepoint={:?} @ {}:{}:{}:{:?}:{}",
                    instruction.id.0,
                    instruction.class,
                    instruction.operation,
                    instruction.embedded_constructs,
                    instruction.effects,
                    instruction.capabilities,
                    instruction.inputs,
                    instruction.outputs,
                    instruction.effect_epoch_output,
                    instruction.safepoint,
                    instruction.site.point.function.0,
                    instruction.site.point.block,
                    instruction.site.point.position,
                    instruction.site.phase,
                    instruction.site.ordinal
                );
            }
            let _ = writeln!(
                output,
                "    -> {:?} {:?} safepoint={:?}",
                block.terminator.class, block.terminator.kind, block.terminator.safepoint
            );
        }
    }
    output
}
