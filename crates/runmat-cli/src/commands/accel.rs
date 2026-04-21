use anyhow::{Context, Result};

#[cfg(feature = "wgpu")]
fn print_threshold_delta(label: &str, entry: &runmat_accelerate::ThresholdDeltaEntry) {
    let percent = entry.ratio.map(|r| (r - 1.0) * 100.0);
    match percent {
        Some(p) => println!(
            "  {:<28} {:>12.6e} -> {:>12.6e} (Δ {:>+12.6e}, {:+6.2}%)",
            label, entry.before, entry.after, entry.absolute, p
        ),
        None => println!(
            "  {:<28} {:>12.6e} -> {:>12.6e} (Δ {:>+12.6e})",
            label, entry.before, entry.after, entry.absolute
        ),
    }
}

#[cfg(feature = "wgpu")]
pub fn dump_provider_telemetry_if_requested() {
    let path = match std::env::var("RUNMAT_TELEMETRY_OUT") {
        Ok(p) if !p.trim().is_empty() => p,
        _ => return,
    };

    let provider = match runmat_accelerate_api::provider() {
        Some(p) => p,
        None => return,
    };

    let info = provider.device_info_struct();
    let telemetry = provider.telemetry_snapshot();

    let mut payload = serde_json::Map::new();
    match serde_json::to_value(&info) {
        Ok(value) => {
            payload.insert("device".to_string(), value);
        }
        Err(err) => log::warn!("Failed to serialize device info for telemetry dump: {err}"),
    }
    match serde_json::to_value(&telemetry) {
        Ok(value) => {
            payload.insert("telemetry".to_string(), value);
        }
        Err(err) => log::warn!("Failed to serialize telemetry snapshot: {err}"),
    }
    if let Some(report) = runmat_accelerate::auto_offload_report() {
        match serde_json::to_value(&report) {
            Ok(value) => {
                payload.insert("auto_offload".to_string(), value);
            }
            Err(err) => log::warn!("Failed to serialize auto-offload report: {err}"),
        }
    }

    let json_payload = serde_json::Value::Object(payload);
    if let Err(err) = std::fs::write(
        &path,
        serde_json::to_string_pretty(&json_payload).unwrap_or_default(),
    ) {
        log::warn!("Failed to write telemetry snapshot to {path}: {err}");
    }

    let reset_flag = std::env::var("RUNMAT_TELEMETRY_RESET")
        .map(|v| {
            matches!(
                v.as_str(),
                "1" | "true" | "TRUE" | "True" | "yes" | "YES" | "Yes" | "on" | "ON"
            )
        })
        .unwrap_or(false);

    if reset_flag {
        provider.reset_telemetry();
        runmat_accelerate::reset_auto_offload_log();
    }
}

#[cfg(not(feature = "wgpu"))]
pub fn dump_provider_telemetry_if_requested() {}

#[cfg(feature = "wgpu")]
pub async fn show_accel_info(json: bool, reset: bool) -> Result<()> {
    if let Some(p) = runmat_accelerate_api::provider() {
        let info = p.device_info_struct();
        let telemetry = p.telemetry_snapshot();

        if json {
            let mut payload = serde_json::Map::new();
            payload.insert("device".to_string(), serde_json::to_value(&info)?);
            payload.insert("telemetry".to_string(), serde_json::to_value(&telemetry)?);
            if let Some(report) = runmat_accelerate::auto_offload_report() {
                payload.insert("auto_offload".to_string(), serde_json::to_value(&report)?);
            }
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::Value::Object(payload))?
            );
        } else {
            println!("Acceleration Provider Info");
            println!("==========================");
            println!(
                "Device: {} ({})",
                info.name,
                info.backend.clone().unwrap_or_default()
            );
            println!(
                "Fused pipeline cache: hits={}, misses={}",
                telemetry.fusion_cache_hits, telemetry.fusion_cache_misses
            );
            println!(
                "Bind group cache: hits={}, misses={}",
                telemetry.bind_group_cache_hits, telemetry.bind_group_cache_misses
            );
            println!(
                "Reduction defaults: two_pass_mode={}, two_pass_threshold={}, workgroup_size={} (env: RUNMAT_REDUCTION_TWO_PASS / RUNMAT_TWO_PASS_THRESHOLD / RUNMAT_REDUCTION_WG)",
                p.reduction_two_pass_mode().as_str(),
                p.two_pass_threshold(),
                p.default_reduction_workgroup_size()
            );
            if let Some(ms) = p.last_warmup_millis() {
                println!("Warmup: last duration ~{} ms", ms);
            }
            let to_ms = |ns: u64| ns as f64 / 1_000_000.0;
            println!("Telemetry:");
            println!(
                "  uploads: {} bytes, downloads: {} bytes",
                telemetry.upload_bytes, telemetry.download_bytes
            );
            println!(
                "  fused_elementwise: count={} wall_ms={:.3}",
                telemetry.fused_elementwise.count,
                to_ms(telemetry.fused_elementwise.total_wall_time_ns)
            );
            println!(
                "  fused_reduction: count={} wall_ms={:.3}",
                telemetry.fused_reduction.count,
                to_ms(telemetry.fused_reduction.total_wall_time_ns)
            );
            println!(
                "  matmul: count={} wall_ms={:.3}",
                telemetry.matmul.count,
                to_ms(telemetry.matmul.total_wall_time_ns)
            );
            println!(
                "  linsolve: count={} wall_ms={:.3}",
                telemetry.linsolve.count,
                to_ms(telemetry.linsolve.total_wall_time_ns)
            );
            println!(
                "  mldivide: count={} wall_ms={:.3}",
                telemetry.mldivide.count,
                to_ms(telemetry.mldivide.total_wall_time_ns)
            );
            println!(
                "  mrdivide: count={} wall_ms={:.3}",
                telemetry.mrdivide.count,
                to_ms(telemetry.mrdivide.total_wall_time_ns)
            );
            if !telemetry.solve_fallbacks.is_empty() {
                println!("  solve_fallbacks:");
                for fallback in &telemetry.solve_fallbacks {
                    println!("    {} => {}", fallback.reason, fallback.count);
                }
            }

            if let Some(report) = runmat_accelerate::auto_offload_report() {
                println!("Auto-offload:");
                println!("  source: {}", report.base_source.as_str());
                println!("  env_overrides_applied: {}", report.env_overrides_applied);
                if let Some(path) = report.cache_path.as_deref() {
                    println!("  cache: {}", path);
                }
                if let Some(ms) = report.calibrate_duration_ms {
                    println!("  last_calibration_ms: {}", ms);
                }
                let thresholds = &report.thresholds;
                println!(
                    "  thresholds: unary={} binary={} reduction={} matmul_flops={} small_batch_dim={} small_batch_min_elems={}",
                    thresholds.unary_min_elems,
                    thresholds.binary_min_elems,
                    thresholds.reduction_min_elems,
                    thresholds.matmul_min_flops,
                    thresholds.small_batch_max_dim,
                    thresholds.small_batch_min_elems
                );
                if !report.decisions.is_empty() {
                    println!("  recent decisions:");
                    for entry in report.decisions.iter().rev().take(5) {
                        println!(
                            "    ts={} op={} decision={:?} reason={:?} elems={:?} flops={:?} batch={:?}",
                            entry.timestamp_ms,
                            entry.operation,
                            entry.decision,
                            entry.reason,
                            entry.elements,
                            entry.flops,
                            entry.batch
                        );
                    }
                }
            }
        }

        if reset {
            p.reset_telemetry();
            runmat_accelerate::reset_auto_offload_log();
        }
    } else if json {
        let payload = serde_json::json!({
            "device": serde_json::Value::Null,
            "telemetry": serde_json::Value::Null,
            "error": "no acceleration provider registered",
        });
        println!("{}", serde_json::to_string_pretty(&payload)?);
    } else {
        println!("Acceleration Provider Info");
        println!("==========================");
        println!("No acceleration provider registered");
    }

    Ok(())
}

#[cfg(not(feature = "wgpu"))]
pub async fn show_accel_info(json: bool, _reset: bool) -> Result<()> {
    if json {
        let payload = serde_json::json!({
            "device": serde_json::Value::Null,
            "telemetry": serde_json::Value::Null,
            "error": "wgpu feature not enabled",
        });
        println!("{}", serde_json::to_string_pretty(&payload)?);
    } else {
        println!("Acceleration Provider Info");
        println!("==========================");
        println!("This build was compiled without the 'wgpu' feature. No GPU provider available.");
    }
    Ok(())
}

#[cfg(feature = "wgpu")]
pub async fn execute_accel_calibrate(
    input: std::path::PathBuf,
    dry_run: bool,
    json: bool,
) -> Result<()> {
    let commit = !dry_run;
    let outcome = runmat_accelerate::apply_auto_offload_calibration_from_file(&input, commit)
        .with_context(|| format!("failed to apply calibration from {}", input.display()))?;

    if json {
        println!("{}", serde_json::to_string_pretty(&outcome)?);
        return Ok(());
    }

    println!("Auto-offload calibration");
    println!("========================");
    println!("Input: {}", input.display());
    if let Some(provider) = &outcome.provider {
        println!(
            "Provider: {} ({}) device_id={}",
            provider.name,
            provider.backend.clone().unwrap_or_default(),
            provider.device_id
        );
    }
    println!("Runs considered: {}", outcome.runs);
    println!("Mode: {}", if commit { "commit" } else { "dry-run" });

    if let Some(delta) = &outcome.delta {
        println!("\nUpdated coefficients (seconds per unit):");
        let mut printed = false;
        if let Some(entry) = &delta.cpu_elem_per_elem {
            print_threshold_delta("cpu_elem_per_elem", entry);
            printed = true;
        }
        if let Some(entry) = &delta.cpu_reduction_per_elem {
            print_threshold_delta("cpu_reduction_per_elem", entry);
            printed = true;
        }
        if let Some(entry) = &delta.cpu_matmul_per_flop {
            print_threshold_delta("cpu_matmul_per_flop", entry);
            printed = true;
        }
        if !printed {
            println!("  (no coefficient changes)");
        }
    } else {
        println!("\nCalibration sample did not yield coefficient adjustments.");
    }

    println!("\nThreshold snapshots:");
    println!(
        "  unary={} -> {}",
        outcome.before.unary_min_elems, outcome.after.unary_min_elems
    );
    println!(
        "  binary={} -> {}",
        outcome.before.binary_min_elems, outcome.after.binary_min_elems
    );
    println!(
        "  reduction={} -> {}",
        outcome.before.reduction_min_elems, outcome.after.reduction_min_elems
    );
    println!(
        "  matmul_flops={} -> {}",
        outcome.before.matmul_min_flops, outcome.after.matmul_min_flops
    );

    if commit {
        if let Some(path) = &outcome.persisted_to {
            println!("\nPersisted calibration cache: {path}");
        }
        println!("Restart RunMat sessions to load the updated thresholds.");
    } else {
        println!(
            "\nDry-run: thresholds were not persisted. Re-run without --dry-run to commit changes."
        );
    }

    Ok(())
}
