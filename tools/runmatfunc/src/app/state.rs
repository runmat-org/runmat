use std::path::PathBuf;

use anyhow::Result;
use indicatif::ProgressBar;
use std::time::Duration;

use crate::app::config::AppConfig;
use crate::builtin::template as builtin_template;
use crate::cli::CliArgs;
use crate::context::{manifest, serialize};
use crate::jobs::{queue as job_queue, scheduler};
use crate::workspace::tests as workspace_tests;

/// Central application context carried throughout the CLI/TUI lifecycle.
pub struct AppContext {
    pub config: AppConfig,
}

impl AppContext {
    pub fn new(args: &CliArgs) -> Result<Self> {
        let config = AppConfig::load(args)?;
        Ok(Self { config })
    }

    pub fn launch_tui(&mut self) -> Result<()> {
        let manifest = manifest::build_manifest()?;
        crate::tui::run(manifest, self.config.clone())
    }

    pub fn print_manifest(&mut self) -> Result<()> {
        let manifest = manifest::build_manifest()?;
        println!("RunMat builtins: {} entries", manifest.builtins.len());
        for record in &manifest.builtins {
            println!(
                "- {:<20} | {:<20} | sink: {} | accel: {}",
                record.name,
                record.category.as_deref().unwrap_or(""),
                record.is_sink,
                record.accel_tags.join(",")
            );
        }
        Ok(())
    }

    pub fn emit_docs(&mut self, out_dir: Option<&str>) -> Result<()> {
        let manifest = manifest::build_manifest()?;
        let out_path = out_dir
            .map(PathBuf::from)
            .unwrap_or_else(|| self.config.docs_output_dir().to_path_buf());
        serialize::write_manifest_files(&manifest, out_path.as_path())?;
        println!(
            "[runmatfunc] wrote documentation manifest to {}",
            out_path.display()
        );
        Ok(())
    }

    pub fn run_builtin(
        &mut self,
        name: &str,
        category: Option<&str>,
        model: Option<String>,
        use_codex: bool,
        show_diff: bool,
        show_doc: bool,
    ) -> Result<()> {
        // If a category was provided, attempt to generate a skeleton builtin before proceeding.
        if let Some(cat) = category {
            match builtin_template::generate_if_missing(name, cat) {
                Ok(outcome) => {
                    if outcome.created {
                        println!(
                            "[runmatfunc] created builtin skeleton at {}",
                            outcome.target_file.display()
                        );
                    }
                }
                Err(err) => {
                    println!(
                        "[runmatfunc] warning: generator failed (continuing): {}",
                        err
                    );
                }
            }
        }

        let resolved_model = model.clone().or_else(|| self.config.default_model.clone());
        let ctx = crate::context::gather::build_authoring_context(name, category, &self.config)?;
        let codex_available = crate::codex::client::is_available();
        println!(
            "Preparing builtin '{}' (model: {})",
            ctx.builtin.name,
            resolved_model.as_deref().unwrap_or("default")
        );
        println!(
            "Codex availability: {}",
            if codex_available {
                "available"
            } else {
                "not available (stub)"
            }
        );
        println!("\nPrompt:\n{}", ctx.prompt);
        if show_doc {
            if let Some(doc) = &ctx.doc_markdown {
                println!("\nDocumentation (DOC_MD):\n{}", doc);
            }
        } else if ctx.doc_markdown.is_some() {
            println!(
                "\n[runmatfunc] Documentation excerpt hidden (rerun with --show-doc to display)."
            );
        }
        println!("\nRelevant sources:");
        for path in &ctx.source_paths {
            println!("- {}", path.display());
        }
        if show_diff {
            match crate::workspace::diff::git_diff(&ctx.source_paths) {
                Ok(Some(diff)) => {
                    println!("\n[runmatfunc] git diff for tracked sources:\n{}", diff);
                }
                Ok(None) => {
                    println!("\n[runmatfunc] no workspace diff for the selected builtin sources");
                }
                Err(err) => {
                    println!("\n[runmatfunc] diff error: {err}");
                }
            }
        }
        let mut spinner = None;
        if use_codex && codex_available {
            let pb = ProgressBar::new_spinner();
            pb.set_message("Contacting Codex…");
            pb.enable_steady_tick(Duration::from_millis(120));
            spinner = Some(pb);
        }

        if use_codex && !codex_available {
            println!(
                "\n[runmatfunc] Codex requested but not available; rerun with --features embedded-codex"
            );
        }
        if use_codex && codex_available {
            match crate::codex::session::run_authoring(&ctx, resolved_model.clone()) {
                Ok(Some(response)) => {
                    if let Some(pb) = spinner.take() {
                        pb.finish_with_message("[runmatfunc] Codex response received");
                    }
                    println!("\n[runmatfunc] Codex response:\n{}", response.summary);
                }
                Ok(None) => {
                    if let Some(pb) = spinner.take() {
                        pb.finish_with_message("[runmatfunc] Codex feature disabled (no response)");
                    }
                    println!("\n[runmatfunc] Codex integration not enabled (requires embedded-codex feature).");
                }
                Err(err) => {
                    if let Some(pb) = spinner.take() {
                        pb.finish_with_message("[runmatfunc] Codex error");
                    }
                    println!("\n[runmatfunc] Codex error: {err}");
                }
            }
        } else if !use_codex {
            println!("\n[runmatfunc] Codex integration skipped (pass --codex to enable).");
        }
        if let Some(pb) = spinner.take() {
            pb.finish_and_clear();
        }
        let mut outcome = match workspace_tests::run_builtin_tests(&ctx, &self.config) {
            Ok(outcome) => outcome,
            Err(err) => {
                println!("[runmatfunc] test execution error: {err}");
                return Ok(());
            }
        };

        // Iterative Codex loop: if tests failed and Codex is available, append failures and retry up to 2 times
        if use_codex && codex_available && !outcome.success {
            let mut attempts = 0usize;
            while attempts < 2 {
                attempts += 1;
                let mut failure_digest = String::new();
                failure_digest.push_str("Tests failed. Here are logs and errors.\n\n");
                for report in &outcome.reports {
                    if report.skipped || report.success {
                        continue;
                    }
                    failure_digest.push_str(&format!("# {}\n", report.label));
                    if !report.stdout.trim().is_empty() {
                        failure_digest.push_str("STDOUT:\n");
                        failure_digest.push_str(&report.stdout);
                        failure_digest.push_str("\n\n");
                    }
                    if !report.stderr.trim().is_empty() {
                        failure_digest.push_str("STDERR:\n");
                        failure_digest.push_str(&report.stderr);
                        failure_digest.push_str("\n\n");
                    }
                }

                let polish_tail = None;
                let _ = crate::codex::session::run_authoring_with_extra(
                    &ctx,
                    resolved_model.clone(),
                    &failure_digest,
                    polish_tail,
                );

                // Re-run tests
                match workspace_tests::run_builtin_tests(&ctx, &self.config) {
                    Ok(new_outcome) => {
                        outcome = new_outcome;
                        if outcome.success {
                            break;
                        }
                    }
                    Err(err) => {
                        println!("[runmatfunc] test execution error after Codex fix: {err}");
                        break;
                    }
                }
            }
        }

        // Polish pass: ask Codex to re-check against BUILTIN_PACKAGING.md and fix any gaps, then test again
        if use_codex && codex_available && outcome.success {
            let polish = "Re-check the implementation against the Builtin Packaging template. Paste the entire template and ensure all required sections (DOC_MD, GPU/Fusion specs, tests, error semantics) are present and correct. Apply changes via apply_patch when necessary. Then stop.";
            let tail = Some("Template is included above in references: crates/runmat-runtime/BUILTIN_PACKAGING.md");
            let _ = crate::codex::session::run_authoring_with_extra(
                &ctx,
                resolved_model.clone(),
                polish,
                tail,
            );
            // Re-run tests
            match workspace_tests::run_builtin_tests(&ctx, &self.config) {
                Ok(new_outcome) => {
                    outcome = new_outcome;
                }
                Err(err) => println!("[runmatfunc] test execution error after polish: {err}"),
            }
        }

        // WGPU provider integration pass: ensure provider hooks and wiring are complete and tested
        if use_codex && codex_available && outcome.success {
            let wgpu_prompt = "Ensure the WGPU provider backend is fully implemented and wired for this builtin: (1) implement or extend provider hooks in crates/runmat-accelerate/src/backend/wgpu/provider_impl.rs matching GPU_SPEC provider_hooks; (2) add dispatch and shader wiring as needed under crates/runmat-accelerate/src/backend/wgpu/{dispatch,shaders,params,types}; (3) add #[cfg(feature=\"wgpu\")] tests in the builtin module verifying GPU parity (use provider registration helper); (4) if you add WGPU-only tests, set requires_feature: 'wgpu' in the DOC_MD frontmatter so runmatfunc can run with the correct Cargo features; (5) keep CPU semantics identical; (6) re-run tests until all pass. Apply changes via apply_patch.";
            let wgpu_tail = Some(
                "References: crates/runmat-accelerate/src/backend/wgpu/provider_impl.rs, .../dispatch, .../shaders, and existing GPU tests in builtins (e.g., math/reduction/{sum,mean}.rs and math/trigonometry/sin.rs). Match GPU_SPEC.provider_hooks for this builtin and ensure provider methods exist and are invoked.",
            );
            let _ = crate::codex::session::run_authoring_with_extra(
                &ctx,
                resolved_model.clone(),
                wgpu_prompt,
                wgpu_tail,
            );
            // Re-run tests to validate provider wiring (DOC_MD may set requires_feature: wgpu)
            match workspace_tests::run_builtin_tests(&ctx, &self.config) {
                Ok(new_outcome) => {
                    outcome = new_outcome;
                }
                Err(err) => println!("[runmatfunc] test execution error after WGPU pass: {err}"),
            }
        }

        // Completion pass: resolve any placeholders and fully implement argument handling
        if use_codex && codex_available && outcome.success {
            let complete_prompt = format!(
                "Within the builtin '{}', eliminate any remaining placeholders and complete the implementation: (1) search changed files and immediate dependencies for TODO, FIXME, unimplemented!, todo!(), panic(\\\"unimplemented\\\"), \\\"for later\\\", \\\"next revision\\\" and replace with working code or proper error paths that match MATLAB semantics; (2) ensure the full argument matrix supported by the spec is implemented (positional, keyword-like strings, size vectors, dtype/logical toggles, 'like' prototype, GPU residency cases); (3) add or extend unit + #[cfg(feature=\\\"wgpu\\\")] tests to cover all supported argument forms and edge cases (including error conditions); (4) ensure DOC_MD frontmatter 'tested' paths reflect unit/integration tests; (5) keep CPU/GPU parity. Apply changes via apply_patch, then stop.",
                ctx.builtin.name
            );
            let complete_tail = Some(
                "References: crates/runmat-runtime/Library.md (argument expectations) and crates/runmat-runtime/BUILTIN_PACKAGING.md. Use existing good builtins (sum, mean, sin, zeros/ones/rand) as templates for argument parsing patterns and tests.",
            );
            let _ = crate::codex::session::run_authoring_with_extra(
                &ctx,
                resolved_model.clone(),
                complete_prompt.as_str(),
                complete_tail,
            );
            // Re-run tests to validate completion of argument handling and removal of placeholders
            match workspace_tests::run_builtin_tests(&ctx, &self.config) {
                Ok(new_outcome) => {
                    outcome = new_outcome;
                }
                Err(err) => {
                    println!("[runmatfunc] test execution error after completion pass: {err}")
                }
            }
        }

        // Documentation quality pass: ask Codex to ensure DOC_MD quality matches existing good builtins
        if use_codex && codex_available && outcome.success {
            let doc_prompt = "Review and improve the DOC_MD for this builtin to match the tone, structure, and completeness of the best existing builtins under crates/runmat-runtime/src/builtins. Ensure: (1) headlines and sections match the style of the best existing builtins; (2) GPU Execution section reflects provider hooks and fusion residency; (3) realistic, runnable Examples; (4) See Also with correct names. Note that the headings have been selected carefully for SEO and communication purposes, so please do not change them. If anything is missing or inconsistent, update DOC_MD via apply_patch. Then stop.";
            let doc_tail = Some("For reference, browse good copies in crates/runmat-runtime/src/builtins (e.g., math/reduction/sum.rs, math/reduction/mean.rs, math/trigonometry/sin.rs, array/zeros.rs/ones.rs/rand.rs). Keep changes scoped to DOC_MD unless code/docs mismatch requires minor fixes.");
            let _ = crate::codex::session::run_authoring_with_extra(
                &ctx,
                resolved_model.clone(),
                doc_prompt,
                doc_tail,
            );
            // Re-run tests to ensure doc changes didn't regress behavior (some tests parse DOC_MD examples)
            match workspace_tests::run_builtin_tests(&ctx, &self.config) {
                Ok(new_outcome) => {
                    outcome = new_outcome;
                }
                Err(err) => println!("[runmatfunc] test execution error after doc pass: {err}"),
            }
        }

        // Report final outcome
        if outcome.success {
            println!(
                "[runmatfunc] tests passed for {} (logs: {})",
                ctx.builtin.name,
                outcome.log_dir.display()
            );
        } else {
            println!(
                "[runmatfunc] tests failed for {} (logs: {})",
                ctx.builtin.name,
                outcome.log_dir.display()
            );
        }

        for report in &outcome.reports {
            let status = if report.skipped {
                "skipped"
            } else if report.success {
                "ok"
            } else {
                "FAILED"
            };
            println!("  {} -> {}", report.label, status);
            if let Some(note) = &report.note {
                println!("    note: {}", note);
            }
            if let Some(path) = &report.stdout_path {
                println!("    stdout log: {}", path.display());
            } else if !report.stdout.trim().is_empty() {
                println!("    stdout:\n{}", indent(&report.stdout, 6));
            }
            if let Some(path) = &report.stderr_path {
                println!("    stderr log: {}", path.display());
            } else if !report.stderr.trim().is_empty() {
                println!("    stderr:\n{}", indent(&report.stderr, 6));
            }
        }
        Ok(())
    }

    pub fn queue_add(
        &mut self,
        builtin: &str,
        model: Option<String>,
        use_codex: bool,
        category: Option<String>,
    ) -> Result<()> {
        let entry = job_queue::QueueEntry::new(
            builtin.to_string(),
            category.clone(),
            model.clone(),
            use_codex,
        );
        let queue = job_queue::add_entry(&self.config, entry)?;
        let model_label = model
            .as_deref()
            .or_else(|| self.config.default_model.as_deref())
            .unwrap_or("default");
        println!(
            "[runmatfunc] queued '{}' (category: {}, model: {}, codex: {}) — total jobs: {}",
            builtin,
            category.as_deref().unwrap_or(""),
            model_label,
            use_codex,
            queue.len()
        );
        if use_codex && !crate::codex::client::is_available() {
            println!(
                "[runmatfunc] warning: Codex feature disabled in this build; job will run without Codex output"
            );
        }
        Ok(())
    }

    pub fn queue_list(&mut self) -> Result<()> {
        let queue = job_queue::load(&self.config)?;
        let queue_path = job_queue::queue_path(&self.config);
        if queue.is_empty() {
            println!(
                "[runmatfunc] queue is empty (file: {})",
                queue_path.display()
            );
            return Ok(());
        }

        println!(
            "[runmatfunc] Codex availability: {}",
            if crate::codex::client::is_available() {
                "available"
            } else {
                "not available (stub)"
            }
        );
        println!(
            "[runmatfunc] queued jobs ({} total) — {}",
            queue.len(),
            queue_path.display()
        );
        for (index, entry) in queue.iter().enumerate() {
            let model_label = entry
                .model
                .as_deref()
                .or_else(|| self.config.default_model.as_deref())
                .unwrap_or("default");
            println!(
                "  {:>3}. {:<20} | model: {:<12} | codex: {:<5} | enqueued: {}",
                index + 1,
                entry.builtin,
                model_label,
                entry.use_codex,
                entry.enqueued_at
            );
        }
        Ok(())
    }

    pub fn queue_run(&mut self, max: Option<usize>) -> Result<()> {
        let mut queue = job_queue::load(&self.config)?;
        if queue.is_empty() {
            println!("[runmatfunc] queue is empty; nothing to run");
            return Ok(());
        }

        let codex_available = crate::codex::client::is_available();
        println!(
            "[runmatfunc] Codex availability: {}",
            if codex_available {
                "available"
            } else {
                "not available (stub)"
            }
        );

        let outcomes = scheduler::run_queue(&self.config, &mut queue, max);
        job_queue::save(&self.config, &queue)?;

        for outcome in outcomes {
            if let Some(err) = outcome.error {
                println!(
                    "[runmatfunc] job '{}' failed: {}",
                    outcome.entry.builtin, err
                );
                continue;
            }
            let transcript_text = outcome
                .transcript_path
                .as_ref()
                .map(|p| p.display().to_string())
                .unwrap_or_else(|| "<not recorded>".to_string());
            if outcome.success {
                println!(
                    "[runmatfunc] job '{}' succeeded (transcript: {})",
                    outcome.entry.builtin, transcript_text
                );
            } else {
                println!(
                    "[runmatfunc] job '{}' completed but tests failed (transcript: {})",
                    outcome.entry.builtin, transcript_text
                );
            }
            if outcome.entry.use_codex {
                if let Some(summary) = &outcome.codex_summary {
                    println!("    Codex summary: {}", summary);
                } else if !codex_available {
                    println!("    Codex skipped (feature not enabled for this build).");
                } else {
                    println!("    Codex returned no response.");
                }
            }
        }

        println!("[runmatfunc] remaining queued jobs: {}", queue.len());
        Ok(())
    }

    pub fn queue_clear(&mut self) -> Result<()> {
        job_queue::clear(&self.config)?;
        println!(
            "[runmatfunc] cleared queue file at {}",
            job_queue::queue_path(&self.config).display()
        );
        Ok(())
    }
}

fn indent(text: &str, spaces: usize) -> String {
    let pad = " ".repeat(spaces);
    text.lines()
        .map(|line| format!("{}{}", pad, line))
        .collect::<Vec<_>>()
        .join("\n")
}
