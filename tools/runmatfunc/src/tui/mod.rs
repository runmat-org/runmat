//! Terminal UI for browsing builtins and triggering helper actions.

use std::time::Duration;

use anyhow::Result;
use crossterm::event::{self, Event, KeyCode};
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use crossterm::{execute, ExecutableCommand};
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::style::{Color, Modifier, Style};
use ratatui::widgets::{Block, Borders, List, ListItem, ListState, Paragraph, Wrap};
use ratatui::Terminal;

use crate::app::config::AppConfig;
use crate::builtin::metadata::{BuiltinManifest, BuiltinRecord};
use crate::context::gather;
use crate::context::types::AuthoringContext;
use crate::context::{manifest as manifest_ctx, serialize};
use crate::workspace::{diff as workspace_diff, tests};

const HELP_TEXT: &str = "↑/↓ to navigate • t=tests • d=emit docs • f=diff • q=quit";

#[derive(Clone)]
struct StatusMessage {
    text: String,
    is_error: bool,
}

impl StatusMessage {
    fn info(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            is_error: false,
        }
    }

    fn error(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            is_error: true,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum DetailMode {
    Context,
    Diff,
}

struct TuiState {
    manifest: BuiltinManifest,
    selected: usize,
    detail: Option<AuthoringContext>,
    status: StatusMessage,
    config: AppConfig,
    detail_mode: DetailMode,
    diff_cache: Option<String>,
}

impl TuiState {
    fn new(manifest: BuiltinManifest, config: AppConfig) -> Result<Self> {
        let mut state = Self {
            manifest,
            selected: 0,
            detail: None,
            status: StatusMessage::info(HELP_TEXT),
            config,
            detail_mode: DetailMode::Context,
            diff_cache: None,
        };
        state.refresh_detail()?;
        Ok(state)
    }

    fn current_record(&self) -> Option<&BuiltinRecord> {
        self.manifest.builtins.get(self.selected)
    }

    fn refresh_detail(&mut self) -> Result<()> {
        if let Some(record) = self.current_record() {
            match gather::build_authoring_context(
                &record.name,
                record.category.as_deref(),
                &self.config,
            ) {
                Ok(ctx) => {
                    self.detail = Some(ctx);
                }
                Err(err) => {
                    self.detail = None;
                    self.status = StatusMessage::error(format!("Failed to load context: {err}"));
                }
            }
        }
        Ok(())
    }

    fn move_selection(&mut self, delta: isize) -> Result<()> {
        let len = self.manifest.builtins.len();
        if len == 0 {
            return Ok(());
        }
        let new_index = ((self.selected as isize) + delta).clamp(0, (len - 1) as isize) as usize;
        if new_index != self.selected {
            self.selected = new_index;
            self.detail_mode = DetailMode::Context;
            self.diff_cache = None;
            self.refresh_detail()?;
        }
        Ok(())
    }

    fn run_tests(&mut self) {
        if let Some(detail) = &self.detail {
            match tests::run_builtin_tests(detail, &self.config) {
                Ok(outcome) => {
                    if outcome.success {
                        self.status = StatusMessage::info(format!(
                            "Tests passed for {} (logs: {})",
                            detail.builtin.name,
                            outcome.log_dir.display()
                        ));
                    } else {
                        let summary: Vec<String> = outcome
                            .reports
                            .iter()
                            .map(|rep| {
                                let status = if rep.skipped {
                                    "skipped"
                                } else if rep.success {
                                    "ok"
                                } else {
                                    "failed"
                                };
                                format!("{}: {}", rep.label, status)
                            })
                            .collect();
                        self.status = StatusMessage::error(format!(
                            "Tests failed for {} -> {} (logs: {})",
                            detail.builtin.name,
                            summary.join(", "),
                            outcome.log_dir.display()
                        ));
                    }
                }
                Err(err) => {
                    self.status = StatusMessage::error(format!("Test run error: {err}"));
                }
            }
        }
    }

    fn emit_docs(&mut self) {
        let docs_dir = self.config.docs_output_dir().to_path_buf();
        match manifest_ctx::build_manifest()
            .and_then(|manifest| serialize::write_manifest_files(&manifest, docs_dir.as_path()))
        {
            Ok(_) => {
                self.status =
                    StatusMessage::info(format!("Docs written to {}", docs_dir.display()));
            }
            Err(err) => {
                self.status = StatusMessage::error(format!("Doc emission failed: {err}"));
            }
        }
    }

    fn toggle_diff(&mut self) {
        match self.detail_mode {
            DetailMode::Context => {
                self.refresh_diff();
                self.detail_mode = DetailMode::Diff;
            }
            DetailMode::Diff => {
                self.detail_mode = DetailMode::Context;
                self.status = StatusMessage::info(HELP_TEXT);
            }
        }
    }

    fn refresh_diff(&mut self) {
        if let Some(detail) = &self.detail {
            match workspace_diff::git_diff(&detail.source_paths) {
                Ok(Some(diff)) => {
                    self.diff_cache = Some(diff);
                    self.status =
                        StatusMessage::info("Showing git diff (press f to return to context view)");
                }
                Ok(None) => {
                    let message = "No workspace differences for selected sources.".to_string();
                    self.diff_cache = Some(message.clone());
                    self.status = StatusMessage::info(message);
                }
                Err(err) => {
                    let message = format!("Diff error: {err}");
                    self.diff_cache = Some(message.clone());
                    self.status = StatusMessage::error(message);
                }
            }
        } else {
            let message = "No builtin selected.".to_string();
            self.diff_cache = Some(message.clone());
            self.status = StatusMessage::info(message);
        }
    }
}

pub fn run(manifest: BuiltinManifest, config: AppConfig) -> Result<()> {
    enable_raw_mode()?;
    let mut stdout = std::io::stdout();
    stdout.execute(EnterAlternateScreen)?;

    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let result = tui_loop(&mut terminal, manifest, config);

    disable_raw_mode()?;
    let mut stdout = std::io::stdout();
    execute!(stdout, LeaveAlternateScreen)?;

    result
}

fn tui_loop(
    terminal: &mut Terminal<CrosstermBackend<std::io::Stdout>>,
    manifest: BuiltinManifest,
    config: AppConfig,
) -> Result<()> {
    let mut state = TuiState::new(manifest, config)?;
    let mut list_state = ListState::default();
    list_state.select(Some(state.selected));

    loop {
        terminal.draw(|frame| draw_ui(frame, &state, &mut list_state))?;

        if event::poll(Duration::from_millis(200))? {
            if let Event::Key(key) = event::read()? {
                match key.code {
                    KeyCode::Char('q') => break,
                    KeyCode::Down => {
                        state.move_selection(1)?;
                    }
                    KeyCode::Up => {
                        state.move_selection(-1)?;
                    }
                    KeyCode::Char('t') => {
                        state.run_tests();
                    }
                    KeyCode::Char('d') => {
                        state.emit_docs();
                    }
                    KeyCode::Char('f') => {
                        state.toggle_diff();
                    }
                    _ => {}
                }
            }
        }
    }

    Ok(())
}

fn draw_ui(frame: &mut ratatui::Frame<'_>, state: &TuiState, list_state: &mut ListState) {
    let outer = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(5), Constraint::Length(3)])
        .split(frame.size());

    let columns = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(35), Constraint::Percentage(65)])
        .split(outer[0]);

    // List of builtins on the left
    let items: Vec<ListItem> = state
        .manifest
        .builtins
        .iter()
        .map(|rec| {
            let label = format!(
                "{:<20} | {:<20}",
                rec.name,
                rec.category.as_deref().unwrap_or("")
            );
            ListItem::new(label)
        })
        .collect();

    list_state.select(Some(state.selected));

    let list = List::new(items)
        .block(Block::default().title("Builtins").borders(Borders::ALL))
        .highlight_style(Style::default().add_modifier(Modifier::REVERSED));

    frame.render_stateful_widget(list, columns[0], list_state);

    // Detail pane on the right
    let detail_text = match state.detail_mode {
        DetailMode::Context => state
            .detail
            .as_ref()
            .map(|ctx| render_context_detail(ctx, &state.config))
            .unwrap_or_else(|| "Loading…".to_string()),
        DetailMode::Diff => state
            .diff_cache
            .clone()
            .unwrap_or_else(|| "Press 'f' to load git diff for source hints.".to_string()),
    };
    let detail = Paragraph::new(detail_text)
        .block(Block::default().title("Details").borders(Borders::ALL))
        .wrap(Wrap { trim: true });
    frame.render_widget(detail, columns[1]);

    // Status/message footer
    let status_style = if state.status.is_error {
        Style::default().fg(Color::Red)
    } else {
        Style::default().fg(Color::Green)
    };
    let status = Paragraph::new(state.status.text.clone())
        .style(status_style)
        .block(Block::default().title("Status").borders(Borders::TOP));
    frame.render_widget(status, outer[1]);
}

fn render_context_detail(ctx: &AuthoringContext, config: &AppConfig) -> String {
    let mut out = String::new();
    if let Some(summary) = &ctx.builtin.summary {
        out.push_str(&format!("Summary: {}\n\n", summary));
    }
    out.push_str(&format!(
        "Codex: {}\n\n",
        if crate::codex::client::is_available() {
            format!(
                "available (default model: {})",
                config.default_model.as_deref().unwrap_or("default")
            )
        } else {
            "not available (enable embedded-codex feature)".to_string()
        }
    ));
    out.push_str(&format!("Prompt:\n{}\n", ctx.prompt));
    if !ctx.source_paths.is_empty() {
        out.push_str("\nSource Hints:\n");
        for path in &ctx.source_paths {
            out.push_str("  • ");
            out.push_str(&path.display().to_string());
            out.push('\n');
        }
    }
    out
}
