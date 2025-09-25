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

use crate::builtin::metadata::{BuiltinManifest, BuiltinRecord};
use crate::context::gather;
use crate::context::types::AuthoringContext;
use crate::context::{manifest as manifest_ctx, serialize};
use crate::workspace::tests;

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

struct TuiState {
    manifest: BuiltinManifest,
    selected: usize,
    detail: Option<AuthoringContext>,
    status: StatusMessage,
}

impl TuiState {
    fn new(manifest: BuiltinManifest) -> Result<Self> {
        let mut state = Self {
            manifest,
            selected: 0,
            detail: None,
            status: StatusMessage::info("↑/↓ to navigate • t=tests • d=emit docs • q=quit"),
        };
        state.refresh_detail()?;
        Ok(state)
    }

    fn current_record(&self) -> Option<&BuiltinRecord> {
        self.manifest.builtins.get(self.selected)
    }

    fn refresh_detail(&mut self) -> Result<()> {
        if let Some(record) = self.current_record() {
            match gather::build_authoring_context(&record.name) {
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
            self.refresh_detail()?;
        }
        Ok(())
    }

    fn run_tests(&mut self) {
        if let Some(detail) = &self.detail {
            match tests::run_builtin_tests(detail) {
                Ok(outcome) => {
                    if outcome.success {
                        self.status = StatusMessage::info(format!(
                            "Tests passed for {}",
                            detail.builtin.name
                        ));
                    } else {
                        let summary: Vec<String> = outcome
                            .reports
                            .iter()
                            .map(|rep| {
                                format!(
                                    "{}: {}",
                                    rep.label,
                                    if rep.success { "ok" } else { "failed" }
                                )
                            })
                            .collect();
                        self.status = StatusMessage::error(format!(
                            "Tests failed for {} -> {}",
                            detail.builtin.name,
                            summary.join(", ")
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
        match manifest_ctx::build_manifest().and_then(|manifest| {
            serialize::write_manifest_files(&manifest, std::path::Path::new("docs/generated"))
        }) {
            Ok(_) => {
                self.status = StatusMessage::info("Docs written to docs/generated");
            }
            Err(err) => {
                self.status = StatusMessage::error(format!("Doc emission failed: {err}"));
            }
        }
    }
}

pub fn run(manifest: BuiltinManifest) -> Result<()> {
    enable_raw_mode()?;
    let mut stdout = std::io::stdout();
    stdout.execute(EnterAlternateScreen)?;

    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let result = tui_loop(&mut terminal, manifest);

    disable_raw_mode()?;
    let mut stdout = std::io::stdout();
    execute!(stdout, LeaveAlternateScreen)?;

    result
}

fn tui_loop(
    terminal: &mut Terminal<CrosstermBackend<std::io::Stdout>>,
    manifest: BuiltinManifest,
) -> Result<()> {
    let mut state = TuiState::new(manifest)?;
    let mut list_state = ListState::default();
    list_state.select(Some(state.selected));

    loop {
        terminal.draw(|frame| draw_ui(frame, &state, &mut list_state))?;

        if event::poll(Duration::from_millis(200))? {
            match event::read()? {
                Event::Key(key) => match key.code {
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
                    _ => {}
                },
                _ => {}
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
    let detail_text = state
        .detail
        .as_ref()
        .map(render_detail_text)
        .unwrap_or_else(|| "Loading…".to_string());
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

fn render_detail_text(ctx: &AuthoringContext) -> String {
    let mut out = String::new();
    if let Some(summary) = &ctx.builtin.summary {
        out.push_str(&format!("Summary: {}\n\n", summary));
    }
    out.push_str(&format!("Prompt:\n{}\n", ctx.prompt));
    if let Some(doc) = &ctx.doc_markdown {
        out.push_str("\nDocumentation Excerpt:\n");
        out.push_str(&truncate(doc, 800));
        out.push('\n');
    }
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

fn truncate(text: &str, max: usize) -> String {
    if text.len() <= max {
        text.to_string()
    } else {
        format!("{}…", &text[..max])
    }
}
