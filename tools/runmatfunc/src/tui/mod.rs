//! Terminal UI scaffolding.

use anyhow::Result;
use crossterm::event::{self, Event, KeyCode};
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use crossterm::{execute, ExecutableCommand};
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Layout};
use ratatui::style::{Modifier, Style};
use ratatui::widgets::{Block, Borders, List, ListItem, ListState};
use ratatui::Terminal;

use crate::builtin::metadata::BuiltinManifest;

pub mod components;
pub mod input;
pub mod layout;

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
    let mut index = 0usize;
    let mut state = ListState::default();
    state.select(Some(0));

    loop {
        terminal.draw(|frame| {
            let chunks = Layout::default()
                .constraints([Constraint::Percentage(100)].as_ref())
                .split(frame.size());

            let items: Vec<ListItem> = manifest
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

            let list = List::new(items)
                .block(
                    Block::default()
                        .title("RunMat Builtins")
                        .borders(Borders::ALL),
                )
                .highlight_style(Style::default().add_modifier(Modifier::REVERSED));

            if manifest.builtins.is_empty() {
                state.select(None);
            } else {
                state.select(Some(index.min(manifest.builtins.len().saturating_sub(1))));
            }
            frame.render_stateful_widget(list, chunks[0], &mut state);
        })?;

        if event::poll(std::time::Duration::from_millis(250))? {
            match event::read()? {
                Event::Key(key) => match key.code {
                    KeyCode::Char('q') => break,
                    KeyCode::Down => {
                        index = (index + 1).min(manifest.builtins.len().saturating_sub(1));
                    }
                    KeyCode::Up => {
                        index = index.saturating_sub(1);
                    }
                    _ => {}
                },
                _ => {}
            }
        }
    }

    Ok(())
}
