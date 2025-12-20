use crate::ReplEngine;
/// Shell-like commands for the REPL (pwd, cd, dir/ls, clear, who/whos, help)
/// These complement the MATLAB expression evaluation engine.
use std::env;
use std::path::{Path, PathBuf};

/// Result of command execution
pub enum CommandResult {
    /// Command was handled, output produced
    Handled(String),
    /// Not a recognized command, let the engine handle it
    NotCommand,
    /// Special command: clear all variables
    Clear,
    /// Special command: exit REPL
    Exit,
}

/// Parse and execute a shell-like command
/// Returns CommandResult if it's a recognized command, or NotCommand if it should be evaluated as MATLAB code
pub fn parse_and_execute(input: &str, engine: &mut ReplEngine) -> CommandResult {
    let trimmed = input.trim();
    let parts: Vec<&str> = trimmed.split_whitespace().collect();

    if parts.is_empty() {
        return CommandResult::NotCommand;
    }

    match parts[0] {
        "pwd" => cmd_pwd(),
        "cd" => cmd_cd(if parts.len() > 1 { parts[1] } else { "." }),
        "dir" | "ls" => cmd_dir_ls(if parts.len() > 1 { parts[1] } else { "." }),
        "clear" | "clearvars" => {
            if parts.len() == 1 {
                // clear all variables
                CommandResult::Clear
            } else {
                // clear specific variables (not implemented yet)
                CommandResult::Handled(
                    "Clearing specific variables not yet implemented. Use 'clear' to clear all."
                        .to_string(),
                )
            }
        }
        "who" => cmd_who(engine),
        "whos" => cmd_whos(engine),
        _ => CommandResult::NotCommand,
    }
}

/// `pwd` — print working directory
fn cmd_pwd() -> CommandResult {
    match env::current_dir() {
        Ok(path) => CommandResult::Handled(format!("{}", path.display())),
        Err(e) => CommandResult::Handled(format!("Error: {e}")),
    }
}

/// `cd <path>` — change directory
fn cmd_cd(path_str: &str) -> CommandResult {
    let path = if path_str == "~" {
        match env::var("HOME").or_else(|_| env::var("USERPROFILE")) {
            Ok(home) => PathBuf::from(home),
            Err(_) => return CommandResult::Handled("Error: HOME not set".to_string()),
        }
    } else {
        PathBuf::from(path_str)
    };

    match env::set_current_dir(&path) {
        Ok(_) => {
            // Print the new directory for feedback
            match env::current_dir() {
                Ok(cwd) => CommandResult::Handled(format!("{}", cwd.display())),
                Err(e) => CommandResult::Handled(format!("Error: {e}")),
            }
        }
        Err(e) => CommandResult::Handled(format!("Error changing directory: {e}")),
    }
}

/// `dir` / `ls <path>` — list directory contents
fn cmd_dir_ls(path_str: &str) -> CommandResult {
    let path = Path::new(path_str);

    match std::fs::read_dir(path) {
        Ok(entries) => {
            let mut files: Vec<String> = entries
                .filter_map(|entry| {
                    entry.ok().and_then(|e| {
                        e.file_name().into_string().ok().or_else(|| {
                            // Try to extract Unicode representation if possible
                            e.file_name().to_str().map(|s| s.to_string())
                        })
                    })
                })
                .collect();

            if files.is_empty() {
                return CommandResult::Handled("(empty directory)".to_string());
            }

            files.sort();
            CommandResult::Handled(files.join("\n"))
        }
        Err(e) => CommandResult::Handled(format!("Error listing directory: {e}")),
    }
}

/// `who` — list variable names (simple format)
fn cmd_who(engine: &ReplEngine) -> CommandResult {
    let vars = engine.get_variables();
    if vars.is_empty() {
        CommandResult::Handled("(no variables defined)".to_string())
    } else {
        let names: Vec<&String> = vars.keys().collect();
        CommandResult::Handled(
            names
                .iter()
                .map(|s| s.as_str())
                .collect::<Vec<_>>()
                .join("  "),
        )
    }
}

/// `whos` — list variables with details (detailed format)
fn cmd_whos(engine: &ReplEngine) -> CommandResult {
    let vars = engine.get_variables();
    if vars.is_empty() {
        return CommandResult::Handled("(no variables defined)".to_string());
    }

    let mut output = String::from("Name            Size          Type\n");
    output.push_str("================================================\n");

    for name in vars.keys() {
        // Format: name, size, type (simplified)
        let type_str = "value";
        output.push_str(&format!("{:<15} {:<13} {}\n", name, "1x1", type_str));
    }

    CommandResult::Handled(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pwd_returns_string() {
        match cmd_pwd() {
            CommandResult::Handled(s) => {
                assert!(!s.is_empty());
                assert!(!s.contains("Error"));
            }
            _ => panic!("pwd should return Handled"),
        }
    }

    #[test]
    fn test_cd_to_current() {
        match cmd_cd(".") {
            CommandResult::Handled(s) => {
                assert!(!s.is_empty());
            }
            _ => panic!("cd . should return Handled"),
        }
    }

    #[test]
    fn test_dir_lists_files() {
        match cmd_dir_ls(".") {
            CommandResult::Handled(s) => {
                // Current directory should have content
                assert!(!s.is_empty());
            }
            _ => panic!("dir . should return Handled"),
        }
    }
}
