use crate::cli::{BatchCommand, Cli, ColorMode, Commands, ConfigCommand};
use owo_colors::{OwoColorize, Style};
use std::ffi::OsString;
use std::fmt::Display;
use std::io::IsTerminal;
use std::sync::OnceLock;
use supports_color::Stream;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ColorLevel {
    None,
    Basic,
    Ansi256,
    TrueColor,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OutputMode {
    Human,
    Machine,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Tone {
    Brand,
    Heading,
    Label,
    Value,
    Path,
    Identifier,
    Error,
    Warning,
    Success,
    Info,
    Help,
    Muted,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ColorEnvironment {
    pub no_color: Option<String>,
    pub clicolor: Option<String>,
    pub clicolor_force: Option<String>,
    pub force_color: Option<String>,
    pub term: Option<String>,
    pub colorterm: Option<String>,
}

impl ColorEnvironment {
    pub fn capture() -> Self {
        Self {
            no_color: environment_value("NO_COLOR"),
            clicolor: environment_value("CLICOLOR"),
            clicolor_force: environment_value("CLICOLOR_FORCE"),
            force_color: environment_value("FORCE_COLOR"),
            term: environment_value("TERM"),
            colorterm: environment_value("COLORTERM"),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct StreamStyles {
    level: ColorLevel,
}

impl StreamStyles {
    pub const fn new(level: ColorLevel) -> Self {
        Self { level }
    }

    pub const fn plain() -> Self {
        Self::new(ColorLevel::None)
    }

    pub const fn level(self) -> ColorLevel {
        self.level
    }

    pub const fn enabled(self) -> bool {
        !matches!(self.level, ColorLevel::None)
    }

    pub fn paint(&self, tone: Tone, value: impl Display) -> String {
        if !self.enabled() {
            return value.to_string();
        }
        format!("{}", value.style(self.tone_style(tone)))
    }

    pub fn brand(&self, value: impl Display) -> String {
        self.paint(Tone::Brand, value)
    }

    pub fn heading(&self, value: impl Display) -> String {
        self.paint(Tone::Heading, value)
    }

    pub fn label(&self, value: impl Display) -> String {
        self.paint(Tone::Label, value)
    }

    pub fn value(&self, value: impl Display) -> String {
        self.paint(Tone::Value, value)
    }

    pub fn path(&self, value: impl Display) -> String {
        self.paint(Tone::Path, value)
    }

    pub fn identifier(&self, value: impl Display) -> String {
        self.paint(Tone::Identifier, value)
    }

    pub fn error(&self, value: impl Display) -> String {
        self.paint(Tone::Error, value)
    }

    pub fn warning(&self, value: impl Display) -> String {
        self.paint(Tone::Warning, value)
    }

    pub fn success(&self, value: impl Display) -> String {
        self.paint(Tone::Success, value)
    }

    pub fn info(&self, value: impl Display) -> String {
        self.paint(Tone::Info, value)
    }

    pub fn help(&self, value: impl Display) -> String {
        self.paint(Tone::Help, value)
    }

    pub fn muted(&self, value: impl Display) -> String {
        self.paint(Tone::Muted, value)
    }

    fn tone_style(self, tone: Tone) -> Style {
        match tone {
            Tone::Brand if self.level == ColorLevel::TrueColor => {
                Style::new().truecolor(194, 108, 255).bold()
            }
            Tone::Brand => Style::new().bright_magenta().bold(),
            Tone::Heading => Style::new().bold(),
            Tone::Label if self.level == ColorLevel::TrueColor => {
                Style::new().truecolor(79, 140, 255).bold()
            }
            Tone::Label => Style::new().bright_blue().bold(),
            Tone::Value => Style::new().bold(),
            Tone::Path => Style::new().cyan(),
            Tone::Identifier => Style::new().bright_blue(),
            Tone::Error => Style::new().red().bold(),
            Tone::Warning => Style::new().yellow().bold(),
            Tone::Success => Style::new().green().bold(),
            Tone::Info => Style::new().cyan(),
            Tone::Help => Style::new().green(),
            Tone::Muted => Style::new().dimmed(),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Presentation {
    stdout: StreamStyles,
    stderr: StreamStyles,
}

impl Presentation {
    pub fn detect(mode: ColorMode) -> Self {
        let environment = ColorEnvironment::capture();
        Self {
            stdout: StreamStyles::new(resolve_color_level(
                mode,
                &environment,
                std::io::stdout().is_terminal(),
                detected_level(Stream::Stdout),
            )),
            stderr: StreamStyles::new(resolve_color_level(
                mode,
                &environment,
                std::io::stderr().is_terminal(),
                detected_level(Stream::Stderr),
            )),
        }
    }

    pub fn for_output(mode: ColorMode, output_mode: OutputMode) -> Self {
        match output_mode {
            OutputMode::Human => Self::detect(mode),
            OutputMode::Machine => Self::plain(),
        }
    }

    pub const fn plain() -> Self {
        Self {
            stdout: StreamStyles::plain(),
            stderr: StreamStyles::plain(),
        }
    }

    pub const fn stdout(&self) -> StreamStyles {
        self.stdout
    }

    pub const fn stderr(&self) -> StreamStyles {
        self.stderr
    }
}

static PRESENTATION: OnceLock<Presentation> = OnceLock::new();
static PLAIN_PRESENTATION: Presentation = Presentation::plain();

pub fn initialize(mode: ColorMode, output_mode: OutputMode) -> &'static Presentation {
    PRESENTATION.get_or_init(|| Presentation::for_output(mode, output_mode))
}

pub fn current() -> &'static Presentation {
    PRESENTATION.get().unwrap_or(&PLAIN_PRESENTATION)
}

pub fn stdout() -> StreamStyles {
    current().stdout()
}

pub fn stderr() -> StreamStyles {
    current().stderr()
}

pub fn cli_output_mode(cli: &Cli) -> OutputMode {
    if cli.generate_config || cli.emit_bytecode.is_some() {
        return OutputMode::Machine;
    }
    match cli.command.as_ref() {
        Some(Commands::Check { json: true, .. })
        | Some(Commands::Run { json: true, .. })
        | Some(Commands::AccelInfo { json: true, .. })
        | Some(Commands::Batch {
            batch_command:
                BatchCommand::Submit { json: true, .. }
                | BatchCommand::List { json: true }
                | BatchCommand::Show { json: true, .. }
                | BatchCommand::Cancel { json: true, .. },
        }) => OutputMode::Machine,
        #[cfg(feature = "wgpu")]
        Some(Commands::AccelCalibrate { json: true, .. }) => OutputMode::Machine,
        Some(Commands::Config {
            config_command: ConfigCommand::Show { .. },
        }) => OutputMode::Machine,
        Some(Commands::Cluster { cluster_command }) if cluster_command.machine_output() => {
            OutputMode::Machine
        }
        Some(Commands::Job { job_command }) if job_command.machine_output() => OutputMode::Machine,
        Some(Commands::Package { package_command }) if package_command.machine_output() => {
            OutputMode::Machine
        }
        _ => OutputMode::Human,
    }
}

pub fn requested_color_mode(args: &[OsString]) -> Option<ColorMode> {
    let mut index = 1;
    let mut selected = None;
    while index < args.len() {
        let arg = args[index].to_string_lossy();
        if arg == "--" {
            break;
        } else if let Some(value) = arg.strip_prefix("--color=") {
            selected = parse_color_mode(value);
        } else if arg == "--color" {
            if let Some(value) = args.get(index + 1) {
                selected = parse_color_mode(&value.to_string_lossy());
                index += 1;
            }
        }
        index += 1;
    }
    selected
}

pub fn clap_color_choice(mode: ColorMode) -> clap::ColorChoice {
    match mode {
        ColorMode::Always => clap::ColorChoice::Always,
        ColorMode::Never => clap::ColorChoice::Never,
        ColorMode::Auto => {
            let environment = ColorEnvironment::capture();
            if no_color_requested(&environment) {
                clap::ColorChoice::Never
            } else if forced_level(&environment).is_some() {
                clap::ColorChoice::Always
            } else if automatic_color_is_disabled(&environment) {
                clap::ColorChoice::Never
            } else {
                clap::ColorChoice::Auto
            }
        }
    }
}

pub fn clap_styles() -> clap::builder::styling::Styles {
    use clap::builder::styling::{AnsiColor, Effects, Styles};

    Styles::styled()
        .header(AnsiColor::BrightBlue.on_default() | Effects::BOLD)
        .usage(AnsiColor::BrightBlue.on_default() | Effects::BOLD)
        .literal(AnsiColor::Cyan.on_default())
        .placeholder(AnsiColor::BrightCyan.on_default())
        .error(AnsiColor::Red.on_default() | Effects::BOLD)
        .valid(AnsiColor::Green.on_default())
        .invalid(AnsiColor::Yellow.on_default())
}

pub fn resolve_color_level(
    mode: ColorMode,
    environment: &ColorEnvironment,
    is_terminal: bool,
    detected: Option<ColorLevel>,
) -> ColorLevel {
    match mode {
        ColorMode::Never => ColorLevel::None,
        ColorMode::Always => forced_level(environment)
            .or(detected)
            .unwrap_or_else(|| inferred_level(environment)),
        ColorMode::Auto => {
            if no_color_requested(environment) {
                return ColorLevel::None;
            }
            if let Some(level) = forced_level(environment) {
                return level;
            }
            if automatic_color_is_disabled(environment) {
                return ColorLevel::None;
            }
            if !is_terminal {
                return ColorLevel::None;
            }
            detected.unwrap_or_else(|| inferred_level(environment))
        }
    }
}

fn parse_color_mode(value: &str) -> Option<ColorMode> {
    match value {
        "auto" => Some(ColorMode::Auto),
        "always" => Some(ColorMode::Always),
        "never" => Some(ColorMode::Never),
        _ => None,
    }
}

fn automatic_color_is_disabled(environment: &ColorEnvironment) -> bool {
    environment.clicolor.as_deref() == Some("0") || environment.term.as_deref() == Some("dumb")
}

fn no_color_requested(environment: &ColorEnvironment) -> bool {
    is_nonempty(environment.no_color.as_deref())
}

fn forced_level(environment: &ColorEnvironment) -> Option<ColorLevel> {
    if let Some(value) = environment.force_color.as_deref() {
        if !value.is_empty() && value != "0" && value != "false" {
            return Some(match value {
                "2" => ColorLevel::Ansi256,
                "3" => ColorLevel::TrueColor,
                _ => ColorLevel::Basic,
            });
        }
    }
    environment
        .clicolor_force
        .as_deref()
        .filter(|value| !value.is_empty() && *value != "0")
        .map(|_| ColorLevel::Basic)
}

fn inferred_level(environment: &ColorEnvironment) -> ColorLevel {
    if environment
        .force_color
        .as_deref()
        .is_some_and(|value| value == "3")
        || environment
            .term
            .as_deref()
            .is_some_and(|term| term.ends_with("direct") || term.ends_with("truecolor"))
        || environment
            .colorterm
            .as_deref()
            .is_some_and(|value| value == "truecolor" || value == "24bit")
    {
        ColorLevel::TrueColor
    } else if environment
        .force_color
        .as_deref()
        .is_some_and(|value| value == "2")
        || environment
            .term
            .as_deref()
            .is_some_and(|term| term.ends_with("256") || term.ends_with("256color"))
    {
        ColorLevel::Ansi256
    } else {
        ColorLevel::Basic
    }
}

fn detected_level(stream: Stream) -> Option<ColorLevel> {
    supports_color::on(stream).map(|level| {
        if level.has_16m {
            ColorLevel::TrueColor
        } else if level.has_256 {
            ColorLevel::Ansi256
        } else {
            ColorLevel::Basic
        }
    })
}

fn is_nonempty(value: Option<&str>) -> bool {
    value.is_some_and(|value| !value.is_empty())
}

fn environment_value(name: &str) -> Option<String> {
    std::env::var_os(name).map(|value| value.to_string_lossy().into_owned())
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    fn environment() -> ColorEnvironment {
        ColorEnvironment::default()
    }

    #[test]
    fn auto_requires_a_terminal() {
        assert_eq!(
            resolve_color_level(ColorMode::Auto, &environment(), false, None),
            ColorLevel::None
        );
        assert_eq!(
            resolve_color_level(
                ColorMode::Auto,
                &environment(),
                true,
                Some(ColorLevel::Ansi256)
            ),
            ColorLevel::Ansi256
        );
    }

    #[test]
    fn no_color_uses_nonempty_presence_semantics() {
        let mut env = environment();
        env.no_color = Some("0".to_string());
        assert_eq!(
            resolve_color_level(ColorMode::Auto, &env, true, Some(ColorLevel::TrueColor)),
            ColorLevel::None
        );

        env.no_color = Some(String::new());
        assert_eq!(
            resolve_color_level(ColorMode::Auto, &env, true, Some(ColorLevel::TrueColor)),
            ColorLevel::TrueColor
        );
    }

    #[test]
    fn explicit_choice_overrides_environment() {
        let mut env = environment();
        env.no_color = Some("1".to_string());
        assert_eq!(
            resolve_color_level(ColorMode::Always, &env, false, None),
            ColorLevel::Basic
        );
        env.force_color = Some("3".to_string());
        assert_eq!(
            resolve_color_level(ColorMode::Never, &env, true, None),
            ColorLevel::None
        );
    }

    #[test]
    fn no_color_wins_over_environment_force_in_auto() {
        let mut env = environment();
        env.no_color = Some("1".to_string());
        env.force_color = Some("3".to_string());
        env.clicolor_force = Some("1".to_string());
        assert_eq!(
            resolve_color_level(ColorMode::Auto, &env, true, None),
            ColorLevel::None
        );
    }

    #[test]
    fn force_variables_work_without_a_terminal() {
        let mut env = environment();
        env.force_color = Some("2".to_string());
        env.term = Some("dumb".to_string());
        env.clicolor = Some("0".to_string());
        assert_eq!(
            resolve_color_level(ColorMode::Auto, &env, false, None),
            ColorLevel::Ansi256
        );
        env.force_color = None;
        env.term = None;
        env.clicolor = None;
        env.clicolor_force = Some("1".to_string());
        assert_eq!(
            resolve_color_level(ColorMode::Auto, &env, false, None),
            ColorLevel::Basic
        );

        env.clicolor_force = Some(String::new());
        assert_eq!(
            resolve_color_level(ColorMode::Auto, &env, false, None),
            ColorLevel::None
        );
    }

    #[test]
    fn dumb_term_and_clicolor_zero_disable_auto() {
        let mut env = environment();
        env.term = Some("dumb".to_string());
        assert_eq!(
            resolve_color_level(ColorMode::Auto, &env, true, None),
            ColorLevel::None
        );
        env.term = None;
        env.clicolor = Some("0".to_string());
        assert_eq!(
            resolve_color_level(ColorMode::Auto, &env, true, None),
            ColorLevel::None
        );
    }

    #[test]
    fn semantic_styles_have_plain_equivalents() {
        let plain = StreamStyles::plain();
        let styled = StreamStyles::new(ColorLevel::Basic);
        assert_eq!(plain.error("error"), "error");
        assert!(styled.error("error").contains("\u{1b}["));
        assert_eq!(strip_ansi(&styled.error("error")), "error");
    }

    #[test]
    fn finds_global_color_argument_anywhere() {
        let args = ["runmat", "check", "main.m", "--color", "never"]
            .into_iter()
            .map(OsString::from)
            .collect::<Vec<_>>();
        assert_eq!(requested_color_mode(&args), Some(ColorMode::Never));

        let args = ["runmat", "--color=always", "info"]
            .into_iter()
            .map(OsString::from)
            .collect::<Vec<_>>();
        assert_eq!(requested_color_mode(&args), Some(ColorMode::Always));

        let args = ["runmat", "run", "main.m", "--", "--color=always"]
            .into_iter()
            .map(OsString::from)
            .collect::<Vec<_>>();
        assert_eq!(requested_color_mode(&args), None);
    }

    #[test]
    fn classifies_explicit_machine_output_modes() {
        let check = Cli::try_parse_from(["runmat", "check", "main.m", "--json"]).unwrap();
        assert_eq!(cli_output_mode(&check), OutputMode::Machine);

        let config = Cli::try_parse_from(["runmat", "config", "show", "--format", "json"]).unwrap();
        assert_eq!(cli_output_mode(&config), OutputMode::Machine);

        let cluster = Cli::try_parse_from(["runmat", "cluster", "list", "--json"]).unwrap();
        assert_eq!(cli_output_mode(&cluster), OutputMode::Machine);

        let recovery = Cli::try_parse_from([
            "runmat",
            "job",
            "recovery",
            "keygen",
            "--output",
            "recovery-key.json",
            "--json",
        ])
        .unwrap();
        assert_eq!(cli_output_mode(&recovery), OutputMode::Machine);

        let package = Cli::try_parse_from(["runmat", "package", "inspect", "--json"]).unwrap();
        assert_eq!(cli_output_mode(&package), OutputMode::Machine);

        let info = Cli::try_parse_from(["runmat", "info"]).unwrap();
        assert_eq!(cli_output_mode(&info), OutputMode::Human);

        let human_recovery = Cli::try_parse_from([
            "runmat",
            "job",
            "recovery",
            "keygen",
            "--output",
            "recovery-key.json",
        ])
        .unwrap();
        assert_eq!(cli_output_mode(&human_recovery), OutputMode::Human);
    }

    fn strip_ansi(value: &str) -> String {
        let mut output = String::new();
        let mut chars = value.chars().peekable();
        while let Some(ch) = chars.next() {
            if ch == '\u{1b}' && chars.peek() == Some(&'[') {
                chars.next();
                for next in chars.by_ref() {
                    if next.is_ascii_alphabetic() {
                        break;
                    }
                }
            } else {
                output.push(ch);
            }
        }
        output
    }
}
