#[derive(Default, Clone, Copy)]
pub struct LexerExtras {
    pub last_was_value: bool,
    pub line_start: bool,
}
