use runmat_parser::{parse_with_options as parse_impl, ParserOptions, Program, SyntaxError};

pub fn parse(input: &str) -> Result<Program, SyntaxError> {
    parse_impl(input, ParserOptions::default())
}
