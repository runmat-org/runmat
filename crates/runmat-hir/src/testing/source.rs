use crate::Span;
use runmat_test::descriptor::{SourceDescriptor, SourceSpan};

use super::SemanticTestSource;

pub(super) fn source_descriptor(
    source: &SemanticTestSource<'_>,
    semantic_path: impl Into<String>,
    span: Span,
) -> SourceDescriptor {
    let (start_line, start_column) = line_column(source.source_text, span.start);
    let (end_line, end_column) = line_column(source.source_text, span.end);
    SourceDescriptor {
        owner_identity: source.owner_identity.to_owned(),
        relative_path: source.relative_source_identity.replace('\\', "/"),
        semantic_path: semantic_path.into(),
        span: SourceSpan {
            start_byte: span.start.min(u32::MAX as usize) as u32,
            end_byte: span.end.min(u32::MAX as usize) as u32,
            start_line,
            start_column,
            end_line,
            end_column,
        },
    }
}

pub(super) fn source_stem(relative_path: &str) -> String {
    relative_path
        .replace('\\', "/")
        .rsplit('/')
        .next()
        .unwrap_or(relative_path)
        .strip_suffix(".m")
        .or_else(|| {
            relative_path
                .rsplit('/')
                .next()
                .and_then(|name| name.strip_suffix(".M"))
        })
        .unwrap_or_else(|| relative_path.rsplit('/').next().unwrap_or(relative_path))
        .to_owned()
}

pub(super) fn is_test_name(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    lower.starts_with("test") || lower.ends_with("test")
}

fn line_column(source: &str, byte: usize) -> (u32, u32) {
    let boundary = byte.min(source.len());
    let prefix = &source[..floor_char_boundary(source, boundary)];
    let line = prefix.bytes().filter(|byte| *byte == b'\n').count() as u32 + 1;
    let column = prefix
        .rsplit_once('\n')
        .map_or(prefix.len(), |(_, tail)| tail.len()) as u32
        + 1;
    (line, column)
}

fn floor_char_boundary(source: &str, mut index: usize) -> usize {
    while index > 0 && !source.is_char_boundary(index) {
        index -= 1;
    }
    index
}
