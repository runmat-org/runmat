use crate::{
    BuiltinDesktopHostFixture, BuiltinDesktopHostInteraction, BuiltinDirectoryDialogOutcome,
    BuiltinFigurePresentationEvent, BuiltinKeyInputOutcome, BuiltinLineInputOutcome,
    BuiltinOpenFileDialogOutcome, BuiltinSaveFileDialogOutcome,
};
use std::collections::BTreeSet;

use super::{push, resource, BuiltinCatalogValidationError};

const MAX_INTERACTIONS: usize = 128;
const MAX_TEXT_BYTES: usize = 64 * 1024;
const MAX_FILTERS: usize = 64;
const MAX_FILTER_PATTERNS: usize = 64;

pub(super) fn validate(
    builtin: &'static str,
    fixture: BuiltinDesktopHostFixture,
    errors: &mut Vec<BuiltinCatalogValidationError>,
) {
    resource::validate_fixture_id(builtin, fixture.id, errors);
    resource::validate_filesystem_entries(builtin, fixture.entries, false, errors);
    let workspace = DesktopWorkspacePaths::from_entries(fixture.entries);
    if fixture.interactions.is_empty() || fixture.interactions.len() > MAX_INTERACTIONS {
        push(
            errors,
            builtin,
            "desktop fixture must contain between 1 and 128 interactions",
        );
    }
    for interaction in fixture.interactions {
        validate_interaction(builtin, interaction, &workspace, errors);
    }
}

fn validate_interaction(
    builtin: &'static str,
    interaction: &BuiltinDesktopHostInteraction,
    workspace: &DesktopWorkspacePaths<'_>,
    errors: &mut Vec<BuiltinCatalogValidationError>,
) {
    match interaction {
        BuiltinDesktopHostInteraction::OpenFileDialog(value) => {
            validate_text(builtin, value.request.title, "desktop dialog title", errors);
            validate_optional_path(builtin, value.request.default_path, errors);
            validate_filters(builtin, value.request.filters, errors);
            match value.outcome {
                BuiltinOpenFileDialogOutcome::Selection {
                    paths,
                    filter_index,
                } => {
                    if paths.is_empty() || paths.len() > 64 {
                        push(
                            errors,
                            builtin,
                            "desktop open-file selection must contain between 1 and 64 paths",
                        );
                    }
                    if !value.request.multiselect && paths.len() != 1 {
                        push(
                            errors,
                            builtin,
                            "desktop single-file dialog must select exactly one path",
                        );
                    }
                    for path in paths {
                        validate_path(builtin, path, errors);
                        if !workspace.files.contains(path) {
                            push(
                                errors,
                                builtin,
                                "desktop open-file selection must name a declared file",
                            );
                        }
                    }
                    validate_filter_index(
                        builtin,
                        filter_index,
                        value.request.filters.len(),
                        errors,
                    );
                }
                BuiltinOpenFileDialogOutcome::Cancel => {}
                BuiltinOpenFileDialogOutcome::Error(message) => {
                    validate_required_text(builtin, message, "desktop dialog error", errors);
                }
            }
        }
        BuiltinDesktopHostInteraction::SaveFileDialog(value) => {
            validate_text(builtin, value.request.title, "desktop dialog title", errors);
            validate_optional_path(builtin, value.request.default_path, errors);
            validate_filters(builtin, value.request.filters, errors);
            match value.outcome {
                BuiltinSaveFileDialogOutcome::Selection { path, filter_index } => {
                    validate_path(builtin, path, errors);
                    if !workspace.directories.contains(parent_path(path)) {
                        push(
                            errors,
                            builtin,
                            "desktop save-file selection parent must exist in the fixture workspace",
                        );
                    }
                    validate_filter_index(
                        builtin,
                        filter_index,
                        value.request.filters.len(),
                        errors,
                    );
                }
                BuiltinSaveFileDialogOutcome::Cancel => {}
                BuiltinSaveFileDialogOutcome::Error(message) => {
                    validate_required_text(builtin, message, "desktop dialog error", errors);
                }
            }
        }
        BuiltinDesktopHostInteraction::DirectoryDialog(value) => {
            validate_text(builtin, value.request.title, "desktop dialog title", errors);
            validate_optional_path(builtin, value.request.default_path, errors);
            match value.outcome {
                BuiltinDirectoryDialogOutcome::Selection { path } => {
                    validate_path(builtin, path, errors);
                    if !workspace.directories.contains(path) {
                        push(
                            errors,
                            builtin,
                            "desktop directory selection must name a materialized fixture directory",
                        );
                    }
                }
                BuiltinDirectoryDialogOutcome::Cancel => {}
                BuiltinDirectoryDialogOutcome::Error(message) => {
                    validate_required_text(builtin, message, "desktop dialog error", errors);
                }
            }
        }
        BuiltinDesktopHostInteraction::LineInput(value) => {
            validate_bounded_text(
                builtin,
                value.prompt,
                "desktop line-input prompt",
                true,
                errors,
            );
            match value.outcome {
                BuiltinLineInputOutcome::Line(line) => {
                    validate_bounded_text(builtin, line, "desktop line-input result", true, errors)
                }
                BuiltinLineInputOutcome::Error(message) => {
                    validate_required_text(builtin, message, "desktop input error", errors);
                }
            }
        }
        BuiltinDesktopHostInteraction::KeyInput(value) => {
            validate_bounded_text(
                builtin,
                value.prompt,
                "desktop key-input prompt",
                true,
                errors,
            );
            if let BuiltinKeyInputOutcome::Error(message) = value.outcome {
                validate_required_text(builtin, message, "desktop input error", errors);
            }
        }
        BuiltinDesktopHostInteraction::FigurePresentation(value) => {
            if value.figure_ordinal == 0 {
                push(errors, builtin, "desktop figure ordinal must be one-based");
            }
            if let Some(snapshot) = value.snapshot {
                validate_text(builtin, snapshot.title, "desktop figure title", errors);
            }
            if matches!(value.event, BuiltinFigurePresentationEvent::Created)
                && value.snapshot.is_none()
            {
                push(
                    errors,
                    builtin,
                    "desktop created-figure interaction requires a snapshot",
                );
            }
        }
    }
}

struct DesktopWorkspacePaths<'a> {
    files: BTreeSet<&'a str>,
    directories: BTreeSet<&'a str>,
}

impl<'a> DesktopWorkspacePaths<'a> {
    fn from_entries(entries: &'a [crate::BuiltinFilesystemEntry]) -> Self {
        let mut files = BTreeSet::new();
        let mut directories = BTreeSet::from([""]);
        for entry in entries {
            let (path, is_file) = match entry {
                crate::BuiltinFilesystemEntry::Directory { relative_path } => {
                    (*relative_path, false)
                }
                crate::BuiltinFilesystemEntry::File { relative_path, .. } => (*relative_path, true),
            };
            if is_file {
                files.insert(path);
            } else {
                directories.insert(path);
            }
            let mut parent = parent_path(path);
            while !parent.is_empty() {
                directories.insert(parent);
                parent = parent_path(parent);
            }
        }
        Self { files, directories }
    }
}

fn parent_path(path: &str) -> &str {
    path.rsplit_once('/').map_or("", |(parent, _)| parent)
}

fn validate_filters(
    builtin: &'static str,
    filters: &[crate::BuiltinDialogFilterExpectation],
    errors: &mut Vec<BuiltinCatalogValidationError>,
) {
    if filters.len() > MAX_FILTERS {
        push(errors, builtin, "desktop dialog has too many filters");
    }
    for filter in filters {
        if filter.patterns.is_empty() || filter.patterns.len() > MAX_FILTER_PATTERNS {
            push(
                errors,
                builtin,
                "desktop dialog filter must contain between 1 and 64 patterns",
            );
        }
        for pattern in filter.patterns {
            validate_required_text(builtin, pattern, "desktop dialog filter pattern", errors);
        }
        validate_text(
            builtin,
            filter.description,
            "desktop dialog filter description",
            errors,
        );
    }
}

fn validate_filter_index(
    builtin: &'static str,
    index: Option<usize>,
    filter_count: usize,
    errors: &mut Vec<BuiltinCatalogValidationError>,
) {
    if index.is_some_and(|value| value >= filter_count) {
        push(
            errors,
            builtin,
            "desktop dialog filter index is outside the declared filters",
        );
    }
}

fn validate_optional_path(
    builtin: &'static str,
    path: Option<&str>,
    errors: &mut Vec<BuiltinCatalogValidationError>,
) {
    if let Some(path) = path {
        validate_path(builtin, path, errors);
    }
}

fn validate_path(
    builtin: &'static str,
    path: &str,
    errors: &mut Vec<BuiltinCatalogValidationError>,
) {
    if !resource::valid_relative_path(path) {
        push(
            errors,
            builtin,
            "desktop fixture paths must be normalized relative paths",
        );
    }
}

fn validate_text(
    builtin: &'static str,
    value: Option<&str>,
    label: &'static str,
    errors: &mut Vec<BuiltinCatalogValidationError>,
) {
    if let Some(value) = value {
        validate_bounded_text(builtin, value, label, true, errors);
    }
}

fn validate_required_text(
    builtin: &'static str,
    value: &str,
    label: &'static str,
    errors: &mut Vec<BuiltinCatalogValidationError>,
) {
    validate_bounded_text(builtin, value, label, false, errors);
}

fn validate_bounded_text(
    builtin: &'static str,
    value: &str,
    label: &'static str,
    allow_empty: bool,
    errors: &mut Vec<BuiltinCatalogValidationError>,
) {
    if (!allow_empty && value.is_empty()) || value.len() > MAX_TEXT_BYTES || value.contains('\0') {
        push(errors, builtin, label);
    }
}
