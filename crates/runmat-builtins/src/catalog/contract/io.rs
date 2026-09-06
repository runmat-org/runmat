use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum IoInferenceRule {
    Console(IoConsoleInferenceRule),
    ReplFs(IoReplFsInferenceRule),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum IoConsoleInferenceRule {
    ClearConsole,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum IoReplFsInferenceRule {
    ChangeDirectory,
    CurrentDirectory,
    DirectoryLifecycle(DirectoryLifecycleInferenceRule),
    Environment(EnvironmentInferenceRule),
    DirectoryListing(DirectoryListingInferenceRule),
    FileTransfer(FileTransferInferenceRule),
    PathPredicate(PathPredicateInferenceRule),
    PathSyntax(PathSyntaxInferenceRule),
    SearchPath(SearchPathInferenceRule),
    TemporaryPath(TemporaryPathInferenceRule),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum DirectoryListingInferenceRule {
    Metadata,
    Names,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum PathPredicateInferenceRule {
    File,
    Folder,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum FileTransferInferenceRule {
    Copy,
    Move,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum DirectoryLifecycleInferenceRule {
    Create,
    Remove,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum EnvironmentInferenceRule {
    Read,
    Exists,
    Set,
    Remove,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum TemporaryPathInferenceRule {
    Directory,
    UniqueName,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum PathSyntaxInferenceRule {
    Join,
    Split,
    FileSeparator,
    PathListSeparator,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum SearchPathInferenceRule {
    QueryOrReplace,
    Add,
    Remove,
    Generate,
    Persist,
}
