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
    WorkingDirectory(WorkingDirectoryInferenceRule),
    Directory(DirectoryInferenceRule),
    Environment(EnvironmentInferenceRule),
    File(FileInferenceRule),
    Path(PathInferenceRule),
    SourceInventory(SourceInventoryInferenceRule),
}

impl IoReplFsInferenceRule {
    pub const fn directory(rule: DirectoryInferenceRule) -> Self {
        Self::Directory(rule)
    }

    pub const fn file(rule: FileInferenceRule) -> Self {
        Self::File(rule)
    }

    pub const fn path(rule: PathInferenceRule) -> Self {
        Self::Path(rule)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum WorkingDirectoryInferenceRule {
    Change,
    Current,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum DirectoryInferenceRule {
    Lifecycle(DirectoryLifecycleInferenceRule),
    Listing(DirectoryListingInferenceRule),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum FileInferenceRule {
    Transfer(FileTransferInferenceRule),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum PathInferenceRule {
    Installation(InstallationPathInferenceRule),
    Predicate(PathPredicateInferenceRule),
    Syntax(PathSyntaxInferenceRule),
    Search(SearchPathInferenceRule),
    Temporary(TemporaryPathInferenceRule),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum InstallationPathInferenceRule {
    Root,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum DirectoryListingInferenceRule {
    Metadata,
    Names,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum SourceInventoryInferenceRule {
    FolderContents,
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
