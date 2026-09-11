use async_trait::async_trait;
use once_cell::sync::OnceCell;
use std::io;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OpenFileDialogFilter {
    pub patterns: Vec<String>,
    pub description: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct OpenFileDialogRequest {
    pub title: Option<String>,
    pub default_path: Option<PathBuf>,
    pub filters: Vec<OpenFileDialogFilter>,
    pub multiselect: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OpenFileDialogSelection {
    pub paths: Vec<PathBuf>,
    pub filter_index: Option<usize>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SaveFileDialogRequest {
    pub title: Option<String>,
    pub default_path: Option<PathBuf>,
    pub filters: Vec<OpenFileDialogFilter>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SaveFileDialogSelection {
    pub path: PathBuf,
    pub filter_index: Option<usize>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DirectoryDialogRequest {
    pub title: Option<String>,
    pub default_path: Option<PathBuf>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DirectoryDialogSelection {
    pub path: PathBuf,
}

/// Host-owned UI service for runtime file and directory selection.
///
/// Storage providers remain responsible for bytes and paths. Native hosts can
/// install this service independently so changing a project's storage backend
/// does not silently remove or replace its dialog implementation.
#[async_trait(?Send)]
pub trait HostDialogProvider: Send + Sync + 'static {
    async fn select_file_open(
        &self,
        _request: &OpenFileDialogRequest,
    ) -> io::Result<Option<OpenFileDialogSelection>> {
        Ok(None)
    }

    async fn select_file_save(
        &self,
        _request: &SaveFileDialogRequest,
    ) -> io::Result<Option<SaveFileDialogSelection>> {
        Ok(None)
    }

    async fn select_directory(
        &self,
        _request: &DirectoryDialogRequest,
    ) -> io::Result<Option<DirectoryDialogSelection>> {
        Ok(None)
    }
}

static HOST_DIALOG_PROVIDER: OnceCell<RwLock<Option<Arc<dyn HostDialogProvider>>>> =
    OnceCell::new();

fn provider_lock() -> &'static RwLock<Option<Arc<dyn HostDialogProvider>>> {
    HOST_DIALOG_PROVIDER.get_or_init(|| RwLock::new(None))
}

pub fn set_host_dialog_provider(provider: Option<Arc<dyn HostDialogProvider>>) {
    *provider_lock()
        .write()
        .expect("host dialog provider lock poisoned") = provider;
}

pub fn replace_host_dialog_provider(
    provider: Option<Arc<dyn HostDialogProvider>>,
) -> HostDialogProviderGuard {
    let previous = std::mem::replace(
        &mut *provider_lock()
            .write()
            .expect("host dialog provider lock poisoned"),
        provider,
    );
    HostDialogProviderGuard { previous }
}

pub(crate) fn current_host_dialog_provider() -> Option<Arc<dyn HostDialogProvider>> {
    provider_lock()
        .read()
        .expect("host dialog provider lock poisoned")
        .clone()
}

pub struct HostDialogProviderGuard {
    previous: Option<Arc<dyn HostDialogProvider>>,
}

impl Drop for HostDialogProviderGuard {
    fn drop(&mut self) {
        set_host_dialog_provider(self.previous.take());
    }
}
