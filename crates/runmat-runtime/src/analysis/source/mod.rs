mod compile;
mod geometry_resolver;
mod host;
mod project_host;

pub use compile::{
    check_fea_path, check_fea_path_in_root, check_fea_source_in_root, check_fea_sources_in_root,
};
pub use host::FilesystemFeaHost;
pub use project_host::ProjectFeaHost;

#[cfg(test)]
mod tests;
