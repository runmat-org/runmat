mod checkout;
mod credentials;
mod fetch;
mod objects;
mod remote;

pub use crate::provider::NativeGitPackageProvider;
pub use checkout::{GitAcquireRequest, NativeGitClient};
pub use credentials::{GitCredential, GitCredentialProvider, NoGitCredentials};
