mod checkout;
mod credentials;
mod fetch;
mod objects;
mod provider;
mod remote;

pub use checkout::{GitAcquireRequest, NativeGitClient};
pub use credentials::{GitCredential, GitCredentialProvider, NoGitCredentials};
pub use provider::NativeGitPackageProvider;
