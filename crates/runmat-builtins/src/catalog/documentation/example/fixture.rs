use serde::Serialize;

mod desktop;
mod foreign;

pub use desktop::*;
pub use foreign::*;

/// Stable identifier for a fixture owned by one catalog example.
///
/// The owning builtin and example IDs provide the global namespace. The local
/// name lets one example select among multiple reviewed fixture definitions
/// without accepting a command, path selector, or runner-specific expression.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct BuiltinExampleFixtureId {
    pub local_name: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum BuiltinExampleFixture {
    None,
    Filesystem(BuiltinFilesystemFixture),
    Loopback(BuiltinLoopbackFixture),
    ForeignAdapter(BuiltinForeignAdapterFixture),
    CliInteraction(BuiltinCliInteractionFixture),
    /// Requires the RunMat Desktop host. Browser and standalone CLI runners
    /// must reject this boundary rather than approximating host behavior.
    DesktopHostOnly(BuiltinDesktopHostFixture),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct BuiltinFilesystemFixture {
    pub id: BuiltinExampleFixtureId,
    pub root: BuiltinFilesystemRoot,
    pub entries: &'static [BuiltinFilesystemEntry],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum BuiltinFilesystemRoot {
    IsolatedWorkspace,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum BuiltinFilesystemEntry {
    Directory {
        relative_path: &'static str,
    },
    File {
        relative_path: &'static str,
        content: BuiltinFixtureContent,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum BuiltinFixtureContent {
    Utf8(&'static str),
    Bytes(&'static [u8]),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct BuiltinLoopbackFixture {
    pub id: BuiltinExampleFixtureId,
    pub scenario: BuiltinLoopbackScenario,
    pub endpoint_substitutions: &'static [BuiltinEndpointSubstitution],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum BuiltinLoopbackScenario {
    Http(BuiltinHttpScenario),
    Tcp(BuiltinTcpScenario),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct BuiltinHttpScenario {
    pub exchanges: &'static [BuiltinHttpExchange],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct BuiltinHttpExchange {
    pub request: BuiltinHttpRequest,
    pub response: BuiltinHttpResponse,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct BuiltinHttpRequest {
    pub method: BuiltinHttpMethod,
    pub path: &'static str,
    pub body: Option<&'static [u8]>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum BuiltinHttpMethod {
    Get,
    Head,
    Post,
    Put,
    Patch,
    Delete,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct BuiltinHttpResponse {
    pub status: u16,
    pub headers: &'static [BuiltinHttpHeader],
    pub body: &'static [u8],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct BuiltinHttpHeader {
    pub name: &'static str,
    pub value: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct BuiltinTcpScenario {
    pub exchanges: &'static [BuiltinTcpExchange],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct BuiltinTcpExchange {
    pub client_bytes: &'static [u8],
    pub server_bytes: &'static [u8],
}

/// Fixed placeholders understood by fixture-aware runners. No fixture can
/// introduce an arbitrary replacement expression.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub enum BuiltinEndpointSubstitution {
    HttpBaseUrl,
    LoopbackHost,
    LoopbackPort,
}

impl BuiltinEndpointSubstitution {
    pub const ALL: [Self; 3] = [Self::HttpBaseUrl, Self::LoopbackHost, Self::LoopbackPort];

    pub const fn source_token(self) -> &'static str {
        match self {
            Self::HttpBaseUrl => "__RUNMAT_HTTP_BASE_URL__",
            Self::LoopbackHost => "__RUNMAT_LOOPBACK_HOST__",
            Self::LoopbackPort => "__RUNMAT_LOOPBACK_PORT__",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct BuiltinCliInteractionFixture {
    pub id: BuiltinExampleFixtureId,
    pub transcript: &'static [BuiltinCliTranscriptStep],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum BuiltinCliTranscriptStep {
    ExpectOutput(&'static str),
    SendLine(&'static str),
    SendBytes(&'static [u8]),
    SendEndOfInput,
    SendInterrupt,
}
