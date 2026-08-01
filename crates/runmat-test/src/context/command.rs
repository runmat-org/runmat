use serde::{Deserialize, Serialize};

use crate::descriptor::ProcedureDescriptor;
use crate::lifecycle::{FixtureScopeKey, QualificationKind};
use crate::result::{Artifact, Diagnostic};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum TestCommand {
    AddTeardown {
        scope: FixtureScopeKey,
        procedure: ProcedureDescriptor,
    },
    Qualify {
        qualification: QualificationKind,
        diagnostic: Diagnostic,
    },
    RecordDiagnostic {
        diagnostic: Diagnostic,
    },
    AttachArtifact {
        artifact: Artifact,
    },
}
