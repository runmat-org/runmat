use runmat_execution::resource::ResourceUsage;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum FeaRunKind {
    Meshing,
    Solve,
    MeshAndSolve,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum FeaRunSite {
    Local,
    Dispatched,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum FeaRunOutcome {
    Completed,
    Cancelled,
    Failed,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FeaRunRequest {
    pub run_id: String,
    pub kind: FeaRunKind,
    pub site: FeaRunSite,
    pub allocated_cpu_millicores: u64,
    pub estimated_wall_seconds: Option<u64>,
    pub study_identity: String,
}

impl FeaRunRequest {
    pub fn local(
        kind: FeaRunKind,
        allocated_cpu_millicores: u64,
        study_identity: runmat_execution::Digest,
    ) -> Result<Self, FeaRunDenied> {
        use std::fmt::Write as _;
        let mut random = [0_u8; 16];
        getrandom::getrandom(&mut random).map_err(|_| FeaRunDenied::LicenseInvalid)?;
        let mut run_id = String::with_capacity(36);
        run_id.push_str("run_");
        for byte in random {
            write!(run_id, "{byte:02x}").expect("formatting into a String");
        }
        let request = Self {
            run_id,
            kind,
            site: FeaRunSite::Local,
            allocated_cpu_millicores,
            estimated_wall_seconds: None,
            study_identity: study_identity.to_string(),
        };
        request.validate()?;
        Ok(request)
    }

    pub(super) fn validate(&self) -> Result<(), FeaRunDenied> {
        let valid_id = !self.run_id.is_empty()
            && self.run_id.len() <= 128
            && self
                .run_id
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b"._:-".contains(&b));
        let valid_estimate = self
            .estimated_wall_seconds
            .is_none_or(|v| (1..=604_800).contains(&v));
        if !valid_id
            || !(1..=4_096_000).contains(&self.allocated_cpu_millicores)
            || !valid_estimate
            || self.study_identity.len() > 256
        {
            return Err(FeaRunDenied::LicenseInvalid);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum FeaRunDenied {
    SignInRequired,
    AllowanceUsed {
        refills_at_unix_seconds: Option<i64>,
    },
    LeaseExpired,
    LicenseInvalid,
    ServerUnreachable,
}

impl FeaRunDenied {
    pub const fn error_code(&self) -> &'static str {
        match self {
            Self::SignInRequired => "RM.FEA.SignInRequired",
            Self::AllowanceUsed { .. } => "RM.FEA.AllowanceUsed",
            Self::LeaseExpired | Self::LicenseInvalid => "RM.FEA.LeaseInvalid",
            Self::ServerUnreachable => "RM.FEA.ServerUnreachable",
        }
    }

    pub fn into_operation_error(
        self,
        operation: &str,
        version: &str,
        context: &crate::operations::OperationContext,
    ) -> crate::operations::OperationErrorEnvelope {
        use crate::operations::*;
        operation_error(
            operation,
            version,
            context,
            OperationErrorSpec {
                error_code: self.error_code(),
                error_type: OperationErrorType::Capacity,
                retryable: matches!(self, Self::ServerUnreachable),
                severity: OperationErrorSeverity::Error,
            },
            self.to_string(),
            Default::default(),
        )
    }
}

impl std::fmt::Display for FeaRunDenied {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SignInRequired => f.write_str("Meshing and solving need a signed-in RunMat account. Run `runmat login`, then run this study again."),
            Self::AllowanceUsed { refills_at_unix_seconds } => {
                let date = refills_at_unix_seconds.and_then(|v| chrono::DateTime::from_timestamp(v, 0));
                f.write_str("You've used this week's core-hours. ")?;
                if let Some(date) = date { write!(f, "They refill on {}. ", date.format("%Y-%m-%d %H:%M UTC"))?; }
                f.write_str("To keep going now, add a core-hour pack or move to Scale: https://runmat.com/pricing")
            }
            Self::LeaseExpired | Self::LicenseInvalid => f.write_str("Your RunMat sign-in needs refreshing. Run `runmat login`, then run this study again."),
            Self::ServerUnreachable => f.write_str("Meshing and solving need a connection to RunMat Server to start. Check the connection, then run this study again."),
        }
    }
}

impl std::error::Error for FeaRunDenied {}

pub trait FeaRunPermit: Send {
    fn heartbeat(&self, _usage: &ResourceUsage) {}
    fn settle(self: Box<Self>, outcome: FeaRunOutcome, usage: &ResourceUsage);
}

pub trait FeaRunAdmission: Send + Sync {
    fn admit(&self, request: &FeaRunRequest) -> Result<Box<dyn FeaRunPermit>, FeaRunDenied>;
}
