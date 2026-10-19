//! Canonical public identifier namespaces shared by server and clients.
use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum IdError {
    #[error("invalid id prefix")]
    InvalidPrefix,
    #[error("invalid id suffix")]
    InvalidSuffix,
}

fn is_valid_suffix(suffix: &str) -> bool {
    suffix.len() == 32
        && suffix
            .bytes()
            .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
}

fn new_suffix() -> String {
    uuid::Uuid::now_v7().simple().to_string()
}

macro_rules! prefixed_id {
    ($name:ident, $prefix:literal) => {
        #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
        #[serde(transparent)]
        pub struct $name(String);

        impl Default for $name {
            fn default() -> Self {
                Self::new()
            }
        }

        impl $name {
            pub const PREFIX: &'static str = $prefix;

            pub fn new() -> Self {
                Self(format!("{}{}", Self::PREFIX, new_suffix()))
            }

            pub fn parse(value: &str) -> Result<Self, IdError> {
                if !value.starts_with(Self::PREFIX) {
                    return Err(IdError::InvalidPrefix);
                }
                let suffix = &value[Self::PREFIX.len()..];
                if !is_valid_suffix(suffix) {
                    return Err(IdError::InvalidSuffix);
                }
                Ok(Self(value.to_string()))
            }

            pub fn as_str(&self) -> &str {
                &self.0
            }

            pub fn from_uuid(value: uuid::Uuid) -> Self {
                Self(format!("{}{}", Self::PREFIX, value.simple()))
            }

            pub fn as_uuid(&self) -> Result<uuid::Uuid, IdError> {
                let suffix = self
                    .0
                    .strip_prefix(Self::PREFIX)
                    .ok_or(IdError::InvalidPrefix)?;
                uuid::Uuid::parse_str(suffix).map_err(|_| IdError::InvalidSuffix)
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(&self.0)
            }
        }

        impl FromStr for $name {
            type Err = IdError;

            fn from_str(s: &str) -> Result<Self, Self::Err> {
                Self::parse(s)
            }
        }

        impl AsRef<str> for $name {
            fn as_ref(&self) -> &str {
                self.as_str()
            }
        }
    };
}

prefixed_id!(OrgId, "org_");
prefixed_id!(ProjectId, "proj_");
prefixed_id!(MembershipId, "mbr_");
prefixed_id!(ProjectMembershipId, "pmbr_");
prefixed_id!(UserId, "usr_");
prefixed_id!(ServiceAccountId, "svc_");
prefixed_id!(ApiKeyId, "key_");
prefixed_id!(FsEntryId, "fse_");
prefixed_id!(FsVersionId, "fsv_");
prefixed_id!(SnapshotId, "snap_");
prefixed_id!(EventId, "evt_");
prefixed_id!(AuditId, "aud_");
prefixed_id!(ReservationId, "res_");
prefixed_id!(ScimTokenId, "scimtok_");
prefixed_id!(ScimUserId, "scimusr_");
prefixed_id!(ScimGroupId, "scimgrp_");
prefixed_id!(VerifyId, "verify_");
prefixed_id!(PackageId, "pkg_");
prefixed_id!(PackageReleaseId, "rel_");
prefixed_id!(PackageArtifactId, "art_");
prefixed_id!(PackagePublicationId, "pub_");
prefixed_id!(PackageGrantId, "pgr_");
prefixed_id!(PackageAdvisoryId, "adv_");
prefixed_id!(PackageGroupId, "grp_");
prefixed_id!(PackageTrustedPublisherId, "ptp_");
prefixed_id!(PackageRecipientKeyId, "pkr_");
prefixed_id!(PackageKeyEnvelopeId, "pke_");
prefixed_id!(ClusterId, "cls_");
prefixed_id!(ClusterNodeId, "node_");
prefixed_id!(ClusterEnrollmentId, "enr_");
prefixed_id!(AllocationLeaseId, "als_");
prefixed_id!(ExecutionArtifactId, "eart_");
prefixed_id!(ExecutionTransferId, "xfr_");
prefixed_id!(ExecutionArtifactGrantId, "eagr_");
prefixed_id!(ExecutionRunId, "run_");
prefixed_id!(ExecutionDriverLeaseId, "drv_");
prefixed_id!(ExecutionCheckpointId, "chk_");
prefixed_id!(ExecutionUsageId, "xuse_");
prefixed_id!(AcademicApplicationId, "acadapp_");
prefixed_id!(AcademicProofId, "acadproof_");
prefixed_id!(AcademicGrantId, "acadgrant_");
prefixed_id!(TeachingAuthorizationId, "teachauth_");
prefixed_id!(AcademicCourseId, "course_");
prefixed_id!(AcademicReviewActionId, "acadreview_");
prefixed_id!(TrialReactivationId, "trialreact_");
prefixed_id!(DomainOutboxId, "outbox_");
prefixed_id!(AgentConnectionId, "aic_");
prefixed_id!(AgentProjectId, "aiproj_");
prefixed_id!(AgentProjectRevisionId, "aiprojr_");
prefixed_id!(AgentProjectCapabilityId, "aiprojc_");
prefixed_id!(AgentEntitlementWindowId, "aiew_");
prefixed_id!(AgentExecutionId, "aiexec_");
prefixed_id!(AgentExecutionLeaseId, "ailease_");
prefixed_id!(AgentExecutionReservationId, "aires_");
prefixed_id!(AgentExecutionEventId, "aievt_");
prefixed_id!(AgentExecutionPackId, "aipack_");
prefixed_id!(AgentExecutionPackEventId, "aipevt_");
prefixed_id!(AgentDeviceId, "aidev_");
prefixed_id!(AgentProjectGrantId, "aigrant_");
prefixed_id!(AgentArtifactId, "aiart_");
prefixed_id!(AgentRuntimeActionId, "aiact_");

pub fn org_id_string(value: uuid::Uuid) -> String {
    OrgId::from_uuid(value).to_string()
}

pub fn project_id_string(value: uuid::Uuid) -> String {
    ProjectId::from_uuid(value).to_string()
}

pub fn membership_id_string(value: uuid::Uuid) -> String {
    MembershipId::from_uuid(value).to_string()
}

pub fn project_membership_id_string(value: uuid::Uuid) -> String {
    ProjectMembershipId::from_uuid(value).to_string()
}

pub fn service_account_id_string(value: uuid::Uuid) -> String {
    ServiceAccountId::from_uuid(value).to_string()
}

pub fn api_key_id_string(value: uuid::Uuid) -> String {
    ApiKeyId::from_uuid(value).to_string()
}

pub fn fs_version_id_string(value: uuid::Uuid) -> String {
    FsVersionId::from_uuid(value).to_string()
}

pub fn snapshot_id_string(value: uuid::Uuid) -> String {
    SnapshotId::from_uuid(value).to_string()
}

pub fn event_id_string(value: uuid::Uuid) -> String {
    EventId::from_uuid(value).to_string()
}

pub fn reservation_id_string(value: uuid::Uuid) -> String {
    ReservationId::from_uuid(value).to_string()
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct PrincipalId(String);

impl PrincipalId {
    pub fn parse(value: &str) -> Result<Self, IdError> {
        if value.starts_with(UserId::PREFIX) {
            UserId::parse(value)?;
            return Ok(Self(value.to_string()));
        }
        if value.starts_with(ServiceAccountId::PREFIX) {
            ServiceAccountId::parse(value)?;
            return Ok(Self(value.to_string()));
        }
        Err(IdError::InvalidPrefix)
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for PrincipalId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl FromStr for PrincipalId {
    type Err = IdError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::parse(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn org_id_roundtrip() {
        let id = OrgId::new();
        assert!(id.as_str().starts_with("org_"));
        let parsed = OrgId::parse(id.as_str()).expect("parse org id");
        assert_eq!(parsed, id);
    }

    #[test]
    fn id_rejects_invalid_suffix() {
        let err = OrgId::parse("org_ABC").expect_err("invalid suffix should fail");
        assert_eq!(err, IdError::InvalidSuffix);
    }

    #[test]
    fn principal_accepts_user_and_service() {
        let user = UserId::new();
        let service = ServiceAccountId::new();
        assert!(PrincipalId::parse(user.as_str()).is_ok());
        assert!(PrincipalId::parse(service.as_str()).is_ok());
        assert!(PrincipalId::parse("org_0123456789abcdef0123456789abcdef").is_err());
    }
}
