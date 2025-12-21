#![allow(dead_code)]

use serde::{Deserialize, Serialize};

pub type DocumentUri = String;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct FusionPlanNodePublic {
    pub id: String,
    pub kind: String,
    pub label: String,
    pub residency: Option<String>,
    pub shape: Vec<usize>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct FusionPlanEdgePublic {
    pub from: String,
    pub to: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct FusionPlanPublic {
    pub nodes: Vec<FusionPlanNodePublic>,
    pub edges: Vec<FusionPlanEdgePublic>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub notes: Option<String>,
}

