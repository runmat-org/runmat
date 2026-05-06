use crate::{MirIndexing, MirLocalId};
use runmat_hir::{BindingId, MemberName};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MirPlace {
    Local(MirLocalId),
    Binding(BindingId),
    Member(Box<MirPlace>, MemberName),
    Index(Box<MirPlace>, MirIndexing),
}
