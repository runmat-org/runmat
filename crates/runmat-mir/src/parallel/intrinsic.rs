use runmat_types::{BuiltinId, CallableIdentity};

/// Parallel operations whose semantics are represented directly in MIR.
///
/// Callable resolution establishes `Builtin` identity before this boundary.
/// User functions and dynamically resolved names with the same spelling must
/// remain ordinary calls.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ParallelIntrinsic {
    Distributed,
    GetLocalPart,
    GetCodistributor,
    Redistribute,
    LabBarrier,
    LabBroadcast,
    LabSend,
    LabReceive,
    LabProbe,
    LabSendReceive,
    Gplus,
    Gcat,
    Gop,
}

impl ParallelIntrinsic {
    pub(crate) fn resolve(identity: &CallableIdentity) -> Option<Self> {
        let CallableIdentity::Builtin(BuiltinId(name)) = identity else {
            return None;
        };
        match name.as_str() {
            "distributed" => Some(Self::Distributed),
            "getLocalPart" => Some(Self::GetLocalPart),
            "getCodistributor" => Some(Self::GetCodistributor),
            "redistribute" => Some(Self::Redistribute),
            "labBarrier" => Some(Self::LabBarrier),
            "spmdBarrier" => Some(Self::LabBarrier),
            "labBroadcast" => Some(Self::LabBroadcast),
            "spmdBroadcast" => Some(Self::LabBroadcast),
            "labSend" => Some(Self::LabSend),
            "spmdSend" => Some(Self::LabSend),
            "labReceive" => Some(Self::LabReceive),
            "spmdReceive" => Some(Self::LabReceive),
            "labProbe" => Some(Self::LabProbe),
            "spmdProbe" => Some(Self::LabProbe),
            "labSendReceive" | "spmdSendReceive" => Some(Self::LabSendReceive),
            "gplus" => Some(Self::Gplus),
            "spmdPlus" => Some(Self::Gplus),
            "gcat" | "spmdCat" => Some(Self::Gcat),
            "gop" | "spmdReduce" => Some(Self::Gop),
            _ => None,
        }
    }

    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Distributed => "distributed",
            Self::GetLocalPart => "getLocalPart",
            Self::GetCodistributor => "getCodistributor",
            Self::Redistribute => "redistribute",
            Self::LabBarrier => "labBarrier",
            Self::LabBroadcast => "labBroadcast",
            Self::LabSend => "labSend",
            Self::LabReceive => "labReceive",
            Self::LabProbe => "labProbe",
            Self::LabSendReceive => "labSendReceive",
            Self::Gplus => "gplus",
            Self::Gcat => "gcat",
            Self::Gop => "gop",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::ParallelIntrinsic;
    use runmat_types::{BuiltinId, CallableIdentity, SymbolName};

    #[test]
    fn only_resolved_builtin_identities_become_intrinsics() {
        assert_eq!(
            ParallelIntrinsic::resolve(&CallableIdentity::Builtin(BuiltinId("labBarrier".into()))),
            Some(ParallelIntrinsic::LabBarrier)
        );
        assert_eq!(
            ParallelIntrinsic::resolve(&CallableIdentity::DynamicName(SymbolName(
                "labBarrier".into()
            ))),
            None
        );
    }
}
