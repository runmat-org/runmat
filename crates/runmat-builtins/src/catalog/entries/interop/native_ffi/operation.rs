//! Internal native-adapter operations; public C symbol names remain text.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NativeFfiOperation {
    Load,
    LoadReport,
    LoadPrepared,
    Unload,
    IsLoaded,
    Functions,
    Pointer,
    Structure,
    PointerValue,
    GetMember,
    SetMember,
    InvokeMember,
    BuildInterface,
    Call,
}

impl NativeFfiOperation {
    pub const fn symbol(self) -> &'static str {
        match self {
            Self::Load => "load",
            Self::LoadReport => "load_report",
            Self::LoadPrepared => "load_prepared",
            Self::Unload => "unload",
            Self::IsLoaded => "is_loaded",
            Self::Functions => "functions",
            Self::Pointer => "pointer",
            Self::Structure => "structure",
            Self::PointerValue => "pointer_value",
            Self::GetMember => "get_member",
            Self::SetMember => "set_member",
            Self::InvokeMember => "invoke_member",
            Self::BuildInterface => "build_interface",
            Self::Call => "call",
        }
    }

    pub fn from_symbol(symbol: &str) -> Option<Self> {
        Some(match symbol {
            "load" => Self::Load,
            "load_report" => Self::LoadReport,
            "load_prepared" => Self::LoadPrepared,
            "unload" => Self::Unload,
            "is_loaded" => Self::IsLoaded,
            "functions" => Self::Functions,
            "pointer" => Self::Pointer,
            "structure" => Self::Structure,
            "pointer_value" => Self::PointerValue,
            "get_member" => Self::GetMember,
            "set_member" => Self::SetMember,
            "invoke_member" => Self::InvokeMember,
            "build_interface" => Self::BuildInterface,
            "call" => Self::Call,
            _ => return None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::NativeFfiOperation;

    #[test]
    fn internal_operation_symbols_round_trip() {
        for operation in [
            NativeFfiOperation::Load,
            NativeFfiOperation::LoadReport,
            NativeFfiOperation::LoadPrepared,
            NativeFfiOperation::Unload,
            NativeFfiOperation::IsLoaded,
            NativeFfiOperation::Functions,
            NativeFfiOperation::Pointer,
            NativeFfiOperation::Structure,
            NativeFfiOperation::PointerValue,
            NativeFfiOperation::GetMember,
            NativeFfiOperation::SetMember,
            NativeFfiOperation::InvokeMember,
            NativeFfiOperation::BuildInterface,
            NativeFfiOperation::Call,
        ] {
            assert_eq!(
                NativeFfiOperation::from_symbol(operation.symbol()),
                Some(operation)
            );
        }
        assert_eq!(NativeFfiOperation::from_symbol("fixture_add"), None);
    }
}
