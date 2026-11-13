/// A networking scope parsed once from the public close selector spelling.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CloseNetworkSelector {
    Clients,
    Servers,
}

impl CloseNetworkSelector {
    pub fn parse(text: &str) -> Option<Self> {
        match text.trim().to_ascii_lowercase().as_str() {
            "client" | "clients" => Some(Self::Clients),
            "server" | "servers" => Some(Self::Servers),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selector_spelling_is_closed_and_case_insensitive() {
        assert_eq!(
            CloseNetworkSelector::parse(" CLIENTS "),
            Some(CloseNetworkSelector::Clients)
        );
        assert_eq!(
            CloseNetworkSelector::parse("server"),
            Some(CloseNetworkSelector::Servers)
        );
        assert_eq!(CloseNetworkSelector::parse("all"), None);
        assert_eq!(CloseNetworkSelector::parse("clientz"), None);
    }
}
