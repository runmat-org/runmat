//! Closed public Name/Value controls shared by TCP catalog and runtime.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NetConstructorOption {
    Timeout,
    ConnectTimeout,
    ByteOrder,
    UserData,
    Name,
    InputBufferSize,
    OutputBufferSize,
}

impl NetConstructorOption {
    pub fn tcpclient(name: &str) -> Option<Self> {
        match name.to_ascii_lowercase().as_str() {
            "timeout" => Some(Self::Timeout),
            "connecttimeout" => Some(Self::ConnectTimeout),
            "byteorder" => Some(Self::ByteOrder),
            "userdata" => Some(Self::UserData),
            "name" => Some(Self::Name),
            "inputbuffersize" => Some(Self::InputBufferSize),
            "outputbuffersize" => Some(Self::OutputBufferSize),
            _ => None,
        }
    }

    pub fn tcpserver(name: &str) -> Option<Self> {
        let option = Self::tcpclient(name)?;
        matches!(
            option,
            Self::Timeout | Self::ByteOrder | Self::UserData | Self::Name
        )
        .then_some(option)
    }

    pub fn accept(name: &str) -> Option<Self> {
        (Self::tcpclient(name)? == Self::Timeout).then_some(Self::Timeout)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identities_accept_only_their_closed_option_sets() {
        assert_eq!(
            NetConstructorOption::tcpclient("CONNECTTIMEOUT"),
            Some(NetConstructorOption::ConnectTimeout)
        );
        assert_eq!(
            NetConstructorOption::tcpserver("ByteOrder"),
            Some(NetConstructorOption::ByteOrder)
        );
        assert_eq!(NetConstructorOption::tcpserver("InputBufferSize"), None);
        assert_eq!(
            NetConstructorOption::accept("tImEoUt"),
            Some(NetConstructorOption::Timeout)
        );
        assert_eq!(NetConstructorOption::accept("ConnectTimeout"), None);
    }
}
