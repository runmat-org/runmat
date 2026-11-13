//! Typed TCP receiver admission. Public object properties never grant socket
//! access; only the sealed native payload can be presented to the service.

use runmat_types::StaticClassIdentity;
use runmat_value::{ForeignRef, Value};

pub(super) fn authority(value: &Value, class: StaticClassIdentity) -> Option<ForeignRef> {
    let Value::Object(object) = value else {
        return None;
    };
    object
        .class_name()
        .is(class)
        .then(|| object.tcp_authority().cloned())
        .flatten()
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_value::ObjectInstance;

    #[test]
    fn visible_class_and_fields_do_not_forge_socket_authority() {
        let detached = Value::Object(ObjectInstance::new(runmat_builtins::TCPCLIENT_CLASS));
        assert!(authority(&detached, runmat_builtins::TCPCLIENT_CLASS).is_none());
        assert!(authority(&detached, runmat_builtins::TCPSERVER_CLASS).is_none());
    }
}
