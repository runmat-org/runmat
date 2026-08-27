use std::rc::Rc;

use crate::{
    value_from_mx_in_context, value_to_mx_for_interface_in_context, MexDiagnostic, MexHostServices,
    MxApiMode, MxArray, MxBoundaryInterface, MxValueContext,
};

/// Native-lane callback boundary.
///
/// Values at this layer are opaque MEX arrays with thread-transferable storage
/// and origin-thread handle tokens. Runtime `Value` conversion belongs to the
/// origin adapter below, never to a worker-thread proxy.
pub trait MexBoundaryHostServices {
    fn eval(&self, command: &str) -> Result<(), MexDiagnostic>;
    fn call(
        &self,
        function: &str,
        arguments: Vec<MxArray>,
        requested_outputs: usize,
    ) -> Result<Vec<MxArray>, MexDiagnostic>;
    fn get_variable(&self, workspace: &str, name: &str) -> Result<Option<MxArray>, MexDiagnostic>;
    fn put_variable(
        &self,
        workspace: &str,
        name: &str,
        value: MxArray,
    ) -> Result<(), MexDiagnostic>;
    fn get_object_property(&self, object: MxArray, name: &str) -> Result<MxArray, MexDiagnostic>;
    fn set_object_property(
        &self,
        object: MxArray,
        name: &str,
        value: MxArray,
    ) -> Result<MxArray, MexDiagnostic>;
}

#[doc(hidden)]
pub struct DirectMexBoundaryHostServices {
    services: Rc<dyn MexHostServices>,
    values: Rc<MxValueContext>,
    mode: MxApiMode,
    interface: MxBoundaryInterface,
}

impl DirectMexBoundaryHostServices {
    pub fn new(
        services: Rc<dyn MexHostServices>,
        values: Rc<MxValueContext>,
        mode: MxApiMode,
        interface: MxBoundaryInterface,
    ) -> Self {
        Self {
            services,
            values,
            mode,
            interface,
        }
    }

    fn decode(&self, value: &MxArray) -> Result<runmat_value::Value, MexDiagnostic> {
        value_from_mx_in_context(value, Some(&self.values)).map_err(conversion_diagnostic)
    }

    fn encode(&self, value: &runmat_value::Value) -> Result<MxArray, MexDiagnostic> {
        value_to_mx_for_interface_in_context(value, self.mode, self.interface, Some(&self.values))
            .map_err(conversion_diagnostic)
    }
}

impl MexBoundaryHostServices for DirectMexBoundaryHostServices {
    fn eval(&self, command: &str) -> Result<(), MexDiagnostic> {
        self.services.eval(command)
    }

    fn call(
        &self,
        function: &str,
        arguments: Vec<MxArray>,
        requested_outputs: usize,
    ) -> Result<Vec<MxArray>, MexDiagnostic> {
        let arguments = arguments
            .iter()
            .map(|argument| self.decode(argument))
            .collect::<Result<Vec<_>, _>>()?;
        self.services
            .call(function, arguments, requested_outputs)?
            .iter()
            .map(|output| self.encode(output))
            .collect()
    }

    fn get_variable(&self, workspace: &str, name: &str) -> Result<Option<MxArray>, MexDiagnostic> {
        self.services
            .get_variable(workspace, name)?
            .as_ref()
            .map(|value| self.encode(value))
            .transpose()
    }

    fn put_variable(
        &self,
        workspace: &str,
        name: &str,
        value: MxArray,
    ) -> Result<(), MexDiagnostic> {
        self.services
            .put_variable(workspace, name, self.decode(&value)?)
    }

    fn get_object_property(&self, object: MxArray, name: &str) -> Result<MxArray, MexDiagnostic> {
        let object = self.decode(&object)?;
        let value = self.services.get_object_property(object, name)?;
        self.encode(&value)
    }

    fn set_object_property(
        &self,
        object: MxArray,
        name: &str,
        value: MxArray,
    ) -> Result<MxArray, MexDiagnostic> {
        let object = self.decode(&object)?;
        let value = self.decode(&value)?;
        let object = self.services.set_object_property(object, name, value)?;
        self.encode(&object)
    }
}

fn conversion_diagnostic(error: crate::MxConversionError) -> MexDiagnostic {
    MexDiagnostic {
        identifier: Some("RunMat:MEX:Conversion".into()),
        message: error.message,
    }
}
