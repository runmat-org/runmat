//! External class static calls share resolution whether parentheses are present or omitted.
use super::*;

impl LoweringCtx {
    pub(super) fn lower_external_static_method(
        &mut self,
        base: &AstExpr,
        name: &str,
        args: &[AstExpr],
        requested_outputs: RequestedOutputCount,
        span: Span,
    ) -> Result<Option<HirExpr>, HirError> {
        // A bound receiver belongs to object/member dispatch, even when its
        // spelling is also the name of a standard or external class.
        let Some(mut class_name) = self.unbound_qualified_member_base(base) else {
            return Ok(None);
        };
        if let AstExpr::Ident(name, _) = base {
            if let Some(imported) = self.resolve_imported_class_name_target(name, base.span())? {
                class_name = imported;
            }
        }
        if self.class_id_for_name(&class_name).is_some() {
            return Ok(None);
        }
        let Some(method) = self
            .external_class_declaration(&class_name)
            .and_then(|declaration| {
                declaration
                    .methods
                    .into_iter()
                    .find(|method| method.name.0 == name)
            })
        else {
            return Ok(None);
        };
        if !method.is_static || method.attributes.access != MemberAccess::Public {
            return Err(HirError::new(format!(
                "method {class_name}.{name} is not accessible as a public static method"
            ))
            .with_span(span));
        }
        let mut call_args = self.lower_call_arguments(args)?;
        if let Some(class_argument) = method.implicit_class_argument {
            call_args.push(HirExpr {
                id: self.alloc_expr_id(),
                kind: HirExprKind::String(StringLiteral(class_argument.display_name().to_owned())),
                span: base.span(),
            });
        }
        let Some(function_name) = method.callable.display_name() else {
            return Err(
                HirError::new(format!("method {class_name}.{name} has no callable name"))
                    .with_span(span),
            );
        };
        Ok(Some(HirExpr {
            id: self.alloc_expr_id(),
            kind: HirExprKind::Call(self.call_for_name(
                &function_name,
                call_args,
                CallSyntax::Plain,
                requested_outputs,
                span,
            )?),
            span,
        }))
    }
}
