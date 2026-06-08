#![allow(unused_imports)]

//! Introspection builtins.

pub mod addlistener;
pub mod arity_check;
pub mod builtin;
pub mod call_bound_method;
pub mod call_method;
pub mod class;
pub mod classref;
pub mod clear;
pub mod clearvars;
pub mod dependent_property;
pub mod dynamic_workspace;
pub mod feval;
pub mod function_handle_text;
pub mod function_metadata;
pub mod getmethod;
pub mod inputname;
pub(crate) mod isa;
pub(crate) mod ischar;
pub(crate) mod isstring;
pub mod isvalid;
pub mod localfunctions;
pub mod make_anon;
pub mod mfilename;
pub mod new_handle_object;
pub mod new_object;
pub mod notify;
pub mod object_indexing;
pub mod test_classes;
pub mod test_methods;
pub(crate) mod type_resolvers;
pub mod which;
pub mod who;
pub mod whos;
pub use class::*;
