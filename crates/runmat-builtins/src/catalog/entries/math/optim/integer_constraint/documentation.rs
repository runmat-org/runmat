use crate::*;
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority:BuiltinDocumentationAuthority::Catalog,title:Some("integerConstraint"),slug:Some("integerConstraint"),
    summary:"Describe integer, semicontinuous and semi-integer decision-variable indices.",
    description:"integerConstraint creates a scalar IntegerConstraint value object for the intcon argument of intlinprog. Optional Integer, SemiContinuous and SemiInteger name/value pairs select disjoint vectors of positive variable indices; omitted properties are empty double vectors.",
    keywords:&["integerConstraint","intlinprog","semicontinuous","semi-integer","MILP"],related:&["intlinprog","optimoptions"],
    sections:&[
        BuiltinDocumentationSection{heading:"Indices and value properties",paragraphs:&[
            "Names ignore case. Each supplied value must be a full real double vector of finite positive integer indices, with no repeated index within or across properties. Scalars are one-element vectors. Single, native integer, logical, complex, sparse and callable inputs are not accepted as property vectors. Indices are checked against the problem variable count when intlinprog consumes the object.",
            "Copies have value semantics: editing one copy does not change another. intlinprog validates the current property values, including their classes, vector shapes and disjoint indices. Values restored from files or transferred between execution workers receive the same validation."]},
        BuiltinDocumentationSection{heading:"Decision domains",paragraphs:&[
            "Integer selects unrestricted integer-valued coordinates subject to the problem's ordinary bounds. SemiContinuous selects zero or any real value in its supplied [lb,ub] interval. SemiInteger selects zero or an integer in that interval. Unspecified coordinates are continuous. Both bounds of every semi domain must be strictly positive and at most 1e5; intlinprog validates these bounds, not the constructor, which has no problem bounds.",
            "Zero remains available in a semi domain even when its active lower bound is positive. The solver keeps this union domain separate from the convex relaxation and branches on forbidden gaps before accepting a feasible point. If the active interval contains no value of the required class, the coordinate must be zero."]},
        BuiltinDocumentationSection{heading:"Execution and transport",paragraphs:&[
            "Construction runs on the host. Automatically placed double arrays can be gathered using the invocation's provider; explicitly created gpuArray inputs require an explicit gather first. Inputs with unknown placement and distributed arrays are unsupported. Construction requires unbounded host workspace and refuses tasks with a finite host-memory budget before transferring any property data.",
            "Existing values can be passed between execution workers and saved in MAT files. Validation uses temporary index-sorting memory charged to the receiving invocation when it has a managed memory budget. Property order is preserved."]},
    ],
    examples:&[
        BuiltinExample{id:"disjoint-domains",title:"Describe three kinds of variables",program:"intcon=integerConstraint('Integer',1,'SemiContinuous',[2 3],'SemiInteger',4)",display_output:None,
            compatibility:BuiltinExampleCompatibility::RunMat,harness:BuiltinExampleHarness::Portable,fixture:BuiltinExampleFixture::None,requirements:BuiltinExampleRequirements::NONE,
            verification:BuiltinExampleVerification::Assertions{source:"assert(isa(intcon,'IntegerConstraint')); assert(intcon.Integer==1); assert(isequal(intcon.SemiContinuous,[2 3])); assert(intcon.SemiInteger==4);"}},
        BuiltinExample{id:"zero-or-interval",title:"Keep zero in a semicontinuous domain",program:"intcon=integerConstraint('SemiContinuous',1);\noptions=optimoptions('intlinprog','Display','off');\n[x,fval,flag]=intlinprog(1,intcon,[],[],[],[],2,5,[],options)",display_output:None,
            compatibility:BuiltinExampleCompatibility::RunMat,harness:BuiltinExampleHarness::Portable,fixture:BuiltinExampleFixture::None,requirements:BuiltinExampleRequirements::NONE,
            verification:BuiltinExampleVerification::Assertions{source:"assert(flag==1); assert(x==0); assert(fval==0);"}},
    ],example_exemption:None,faqs:&[],
    links:&[BuiltinDocumentationLink{label:"MATLAB integerConstraint",target:BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/optim/ug/integerconstraint.html")}],media:&[],
    evidence:BuiltinDocumentationEvidence{implementation:&[BuiltinDocumentationLink{label:"Runtime constructor and schema",target:BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/optim/integer_constraint.rs")}],verification:&[BuiltinEvidenceReference{kind:BuiltinEvidenceKind::UnitTest,label:"Domain object and disjunction controls",location:"builtins::math::optim::integer_constraint::tests"}],notes:&[]},
    introduced:None,status:Some(BuiltinDocumentationStatus::Partial),
};
