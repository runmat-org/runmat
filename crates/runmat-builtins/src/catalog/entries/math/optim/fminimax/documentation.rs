use crate::*;

pub(super) const DOCUMENTATION:BuiltinDocumentation=BuiltinDocumentation{
authority:BuiltinDocumentationAuthority::Catalog,
title:Some("fminimax"),
slug:Some("fminimax"),
summary:"Minimize the largest of a set of constrained objective values.",
description:"fminimax solves a smooth goal-attainment epigraph for a vector objective, retaining every original constraint. The solver uses genuine constrained SQP, not a fixed penalty or a nonsmooth max callback.",
keywords:&["fminimax",
"multiobjective",
"epigraph",
"goal attainment",
"optimization"],
related:&["fmincon",
"fgoalattain",
"optimoptions",
"optimset"],
sections:super::super::multiobjective_documentation::SECTIONS,

examples:&[BuiltinExample{
id:"balanced-maximum",
title:"Minimize the larger quadratic",
program:"fun=@(x)[(x-1)^2 (x+1)^2];\n[x,F,m,flag]=fminimax(fun,2)",
display_output:None,
compatibility:BuiltinExampleCompatibility::RunMat,
harness:BuiltinExampleHarness::Portable,
fixture:BuiltinExampleFixture::None,
requirements:BuiltinExampleRequirements::NONE,
verification:BuiltinExampleVerification::Assertions{
source:"assert(flag>0); assert(abs(x)<1e-4); assert(norm(F-[1 1],Inf)<1e-4); assert(abs(m-1)<1e-4); assert(isequal(size(F),[1 2]));"
}
},

BuiltinExample{
id:"absolute-subset",
title:"Minimize an absolute objective while reporting original F",
program:"options=optimoptions('fminimax','AbsoluteMaxObjectiveCount',1,'Display','off');\n[x,F,m,flag]=fminimax(@(x)[x;-5],-1.5,[],[],[],[],-2,-1,[],options)",
display_output:None,
compatibility:BuiltinExampleCompatibility::RunMat,
harness:BuiltinExampleHarness::Portable,
fixture:BuiltinExampleFixture::None,
requirements:BuiltinExampleRequirements::NONE,
verification:BuiltinExampleVerification::Assertions{
source:"assert(flag>0); assert(abs(x+1)<1e-4); assert(norm(F-[-1;-5],Inf)<1e-4); assert(abs(m+1)<1e-4);"
}
}],
example_exemption:None,
faqs:&[],
links:&[BuiltinDocumentationLink{
label:"MATLAB fminimax",
target:BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/optim/ug/fminimax.html")
},
BuiltinDocumentationLink{
label:"Multiobjective algorithms",
target:BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/optim/ug/multiobjective-optimization-algorithms.html")
}],
media:&[],
evidence:BuiltinDocumentationEvidence{
implementation:&[BuiltinDocumentationLink{
label:"Shared constrained epigraph solver",
target:BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/optim/fmincon/multiobjective/mod.rs")
}],
verification:&[BuiltinEvidenceReference{
kind:BuiltinEvidenceKind::UnitTest,
label:"Original vector shapes, epigraph constraints, derivatives and callback budgets",
location:"builtins::math::optim::fmincon::multiobjective::tests"
}],
notes:&[]
},
introduced:None,
status:Some(BuiltinDocumentationStatus::Partial)
};
