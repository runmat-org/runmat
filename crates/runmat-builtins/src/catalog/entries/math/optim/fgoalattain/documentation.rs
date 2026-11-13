use crate::*;

pub(super) const DOCUMENTATION:BuiltinDocumentation=BuiltinDocumentation{
authority:BuiltinDocumentationAuthority::Catalog,
title:Some("fgoalattain"),
slug:Some("fgoalattain"),
summary:"Find a constrained weighted goal-attainment solution.",
description:"fgoalattain minimizes a smooth attainment factor for a vector objective, retaining original hard constraints and objective identities. The solver uses genuine constrained SQP, not a fixed penalty or a nonsmooth max callback.",
keywords:&["fgoalattain",
"multiobjective",
"epigraph",
"goal attainment",
"optimization"],
related:&["fmincon",
"fminimax",
"optimoptions",
"optimset"],
sections:super::super::multiobjective_documentation::SECTIONS,

examples:&[BuiltinExample{
id:"signed-goals-hard-component",
title:"Combine minimizing, maximizing and hard goals",
program:"fun=@(x)[x;x;x];\n[x,F,a,flag]=fgoalattain(fun,0.5,[0;2;1.5],[1;-1;0],[],[],[],[],0,3)",
display_output:None,
compatibility:BuiltinExampleCompatibility::RunMat,
harness:BuiltinExampleHarness::Portable,
fixture:BuiltinExampleFixture::None,
requirements:BuiltinExampleRequirements::NONE,
verification:BuiltinExampleVerification::Assertions{
source:"assert(flag>0); assert(abs(x-1)<1e-4); assert(norm(F-[1;1;1],Inf)<1e-4); assert(abs(a-1)<1e-4); assert(F(3)<=1.5+1e-6);"
}
},
BuiltinExample{
id:"negative-equality-goal",
title:"Minimize absolute first-goal deviation with a signed weight",
program:"fun=@(x)[x;2-x];\noptions=optimoptions('fgoalattain','EqualityGoalCount',1,'Display','off');\n[x,F,a,flag]=fgoalattain(fun,1.5,[1;0],[-2;1],[],[],[],[],0,3,[],options)",
display_output:None,
compatibility:BuiltinExampleCompatibility::RunMat,
harness:BuiltinExampleHarness::Portable,
fixture:BuiltinExampleFixture::None,
requirements:BuiltinExampleRequirements::NONE,
verification:BuiltinExampleVerification::Assertions{
source:"assert(flag>0); assert(abs(x-5/3)<1e-4); assert(norm(F-[5/3;1/3],Inf)<1e-4); assert(abs(a-1/3)<1e-4);"
}
},
BuiltinExample{
id:"balanced-goals",
title:"Balance two quadratic goals",
program:"fun=@(x)[(x-2)^2;(x+2)^2];\n[x,F,a,flag]=fgoalattain(fun,1,[1;1],[1;1])",
display_output:None,
compatibility:BuiltinExampleCompatibility::RunMat,
harness:BuiltinExampleHarness::Portable,
fixture:BuiltinExampleFixture::None,
requirements:BuiltinExampleRequirements::NONE,
verification:BuiltinExampleVerification::Assertions{
source:"assert(flag>0); assert(abs(x)<1e-4); assert(norm(F-[4;4],Inf)<1e-4); assert(abs(a-3)<1e-4);"
}
},

BuiltinExample{
id:"hard-goal",
title:"Keep a zero-weight goal as a hard constraint",
program:"fun=@(x)[(x-2)^2;x];\n[x,F,a,flag]=fgoalattain(fun,0,[0;1],[1;0],[],[],[],[],0,3)",
display_output:None,
compatibility:BuiltinExampleCompatibility::RunMat,
harness:BuiltinExampleHarness::Portable,
fixture:BuiltinExampleFixture::None,
requirements:BuiltinExampleRequirements::NONE,
verification:BuiltinExampleVerification::Assertions{
source:"assert(flag>0); assert(abs(x-1)<1e-4); assert(norm(F-[1;1],Inf)<1e-4); assert(abs(a-1)<1e-4);"
}
}],
example_exemption:None,
faqs:&[],
links:&[BuiltinDocumentationLink{
label:"MATLAB fgoalattain",
target:BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/optim/ug/fgoalattain.html")
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
