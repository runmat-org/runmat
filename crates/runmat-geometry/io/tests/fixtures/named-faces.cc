#include <BRepPrimAPI_MakeBox.hxx>
#include <TopExp_Explorer.hxx>
#include <BRepGProp.hxx>
#include <GProp_GProps.hxx>
#include <TDocStd_Document.hxx>
#include <XCAFApp_Application.hxx>
#include <XCAFDoc_DocumentTool.hxx>
#include <XCAFDoc_ShapeTool.hxx>
#include <XCAFDoc_ColorTool.hxx>
#include <TDataStd_Name.hxx>
#include <STEPCAFControl_Writer.hxx>
#include <Quantity_Color.hxx>
#include <iostream>
#include <Interface_Static.hxx>
#include <DESTEP_Parameters.hxx>
int main(int, char** argv){
  TopoDS_Shape box=BRepPrimAPI_MakeBox(gp_Pnt(0,0,0),100,20,10).Shape();
  Handle(TDocStd_Document) doc; XCAFApp_Application::GetApplication()->NewDocument("MDTV-XCAF",doc);
  Handle(XCAFDoc_ShapeTool) st=XCAFDoc_DocumentTool::ShapeTool(doc->Main());
  Handle(XCAFDoc_ColorTool) ct=XCAFDoc_DocumentTool::ColorTool(doc->Main());
  TDF_Label root=st->AddShape(box,Standard_False); TDataStd_Name::Set(root,"Bar");
  for(TopExp_Explorer f(box,TopAbs_FACE);f.More();f.Next()){ GProp_GProps p;BRepGProp::SurfaceProperties(f.Current(),p); double x=p.CentreOfMass().X();
    const char* n = x<0.1 ? "Fixed end" : (x>99.9 ? "Loaded end" : nullptr); if(!n) continue;
    TDF_Label fl=st->AddSubShape(root,f.Current()); TDataStd_Name::Set(fl,n); ct->SetColor(fl,Quantity_Color(0.8,0.5,0.2,Quantity_TOC_RGB),XCAFDoc_ColorSurf);}
  Interface_Static::SetIVal("write.stepcaf.subshapes.name",1);
  STEPCAFControl_Writer w; w.SetNameMode(true); w.SetColorMode(true); DESTEP_Parameters prm; prm.InitFromStatic(); prm.WriteSubshapeNames=true; w.Transfer(doc,prm,STEPControl_AsIs); w.Write(argv[1]);
}
