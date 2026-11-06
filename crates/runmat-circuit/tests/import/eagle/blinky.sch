<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE eagle SYSTEM "eagle.dtd">
<!-- A hand-written test fixture: a header, a resistor and an LED. Written for RunMat's tests from the public format description; no third-party content. -->
<eagle version="9.6.2">
<drawing>
<settings><setting alwaysvectorfont="no"/></settings>
<grid distance="0.1" unitdist="inch" unit="inch"/>
<layers>
<layer number="1" name="Top" color="4" fill="1" visible="yes" active="yes"/>
<layer number="16" name="Bottom" color="1" fill="1" visible="yes" active="yes"/>
<layer number="20" name="Dimension" color="24" fill="1" visible="yes" active="yes"/>
<layer number="21" name="tPlace" color="7" fill="1" visible="yes" active="yes"/>
</layers>
<schematic>
<libraries>
<library name="passives">
<packages>
<package name="0603">
<description>Chip resistor, 0603</description>
<smd name="1" x="-0.8" y="0" dx="0.9" dy="0.95" layer="1"/>
<smd name="2" x="0.8" y="0" dx="0.9" dy="0.95" layer="1"/>
<wire x1="-1.5" y1="0.7" x2="1.5" y2="0.7" width="0.12" layer="21"/>
<wire x1="-1.5" y1="-0.7" x2="1.5" y2="-0.7" width="0.12" layer="21"/>
<text x="0" y="1.2" size="0.8" layer="25">&gt;NAME</text>
</package>
</packages>
<symbols>
<symbol name="R">
<pin name="1" x="-5.08" y="0" length="short" direction="pas"/>
<pin name="2" x="5.08" y="0" length="short" direction="pas" rot="R180"/>
</symbol>
</symbols>
<devicesets>
<deviceset name="R" prefix="R">
<description>&lt;b&gt;Resistor&lt;/b&gt; thick film</description>
<gates><gate name="G$1" symbol="R" x="0" y="0"/></gates>
<devices>
<device name="0603" package="0603">
<connects><connect gate="G$1" pin="1" pad="1"/><connect gate="G$1" pin="2" pad="2"/></connects>
<technologies><technology name=""/></technologies>
</device>
</devices>
</deviceset>
</devicesets>
</library>
<library name="leds">
<packages>
<package name="LED0603">
<smd name="A" x="-0.8" y="0" dx="0.9" dy="0.95" layer="1"/>
<smd name="C" x="0.8" y="0" dx="0.9" dy="0.95" layer="1" roundness="50"/>
<circle x="-1.6" y="0" radius="0.15" width="0" layer="21"/>
</package>
</packages>
<symbols>
<symbol name="LED">
<pin name="A" x="-5.08" y="0" length="short" direction="pas"/>
<pin name="C" x="5.08" y="0" length="short" direction="pas" rot="R180"/>
</symbol>
</symbols>
<devicesets>
<deviceset name="LED" prefix="D">
<gates><gate name="G$1" symbol="LED" x="0" y="0"/></gates>
<devices>
<device name="0603" package="LED0603">
<connects><connect gate="G$1" pin="A" pad="A"/><connect gate="G$1" pin="C" pad="C"/></connects>
<technologies><technology name="RED"><attribute name="MPN" value="LED-RED-0603"/><attribute name="COLOR" value="red"/></technology></technologies>
</device>
</devices>
</deviceset>
</devicesets>
</library>
<library name="connectors">
<packages>
<package name="HDR2">
<pad name="1" x="-1.27" y="0" drill="1" diameter="1.7" shape="square"/>
<pad name="2" x="1.27" y="0" drill="1"/>
<wire x1="-2.54" y1="1.27" x2="2.54" y2="1.27" width="0.15" layer="21"/>
</package>
</packages>
<symbols>
<symbol name="HDR2">
<pin name="1" x="-7.62" y="0" length="middle"/>
<pin name="2" x="-7.62" y="-2.54" length="middle"/>
</symbol>
</symbols>
<devicesets>
<deviceset name="HDR2" prefix="J">
<gates><gate name="G$1" symbol="HDR2" x="0" y="0"/></gates>
<devices>
<device name="" package="HDR2">
<connects><connect gate="G$1" pin="1" pad="1"/><connect gate="G$1" pin="2" pad="2"/></connects>
<technologies><technology name=""><attribute name="MPN" value="HDR-1X2-2.54"/></technology></technologies>
</device>
</devices>
</deviceset>
</devicesets>
</library>
<library name="supply">
<symbols>
<symbol name="VCC"><pin name="VCC" x="0" y="-2.54" visible="off" length="short" direction="sup" rot="R90"/></symbol>
<symbol name="GND"><pin name="GND" x="0" y="2.54" visible="off" length="short" direction="sup" rot="R270"/></symbol>
</symbols>
<devicesets>
<deviceset name="VCC" prefix="SUPPLY"><gates><gate name="G$1" symbol="VCC" x="0" y="0"/></gates><devices><device name=""><technologies><technology name=""/></technologies></device></devices></deviceset>
<deviceset name="GND" prefix="SUPPLY"><gates><gate name="G$1" symbol="GND" x="0" y="0"/></gates><devices><device name=""><technologies><technology name=""/></technologies></device></devices></deviceset>
</devicesets>
</library>
</libraries>
<variantdefs><variantdef name="lite"/></variantdefs>
<classes>
<class number="0" name="default" width="0" drill="0"/>
<class number="1" name="power" width="0.4" drill="0"><clearance class="1" value="0.2"/></class>
</classes>
<parts>
<part name="J1" library="connectors" deviceset="HDR2" device=""/>
<part name="R1" library="passives" deviceset="R" device="0603" value="1k"/>
<part name="D1" library="leds" deviceset="LED" device="0603" technology="RED" value="red"><variant name="lite" populate="no"/></part>
<part name="SUP1" library="supply" deviceset="VCC" device=""/>
<part name="SUP2" library="supply" deviceset="GND" device=""/>
</parts>
<sheets>
<sheet>
<instances>
<instance part="J1" gate="G$1" x="10" y="20"/>
<instance part="R1" gate="G$1" x="30" y="25"/>
<instance part="D1" gate="G$1" x="45" y="25"/>
<instance part="SUP1" gate="G$1" x="20" y="30"/>
<instance part="SUP2" gate="G$1" x="20" y="10"/>
</instances>
<nets>
<net name="VCC" class="1">
<segment><pinref part="SUP1" gate="G$1" pin="VCC"/><pinref part="J1" gate="G$1" pin="1"/><wire x1="20" y1="27.46" x2="20" y2="20" width="0.1524" layer="91"/></segment>
<segment><pinref part="R1" gate="G$1" pin="1"/><label x="24.92" y="25" size="1.778" layer="95"/></segment>
</net>
<net name="N$1" class="0">
<segment><pinref part="R1" gate="G$1" pin="2"/><pinref part="D1" gate="G$1" pin="A"/><wire x1="35.08" y1="25" x2="39.92" y2="25" width="0.1524" layer="91"/></segment>
</net>
</nets>
</sheet>
<sheet>
<nets>
<net name="GND" class="0">
<segment><pinref part="SUP2" gate="G$1" pin="GND"/><pinref part="J1" gate="G$1" pin="2"/></segment>
<segment><pinref part="D1" gate="G$1" pin="C"/></segment>
</net>
</nets>
</sheet>
</sheets>
</schematic>
</drawing>
</eagle>
