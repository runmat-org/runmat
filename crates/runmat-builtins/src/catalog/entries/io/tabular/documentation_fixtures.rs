//! Deterministic, minimal XLSX inputs for the tabular documentation examples.
//! Stored ZIP/XML structure follows table::tests::write_minimal_xlsx.
use crate::*;

const COUNTERS_BYTES: &[u8] = b"PK\x03\x04\x14\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1ct\xfc8\xf6\x01\x00\x00\xf6\x01\x00\
    \x00\x13\x00\x00\x00[Content_Types].xml<Types xmlns=\"http://schemas.openxmlformats.org/\
    package/2006/content-types\"><Default Extension=\"rels\" ContentType=\"application/vnd.o\
    penxmlformats-package.relationships+xml\"/><Default Extension=\"xml\" ContentType=\"appl\
    ication/xml\"/><Override PartName=\"/xl/workbook.xml\" ContentType=\"application/vnd.ope\
    nxmlformats-officedocument.spreadsheetml.sheet.main+xml\"/><Override PartName=\"/xl/work\
    sheets/sheet1.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.spreadsh\
    eetml.worksheet+xml\"/></Types>PK\x03\x04\x14\x00\x00\x00\x00\x00\x00\x00\x00\x00\xfe[\
    \x86r\xf0\x00\x00\x00\xf0\x00\x00\x00\x0b\x00\x00\x00_rels/.rels<Relationships xmlns=\"h\
    ttp://schemas.openxmlformats.org/package/2006/relationships\"><Relationship Id=\"rId1\" \
    Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocumen\
    t\" Target=\"xl/workbook.xml\"/></Relationships>PK\x03\x04\x14\x00\x00\x00\x00\x00\x00\
    \x00\x00\x00u\xf3\xd4\x1d\xe2\x00\x00\x00\xe2\x00\x00\x00\x0f\x00\x00\x00xl/workbook.xml\
    <workbook xmlns=\"http://schemas.openxmlformats.org/spreadsheetml/2006/main\" xmlns:r=\"\
    http://schemas.openxmlformats.org/officeDocument/2006/relationships\"><sheets><sheet nam\
    e=\"Data\" sheetId=\"1\" r:id=\"rId1\"/></sheets></workbook>PK\x03\x04\x14\x00\x00\x00\
    \x00\x00\x00\x00\x00\x00\xb6T\xaa\xcf\xf1\x00\x00\x00\xf1\x00\x00\x00\x1a\x00\x00\x00xl/\
    _rels/workbook.xml.rels<Relationships xmlns=\"http://schemas.openxmlformats.org/package/\
    2006/relationships\"><Relationship Id=\"rId1\" Type=\"http://schemas.openxmlformats.org/\
    officeDocument/2006/relationships/worksheet\" Target=\"worksheets/sheet1.xml\"/></Relati\
    onships>PK\x03\x04\x14\x00\x00\x00\x00\x00\x00\x00\x00\x00\xa26\xf5\x89\x0e\x01\x00\x00\
    \x0e\x01\x00\x00\x18\x00\x00\x00xl/worksheets/sheet1.xml<worksheet xmlns=\"http://schema\
    s.openxmlformats.org/spreadsheetml/2006/main\"><sheetData><row r=\"1\"><c r=\"A1\" t=\"i\
    nlineStr\"><is><t>9007199254740993</t></is></c></row><row r=\"2\"><c r=\"A2\" t=\"inline\
    Str\"><is><t>18446744073709551615</t></is></c></row></sheetData></worksheet>PK\x01\x02\
    \x14\x00\x14\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1ct\xfc8\xf6\x01\x00\x00\xf6\x01\x00\
    \x00\x13\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00[Content_Typ\
    es].xmlPK\x01\x02\x14\x00\x14\x00\x00\x00\x00\x00\x00\x00\x00\x00\xfe[\x86r\xf0\x00\x00\
    \x00\xf0\x00\x00\x00\x0b\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'\x02\x00\
    \x00_rels/.relsPK\x01\x02\x14\x00\x14\x00\x00\x00\x00\x00\x00\x00\x00\x00u\xf3\xd4\x1d\
    \xe2\x00\x00\x00\xe2\x00\x00\x00\x0f\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\
    @\x03\x00\x00xl/workbook.xmlPK\x01\x02\x14\x00\x14\x00\x00\x00\x00\x00\x00\x00\x00\x00\
    \xb6T\xaa\xcf\xf1\x00\x00\x00\xf1\x00\x00\x00\x1a\x00\x00\x00\x00\x00\x00\x00\x00\x00\
    \x00\x00\x00\x00O\x04\x00\x00xl/_rels/workbook.xml.relsPK\x01\x02\x14\x00\x14\x00\x00\
    \x00\x00\x00\x00\x00\x00\x00\xa26\xf5\x89\x0e\x01\x00\x00\x0e\x01\x00\x00\x18\x00\x00\
    \x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00x\x05\x00\x00xl/worksheets/sheet1.xmlPK\x05\
    \x06\x00\x00\x00\x00\x05\x00\x05\x00E\x01\x00\x00\xbc\x06\x00\x00\x00\x00";
pub(super) const COUNTERS: BuiltinExampleFixture =
    BuiltinExampleFixture::Filesystem(BuiltinFilesystemFixture {
        id: BuiltinExampleFixtureId {
            local_name: "tabular-counters",
        },
        root: BuiltinFilesystemRoot::IsolatedWorkspace,
        entries: &[BuiltinFilesystemEntry::File {
            relative_path: "counters.xlsx",
            content: BuiltinFixtureContent::Bytes(COUNTERS_BYTES),
        }],
    });

const AMOUNTS_BYTES: &[u8] = b"PK\x03\x04\x14\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1ct\xfc8\xf6\x01\x00\x00\xf6\x01\x00\
    \x00\x13\x00\x00\x00[Content_Types].xml<Types xmlns=\"http://schemas.openxmlformats.org/\
    package/2006/content-types\"><Default Extension=\"rels\" ContentType=\"application/vnd.o\
    penxmlformats-package.relationships+xml\"/><Default Extension=\"xml\" ContentType=\"appl\
    ication/xml\"/><Override PartName=\"/xl/workbook.xml\" ContentType=\"application/vnd.ope\
    nxmlformats-officedocument.spreadsheetml.sheet.main+xml\"/><Override PartName=\"/xl/work\
    sheets/sheet1.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.spreadsh\
    eetml.worksheet+xml\"/></Types>PK\x03\x04\x14\x00\x00\x00\x00\x00\x00\x00\x00\x00\xfe[\
    \x86r\xf0\x00\x00\x00\xf0\x00\x00\x00\x0b\x00\x00\x00_rels/.rels<Relationships xmlns=\"h\
    ttp://schemas.openxmlformats.org/package/2006/relationships\"><Relationship Id=\"rId1\" \
    Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocumen\
    t\" Target=\"xl/workbook.xml\"/></Relationships>PK\x03\x04\x14\x00\x00\x00\x00\x00\x00\
    \x00\x00\x00u\xf3\xd4\x1d\xe2\x00\x00\x00\xe2\x00\x00\x00\x0f\x00\x00\x00xl/workbook.xml\
    <workbook xmlns=\"http://schemas.openxmlformats.org/spreadsheetml/2006/main\" xmlns:r=\"\
    http://schemas.openxmlformats.org/officeDocument/2006/relationships\"><sheets><sheet nam\
    e=\"Data\" sheetId=\"1\" r:id=\"rId1\"/></sheets></workbook>PK\x03\x04\x14\x00\x00\x00\
    \x00\x00\x00\x00\x00\x00\xb6T\xaa\xcf\xf1\x00\x00\x00\xf1\x00\x00\x00\x1a\x00\x00\x00xl/\
    _rels/workbook.xml.rels<Relationships xmlns=\"http://schemas.openxmlformats.org/package/\
    2006/relationships\"><Relationship Id=\"rId1\" Type=\"http://schemas.openxmlformats.org/\
    officeDocument/2006/relationships/worksheet\" Target=\"worksheets/sheet1.xml\"/></Relati\
    onships>PK\x03\x04\x14\x00\x00\x00\x00\x00\x00\x00\x00\x00\xfb\xa97\xfd)\x11\x00\x00)\
    \x11\x00\x00\x18\x00\x00\x00xl/worksheets/sheet1.xml<worksheet xmlns=\"http://schemas.op\
    enxmlformats.org/spreadsheetml/2006/main\"><sheetData><row r=\"1\"><c r=\"A1\" t=\"inlin\
    eStr\"><is><t>Item</t></is></c><c r=\"B1\" t=\"inlineStr\"><is><t>Value</t></is></c><c r\
    =\"C1\" t=\"inlineStr\"><is><t>Amount</t></is></c></row><row r=\"2\"><c r=\"A2\" t=\"inl\
    ineStr\"><is><t>row1</t></is></c><c r=\"B2\"><v>1</v></c><c r=\"C2\"><v>1</v></c></row><\
    row r=\"3\"><c r=\"A3\" t=\"inlineStr\"><is><t>row2</t></is></c><c r=\"B3\"><v>2</v></c>\
    <c r=\"C3\"><v>2</v></c></row><row r=\"4\"><c r=\"A4\" t=\"inlineStr\"><is><t>row3</t></\
    is></c><c r=\"B4\"><v>3</v></c><c r=\"C4\"><v>3</v></c></row><row r=\"5\"><c r=\"A5\" t=\
    \"inlineStr\"><is><t>row4</t></is></c><c r=\"B5\"><v>4</v></c><c r=\"C5\"><v>4</v></c></\
    row><row r=\"6\"><c r=\"A6\" t=\"inlineStr\"><is><t>row5</t></is></c><c r=\"B6\"><v>5</v\
    ></c><c r=\"C6\"><v>5</v></c></row><row r=\"7\"><c r=\"A7\" t=\"inlineStr\"><is><t>row6<\
    /t></is></c><c r=\"B7\"><v>6</v></c><c r=\"C7\"><v>6</v></c></row><row r=\"8\"><c r=\"A8\
    \" t=\"inlineStr\"><is><t>row7</t></is></c><c r=\"B8\"><v>7</v></c><c r=\"C8\"><v>7</v><\
    /c></row><row r=\"9\"><c r=\"A9\" t=\"inlineStr\"><is><t>row8</t></is></c><c r=\"B9\"><v\
    >8</v></c><c r=\"C9\"><v>8</v></c></row><row r=\"10\"><c r=\"A10\" t=\"inlineStr\"><is><\
    t>row9</t></is></c><c r=\"B10\"><v>9</v></c><c r=\"C10\"><v>9</v></c></row><row r=\"11\"\
    ><c r=\"A11\" t=\"inlineStr\"><is><t>row10</t></is></c><c r=\"B11\"><v>10</v></c><c r=\"\
    C11\"><v>10</v></c></row><row r=\"12\"><c r=\"A12\" t=\"inlineStr\"><is><t>row11</t></is\
    ></c><c r=\"B12\"><v>11</v></c><c r=\"C12\"><v>11</v></c></row><row r=\"13\"><c r=\"A13\
    \" t=\"inlineStr\"><is><t>row12</t></is></c><c r=\"B13\"><v>12</v></c><c r=\"C13\"><v>12\
    </v></c></row><row r=\"14\"><c r=\"A14\" t=\"inlineStr\"><is><t>row13</t></is></c><c r=\
    \"B14\"><v>13</v></c><c r=\"C14\"><v>13</v></c></row><row r=\"15\"><c r=\"A15\" t=\"inli\
    neStr\"><is><t>row14</t></is></c><c r=\"B15\"><v>14</v></c><c r=\"C15\"><v>14</v></c></r\
    ow><row r=\"16\"><c r=\"A16\" t=\"inlineStr\"><is><t>row15</t></is></c><c r=\"B16\"><v>1\
    5</v></c><c r=\"C16\"><v>15</v></c></row><row r=\"17\"><c r=\"A17\" t=\"inlineStr\"><is>\
    <t>row16</t></is></c><c r=\"B17\"><v>16</v></c><c r=\"C17\"><v>16</v></c></row><row r=\"\
    18\"><c r=\"A18\" t=\"inlineStr\"><is><t>row17</t></is></c><c r=\"B18\"><v>17</v></c><c \
    r=\"C18\"><v>17</v></c></row><row r=\"19\"><c r=\"A19\" t=\"inlineStr\"><is><t>row18</t>\
    </is></c><c r=\"B19\"><v>18</v></c><c r=\"C19\"><v>18</v></c></row><row r=\"20\"><c r=\"\
    A20\" t=\"inlineStr\"><is><t>row19</t></is></c><c r=\"B20\"><v>19</v></c><c r=\"C20\"><v\
    >19</v></c></row><row r=\"21\"><c r=\"A21\" t=\"inlineStr\"><is><t>row20</t></is></c><c \
    r=\"B21\"><v>20</v></c><c r=\"C21\"><v>20</v></c></row><row r=\"22\"><c r=\"A22\" t=\"in\
    lineStr\"><is><t>row21</t></is></c><c r=\"B22\"><v>21</v></c><c r=\"C22\"><v>21</v></c><\
    /row><row r=\"23\"><c r=\"A23\" t=\"inlineStr\"><is><t>row22</t></is></c><c r=\"B23\"><v\
    >22</v></c><c r=\"C23\"><v>22</v></c></row><row r=\"24\"><c r=\"A24\" t=\"inlineStr\"><i\
    s><t>row23</t></is></c><c r=\"B24\"><v>23</v></c><c r=\"C24\"><v>23</v></c></row><row r=\
    \"25\"><c r=\"A25\" t=\"inlineStr\"><is><t>row24</t></is></c><c r=\"B25\"><v>24</v></c><\
    c r=\"C25\"><v>24</v></c></row><row r=\"26\"><c r=\"A26\" t=\"inlineStr\"><is><t>row25</\
    t></is></c><c r=\"B26\"><v>25</v></c><c r=\"C26\"><v>25</v></c></row><row r=\"27\"><c r=\
    \"A27\" t=\"inlineStr\"><is><t>row26</t></is></c><c r=\"B27\"><v>26</v></c><c r=\"C27\">\
    <v>26</v></c></row><row r=\"28\"><c r=\"A28\" t=\"inlineStr\"><is><t>row27</t></is></c><\
    c r=\"B28\"><v>27</v></c><c r=\"C28\"><v>27</v></c></row><row r=\"29\"><c r=\"A29\" t=\"\
    inlineStr\"><is><t>row28</t></is></c><c r=\"B29\"><v>28</v></c><c r=\"C29\"><v>28</v></c\
    ></row><row r=\"30\"><c r=\"A30\" t=\"inlineStr\"><is><t>row29</t></is></c><c r=\"B30\">\
    <v>29</v></c><c r=\"C30\"><v>29</v></c></row><row r=\"31\"><c r=\"A31\" t=\"inlineStr\">\
    <is><t>row30</t></is></c><c r=\"B31\"><v>30</v></c><c r=\"C31\"><v>30</v></c></row><row \
    r=\"32\"><c r=\"A32\" t=\"inlineStr\"><is><t>row31</t></is></c><c r=\"B32\"><v>31</v></c\
    ><c r=\"C32\"><v>31</v></c></row><row r=\"33\"><c r=\"A33\" t=\"inlineStr\"><is><t>row32\
    </t></is></c><c r=\"B33\"><v>32</v></c><c r=\"C33\"><v>32</v></c></row><row r=\"34\"><c \
    r=\"A34\" t=\"inlineStr\"><is><t>row33</t></is></c><c r=\"B34\"><v>33</v></c><c r=\"C34\
    \"><v>33</v></c></row><row r=\"35\"><c r=\"A35\" t=\"inlineStr\"><is><t>row34</t></is></\
    c><c r=\"B35\"><v>34</v></c><c r=\"C35\"><v>34</v></c></row><row r=\"36\"><c r=\"A36\" t\
    =\"inlineStr\"><is><t>row35</t></is></c><c r=\"B36\"><v>35</v></c><c r=\"C36\"><v>35</v>\
    </c></row><row r=\"37\"><c r=\"A37\" t=\"inlineStr\"><is><t>row36</t></is></c><c r=\"B37\
    \"><v>36</v></c><c r=\"C37\"><v>36</v></c></row></sheetData></worksheet>PK\x01\x02\x14\
    \x00\x14\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1ct\xfc8\xf6\x01\x00\x00\xf6\x01\x00\x00\
    \x13\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00[Content_Types].\
    xmlPK\x01\x02\x14\x00\x14\x00\x00\x00\x00\x00\x00\x00\x00\x00\xfe[\x86r\xf0\x00\x00\x00\
    \xf0\x00\x00\x00\x0b\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'\x02\x00\x00_re\
    ls/.relsPK\x01\x02\x14\x00\x14\x00\x00\x00\x00\x00\x00\x00\x00\x00u\xf3\xd4\x1d\xe2\x00\
    \x00\x00\xe2\x00\x00\x00\x0f\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00@\x03\
    \x00\x00xl/workbook.xmlPK\x01\x02\x14\x00\x14\x00\x00\x00\x00\x00\x00\x00\x00\x00\xb6T\
    \xaa\xcf\xf1\x00\x00\x00\xf1\x00\x00\x00\x1a\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\
    \x00\x00O\x04\x00\x00xl/_rels/workbook.xml.relsPK\x01\x02\x14\x00\x14\x00\x00\x00\x00\
    \x00\x00\x00\x00\x00\xfb\xa97\xfd)\x11\x00\x00)\x11\x00\x00\x18\x00\x00\x00\x00\x00\x00\
    \x00\x00\x00\x00\x00\x00\x00x\x05\x00\x00xl/worksheets/sheet1.xmlPK\x05\x06\x00\x00\x00\
    \x00\x05\x00\x05\x00E\x01\x00\x00\xd7\x16\x00\x00\x00\x00";
pub(super) const AMOUNTS: BuiltinExampleFixture =
    BuiltinExampleFixture::Filesystem(BuiltinFilesystemFixture {
        id: BuiltinExampleFixtureId {
            local_name: "tabular-amounts",
        },
        root: BuiltinFilesystemRoot::IsolatedWorkspace,
        entries: &[BuiltinFilesystemEntry::File {
            relative_path: "data.xlsx",
            content: BuiltinFixtureContent::Bytes(AMOUNTS_BYTES),
        }],
    });

const WORKBOOK_BYTES: &[u8] = b"PK\x03\x04\x14\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1ct\xfc8\xf6\x01\x00\x00\xf6\x01\x00\
    \x00\x13\x00\x00\x00[Content_Types].xml<Types xmlns=\"http://schemas.openxmlformats.org/\
    package/2006/content-types\"><Default Extension=\"rels\" ContentType=\"application/vnd.o\
    penxmlformats-package.relationships+xml\"/><Default Extension=\"xml\" ContentType=\"appl\
    ication/xml\"/><Override PartName=\"/xl/workbook.xml\" ContentType=\"application/vnd.ope\
    nxmlformats-officedocument.spreadsheetml.sheet.main+xml\"/><Override PartName=\"/xl/work\
    sheets/sheet1.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.spreadsh\
    eetml.worksheet+xml\"/></Types>PK\x03\x04\x14\x00\x00\x00\x00\x00\x00\x00\x00\x00\xfe[\
    \x86r\xf0\x00\x00\x00\xf0\x00\x00\x00\x0b\x00\x00\x00_rels/.rels<Relationships xmlns=\"h\
    ttp://schemas.openxmlformats.org/package/2006/relationships\"><Relationship Id=\"rId1\" \
    Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocumen\
    t\" Target=\"xl/workbook.xml\"/></Relationships>PK\x03\x04\x14\x00\x00\x00\x00\x00\x00\
    \x00\x00\x00u\xf3\xd4\x1d\xe2\x00\x00\x00\xe2\x00\x00\x00\x0f\x00\x00\x00xl/workbook.xml\
    <workbook xmlns=\"http://schemas.openxmlformats.org/spreadsheetml/2006/main\" xmlns:r=\"\
    http://schemas.openxmlformats.org/officeDocument/2006/relationships\"><sheets><sheet nam\
    e=\"Data\" sheetId=\"1\" r:id=\"rId1\"/></sheets></workbook>PK\x03\x04\x14\x00\x00\x00\
    \x00\x00\x00\x00\x00\x00\xb6T\xaa\xcf\xf1\x00\x00\x00\xf1\x00\x00\x00\x1a\x00\x00\x00xl/\
    _rels/workbook.xml.rels<Relationships xmlns=\"http://schemas.openxmlformats.org/package/\
    2006/relationships\"><Relationship Id=\"rId1\" Type=\"http://schemas.openxmlformats.org/\
    officeDocument/2006/relationships/worksheet\" Target=\"worksheets/sheet1.xml\"/></Relati\
    onships>PK\x03\x04\x14\x00\x00\x00\x00\x00\x00\x00\x00\x00\xe6l\x0cb\x98\x01\x00\x00\x98\
    \x01\x00\x00\x18\x00\x00\x00xl/worksheets/sheet1.xml<worksheet xmlns=\"http://schemas.op\
    enxmlformats.org/spreadsheetml/2006/main\"><sheetData><row r=\"1\"><c r=\"A1\" t=\"inlin\
    eStr\"><is><t>Item Name</t></is></c><c r=\"B1\" t=\"inlineStr\"><is><t>Value</t></is></c\
    ></row><row r=\"2\"><c r=\"A2\" t=\"inlineStr\"><is><t>alpha</t></is></c><c r=\"B2\"><v>\
    10</v></c></row><row r=\"3\"><c r=\"A3\" t=\"inlineStr\"><is><t>beta</t></is></c><c r=\"\
    B3\"><v>20</v></c></row></sheetData></worksheet>PK\x01\x02\x14\x00\x14\x00\x00\x00\x00\
    \x00\x00\x00\x00\x00\x1ct\xfc8\xf6\x01\x00\x00\xf6\x01\x00\x00\x13\x00\x00\x00\x00\x00\
    \x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00[Content_Types].xmlPK\x01\x02\x14\x00\
    \x14\x00\x00\x00\x00\x00\x00\x00\x00\x00\xfe[\x86r\xf0\x00\x00\x00\xf0\x00\x00\x00\x0b\
    \x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'\x02\x00\x00_rels/.relsPK\x01\x02\
    \x14\x00\x14\x00\x00\x00\x00\x00\x00\x00\x00\x00u\xf3\xd4\x1d\xe2\x00\x00\x00\xe2\x00\
    \x00\x00\x0f\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00@\x03\x00\x00xl/workbook\
    .xmlPK\x01\x02\x14\x00\x14\x00\x00\x00\x00\x00\x00\x00\x00\x00\xb6T\xaa\xcf\xf1\x00\x00\
    \x00\xf1\x00\x00\x00\x1a\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00O\x04\x00\
    \x00xl/_rels/workbook.xml.relsPK\x01\x02\x14\x00\x14\x00\x00\x00\x00\x00\x00\x00\x00\x00\
    \xe6l\x0cb\x98\x01\x00\x00\x98\x01\x00\x00\x18\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\
    \x00\x00\x00x\x05\x00\x00xl/worksheets/sheet1.xmlPK\x05\x06\x00\x00\x00\x00\x05\x00\x05\
    \x00E\x01\x00\x00F\x07\x00\x00\x00\x00";
pub(super) const WORKBOOK: BuiltinExampleFixture =
    BuiltinExampleFixture::Filesystem(BuiltinFilesystemFixture {
        id: BuiltinExampleFixtureId {
            local_name: "tabular-workbook",
        },
        root: BuiltinFilesystemRoot::IsolatedWorkspace,
        entries: &[BuiltinFilesystemEntry::File {
            relative_path: "workbook.xlsx",
            content: BuiltinFixtureContent::Bytes(WORKBOOK_BYTES),
        }],
    });
