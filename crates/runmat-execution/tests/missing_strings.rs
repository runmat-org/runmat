use runmat_execution::value::{InlineValue, StringElement, ValueLimits, ValuePayload};

#[test]
fn portable_strings_preserve_presence_json_identity_and_budgets() {
    let cases = [
        StringElement(None),
        "<missing>".into(),
        "<MISSING>".into(),
        "".into(),
        "雪".into(),
    ];
    let mut digests = Vec::new();
    for element in &cases {
        let payload = ValuePayload::Inline(Box::new(InlineValue::String(element.clone())));
        let json = serde_json::to_value(&payload).unwrap();
        assert_eq!(
            json["value"]["value"],
            serde_json::to_value(element).unwrap()
        );
        assert_eq!(
            serde_json::from_value::<ValuePayload>(json).unwrap(),
            payload
        );
        let digest = payload.logical_digest().unwrap();
        assert!(!digests.contains(&digest));
        digests.push(digest);
    }
    let array = ValuePayload::Inline(Box::new(InlineValue::StringArray {
        shape: vec![1, cases.len() as u64],
        values: cases.to_vec(),
    }));
    let json = serde_json::to_value(&array).unwrap();
    assert_eq!(
        json["value"]["value"]["values"],
        serde_json::json!([null, "<missing>", "<MISSING>", "", "雪"])
    );
    assert_eq!(serde_json::from_value::<ValuePayload>(json).unwrap(), array);
    array.validate(ValueLimits::default()).unwrap();
    let missing = ValuePayload::Inline(Box::new(InlineValue::String(StringElement(None))));
    missing
        .validate(ValueLimits {
            max_text_bytes: 0,
            ..ValueLimits::default()
        })
        .unwrap();
    assert!(serde_json::from_value::<StringElement>(serde_json::json!(false)).is_err());
}
