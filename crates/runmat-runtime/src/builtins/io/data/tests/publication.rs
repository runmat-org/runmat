use super::*;

#[test]
fn rejected_manifest_preserves_published_array_objects_across_mutations() {
    let _serial = serial_test_guard();
    let provider = CountingDataUploadProvider::default();
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    futures::executor::block_on(async {
        let payload = DataArrayPayload {
            dtype: "f64".into(),
            shape: vec![2, 2],
            values: crate::data::DataArrayValues::F64(vec![1., 3., 2., 4.]),
            imaginary_values: None,
        };
        let mut budget = None;
        let (data, index) = crate::data::write_array_payload_on(
            &provider,
            root,
            "samples",
            &payload,
            &[1, 1],
            &mut budget,
        )
        .await
        .unwrap();
        let meta = DataArrayMeta {
            dtype: "f64".into(),
            shape: vec![2, 2],
            chunk_shape: vec![1, 1],
            order: "column_major".into(),
            codec: "none".into(),
            data_path: make_rel_data_path(root, &data).unwrap(),
            chunk_index_path: Some(make_rel_data_path(root, &index).unwrap()),
        };
        let manifest = DataManifest {
            schema_version: 1,
            format: "runmat-data".into(),
            dataset_id: "publication-test".into(),
            name: None,
            created_at: now_rfc3339(),
            updated_at: now_rfc3339(),
            arrays: [("samples".into(), meta.clone())].into(),
            attrs: Default::default(),
            txn_sequence: 1,
        };
        crate::data::write_manifest_on(&provider, root, &manifest)
            .await
            .unwrap();
        let index_bytes = provider.read(&index).await.unwrap();
        let chunks: DataChunkIndex = serde_json::from_slice(&index_bytes).unwrap();
        let mut objects = vec![
            (
                root.join("manifest.json"),
                provider.read(&root.join("manifest.json")).await.unwrap(),
            ),
            (
                data,
                provider.read(&root.join(&meta.data_path)).await.unwrap(),
            ),
            (index, index_bytes),
        ];
        for chunk in chunks.chunks {
            let path = root.join(chunk.data_path);
            objects.push((path.clone(), provider.read(&path).await.unwrap()));
        }
        provider
            .reject_manifest_writes
            .store(true, std::sync::atomic::Ordering::SeqCst);
        let rhs = Value::Tensor(Tensor::new(vec![9.; 4], vec![2, 2]).unwrap());
        let slice = Value::Cell(
            CellArray::new(
                vec![
                    Value::Tensor(Tensor::new(vec![1., 2.], vec![1, 2]).unwrap()),
                    Value::Tensor(Tensor::new(vec![1., 2.], vec![1, 2]).unwrap()),
                ],
                1,
                2,
            )
            .unwrap(),
        );
        for mutation in 0..5 {
            let mut candidate = manifest.clone();
            match mutation {
                0 | 1 => apply_write_to_manifest_on(
                    &provider,
                    root,
                    &mut candidate,
                    "samples",
                    (mutation == 1).then_some(&slice),
                    &rhs,
                    &mut budget,
                )
                .await
                .unwrap(),
                2 => resize_array_in_manifest(
                    &provider,
                    root,
                    &mut candidate,
                    "samples",
                    vec![3, 3],
                    &mut budget,
                )
                .await
                .unwrap(),
                3 => fill_array_in_manifest(
                    &provider,
                    root,
                    &mut candidate,
                    "samples",
                    None,
                    &Value::Num(8.),
                    &mut budget,
                )
                .await
                .unwrap(),
                4 => delete_array_in_manifest_on(
                    &provider,
                    root,
                    &mut candidate,
                    "samples",
                    &mut budget,
                )
                .await
                .unwrap(),
                _ => unreachable!(),
            }
            crate::data::write_manifest_on(&provider, root, &candidate)
                .await
                .expect_err("head conflict rejects publication");
            for (path, bytes) in &objects {
                assert_eq!(
                    &provider.read(path).await.unwrap(),
                    bytes,
                    "mutation {mutation} changed {}",
                    path.display()
                );
            }
            let published = crate::data::read_array_payload_on(&provider, root, &meta)
                .await
                .unwrap();
            assert_eq!(published.values, payload.values);
        }
    });
}
