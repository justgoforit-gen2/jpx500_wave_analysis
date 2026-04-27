# Implementation Notes — JPX500 × naibu 統合の実装ログ

**正本**: `naibu-ryuho-app/IMPLEMENTATION_NOTES.md` を参照。本書は jpx500側で発見した事項のみ追記する従ファイル。

---

## jpx500側で関係する重要事項（要約）

詳細は正本 [naibu-ryuho-app/IMPLEMENTATION_NOTES.md](../../python_projects/dify_projects/naibu-ryuho-app/IMPLEMENTATION_NOTES.md) を参照。

### 証券コード形式
- 当アプリ（jpx500）: **4桁ティッカー**（例: `7203`、`results.csv`/`jpx500_list.csv`/API パス全般）
- naibu 側: **5桁 EDINET形式**（例: `72030`、末尾"0"付与）
- **変換責務は naibu 側**。jpx500 は4桁を維持。

### API表面
- 提供エンドポイントは [api_server.py](api_server.py) 参照。`/api/wave/{code}` 等は引き続き4桁前提。
- `results.csv` の `wave_types` 列は `|` 区切りで複数ラベルが入る（例: `"上昇トレンド|ブレイク気味"`）。

### ETFが100件含まれる
- `data/jpx500_list.csv` 全598行 = JPX500構成498社 + ETF 100行。
- 当アプリ的には全部 universe扱いで OK（テクニカル分析の対象として有効）。
- naibu 側で companies と結合する際は ETF が自然に脱落（EDINET フィリング無し）。
