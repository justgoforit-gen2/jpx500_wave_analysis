# LINEへ日次サマリ通知（自分のLINE）セットアップ

このプロジェクトは GitHub Actions で日次更新を実行し、結果サマリを LINE Messaging API の Push で送信します。

## 1) LINE Developers 側の準備

1. LINE Developers にログインして Provider を作成
2. 「Messaging API」チャネルを作成
3. Messaging API 画面で以下を確認/設定
   - **Channel access token (long-lived)** を発行（これが `LINE_CHANNEL_ACCESS_TOKEN`）
   - 「Push messages」を **有効**（表示/権限は画面表記に従ってON）

## 2) 自分の userId（LINE_TO）を取得

Push 送信には「あなたの userId」が必要です。これは **ボットがあなたからのイベントを受け取って初めて分かる** ため、最初だけWebhookで1回取得します。

最短手順（外部のHTTP受け口を使う）:

1. Pipedream（または RequestBin / webhook.site）で **HTTP受信エンドポイント** を作成
   - Pipedream例: 「HTTP / Webhook」トリガーを作るだけでOK
2. LINE Developers → Messaging API → Webhook settings
   - Webhook URL に (1) のURLを設定
   - 「Use webhook」をON
   - 「Verify」を実行（成功すればOK）
3. LINEアプリで、作成したボット（公式アカウント）を **友だち追加**
4. ボットに何でもいいので1通メッセージを送る
5. Pipedream側の受信ログ（payload）から `source.userId` を探す
   - 例: `events[0].source.userId`
   - これが `LINE_TO`

取得が終わったら、Webhook URL は残しても消しても構いません（Push通知だけなら不要です）。

## 3) GitHub に Secret/Variable を設定

GitHub リポジトリ → Settings → Secrets and variables → Actions

- Secrets
  - `LINE_CHANNEL_ACCESS_TOKEN`: 1) で発行したトークン
  - `LINE_TO`: 2) で取得した userId
- Variables（任意）
  - `STREAMLIT_APP_URL`: Streamlit Community Cloud のアプリURL（通知文末に表示）

## 4) 動作確認

GitHub → Actions → "JPX500 Daily Update + LINE Notify" → Run workflow

- 成功すると自分のLINEに「JPX500 日次更新」サマリが届きます
- スケジュール実行は毎日 07:00 JST です

## Streamlit Community Cloud 側が更新されない場合

Streamlit Community Cloud は、あなたのPCのタスクスケジューラで生成した `data/results.csv` などを自動では参照できません。
Cloudに反映させるには **GitHub上のリポジトリに更新済みCSVがpushされる必要**があります。

このリポジトリでは、GitHub Actions が日次更新後に以下を自動でコミット＆pushするようにしてあります（pushによりStreamlit Cloudが再デプロイし、最新CSVに追随します）。

- `jpx500_wave_analysis/data/results.csv`
- `jpx500_wave_analysis/data/daily_picks.csv`
- `jpx500_wave_analysis/data/earnings/earnings_dates.csv`（存在する場合）

注意:
- `main` ブランチが保護されていて Actions からの push が禁止されている場合は失敗します。その場合はブランチ保護を調整するか、PR作成方式に変更します。
