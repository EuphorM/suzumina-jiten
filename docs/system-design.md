# システム設計書

## アーキテクチャ概要

```
GitHub リポジトリ（ソースコード・語録データ）
    ↓ push するたびに自動ビルド・デプロイ
Vercel（ホスティング）
    ↓ 公開
すずみな語録辞典サイト
```

| 役割 | 使用技術 |
|---|---|
| フレームワーク | Astro |
| ホスティング | Vercel |
| コンテンツ管理 | Markdownファイル（GitHubリポジトリ） |
| 投稿受付 | Vercel Serverless Function → GitHub API |

## 投稿フロー

```
視聴者がフォームに入力して送信
    ↓
Vercel の API（/api/submit）が GitHub API を呼び出す
    ↓
GitHub Issue が自動作成される（ラベル: submission）
    ↓
GitHub Actions が自動でMDドラフトのブランチ・PRを作成する
    ↓
モデレーターが PR の内容を確認・修正（ファイル名変更・画像追加など）する
    ↓
承認 → PRをマージ
    ↓
Vercel が自動でビルド・デプロイ
    ↓
サイトに語録が公開される
```

## ディレクトリ構成

```
suzumina-jiten/
├── .github/                     # GitHubの設定
│   ├── ISSUE_TEMPLATE/          # Issueテンプレート
│   ├── workflows/               # GitHub Actions（自動PR生成など）
│   └── PULL_REQUEST_TEMPLATE.md
├── docs/                        # ドキュメント
├── public/
│   └── images/
│       └── quotes/              # 語録の画像ファイル
├── src/
│   ├── content/
│   │   └── quotes/              # 語録データ（Markdownファイル）
│   ├── components/              # UIパーツ
│   ├── layouts/                 # ページ共通レイアウト
│   ├── pages/
│   │   ├── index.astro          # トップページ
│   │   ├── about.astro          # このサイトについて
│   │   ├── quotes/              # 語録一覧・詳細ページ
│   │   ├── submit.astro         # 投稿フォーム
│   │   └── api/submit.ts        # 投稿受付API
│   └── styles/
│       └── global.css           # サイト全体のスタイル
├── .env.example                 # 環境変数のテンプレート
├── astro.config.mjs             # Astro設定
└── package.json
```

## 語録データの構造

語録は `src/content/quotes/` 以下のMarkdownファイルで管理する。ファイル名はローマ字表記。

主なフィールド：

| フィールド | 説明 |
|---|---|
| `title` | 語録のタイトル |
| `reading` | ひらがなの読み |
| `meaning` | 一言で意味を説明 |
| `tags` | 任意のタグ（配列） |
| `rarity` | レア度 1〜5 |
| `updated_at` | 掲載日・更新日（YYYY-MM-DD）。MDファイル作成時に入力。トップページの新着順に使用 |
| `first_date` | 後方互換のため保持。非表示。新規ファイルでは空欄でよい |
| `first_appearance` | 初出の配信タイトル。未設定時はサイト上に「調査中」と表示 |
| `youtube_url` | 後方互換のため保持。非表示。新規ファイルでは `youtube_urls` を使う |
| `image` | 配信シーンのスクリーンショット（`public/images/quotes/` に配置） |
| `usage` | 使用例。`{text, date}` 形式。dateがある場合は吹き出しに引用元を表示 |
| `contributor` | 情報提供者のハンドルネーム（配列） |
| `youtube_urls` | 関連動画。`{label, url}` 形式 |

## 環境変数

Vercel のプロジェクト設定と、ローカル開発時の `.env` ファイルに以下を設定する。

| 変数名 | 説明 |
|---|---|
| `GITHUB_TOKEN` | GitHub Personal Access Token（Issues: Read and write 権限） |
| `GITHUB_OWNER` | GitHubのユーザー名（例: EuphorM） |
| `GITHUB_REPO` | リポジトリ名（suzumina-jiten） |

## デプロイ

`main` ブランチへのコミット・プッシュで Vercel が自動的にビルド・デプロイを実行する。特別な操作は不要。
