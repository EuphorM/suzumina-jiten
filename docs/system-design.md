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
モデレーターが Issue の内容を確認・審査する
    ↓
承認 → src/content/quotes/ にMarkdownファイルを作成してコミット
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

## 環境変数

Vercel のプロジェクト設定と、ローカル開発時の `.env` ファイルに以下を設定する。

| 変数名 | 説明 |
|---|---|
| `GITHUB_TOKEN` | GitHub Personal Access Token（Issues: Read and write 権限） |
| `GITHUB_OWNER` | GitHubのユーザー名（例: EuphorM） |
| `GITHUB_REPO` | リポジトリ名（suzumina-jiten） |

## デプロイ

`main` ブランチへのコミット・プッシュで Vercel が自動的にビルド・デプロイを実行する。特別な操作は不要。
