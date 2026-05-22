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
├── content/                     ← 運営が編集する場所（エンジニア不要）
│   ├── images/                  # 語録の画像ファイル
│   └── quotes/                  # 語録データ（Markdownファイル）
├── docs/                        # ドキュメント
├── src/                         ← 開発者のみ触る領域
│   ├── content/
│   │   └── quotes/              # YAMLコレクション（旧形式・参照のみ）
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

## 画像パスの仕組み

`astro.config.mjs` に `publicDir: 'content'` を設定しているため、Astroの静的ファイルルートは `content/` ディレクトリになる。

これにより：
- `content/images/foo.jpg` が存在すると、ブラウザからは `/images/foo.jpg` でアクセスできる
- MDファイル内で `/images/foo.jpg` と書くとAstroサイトでは動くが、GitHubのMarkdownプレビューでは画像が表示されない（GitHubはリポジトリルートを基点に解決するため）

**統一ルール：MDファイル内の画像パスは `../images/` の相対パスで記述する。**

相対パスの場合、GitHubは `content/quotes/` を基点に `content/images/` を解決するため、GitHub PreviewでもAstroサイトでも両方で画像が表示される。

## 語録データの構造

語録は `content/quotes/` 以下のMarkdownファイルで管理する。ファイル名はローマ字表記。

主なフィールド（YAMLフロントマター）：

| フィールド | 説明 |
|---|---|
| `title` | 語録のタイトル |
| `reading` | ひらがなの読み |
| `meaning` | 一言で意味を説明 |
| `tags` | 任意のタグ（配列） |
| `rarity` | レア度 1〜5 |
| `origin` | 由来の配信タイトル・URL・日付（MDリンク記法）。不明な場合は `調査中` |
| `updated_at` | 掲載日・更新日（YYYY-MM-DD）。トップページの新着順に使用 |

使用例・関連リンク・情報提供者はMarkdown本文に `## 使用例` `## 関連リンク` `情報提供者：` として記載する。

## スキーマフィールドを変更・追加するときの影響範囲

フィールドの追加・削除・リネームを行う場合は、以下のファイルをすべて確認・更新する。

| ファイル | 対応内容 |
|---|---|
| `src/content.config.ts` | スキーマ定義を変更する |
| `content/quotes/*.md` | フロントマターのキー名を変更する |
| `src/pages/quotes/[slug].astro` | デストラクチャリングと使用箇所を変更する |
| `src/pages/api/submit.ts` | Issue作成時のyamlテンプレートを変更する |
| `.github/workflows/auto-pr-from-issue.yml` | MDドラフト生成スクリプトを確認する |
| `.github/ISSUE_TEMPLATE/quote_proposal.md` | Issueテンプレートを確認する |
| `README.md` | データ構造のサンプルを更新する |
| `docs/content-guide.md` | フィールド説明・テンプレートを更新する |
| `docs/moderator-guide.md` | 記載例を更新する |
| `docs/system-design.md` | フィールド一覧を更新する |

## 環境変数

Vercel のプロジェクト設定と、ローカル開発時の `.env` ファイルに以下を設定する。

| 変数名 | 説明 |
|---|---|
| `GITHUB_TOKEN` | GitHub Personal Access Token（Issues: Read and write 権限） |
| `GITHUB_OWNER` | GitHubのユーザー名（例: EuphorM） |
| `GITHUB_REPO` | リポジトリ名（suzumina-jiten） |

## デプロイ

`main` ブランチへのコミット・プッシュで Vercel が自動的にビルド・デプロイを実行する。特別な操作は不要。
