# すずみな語録辞典（非公式）

涼花みなせさんの配信で生まれた語録・印象的なフレーズを体系的に記録・整理するファンサイト（非公式）です。

## 概要

涼花みなせさんは配信において独特なフレーズを多く用います。このサイトはそれらの語録を既存のファンにも新規のファンにも分かりやすい形で提供することを目的としています。視聴者からの語録投稿も受け付けています。

## 技術スタック

- **フレームワーク**: [Astro](https://astro.build/)
- **ホスティング**: [Vercel](https://vercel.com/)
- **コンテンツ管理**: Markdown ファイル（`src/content/quotes/`）
- **投稿受付**: Vercel Serverless Function → GitHub Issues

## 語録の追加方法（モデレーター向け）

`src/content/quotes/` にMarkdownファイルを作成してください。ファイル名はローマ字表記を推奨します。

```markdown
---
title: 語録のタイトル
reading: よみがな
meaning: 意味の説明
tags: [タグ1, タグ2]
rarity: 3
first_appearance: "配信タイトルや日付"
youtube_url: "https://www.youtube.com/..."
---

## 解説

語録が生まれた背景や経緯。

## 使用例

- 実際に使われた文脈や例文
```

**rarity（レア度）の目安**

| 値 | 目安 |
|---|---|
| 1 | よく使う定番フレーズ |
| 2 | たまに使う |
| 3 | 印象的な場面で登場 |
| 4 | なかなか聞けない |
| 5 | 激レア |

## ローカル開発

```bash
# 依存パッケージのインストール
npm install

# 開発サーバー起動
npx astro dev
```

### 環境変数の設定

`.env.example` をコピーして `.env` を作成し、各値を設定してください。

```
GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxx
GITHUB_OWNER=GitHubユーザー名
GITHUB_REPO=suzumina-jiten
```

`GITHUB_TOKEN` は GitHub の Personal Access Token（Fine-grained）で、対象リポジトリの **Issues: Read and write** 権限が必要です。

## ライセンス

このサイトは涼花みなせさんの非公式ファンサイトです。掲載内容に関するお問い合わせはIssueにてお知らせください。
