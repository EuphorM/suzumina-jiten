# モデレーター運用マニュアル

## モデレーターの役割

- 視聴者から投稿された語録を審査し、掲載するかどうかを判断する
- 掲載が決まった語録をMarkdownファイルとして追加・公開する
- コンテンツの品質・正確性を維持する

---

## 投稿の確認方法

語録が投稿されると、GitHubのIssueに自動で記録される。

1. `https://github.com/EuphorM/suzumina-jiten/issues` を開く
2. `submission` ラベルのついたIssueが投稿された語録
3. Issueを開いて内容を確認する

---

## 審査の基準

[コンテンツガイドライン](./content-guide.md) を参照。

**承認の目安:**
- 涼花みなせさんが実際に発言したフレーズである
- 意味・背景の説明が適切である

**却下の目安:**
- 発言の事実確認が取れない
- 不適切・誹謗中傷にあたる内容を含む
- すでに同じ語録が掲載済み（ただし新たな情報・補足がある場合は既存の語録に追記する）

---

## 権限の整理

| ロール | できること |
|--------|-----------|
| Admin（EuphorM） | PRのマージ、mainへの直接push |
| モデレーター（write） | Draft PRの確認・編集、ドラフト解除 |

---

## 語録を公開する手順

投稿Issueに`submission`ラベルが付与されると、GitHub Actionsが自動でブランチ・MDドラフト・Draft PRを作成する。モデレーターはそのDraft PRを確認・編集してからAdmin（EuphorM）のマージを待つ。

**1. Draft PRを確認する**

- IssueにコメントされたPRリンクを開く
- または `https://github.com/EuphorM/suzumina-jiten/pulls` のDraft PR一覧から探す

**2. Draft PRを編集する**

1. PRの「Files changed」タブを開く
2. `content/quotes/quote-issue-{番号}.md` の鉛筆アイコンをクリック
3. ファイル名をローマ字にリネーム（例: `mushikera.md`）
4. 内容（rarity・origin・解説文など）を確認・修正する
5. 「Commit directly to `quote/issue-{番号}`」でコミット

ファイル名の付け方・各フィールドの詳細は [コンテンツガイドライン](./content-guide.md) を参照。

**3. （任意）画像を追加する**

語録に画像を添付する場合は、`content/images/` フォルダにファイルをアップロードしてからMDファイルに以下の形式で追記する。

```markdown
![語録名](../images/ファイル名.jpg)
```

パスは必ず `../images/` の形式にすること（`/images/` ではない）。詳細は [コンテンツガイドライン](./content-guide.md) の「画像の追加方法」を参照。

**4. ドラフトを解除する**

PRページ下部の「Ready for review」ボタンをクリックする。

**5. Adminのマージを待つ**

Admin（EuphorM）が確認してマージする。マージ後、Vercelが自動でビルド・デプロイを開始する（1〜2分程度）。

---

### ローカルで編集する場合（エンジニア向け）

```bash
git fetch origin
git checkout quote/issue-XX   # 対象のIssue番号に合わせる

# ファイルを編集後
git add content/quotes/
git commit -m "fix: 語録「語録名」の内容を修正"
git push origin quote/issue-XX
# GitHub上でDraft PRを「Ready for review」に変更する
```

---

## 既存の語録に追記・修正する場合

投稿内容がすでに掲載済みの語録への追加情報である場合は、新規ファイルを作成せず既存ファイルを編集する。

1. `content/quotes/` から対象の `.md` ファイルを開く
2. 鉛筆アイコンをクリックして編集する
3. 情報を追記・修正してコミットする
4. Issueに「〇〇の語録に追記しました」とコメントしてクローズする

**よくある追記パターン（コピペ用）**

解説を追加する場合：
```markdown
## 解説

ここに背景・経緯を書く。
```

使用例を追加する場合：
```markdown
## 使用例

> セリフ<br>——YYYY年M月D日 涼花みなせ
```

関連リンクを追加する場合：
```markdown
## 関連リンク

- [配信タイトルなど](https://www.youtube.com/...)
```

由来を追加する場合（`---` の中に追記する）：
```yaml
origin: "[配信タイトル](https://www.youtube.com/...)（YYYY年M月D日）"
```

情報提供者を追加する場合（本文末尾に記載）：
```
情報提供者：ハンドルネーム
```

---

## Issueをクローズする手順

語録を公開したらIssueをクローズして管理する。

1. 対象のIssueを開く
2. コメント欄に「掲載しました」などひとこと入力（任意）
3. 「Close issue」ボタンをクリック

> コミットメッセージに `closes #Issue番号` と書いた場合は、プッシュ時にIssueが自動でクローズされる。

---

## 投稿を却下する場合

1. 対象のIssueを開く
2. 却下理由をコメントする（任意だが投稿者への配慮として推奨）
3. 「Close issue」ボタンをクリック

---

## よくある操作まとめ

| やること | 操作 |
|---|---|
| 投稿を確認する | GitHub Issues を開いて `submission` ラベルで絞り込む |
| Draft PRを確認する | IssueのPRリンク、またはPulls一覧からDraft PRを開く |
| Draft PRを編集する | PRの「Files changed」タブから鉛筆アイコンで編集 |
| ドラフトを解除する | PRページ下部「Ready for review」をクリック |
| 掲載済みの語録を修正する | `content/quotes/` の対象ファイルを開き、鉛筆アイコンから編集してPR作成 |
| Issueをクローズする | マージ時に自動クローズ（コミットに `closes #番号` が含まれるため） |
| 参加希望を確認する | GitHub Issues を開いて `参加希望` ラベルで絞り込む |
