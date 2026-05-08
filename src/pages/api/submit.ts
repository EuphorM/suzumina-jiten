export const prerender = false;

interface SubmissionBody {
  title: string;
  reading?: string;
  meaning: string;
  tags?: string;
  first_appearance?: string;
  youtube_url?: string;
  explanation?: string;
  examples?: string;
  contributor?: string;
}

function buildIssueBody(data: SubmissionBody): string {
  const lines: string[] = ['## 語録投稿', ''];

  lines.push(`**語録**: ${data.title}`);
  if (data.reading) lines.push(`**読み**: ${data.reading}`);
  lines.push(`**意味**: ${data.meaning}`);
  if (data.tags) lines.push(`**タグ**: ${data.tags}`);
  if (data.first_appearance) lines.push(`**初出**: ${data.first_appearance}`);
  if (data.youtube_url) lines.push(`**URL**: ${data.youtube_url}`);
  const contributors = data.contributor
    ? data.contributor.split(',').map((s) => s.trim()).filter(Boolean)
    : [];
  if (contributors.length > 0) lines.push(`**投稿者**: ${contributors.join(', ')}`);

  if (data.explanation) {
    lines.push('', '### 解説', data.explanation);
  }

  if (data.examples) {
    lines.push('', '### 使用例');
    data.examples
      .split('\n')
      .filter(Boolean)
      .forEach((ex) => lines.push(`- ${ex.trim()}`));
  }

  lines.push(
    '',
    '---',
    '*このIssueは投稿フォームから自動作成されました。*',
    '',
    '**モデレーター向け:** 内容を確認し、問題なければ `src/content/quotes/` にMarkdownファイルを作成してください。',
    '',
    '```yaml',
    '---',
    `title: ${data.title}`,
    `reading: ${data.reading ?? ''}`,
    `meaning: ${data.meaning}`,
    `tags: [${data.tags ?? ''}]`,
    'rarity: 1',
    data.first_appearance ? `first_appearance: "${data.first_appearance}"` : '',
    data.youtube_url ? `youtube_url: "${data.youtube_url}"` : '',
    contributors.length > 0 ? `contributor: [${contributors.map((c) => `"${c}"`).join(', ')}]` : '',
    '---',
    '',
    '## 解説',
    data.explanation ?? '（追記してください）',
    '',
    '## 使用例',
    data.examples
      ? data.examples
          .split('\n')
          .filter(Boolean)
          .map((ex) => `- ${ex.trim()}`)
          .join('\n')
      : '- （追記してください）',
    '```',
  );

  return lines.filter((l) => l !== undefined).join('\n');
}

export async function POST({ request }: { request: Request }) {
  const body: SubmissionBody = await request.json().catch(() => null);

  if (!body || !body.title?.trim() || !body.meaning?.trim()) {
    return Response.json({ error: '語録と意味は必須です' }, { status: 400 });
  }

  const token = import.meta.env.GITHUB_TOKEN;
  const owner = import.meta.env.GITHUB_OWNER;
  const repo = import.meta.env.GITHUB_REPO;

  if (!token || !owner || !repo) {
    console.error('GitHub環境変数が設定されていません');
    return Response.json({ error: 'サーバー設定エラーが発生しました' }, { status: 500 });
  }

  const issueRes = await fetch(`https://api.github.com/repos/${owner}/${repo}/issues`, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${token}`,
      'Content-Type': 'application/json',
      Accept: 'application/vnd.github.v3+json',
      'X-GitHub-Api-Version': '2022-11-28',
    },
    body: JSON.stringify({
      title: `[語録投稿] ${body.title.trim()}`,
      body: buildIssueBody(body),
      labels: ['submission'],
    }),
  });

  if (!issueRes.ok) {
    console.error('GitHub API error:', await issueRes.text());
    return Response.json({ error: '投稿に失敗しました。時間をおいて再度お試しください。' }, { status: 500 });
  }

  return Response.json({ success: true }, { status: 200 });
}
