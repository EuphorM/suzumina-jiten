import { defineCollection, z } from 'astro:content';
import { glob } from 'astro/loaders';

const quotes = defineCollection({
  loader: glob({ pattern: '**/*.md', base: './src/content/quotes' }),
  schema: z.object({
    title: z.string(),
    reading: z.string(),
    meaning: z.string(),
    tags: z.array(z.string()).default([]),
    rarity: z.number().min(1).max(5).default(1),
    first_appearance: z.preprocess((val) => val === '' ? undefined : val, z.string().optional()),
    first_date: z.preprocess((val) => val === '' ? undefined : val, z.string().optional()),
    youtube_url: z.preprocess((val) => val === '' ? undefined : val, z.string().url().optional()),
    image: z.preprocess((val) => val === '' ? undefined : val, z.string().optional()),
    usage: z.preprocess(
      (val) => Array.isArray(val) ? val.filter((s: any) => s !== '') : val,
      z.array(z.string()).default([])
    ),
    contributor: z.array(z.string()).default([]),
    youtube_urls: z.preprocess(
      (val) => Array.isArray(val) ? val.filter((item: any) => item?.url && item.url !== '') : val,
      z.array(z.object({ label: z.string(), url: z.string().url() })).default([])
    ),
  }),
});

export const collections = { quotes };
