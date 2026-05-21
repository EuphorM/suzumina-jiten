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
    updated_at: z.string().optional(),
    first_appearance: z.preprocess((val) => val === '' ? undefined : val, z.string().optional()),
    first_date: z.preprocess((val) => val === '' ? undefined : val, z.string().optional()),
    first_appearance_url: z.preprocess((val) => val === '' ? undefined : val, z.string().url().optional()),
    image: z.preprocess((val) => val === '' ? undefined : val, z.string().optional()),
    usage: z.preprocess(
      (val) => {
        if (!Array.isArray(val)) return val;
        return val
          .map((item: any) => {
            if (typeof item === 'string') return item === '' ? null : { text: item };
            return item?.text ? item : null;
          })
          .filter(Boolean);
      },
      z.array(z.object({
        text: z.string(),
        date: z.preprocess((val) => val === '' ? undefined : val, z.string().optional()),
      })).default([])
    ),
    contributor: z.array(z.string()).default([]),
    related_links: z.preprocess(
      (val) => Array.isArray(val) ? val.filter((item: any) => item?.url && item.url !== '') : val,
      z.array(z.object({ label: z.string(), url: z.string().url() })).default([])
    ),
  }),
});

const quotesMarkdown = defineCollection({
  loader: glob({ pattern: '**/*.md', base: './content/quotes' }),
  schema: z.object({
    title: z.string(),
    reading: z.string(),
    meaning: z.string(),
    tags: z.array(z.string()).default([]),
    rarity: z.number().min(1).max(5).default(1),
    updated_at: z.string().optional(),
  }),
});

export const collections = { quotes, quotesMarkdown };
