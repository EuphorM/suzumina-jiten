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
    first_appearance: z.string().optional(),
    youtube_url: z.string().url().optional(),
    image: z.string().optional(),
    usage: z.array(z.string()).default([]),
    contributor: z.array(z.string()).default([]),
    youtube_urls: z.array(z.object({
      label: z.string(),
      url: z.string().url(),
    })).default([]),
  }),
});

export const collections = { quotes };
