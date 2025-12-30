# Diego Porres - ML Researcher Website

Personal academic website for Diego Porres, showcasing research in autonomous driving, computer vision, and generative AI.

## Tech Stack

- **Framework**: Jekyll 4.2.0 (GitHub Pages)
- **Theme**: Jekyll Theme Cayman (heavily customized)
- **Typography**: Inter (body), JetBrains Mono (code)
- **Math**: KaTeX
- **Plots**: Plotly.js
- **Diagrams**: TikZJax

## Site Structure

```
/                       # Homepage - hero, featured research, recent posts
/research/              # Research & publications page
/projects/              # Open-source projects page
/YYYY/MM/DD/post-name/  # Blog posts
```

## Key Files

### Data Files
- `_data/papers.yml` - Research papers with thumbnails, links, BibTeX
- `_data/projects.yml` - GitHub projects with star counts, thumbnails

### Includes
- `_includes/head-custom.html` - All custom CSS and JavaScript (inline)
- `_includes/paper-card.html` - Paper card component with thumbnail support
- `_includes/project-card.html` - Project card component with star count
- `_includes/page-header.html` - Header with mobile hamburger menu
- `_includes/plotly.html` - Interactive plot embedding
- `_includes/video-embed.html` - YouTube/Vimeo/local video embedding

### Pages
- `index.html` - Homepage with hero section
- `research/index.html` - Research papers by year
- `projects/index.html` - Projects by category

## Adding Content

### Add a New Paper
Edit `_data/papers.yml`:
```yaml
- title: "Paper Title"
  authors: ["Author 1", "Diego Porres", "Author 3"]
  venue: "Conference Full Name"
  venue_short: "CONF 2025"
  year: 2025
  abstract: "Brief description"
  pdf: "https://arxiv.org/pdf/..."
  arxiv: "https://arxiv.org/abs/..."
  code: "https://github.com/..."
  page: null  # or project page URL
  blog_post: null  # or /YYYY/MM/DD/post-name/
  tags: [autonomous-driving, computer-vision]
  featured: true
  thumbnail: "img/paper_thumbnails/filename.png"
```

### Add a New Project
Edit `_data/projects.yml`:
```yaml
- title: "Project Name"
  description: "Brief description"
  category: "Research Implementation"  # or "Tool", "Demo"
  tags: [tag1, tag2]
  github: "https://github.com/PDillis/repo"
  demo: null  # or demo URL
  paper: null  # or arxiv URL
  featured: true
  image: "img/project_thumbnails/filename.png"
```

### Add Thumbnails
1. Place images in `img/paper_thumbnails/` or `img/project_thumbnails/`
2. Recommended size: 360x240px for papers, 320x200px for projects
3. Use JPG/PNG format

### GitHub Star Counts
Star counts are fetched automatically from the GitHub API:
- Cached in localStorage for 24 hours to avoid rate limits
- Displays formatted counts (e.g., "1.2k" for 1200 stars)
- Falls back gracefully if API is unavailable

## Features

### Dark Mode
- Persists across page navigation via localStorage
- Uses `html.dark-mode` class for instant loading (no flash)
- Toggle button in header (☀️/🌙)
- Respects system preference on first visit

### Mobile Navigation
- Hamburger menu (☰) on screens < 768px
- Collapses all navigation into dropdown

### Code Blocks
- Copy button appears on hover
- Syntax highlighting via Rouge
- JetBrains Mono font

### Table of Contents
- Auto-generated for posts with 3+ headings
- Smooth scroll navigation
- Nested hierarchy for h2/h3 headings

### Lazy Loading
- Images in posts automatically get `loading="lazy"`
- Improves initial page load performance

### Math Rendering
- KaTeX for fast math rendering
- Supports `$...$` (inline) and `$$...$$` (display)

### Plotly Charts
```liquid
{% include plotly.html
   id="unique-id"
   data='[{"x": [1,2,3], "y": [4,5,6], "type": "scatter"}]'
   layout='{"title": "My Plot"}'
   caption="Optional caption"
%}
```

### Video Embedding
```liquid
{% include video-embed.html
   url="https://youtube.com/watch?v=VIDEO_ID"
   caption="Optional caption"
%}
```

## Local Development

```bash
# Install dependencies
bundle install

# Run local server
bundle exec jekyll serve

# View at http://localhost:4000
```

## Color Palette

### Light Mode
- Primary: `#2563eb` (Electric Blue)
- Secondary: `#0891b2` (Cyan)
- Accent: `#7c3aed` (Purple)
- Background: `#ffffff`
- Text: `#0f172a`

### Dark Mode
- Primary: `#3b82f6`
- Secondary: `#06b6d4`
- Accent: `#a78bfa`
- Background: `#0f172a` (Navy)
- Text: `#f1f5f9`

---

## Domain Migration Guide

### Migrating from blog.diegoporres.com to diegoporres.com

#### Step 1: Update DNS Records
In your domain registrar (e.g., Namecheap, Cloudflare, Google Domains):

1. Remove or update the CNAME record for `blog` subdomain
2. Add these records for the apex domain (`diegoporres.com`):
   ```
   Type: A      Name: @    Value: 185.199.108.153
   Type: A      Name: @    Value: 185.199.109.153
   Type: A      Name: @    Value: 185.199.110.153
   Type: A      Name: @    Value: 185.199.111.153
   Type: CNAME  Name: www  Value: pdillis.github.io
   ```

#### Step 2: Update GitHub Pages Settings
1. Go to repository Settings → Pages
2. Under "Custom domain", enter: `diegoporres.com`
3. Check "Enforce HTTPS" (may take a few minutes to become available)
4. GitHub will create/update the `CNAME` file

#### Step 3: Update _config.yml
```yaml
url: "https://diegoporres.com"
baseurl: ""  # Empty for apex domain
```

#### Step 4: Set Up Redirect from Old Domain
Option A: **DNS Redirect** (preferred)
- In your DNS provider, set up a redirect from `blog.diegoporres.com` to `diegoporres.com`

Option B: **Cloudflare Page Rule** (if using Cloudflare)
- Create a Page Rule: `blog.diegoporres.com/*` → `https://diegoporres.com/$1` (301 redirect)

#### Step 5: Update Internal Links
Search and replace in the codebase:
- `blog.diegoporres.com` → `diegoporres.com`
- Update any hardcoded URLs in papers.yml, projects.yml, etc.

#### Step 6: Update External References
- Google Scholar profile
- LinkedIn
- GitHub profile
- Any published papers with website links

#### Step 7: Verify
1. Visit `https://diegoporres.com` - should load the site
2. Visit `https://www.diegoporres.com` - should redirect to apex
3. Visit `https://blog.diegoporres.com` - should redirect to new domain
4. Check HTTPS certificate is valid

#### Step 8: Update Google Search Console
1. Add `diegoporres.com` as a new property
2. Submit new sitemap
3. Request indexing of key pages

---

## Changelog

### 2025-12-27: Expanded Search & Interactive Generative Models Post
- ✅ Added: Search now includes papers and projects (not just blog posts)
- ✅ Added: Type badges (Post, Paper, Project) in search results
- ✅ Added: Search filters to filter by content type
- ✅ Added: Author and venue search for papers
- ✅ Added: New blog post "Interactive 1D Generative Models" with TensorFlow.js
- ✅ Added: In-browser GAN training with real-time visualization

### 2025-12-14: GitHub Stars API, ToC & Optimizations
- ✅ Added: Dynamic GitHub star counts via API (cached 24h)
- ✅ Added: Auto-generated table of contents for long posts
- ✅ Added: Lazy loading for images
- ✅ Changed: Site title to "Diego Porres" (professional)
- ✅ Fixed: Paper/project thumbnails now display correctly
- ✅ Fixed: Theme persistence across page navigation (uses `html.dark-mode`)
- ✅ Added: Mobile hamburger menu for better navigation
- ✅ Fixed: papers.yml syntax error
- ✅ Created: `project-card.html` include component

### 2025-12-09: Phase 3 Complete - Research Showcase
- ✅ Research page with paper cards
- ✅ Projects page with project cards
- ✅ Homepage redesign with hero section
- ✅ Enhanced navigation

### 2025-12-08: Phase 1-2 Complete
- ✅ Modern typography (Inter, JetBrains Mono)
- ✅ Color system with CSS variables
- ✅ Dark mode with persistence
- ✅ Blue gradient header
- ✅ Plotly.js integration
- ✅ Video embedding
- ✅ Reading time estimates

---

## TODO

- [ ] Improve syntax highlighting colors for dark mode
- [x] Consider adding search functionality (expanded to include papers/projects with type badges)
- [ ] Anchor table of contents to left side of screen (sticky TOC that follows scroll)
- [ ] Make TOC collapsible
