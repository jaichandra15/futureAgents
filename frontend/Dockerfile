# Multi-stage build for Next.js 14 frontend
# Optimized for standalone output with Tailwind CSS

# Stage 1: Dependencies
FROM node:20-alpine AS deps

WORKDIR /app

# Install libc compatibility for native modules
RUN apk add --no-cache libc6-compat

# Copy package files and install dependencies
COPY package.json package-lock.json* yarn.lock* ./

RUN if [ -f yarn.lock ]; then yarn install --frozen-lockfile; \
    elif [ -f package-lock.json ]; then npm ci; \
    else npm install; fi

# Stage 2: Builder
FROM node:20-alpine AS builder

WORKDIR /app

# Install libc compatibility
RUN apk add --no-cache libc6-compat

# Copy dependencies from first stage
COPY --from=deps /app/node_modules ./node_modules

# Copy application code
COPY . .

# Build time environment variables
ENV NEXT_PUBLIC_API_URL=http://localhost:8000
ENV NEXT_TELEMETRY_DISABLED=1

# Build the Next.js application
RUN npm run build

# Stage 3: Production runtime
FROM node:20-alpine AS runner

WORKDIR /app

ENV NODE_ENV=production
ENV NEXT_TELEMETRY_DISABLED=1
ENV PORT=3000
ENV HOSTNAME=0.0.0.0

RUN addgroup --system --gid 1001 nodejs && \
    adduser --system --uid 1001 nextjs

# Copy public assets
COPY --from=builder --chown=nextjs:nodejs /app/public ./public

# Copy Next.js standalone output and static files
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static

USER nextjs

EXPOSE 3000

CMD ["node", "server.js"]
