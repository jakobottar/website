# Keep in sync with HUGO_VERSION in .github/workflows/ci.yml (CI passes it as a build arg).
ARG HUGO_VERSION=0.168.0

FROM ghcr.io/gohugoio/hugo:v${HUGO_VERSION} AS build

# The image defaults to a non-root user and a /project volume; this stage is
# throwaway, so build as root in a plain directory.
USER root
WORKDIR /site
COPY ./ /site
RUN hugo

FROM nginxinc/nginx-unprivileged:1.30-alpine AS runtime

ENV NGINX_ENTRYPOINT_QUIET_LOGS=true

COPY nginx.conf /etc/nginx/nginx.conf
COPY --from=build /site/public /usr/share/nginx/html

EXPOSE 8080
