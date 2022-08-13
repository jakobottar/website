FROM alpine:latest as build

RUN apk update && apk add hugo

WORKDIR /site
COPY ./ /site
RUN hugo

FROM nginxinc/nginx-unprivileged:alpine as runtime

ENV NGINX_ENTRYPOINT_QUIET_LOGS=true

COPY --from=build --chown=nginx:nginx /site/nginx.conf /etc/nginx/nginx.conf
COPY --from=build /site/public /usr/share/nginx/html

# TODO: use non-root user
EXPOSE 8080
