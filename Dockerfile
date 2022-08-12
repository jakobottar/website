FROM alpine:latest as build

RUN apk update && apk add hugo

WORKDIR /site
COPY ./ /site
RUN hugo

# Gen default self signed certs
# Prod should replace these certs
FROM alpine:latest as certs
RUN apk update \
  && apk add openssl

RUN openssl req -x509 -nodes -days 365 -newkey rsa:2048 -subj '/CN=localhost' -keyout /selfsigned.key -out /selfsigned.crt

FROM nginxinc/nginx-unprivileged:alpine as runtime

# # Temp fix for CVE-2021-22945 issues. Should be fixed in next nginx release. https://github.com/alpinelinux/docker-alpine/issues/204
# RUN apk add --update --no-cache 'libcurl>=7.79.0-r0' && \
#     apk add --update --no-cache 'curl>=7.79.0-r0'
# RUN rm /etc/nginx/conf.d/default.conf

# Suppress "10-listen-on-ipv6-by-default.sh: info: /etc/nginx/conf.d/default.conf differs from the packaged version" warning
# We override default.conf so of course it's different.
ENV NGINX_ENTRYPOINT_QUIET_LOGS=true

COPY --from=certs --chown=nginx:nginx /selfsigned.key /etc/ssl/private/cert.key
COPY --from=certs --chown=nginx:nginx /selfsigned.crt /etc/ssl/certs/cert.crt
COPY --from=build --chown=nginx:nginx /site/nginx.conf /etc/nginx/nginx.conf
COPY --from=build /site/public /usr/share/nginx/html
# COPY --from=build /robots.txt /usr/share/nginx/html/robots.txt

# TODO: use non-root user
EXPOSE 8080
EXPOSE 4443